use crate::qwen::audio::AudioProcessor;
use crate::qwen::backend::DecoderBackend;
use crate::qwen::error::QwenError;
use llama_cpp_sys_2 as ll;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::ffi::CString;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedToken {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub units: Vec<AlignedToken>,
    pub elapsed_ms: u64,
    /// 对齐质量: "ForcedAligned" (真实 GGUF 对齐, 可能含部分段级线性回退)
    /// / "LinearFallback"
    pub align_quality: String,
}

/// 单次对齐推理产物: 词在给定音频轴上的本地时间 + 置信度 
#[derive(Debug, Clone)]
struct AlignUnit {
    /// 词表索引
    word_idx: usize,
    start_local_ms: u64,
    end_local_ms: u64,
    /// 两个 <timestamp> softmax 峰值置信度的最小值
    confidence: f32,
}

/// 锚点置信度: 词置信 >= 此值视为可信锚点
const CONF_ANCHOR: f32 = 0.3;
/// 低置信阈值: 词置信 < 此值视为可疑 (文本与音频大概率不一致)
const CONF_LOW: f32 = 0.05;
/// 整块低置信词比例上限 (超过 -> 块级结果不可信, 走逐段重对齐)
const LOW_CONF_RATIO_GLOBAL: f32 = 0.6;
/// 段级低置信词比例上限 (超过 -> 该段重对齐结果不可信, 回退线性)
const LOW_CONF_RATIO_SPAN: f32 = 0.5;
/// 词时长下限 (低于视为挤压错位)
const DUR_MIN_MS: u64 = 20;
/// 段级重对齐的最小段时长 (过短的段不值得重对齐)
const SPAN_MIN_REFINE_MS: u64 = 400;

/// 拼接音频 -> 媒体时间轴的分段映射。
/// VAD 切出的短段拼接成块时, 中间的静音被过滤, 拼接轴的局部时间
/// 必须通过此映射还原到原视频时间轴。
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct TimelineSpan {
    /// 本段在拼接音频中的采样起点 (16k)
    pub concat_start_sample: usize,
    /// 本段在拼接音频中的采样终点 (16k)
    pub concat_end_sample: usize,
    /// 本段在媒体时间轴上的起点 (ms)
    pub source_start_ms: i64,
    /// 本段在媒体时间轴上的终点 (ms)
    pub source_end_ms: i64,
}

/// 对齐器音频编码器参数 (与转换模型一致)
const CHUNK_FRAMES: usize = 100; // 每块 mel 帧数
const N_MELS: usize = 128;
const D_MODEL: usize = 1024; // 对齐器 audio 特征维度
/// 时间戳类数上限 (HaujetZhao 参考实现取 logits[:4000]; 4000×80ms=320s)
const TIMESTAMP_CLASSES: usize = 4000;
/// 每 100 mel 帧的 audio token 数
const TOKENS_PER_CHUNK: usize = 13;

static BACKEND_INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

extern "C" fn llama_log_callback(
    level: ll::ggml_log_level,
    text: *const std::os::raw::c_char,
    _user_data: *mut std::ffi::c_void,
) {
    if text.is_null() {
        return;
    }
    let cstr = unsafe { std::ffi::CStr::from_ptr(text) };
    let s = cstr.to_string_lossy();
    let tag = match level {
        ll::GGML_LOG_LEVEL_ERROR => "[aligner-llm:ERROR]",
        ll::GGML_LOG_LEVEL_WARN => "[aligner-llm:WARN] ",
        _ => "[aligner-llm]     ",
    };
    print!("{} {}", tag, s);
    if !s.ends_with('\n') {
        println!();
    }
}

fn ensure_backend_init() {
    BACKEND_INITIALIZED.get_or_init(|| unsafe {
        ll::llama_log_set(Some(llama_log_callback), std::ptr::null_mut());
        ll::llama_backend_init();
    });
}

/// 运行时探测是否有可用的 GPU 设备 (与 decoder 的判定一致)。
/// 后端编译进 cdylib ≠ 运行时驱动可用; 无设备时 llama.cpp 会把 offload 层数归零,
/// 模型实际跑在 CPU。此检查用于决定 GGUF 是否真正 offload。
fn gpu_device_available() -> bool {
    unsafe {
        !ll::ggml_backend_dev_by_type(ll::GGML_BACKEND_DEVICE_TYPE_GPU).is_null()
            || !ll::ggml_backend_dev_by_type(ll::GGML_BACKEND_DEVICE_TYPE_IGPU).is_null()
    }
}

/// Qwen3-ForcedAligner-0.6B (GGUF 版) Rust 推理:
///   - frontend ONNX: 100 mel 帧 -> 13 个 audio token (int4)
///   - backend ONNX:  块特征 + 全零注意力掩码 -> 1024 维音频特征 (int4)
///   - GGUF LLM:      文本解码器 (qwen3vl, 28 层, 输出 152064 类,
///                     前 4000 类为时间戳 0..3999, ×80ms)
///   - 时间戳: 每个词后两个 <timestamp> 标记, argmax × 80ms + LIS 修正
pub struct QwenAligner {
    frontend: Option<Session>,
    backend: Option<Session>,
    model: *mut ll::llama_model,
    context: *mut ll::llama_context,
    vocab: *const ll::llama_vocab,
    n_embd: i32,
    model_path: String,
    audio_start_id: i32,
    audio_end_id: i32,
    timestamp_id: i32,
    step_ms: f64,
    #[allow(dead_code)]
    pub use_gpu: bool,
}

unsafe impl Send for QwenAligner {}
unsafe impl Sync for QwenAligner {}

impl Drop for QwenAligner {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                ll::llama_free(self.context);
            }
            if !self.model.is_null() {
                ll::llama_model_free(self.model);
            }
        }
    }
}

impl QwenAligner {
    pub fn load(model_dir: &str, decoder_backend: DecoderBackend) -> Result<Self, QwenError> {
        let dir = Path::new(model_dir);
        let fe_path = dir.join("qwen3_aligner_encoder_frontend.int4.onnx");
        let be_path = dir.join("qwen3_aligner_encoder_backend.int4.onnx");

        // 1. ONNX 编码器 (frontend + backend)
        //    显式注册 CPU EP 并禁用 arena: 推理中间缓冲用完即释放, 避免 arena 保留峰值内存
        let (frontend, backend) = if fe_path.exists() && be_path.exists() {
            let cpu_ep = || ort::ep::CPU::default().with_arena_allocator(false).build();
            let fe = Session::builder()
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend builder: {}", e)))?
                .with_intra_threads(2)
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend intra_threads: {}", e)))?
                .with_execution_providers([cpu_ep()])
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend EP: {}", e)))?
                .commit_from_file(&fe_path)
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend commit: {}", e)))?;
            let be = Session::builder()
                .map_err(|e| QwenError::OnnxError(format!("aligner backend builder: {}", e)))?
                .with_intra_threads(2)
                .map_err(|e| QwenError::OnnxError(format!("aligner backend intra_threads: {}", e)))?
                .with_execution_providers([cpu_ep()])
                .map_err(|e| QwenError::OnnxError(format!("aligner backend EP: {}", e)))?
                .commit_from_file(&be_path)
                .map_err(|e| QwenError::OnnxError(format!("aligner backend commit: {}", e)))?;
            (Some(fe), Some(be))
        } else {
            (None, None)
        };

        // 2. GGUF LLM (完整词表, 含 <timestamp>)
        ensure_backend_init();
        let gguf_names = [
            "qwen3_aligner_llm.q4_k.gguf",
            "qwen3_aligner_llm.gguf",
            "aligner_llm.gguf",
        ];
        let chosen = gguf_names
            .iter()
            .map(|f| dir.join(f))
            .find(|p| p.exists())
            .ok_or_else(|| QwenError::ModelNotFound(format!("No aligner GGUF in {}", model_dir)))?;
        let chosen_str = chosen.to_string_lossy().to_string();

        let supports_offload = unsafe { ll::llama_supports_gpu_offload() };
        let use_gpu = !matches!(decoder_backend, DecoderBackend::Cpu)
            && supports_offload
            && gpu_device_available();

        let path_c = CString::new(chosen_str.clone())
            .map_err(|e| QwenError::DecoderError(format!("aligner path CString: {}", e)))?;
        let mut model_params = unsafe { ll::llama_model_default_params() };
        model_params.n_gpu_layers = if use_gpu { -1 } else { 0 };
        model_params.use_mmap = true;
        let model = unsafe { ll::llama_model_load_from_file(path_c.as_ptr(), model_params) };
        if model.is_null() {
            return Err(QwenError::DecoderError(format!(
                "aligner llama_model_load_from_file returned NULL for {}",
                chosen_str
            )));
        }
        let vocab = unsafe { ll::llama_model_get_vocab(model) };
        let n_embd = unsafe { ll::llama_model_n_embd(model) };
        println!(
            "[aligner] loaded GGUF: {} (n_embd={}, vocab={}, gpu={})",
            chosen_str,
            n_embd,
            unsafe { ll::llama_vocab_n_tokens(vocab) },
            use_gpu
        );

        let mut ctx_params = unsafe { ll::llama_context_default_params() };
        // 对齐序列 ≈ audio tokens (45s≈585) + 词 tokens + 双 <timestamp> 标记, 2048 足够;
        // GPU 模式下 KV 在显存, 4096→2048 使显存 KV 占用减半
        ctx_params.n_ctx = 2048;
        ctx_params.n_batch = 2048;
        ctx_params.n_ubatch = 512;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = 8;
        ctx_params.n_threads_batch = 8;
        ctx_params.flash_attn_type = ll::LLAMA_FLASH_ATTN_TYPE_ENABLED as _;
        ctx_params.pooling_type = ll::LLAMA_POOLING_TYPE_NONE as _;
        ctx_params.embeddings = false;
        let context = unsafe { ll::llama_init_from_model(model, ctx_params) };
        if context.is_null() {
            unsafe { ll::llama_model_free(model) };
            return Err(QwenError::DecoderError(
                "aligner llama_init_from_model returned NULL".into(),
            ));
        }

        // 3. 特殊 token id (词表含完整 added tokens)
        let audio_start_id = Self::tokenize_one(vocab, "<|audio_start|>");
        let audio_end_id = Self::tokenize_one(vocab, "<|audio_end|>");
        let timestamp_id = Self::tokenize_one(vocab, "<timestamp>");
        println!(
            "[aligner] special ids: audio_start={} audio_end={} timestamp={}",
            audio_start_id, audio_end_id, timestamp_id
        );

        Ok(Self {
            frontend,
            backend,
            model,
            context,
            vocab,
            n_embd,
            model_path: chosen_str,
            audio_start_id,
            audio_end_id,
            timestamp_id,
            step_ms: 80.0,
            use_gpu,
        })
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    /// 用对齐器 GGUF 词表对文本做 BPE 编码 (parse_special=true)
    fn tokenize(&self, text: &str) -> Result<Vec<ll::llama_token>, QwenError> {
        let text_c = CString::new(text)
            .map_err(|e| QwenError::DecoderError(format!("aligner tokenize CString: {}", e)))?;
        let len_i32 = text.len() as i32;
        let needed = unsafe {
            ll::llama_tokenize(
                self.vocab,
                text_c.as_ptr(),
                len_i32,
                std::ptr::null_mut(),
                0,
                false,
                true,
            )
        };
        let n_needed = needed.abs() as usize;
        if n_needed == 0 {
            return Ok(Vec::new());
        }
        let mut tokens = vec![0_i32; n_needed + 8];
        let rc = unsafe {
            ll::llama_tokenize(
                self.vocab,
                text_c.as_ptr(),
                len_i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                false,
                true,
            )
        };
        let n = rc.abs() as usize;
        tokens.truncate(n);
        Ok(tokens)
    }

    fn tokenize_one(vocab: *const ll::llama_vocab, text: &str) -> i32 {
        let text_c = CString::new(text).unwrap_or_default();
        let len_i32 = text.len() as i32;
        let needed = unsafe {
            ll::llama_tokenize(
                vocab,
                text_c.as_ptr(),
                len_i32,
                std::ptr::null_mut(),
                0,
                false,
                true,
            )
        };
        let n_needed = needed.abs() as usize;
        if n_needed == 0 {
            return -1;
        }
        let mut tokens = vec![0_i32; n_needed + 8];
        let rc = unsafe {
            ll::llama_tokenize(
                vocab,
                text_c.as_ptr(),
                len_i32,
                tokens.as_mut_ptr(),
                tokens.len() as i32,
                false,
                true,
            )
        };
        let n = rc.abs() as usize;
        if n > 0 {
            tokens[0]
        } else {
            -1
        }
    }

    /// 真实强制对齐 (GGUF 路线)。ONNX/GGUF 缺失时回退线性字符平分。
    /// `timeline` 把拼接音频的局部时间映射回媒体时间轴。
    ///
    /// 鲁棒性策略 (ASR 文本与音频不一致时, 防止单点错误污染整块):
    ///   1. 先做块级对齐, 每个词附带时间戳 softmax 置信度;
    ///   2. 块级结果健康时: 按时间中点把词归属到各 VAD 段, 仅对不健康的段局部重对齐;
    ///   3. 块级结果不健康时: 所有段逐段独立重对齐 (文本缺失导致的未覆盖段
    ///      由管线层"未覆盖段重解"补齐, 这里绝不把词硬塞进无关音频);
    ///   4. 段级重对齐仍不健康时: 该段回退线性平分。
    ///   错误因此被限制在单个 VAD 段 (秒级) 内, 不再波及整块。
    pub fn align(
        &mut self,
        samples_16k: &[f32],
        text: &str,
        segment_start_ms: u64,
        segment_end_ms: u64,
        _language: Option<&str>,
        timeline: &[TimelineSpan],
        cancel: &AtomicBool,
    ) -> Result<AlignmentResult, QwenError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(QwenError::Cancelled);
        }
        let start_time = std::time::Instant::now();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
                align_quality: "ForcedAligned".into(),
            });
        }

        if self.frontend.is_none() || self.backend.is_none() {
            return Ok(AlignmentResult {
                units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                elapsed_ms: 0,
                align_quality: "LinearFallback".into(),
            });
        }
        if self.model.is_null() || self.context.is_null() || self.audio_start_id < 0 {
            return Ok(AlignmentResult {
                units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                elapsed_ms: 0,
                align_quality: "LinearFallback".into(),
            });
        }

        // 1. 词表切分 (CJK 逐字, 拉丁按词; 日语/韩语逐字符回退)
        let words = tokenize_for_align(trimmed, _language);
        if words.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
                align_quality: "ForcedAligned".into(),
            });
        }

        // 2. 块级对齐 (拼接轴本地时间 + 每词置信度)
        let block_pass = match self.align_once(samples_16k, &words, cancel) {
            Ok(units) => units,
            Err(QwenError::Cancelled) => return Err(QwenError::Cancelled),
            Err(e) => {
                println!(
                    "[aligner] block pass failed ({}), whole-block linear fallback",
                    e
                );
                return Ok(AlignmentResult {
                    units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                    elapsed_ms: start_time.elapsed().as_millis() as u64,
                    align_quality: "LinearFallback".into(),
                });
            }
        };

        // 3. 无多段时间轴 (VAD 关闭/单段块): 块级健康直接用, 否则线性平分
        if timeline.len() <= 1 {
            if sequence_healthy(&block_pass, LOW_CONF_RATIO_GLOBAL) {
                let items = block_pass
                    .iter()
                    .map(|u| AlignedToken {
                        text: words[u.word_idx].clone(),
                        start_ms: map_to_media(timeline, u.start_local_ms),
                        end_ms: map_to_media(timeline, u.end_local_ms),
                        confidence: Some(u.confidence),
                    })
                    .collect();
                return Ok(AlignmentResult {
                    units: reconcile(trimmed, items),
                    elapsed_ms: start_time.elapsed().as_millis() as u64,
                    align_quality: "ForcedAligned".into(),
                });
            }
            return Ok(AlignmentResult {
                units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                elapsed_ms: start_time.elapsed().as_millis() as u64,
                align_quality: "LinearFallback".into(),
            });
        }

        // 4. 多段块: 始终按时间中点归属到各段。全局健康时保留健康段的块级时间,
        //    否则所有段逐段重对齐。不做按段时长比例分配词
        let n_words = words.len();
        let mids: Vec<u64> = block_pass
            .iter()
            .map(|u| (u.start_local_ms + u.end_local_ms) / 2)
            .collect();
        let attributed = attribute_by_time(&mids, timeline);
        let global_healthy = sequence_healthy(&block_pass, LOW_CONF_RATIO_GLOBAL);

        // 5. 逐段处理: 健康段保留块级时间, 其余段局部重对齐 (失败则该段线性平分)
        let mut slots: Vec<Option<AlignedToken>> = vec![None; n_words];
        let (mut kept, mut refined, mut linear_spans) = (0usize, 0usize, 0usize);
        for (si, span) in timeline.iter().enumerate() {
            if cancel.load(Ordering::Relaxed) {
                return Err(QwenError::Cancelled);
            }
            let idxs = &attributed[si];
            if idxs.is_empty() {
                continue;
            }
            let span_ms = (span
                .concat_end_sample
                .saturating_sub(span.concat_start_sample) as u64)
                * 1000
                / 16000;

            // 健康块内, 归属段也健康 (或段过短不值得重对齐) -> 保留块级时间
            if global_healthy {
                let span_units: Vec<AlignUnit> = idxs.iter().map(|&i| block_pass[i].clone()).collect();
                if span_ms < SPAN_MIN_REFINE_MS
                    || sequence_healthy(&span_units, LOW_CONF_RATIO_SPAN)
                {
                    for &i in idxs {
                        let u = &block_pass[i];
                        slots[i] = Some(AlignedToken {
                            text: words[i].clone(),
                            start_ms: map_to_media(timeline, u.start_local_ms),
                            end_ms: map_to_media(timeline, u.end_local_ms),
                            confidence: Some(u.confidence),
                        });
                    }
                    kept += 1;
                    continue;
                }
            }

            // 局部重对齐: 只对齐本段音频 + 本段词表, 段间互不影响
            let span_words: Vec<String> = idxs.iter().map(|&i| words[i].clone()).collect();
            let sub_samples = span_samples(samples_16k, span);
            let refined_units = self.align_once(sub_samples, &span_words, cancel);
            let units: Vec<AlignedToken> = match refined_units {
                Ok(units) if sequence_healthy(&units, LOW_CONF_RATIO_SPAN) => {
                    refined += 1;
                    units
                        .iter()
                        .map(|u| {
                            let start_media = (span.source_start_ms + u.start_local_ms as i64)
                                .clamp(span.source_start_ms, span.source_end_ms);
                            let end_media = (span.source_start_ms + u.end_local_ms as i64)
                                .clamp(span.source_start_ms, span.source_end_ms);
                            AlignedToken {
                                text: span_words[u.word_idx].clone(),
                                start_ms: start_media as u64,
                                end_ms: end_media as u64,
                                confidence: Some(u.confidence),
                            }
                        })
                        .collect()
                }
                Err(QwenError::Cancelled) => return Err(QwenError::Cancelled),
                Ok(_) | Err(_) => {
                    linear_spans += 1;
                    linear_align(
                        &span_words.join(" "),
                        span.source_start_ms as u64,
                        span.source_end_ms as u64,
                    )
                }
            };
            for (k, &i) in idxs.iter().enumerate() {
                if let Some(tok) = units.get(k) {
                    slots[i] = Some(tok.clone());
                }
            }
        }

        // 6. 组装 (每词必有归属段, None 仅为防御性兜底)
        let mut items: Vec<AlignedToken> = Vec::with_capacity(n_words);
        for i in 0..n_words {
            match &slots[i] {
                Some(tok) => items.push(tok.clone()),
                None => items.push(AlignedToken {
                    text: words[i].clone(),
                    start_ms: segment_start_ms,
                    end_ms: segment_start_ms + 1,
                    confidence: None,
                }),
            }
        }
        items = reconcile(trimmed, items);

        let quality = if kept == 0 && refined == 0 && linear_spans > 0 {
            "LinearFallback"
        } else {
            "ForcedAligned"
        };
        println!(
            "[aligner] block aligned {} words in {}ms (global_healthy={}, kept={}, refined={}, linear_spans={})",
            words.len(),
            start_time.elapsed().as_millis(),
            global_healthy,
            kept,
            refined,
            linear_spans
        );

        Ok(AlignmentResult {
            units: items,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            align_quality: quality.into(),
        })
    }

    /// 单次对齐推理: 音频 + 词表 -> 词级本地时间戳与置信度。
    /// 不映射媒体时间轴; 词表为空返回空向量。
    fn align_once(
        &mut self,
        samples_16k: &[f32],
        words: &[String],
        cancel: &AtomicBool,
    ) -> Result<Vec<AlignUnit>, QwenError> {
        if words.is_empty() {
            return Ok(Vec::new());
        }
        let pass_start = std::time::Instant::now();

        // 1. mel (与 encoder 相同的归一化 log-mel; 行步长取返回的实际帧数
        //    = samples.len()/160 + 1, 有效列 = samples.len()/160)
        let (mel, mel_stride) = AudioProcessor::log_mel(samples_16k)?;
        let n_frames = samples_16k.len() / 160;
        if n_frames == 0 {
            return Ok(Vec::new());
        }
        let audio_end_ms = (n_frames * 10) as i64;

        // 2-3. ONNX 编码器 (mel -> frontend -> backend), 借用在块结束后释放
        let features = {
            let frontend = self.frontend.as_mut().unwrap();
            let backend = self.backend.as_mut().unwrap();

            // 2. frontend: 100 帧/块 -> 13 token/块, 拼接后按有效长度截断
            let pad_len = (CHUNK_FRAMES - (n_frames % CHUNK_FRAMES)) % CHUNK_FRAMES;
            let t_padded = n_frames + pad_len;
            let n_chunks = t_padded / CHUNK_FRAMES;
            let n_audio = feat_output_lengths(n_frames as i64) as usize;
            let mut hidden: Vec<f32> = Vec::with_capacity(n_chunks * TOKENS_PER_CHUNK * D_MODEL);
            for c in 0..n_chunks {
                if cancel.load(Ordering::Relaxed) {
                    return Err(QwenError::Cancelled);
                }
                let base = c * CHUNK_FRAMES;
                let mut chunk = vec![0.0f32; N_MELS * CHUNK_FRAMES];
                for m in 0..N_MELS {
                    let src = m * mel_stride;
                    let dst = m * CHUNK_FRAMES;
                    for t in 0..CHUNK_FRAMES {
                        let idx = base + t;
                        chunk[dst + t] = if idx < n_frames { mel[src + idx] } else { 0.0 };
                    }
                }
                let chunk_t = Tensor::<f32>::from_array((
                    [1i64, N_MELS as i64, CHUNK_FRAMES as i64],
                    chunk.into_boxed_slice(),
                ))
                .map_err(|e| QwenError::OnnxError(format!("aligner chunk tensor: {}", e)))?;
                let out = frontend
                    .run(ort::inputs!["chunk_mel" => chunk_t])
                    .map_err(|e| QwenError::OnnxError(format!("aligner frontend run: {:?}", e)))?;
                let mut iter = out.iter();
                let (_, value) = iter
                    .next()
                    .ok_or_else(|| QwenError::OnnxError("aligner frontend: no output".into()))?;
                let (_, data) = value
                    .try_extract_tensor::<f32>()
                    .map_err(|e| QwenError::OnnxError(format!("aligner frontend extract: {}", e)))?;
                hidden.extend_from_slice(data);
            }
            hidden.truncate(n_audio * D_MODEL);
            if hidden.len() != n_audio * D_MODEL {
                return Err(QwenError::OnnxError(format!(
                    "aligner: frontend features {} < n_audio {}",
                    hidden.len() / D_MODEL,
                    n_audio
                )));
            }

            // 3. backend: 全零注意力掩码 (全局注意力, 与参考实现一致)
            let hidden_t = Tensor::<f32>::from_array((
                [1i64, n_audio as i64, D_MODEL as i64],
                hidden.into_boxed_slice(),
            ))
            .map_err(|e| QwenError::OnnxError(format!("aligner hidden tensor: {}", e)))?;
            let mask = vec![0.0f32; n_audio * n_audio];
            let mask_t = Tensor::<f32>::from_array((
                [1i64, 1i64, n_audio as i64, n_audio as i64],
                mask.into_boxed_slice(),
            ))
            .map_err(|e| QwenError::OnnxError(format!("aligner mask tensor: {}", e)))?;
            let out = backend
                .run(ort::inputs![
                    "hidden_states" => hidden_t,
                    "attention_mask" => mask_t,
                ])
                .map_err(|e| QwenError::OnnxError(format!("aligner backend run: {:?}", e)))?;
            let mut iter = out.iter();
            let (_, value) = iter
                .next()
                .ok_or_else(|| QwenError::OnnxError("aligner backend: no output".into()))?;
            let (_, data) = value
                .try_extract_tensor::<f32>()
                .map_err(|e| QwenError::OnnxError(format!("aligner backend extract: {}", e)))?;
            let mut features = data.to_vec();
            features.truncate(n_audio * D_MODEL);
            if features.len() != n_audio * D_MODEL {
                return Err(QwenError::OnnxError(format!(
                    "aligner: backend features {} < n_audio {}",
                    features.len() / D_MODEL,
                    n_audio
                )));
            }
            (features, n_audio, n_chunks)
        };
        let (features, n_audio, n_chunks) = features;

        // 4. 组装序列: [audio_start] + audio×N + [audio_end] + w1 + ts + ts + w2 + ts + ts ...
        //    ts 标记紧跟词后 (HaujetZhao GGUF 参考实现顺序)
        let mut post_ids: Vec<ll::llama_token> = vec![self.audio_end_id];
        let mut ts_pos_in_post: Vec<usize> = Vec::with_capacity(words.len() * 2);
        let mut post_len = 1usize; // audio_end
        for w in words {
            let word_tokens = self.tokenize(w)?;
            post_ids.extend_from_slice(&word_tokens);
            post_len += word_tokens.len();
            ts_pos_in_post.push(post_len);
            post_ids.push(self.timestamp_id);
            post_len += 1;
            ts_pos_in_post.push(post_len);
            post_ids.push(self.timestamp_id);
            post_len += 1;
        }

        // 5. llama 推理: 前缀 token + 音频嵌入 (4 轴 M-RoPE) + 后缀 token (ts 位置取 logits)
        let mem = unsafe { ll::llama_get_memory(self.context) };
        unsafe { ll::llama_memory_clear(mem, true) };

        self.submit_token_batch(&[self.audio_start_id], 0)?;
        let pos = 1usize;
        self.submit_embedding_batch(&features, n_audio, pos)?;
        let post_start = 1 + n_audio;
        self.submit_token_batch_logits(&post_ids, post_start, &ts_pos_in_post)?;

        // 6. 读取时间戳 logits: argmax(logits[:4000]) × 80ms + softmax 峰值置信度
        let mut raw_ts: Vec<f64> = Vec::with_capacity(ts_pos_in_post.len());
        let mut ts_conf: Vec<f32> = Vec::with_capacity(ts_pos_in_post.len());
        for &i in &ts_pos_in_post {
            let logits_ptr = unsafe { ll::llama_get_logits_ith(self.context, i as i32) };
            if logits_ptr.is_null() {
                return Err(QwenError::DecoderError(
                    "aligner: llama_get_logits_ith returned NULL".into(),
                ));
            }
            let logits = unsafe { std::slice::from_raw_parts(logits_ptr, TIMESTAMP_CLASSES) };
            let (argmax, conf) = timestamp_argmax_conf(logits);
            raw_ts.push(argmax as f64 * self.step_ms);
            ts_conf.push(conf);
        }

        // 7. LIS 修正 + 锚点插值 (低置信词用邻近高置信锚点插值, 阻断错误传播)
        let mut fixed = fix_timestamp(&raw_ts);
        let word_confs: Vec<f32> = words
            .iter()
            .enumerate()
            .map(|(i, _)| ts_conf[2 * i].min(ts_conf[2 * i + 1]))
            .collect();
        anchor_interpolate(&mut fixed, &word_confs, audio_end_ms);

        let units = words
            .iter()
            .enumerate()
            .map(|(i, _)| AlignUnit {
                word_idx: i,
                start_local_ms: fixed[2 * i].max(0) as u64,
                end_local_ms: fixed[2 * i + 1].max(0) as u64,
                confidence: word_confs[i],
            })
            .collect();

        println!(
            "[aligner] aligned {} words ({} frames, {} audio tokens, {} chunks) in {}ms",
            words.len(),
            n_frames,
            n_audio,
            n_chunks,
            pass_start.elapsed().as_millis()
        );
        Ok(units)
    }

    /// 提交 token 批次, 位置递增; 与解码器相同 (文本 token 的 M-RoPE 由 llama.cpp 自动处理)
    fn submit_token_batch(&mut self, tokens: &[ll::llama_token], start_pos: usize) -> Result<(), QwenError> {
        self.submit_token_batch_logits(tokens, start_pos, &[])
    }

    /// 提交 token 批次; `logits_pos` 为批次内需要输出 logits 的位置 (0-based)
    fn submit_token_batch_logits(
        &mut self,
        tokens: &[ll::llama_token],
        start_pos: usize,
        logits_pos: &[usize],
    ) -> Result<(), QwenError> {
        if tokens.is_empty() {
            return Ok(());
        }
        let n = tokens.len() as i32;
        let mut batch = unsafe { ll::llama_batch_init(n, 0, 1) };
        for (i, tok) in tokens.iter().enumerate() {
            unsafe {
                if !batch.token.is_null() {
                    *batch.token.add(i) = *tok;
                }
                if !batch.pos.is_null() {
                    *batch.pos.add(i) = (start_pos + i) as ll::llama_pos;
                }
                if !batch.n_seq_id.is_null() {
                    *batch.n_seq_id.add(i) = 1;
                }
                if !batch.seq_id.is_null() {
                    let seq_ptr = *batch.seq_id.add(i);
                    if !seq_ptr.is_null() {
                        *seq_ptr = 0;
                    }
                }
                if !batch.logits.is_null() {
                    *batch.logits.add(i) = if logits_pos.contains(&i) { 1 } else { 0 };
                }
            }
        }
        batch.n_tokens = n;
        let rc = unsafe { ll::llama_decode(self.context, batch) };
        unsafe { ll::llama_batch_free(batch) };
        if rc != 0 {
            return Err(QwenError::DecoderError(format!(
                "aligner llama_decode rc={}",
                rc
            )));
        }
        Ok(())
    }

    /// 将音频特征按 4 轴 M-RoPE 位置布局注入 (与解码器 submit_embedding_batch 相同)
    fn submit_embedding_batch(
        &mut self,
        embeddings: &[f32],
        n_tokens: usize,
        start_pos: usize,
    ) -> Result<(), QwenError> {
        if n_tokens == 0 || embeddings.is_empty() {
            return Ok(());
        }
        let n_embd = self.n_embd as usize;
        if embeddings.len() < n_tokens * n_embd {
            return Err(QwenError::DecoderError(format!(
                "aligner embeddings len ({}) < n_tokens ({}) * n_embd ({})",
                embeddings.len(),
                n_tokens,
                n_embd
            )));
        }
        let physical_batch = 256.min(n_tokens);
        const N_POS_PER_EMBD: usize = 4;
        for token_offset in (0..n_tokens).step_by(physical_batch) {
            let chunk_tokens = (n_tokens - token_offset).min(physical_batch);
            let n_alloc = chunk_tokens * N_POS_PER_EMBD;
            let mut batch = unsafe { ll::llama_batch_init(n_alloc as i32, self.n_embd, 1) };
            unsafe {
                if batch.embd.is_null() {
                    ll::llama_batch_free(batch);
                    return Err(QwenError::DecoderError(
                        "aligner batch: null embedding buffer".into(),
                    ));
                }
                std::ptr::copy_nonoverlapping(
                    embeddings.as_ptr().add(token_offset * n_embd),
                    batch.embd,
                    chunk_tokens * n_embd,
                );
                for i in 0..chunk_tokens {
                    if !batch.n_seq_id.is_null() {
                        *batch.n_seq_id.add(i) = 1;
                    }
                    if !batch.seq_id.is_null() {
                        let seq_ptr = *batch.seq_id.add(i);
                        if !seq_ptr.is_null() {
                            *seq_ptr = 0;
                        }
                    }
                    if !batch.logits.is_null() {
                        *batch.logits.add(i) = 0;
                    }
                }
                if !batch.pos.is_null() {
                    for i in 0..chunk_tokens {
                        let p = (start_pos + token_offset + i) as ll::llama_pos;
                        // M-RoPE 四轴布局与参考实现一致: T/H/W = 位置, E 轴 = 0
                        for j in 0..3 {
                            *batch.pos.add(j * chunk_tokens + i) = p;
                        }
                        *batch.pos.add(3 * chunk_tokens + i) = 0;
                    }
                }
                batch.n_tokens = chunk_tokens as i32;
            }
            let rc = unsafe { ll::llama_decode(self.context, batch) };
            batch.n_tokens = n_alloc as i32;
            unsafe { ll::llama_batch_free(batch) };
            if rc != 0 {
                return Err(QwenError::DecoderError(format!(
                    "aligner embedding decode rc={} at audio token {}",
                    rc, token_offset
                )));
            }
        }
        Ok(())
    }
}

/// 时间戳 logits 的 argmax + softmax 峰值置信度 (前 TIMESTAMP_CLASSES 类)。
fn timestamp_argmax_conf(logits: &[f32]) -> (usize, f32) {
    let mut max_v = f32::NEG_INFINITY;
    let mut argmax = 0usize;
    for (k, &v) in logits.iter().enumerate() {
        if v > max_v {
            max_v = v;
            argmax = k;
        }
    }
    let mut sum = 0.0f32;
    for &v in logits.iter() {
        sum += (v - max_v).exp();
    }
    (argmax, 1.0 / sum.max(1e-6))
}

/// 低置信词的时间戳用邻近高置信锚点线性插值:
/// 左右锚点都存在时在 [左锚词尾, 右锚词首] 内均分;
/// 单侧锚点时另一侧用序列端点 (0 / audio_end_ms)。
/// 锚点不足 2 个时不做任何处理 (由上层健康门控兜底)。
fn anchor_interpolate(times: &mut [i64], word_confs: &[f32], audio_end_ms: i64) {
    let n_words = word_confs.len();
    if times.len() != n_words * 2 || n_words == 0 {
        return;
    }
    let anchor_count = word_confs.iter().filter(|&&c| c >= CONF_ANCHOR).count();
    if anchor_count < 2 {
        return;
    }
    let mut i = 0usize;
    while i < n_words {
        if word_confs[i] >= CONF_LOW {
            i += 1;
            continue;
        }
        let mut j = i;
        while j < n_words && word_confs[j] < CONF_LOW {
            j += 1;
        }
        let left = (0..i).rev().find(|&k| word_confs[k] >= CONF_ANCHOR);
        let right = (j..n_words).find(|&k| word_confs[k] >= CONF_ANCHOR);
        let l = left.map(|k| times[2 * k + 1]).unwrap_or(0);
        let r = right.map(|k| times[2 * k]).unwrap_or(audio_end_ms);
        if r > l {
            let step = (r - l) as f64 / (j - i) as f64;
            for k in i..j {
                times[2 * k] = l + (step * (k - i) as f64).round() as i64;
                times[2 * k + 1] = l + (step * (k - i + 1) as f64).round() as i64;
            }
        } else {
            for k in i..j {
                times[2 * k] = l;
                times[2 * k + 1] = l;
            }
        }
        i = j;
    }
}

/// 序列健康度: 时间单调 + 词时长合理 + 低置信词比例不超过 `max_low_ratio`。
fn sequence_healthy(units: &[AlignUnit], max_low_ratio: f32) -> bool {
    if units.is_empty() {
        return true;
    }
    let mut prev_start = 0u64;
    let mut prev_end = 0u64;
    let mut durs: Vec<u64> = Vec::with_capacity(units.len());
    for u in units {
        if u.end_local_ms < u.start_local_ms {
            return false;
        }
        if u.start_local_ms < prev_start || u.end_local_ms < prev_end {
            return false;
        }
        prev_start = u.start_local_ms;
        prev_end = u.end_local_ms;
        durs.push(u.end_local_ms - u.start_local_ms);
    }
    if durs.iter().any(|&d| d < DUR_MIN_MS) {
        return false;
    }
    durs.sort_unstable();
    let median = durs[durs.len() / 2];
    let max_d = durs[durs.len() - 1];
    if max_d > median.saturating_mul(6).saturating_add(500) || max_d > 5000 {
        return false;
    }
    let low = units.iter().filter(|u| u.confidence < CONF_LOW).count();
    (low as f32) / (units.len() as f32) <= max_low_ratio
}

/// 按词时间中点 (拼接轴本地 ms) 归属到 VAD 段; 越界词钳到首/末段。
fn attribute_by_time(word_mid_ms: &[u64], spans: &[TimelineSpan]) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); spans.len()];
    if spans.is_empty() {
        return out;
    }
    for (wi, &mid_ms) in word_mid_ms.iter().enumerate() {
        let mid_sample = mid_ms.saturating_mul(16);
        let mut target = 0usize;
        if mid_sample >= spans[0].concat_start_sample as u64 {
            target = spans.len() - 1;
            for (si, s) in spans.iter().enumerate() {
                if mid_sample >= s.concat_start_sample as u64
                    && mid_sample < s.concat_end_sample as u64
                {
                    target = si;
                    break;
                }
            }
        }
        out[target].push(wi);
    }
    out
}

/// 取某段在拼接块内的音频切片 (越界钳制)。
fn span_samples<'a>(samples: &'a [f32], span: &TimelineSpan) -> &'a [f32] {
    let s = span.concat_start_sample.min(samples.len());
    let e = span.concat_end_sample.min(samples.len());
    if s < e {
        &samples[s..e]
    } else {
        &[]
    }
}

/// 音频卷积下采样后的 token 数: 每 100 mel 帧 13 个 (Python `//` 向下取整语义)
fn feat_output_lengths(input_lengths: i64) -> i64 {
    let leave = input_lengths % 100;
    let feat = (leave - 1).div_euclid(2) + 1;
    ((feat - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1 + (input_lengths / 100) * TOKENS_PER_CHUNK as i64
}

/// 把拼接音频上的局部时间 (ms, 相对块起点) 映射回媒体时间轴。
/// 时间落在某分段内时线性映射; 超出块范围时钳制到末段终点。
pub(crate) fn map_to_media(timeline: &[TimelineSpan], local_ms: u64) -> u64 {
    if timeline.is_empty() {
        return local_ms;
    }
    let sample_pos = local_ms.saturating_mul(16);
    for span in timeline {
        if sample_pos < span.concat_end_sample as u64 {
            let local_in_span = sample_pos.saturating_sub(span.concat_start_sample as u64);
            let media = span.source_start_ms + (local_in_span * 1000 / 16000) as i64;
            return media.clamp(span.source_start_ms, span.source_end_ms) as u64;
        }
    }
    timeline
        .last()
        .map(|s| s.source_end_ms as u64)
        .unwrap_or(local_ms)
}

fn is_kept_char(c: char) -> bool {
    if c == '\'' {
        return true;
    }
    c.is_alphabetic() || c.is_numeric()
}

pub(crate) fn is_cjk_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

/// 通用分词: 按空白切词, 过滤非字母/数字, CJK 逐字拆出 (中英混排适用)。
/// 日语/韩语: 官方使用 nagisa/soynlp 形态素分词, 无依赖时逐字符回退
/// (与 HaujetZhao 参考实现的 ImportError 回退一致)。
pub(crate) fn tokenize_for_align(text: &str, language: Option<&str>) -> Vec<String> {
    let lang = language.unwrap_or("").to_lowercase();
    let per_char = lang == "japanese" || lang == "korean";
    let mut tokens: Vec<String> = Vec::new();
    for seg in text.split_whitespace() {
        let cleaned: String = seg.chars().filter(|c| is_kept_char(*c)).collect();
        if cleaned.is_empty() {
            continue;
        }
        let mut buf = String::new();
        for ch in cleaned.chars() {
            if per_char || is_cjk_char(ch) {
                if !buf.is_empty() {
                    tokens.push(std::mem::take(&mut buf));
                }
                tokens.push(ch.to_string());
            } else {
                buf.push(ch);
            }
        }
        if !buf.is_empty() {
            tokens.push(buf);
        }
    }
    tokens
}

/// 官方 fix_timestamp (LIS): 正常点保留, 异常点 (≤2 个) 取邻近值, 多个线性插值
fn fix_timestamp(data: &[f64]) -> Vec<i64> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let mut dp = vec![1i32; n];
    let mut parent = vec![-1i32; n];
    for i in 1..n {
        for j in 0..i {
            if data[j] <= data[i] && dp[j] + 1 > dp[i] {
                dp[i] = dp[j] + 1;
                parent[i] = j as i32;
            }
        }
    }
    let max_len = dp.iter().copied().max().unwrap_or(1);
    let max_idx = dp.iter().position(|&x| x == max_len).unwrap_or(0);
    let mut lis = Vec::new();
    let mut idx = max_idx as i32;
    while idx != -1 {
        lis.push(idx as usize);
        idx = parent[idx as usize];
    }
    lis.reverse();
    let mut is_normal = vec![false; n];
    for &i in &lis {
        is_normal[i] = true;
    }
    let mut result = data.to_vec();
    let mut i = 0;
    while i < n {
        if !is_normal[i] {
            let mut j = i;
            while j < n && !is_normal[j] {
                j += 1;
            }
            let anomaly_count = j - i;
            let left_val = (0..i).rev().find(|&k| is_normal[k]).map(|k| result[k]);
            let right_val = (j..n).find(|&k| is_normal[k]).map(|k| result[k]);
            if anomaly_count <= 2 {
                for k in i..j {
                    result[k] = match (left_val, right_val) {
                        (None, Some(r)) => r,
                        (Some(l), None) => l,
                        (Some(l), Some(r)) => {
                            if (k - i + 1) as f64 <= (j - k) as f64 {
                                l
                            } else {
                                r
                            }
                        }
                        (None, None) => result[k],
                    };
                }
            } else {
                match (left_val, right_val) {
                    (Some(l), Some(r)) => {
                        let step = (r - l) / (anomaly_count as f64 + 1.0);
                        for k in i..j {
                            result[k] = l + step * (k - i + 1) as f64;
                        }
                    }
                    (Some(l), None) => {
                        for k in i..j {
                            result[k] = l;
                        }
                    }
                    (None, Some(r)) => {
                        for k in i..j {
                            result[k] = r;
                        }
                    }
                    (None, None) => {}
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    result.iter().map(|&v| v as i64).collect()
}

/// reconcile: 从原始文本找回标点/空格 (对齐项按原文形态重组, 标点为零时长项)
pub(crate) fn reconcile(original_text: &str, items: Vec<AlignedToken>) -> Vec<AlignedToken> {
    if items.is_empty() {
        return if original_text.trim().is_empty() {
            Vec::new()
        } else {
            vec![AlignedToken {
                text: original_text.to_string(),
                start_ms: 0,
                end_ms: 0,
                confidence: None,
            }]
        };
    }
    let orig: Vec<char> = original_text.chars().collect();
    let mut out: Vec<AlignedToken> = Vec::new();
    let mut curr_ptr = 0usize;
    let mut last_ts = items[0].start_ms;

    for item in &items {
        if let Some((s, e)) = find_token_indices(&orig, &item.text, curr_ptr) {
            if s > curr_ptr {
                let gap: String = orig[curr_ptr..s].iter().collect();
                out.push(AlignedToken {
                    text: gap,
                    start_ms: last_ts,
                    end_ms: last_ts,
                    confidence: None,
                });
            }
            let matched: String = orig[s..e].iter().collect();
            out.push(AlignedToken {
                text: matched,
                start_ms: item.start_ms,
                end_ms: item.end_ms,
                confidence: item.confidence,
            });
            curr_ptr = e;
            last_ts = item.end_ms;
        } else {
            out.push(item.clone());
            last_ts = item.end_ms;
        }
    }
    if curr_ptr < orig.len() {
        let tail: String = orig[curr_ptr..].iter().collect();
        out.push(AlignedToken {
            text: tail,
            start_ms: last_ts,
            end_ms: last_ts,
            confidence: None,
        });
    }
    out
}

/// 在原文中寻找 target 的最小区间, 允许穿插非保留字符 (返回 (start, end))
fn find_token_indices(
    orig: &[char],
    target: &str,
    start_index: usize,
) -> Option<(usize, usize)> {
    let target: Vec<char> = target.chars().collect();
    if target.is_empty() {
        return None;
    }
    let mut t_ptr = 0usize;
    let mut first_match: Option<usize> = None;
    let mut i = start_index;
    while i < orig.len() {
        let ch = orig[i];
        if ch == target[t_ptr] {
            if t_ptr == 0 {
                first_match = Some(i);
            }
            t_ptr += 1;
            if t_ptr == target.len() {
                return Some((first_match.unwrap_or(i), i + 1));
            }
        } else if is_kept_char(ch) {
            if first_match.is_some() {
                // 回退重试
                i = first_match.unwrap();
                first_match = None;
                t_ptr = 0;
            }
        }
        i += 1;
    }
    None
}

fn linear_align(text: &str, segment_start_ms: u64, segment_end_ms: u64) -> Vec<AlignedToken> {
    // 智能分词: CJK 逐字符, 西文按完整词(含空白/标点)切分
    let mut tokens: Vec<String> = Vec::new();
    let mut current_word = String::new();
    for ch in text.chars() {
        if is_cjk_char(ch) {
            if !current_word.is_empty() {
                tokens.push(std::mem::take(&mut current_word));
            }
            tokens.push(ch.to_string());
        } else if ch.is_whitespace() {
            current_word.push(ch);
            tokens.push(std::mem::take(&mut current_word));
        } else {
            current_word.push(ch);
        }
    }
    if !current_word.is_empty() {
        tokens.push(current_word);
    }

    let total_chars: usize = tokens.iter().map(|t| t.chars().count()).sum();
    let duration = segment_end_ms.saturating_sub(segment_start_ms);
    let step_per_char = if total_chars > 0 {
        duration as f64 / total_chars as f64
    } else {
        0.0
    };

    let mut units = Vec::with_capacity(tokens.len());
    let mut accumulated_chars = 0usize;
    let n = tokens.len();
    for (i, tok) in tokens.into_iter().enumerate() {
        let n_chars = tok.chars().count();
        let start = segment_start_ms + (accumulated_chars as f64 * step_per_char) as u64;
        accumulated_chars += n_chars;
        let end = if i + 1 == n {
            segment_end_ms
        } else {
            segment_start_ms + (accumulated_chars as f64 * step_per_char) as u64
        };
        units.push(AlignedToken {
            text: tok,
            start_ms: start,
            end_ms: end.max(start + 1),
            confidence: None,
        });
    }
    units
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feat_length_formula() {
        assert_eq!(feat_output_lengths(100), 13);
        assert_eq!(feat_output_lengths(50), 7);
        assert_eq!(feat_output_lengths(200), 26);
        assert_eq!(feat_output_lengths(1), 1);
    }

    #[test]
    fn cjk_tokenization() {
        assert_eq!(
            tokenize_for_align("今天你好 world", None),
            vec![
                "今".to_string(),
                "天".to_string(),
                "你".to_string(),
                "好".to_string(),
                "world".to_string()
            ]
        );
        assert_eq!(
            tokenize_for_align("Hello, world!", None),
            vec!["Hello".to_string(), "world".to_string()]
        );
        // 日语逐字符 (无形态素分词依赖时的回退)
        let ja = tokenize_for_align("こんにちは", Some("Japanese"));
        assert_eq!(ja.len(), 5);
    }

    #[test]
    fn timeline_mapping() {
        use super::TimelineSpan;
        // 语音 A 5s + 静音 8s + 语音 B 5s, 拼接轴只保留两段语音
        let spans = vec![
            TimelineSpan {
                concat_start_sample: 0,
                concat_end_sample: 80000,
                source_start_ms: 0,
                source_end_ms: 5000,
            },
            TimelineSpan {
                concat_start_sample: 80000,
                concat_end_sample: 160000,
                source_start_ms: 13000,
                source_end_ms: 18000,
            },
        ];
        assert_eq!(map_to_media(&spans, 0), 0);
        assert_eq!(map_to_media(&spans, 4000), 4000);
        // 第二段语音局部 0ms (拼接 5000ms 处) -> 媒体 13000ms
        assert_eq!(map_to_media(&spans, 5000), 13000);
        // 第二段语音局部 1s -> 媒体 14000ms
        assert_eq!(map_to_media(&spans, 6000), 14000);
        // 超出块尾 -> 末段终点
        assert_eq!(map_to_media(&spans, 12000), 18000);
    }

    #[test]
    fn fix_timestamp_works() {
        let data = vec![100.0, 200.0, 300.0, 400.0];
        assert_eq!(fix_timestamp(&data), vec![100, 200, 300, 400]);
        let data = vec![100.0, 5000.0, 300.0, 400.0];
        let fixed = fix_timestamp(&data);
        assert_eq!(fixed.len(), 4);
        assert!(fixed[1] <= fixed[2] || fixed[1] == fixed[2]);
    }

    #[test]
    fn reconcile_restores_punctuation() {
        // "你好世界。" -> 字对齐: 你/好/世/界 + 句号
        let items = vec![
            AlignedToken { text: "你".into(), start_ms: 0, end_ms: 100, confidence: None },
            AlignedToken { text: "好".into(), start_ms: 100, end_ms: 200, confidence: None },
            AlignedToken { text: "世".into(), start_ms: 200, end_ms: 300, confidence: None },
            AlignedToken { text: "界".into(), start_ms: 300, end_ms: 400, confidence: None },
        ];
        let out = reconcile("你好世界。", items);
        assert_eq!(out.len(), 5);
        assert_eq!(out[4].text, "。");
        assert_eq!(out[4].start_ms, 400);
    }

    #[test]
    fn timestamp_conf_peaked_and_flat() {
        let (argmax, conf) = timestamp_argmax_conf(&[0.0, 10.0, 0.0]);
        assert_eq!(argmax, 1);
        assert!(conf > 0.99);
        let (_, conf_flat) = timestamp_argmax_conf(&[0.0, 0.0, 0.0]);
        assert!((conf_flat - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn anchor_interpolate_repairs_low_conf_run() {
        let mut times = vec![0, 100, 100, 200, 200, 300, 300, 400, 400, 500];
        let confs = vec![0.9, 0.01, 0.01, 0.9, 0.9];
        anchor_interpolate(&mut times, &confs, 500);
        // 词 1..2 在锚点 0 (尾 100) 与锚点 3 (首 300) 之间均分
        assert_eq!(times[2], 100);
        assert_eq!(times[3], 200);
        assert_eq!(times[4], 200);
        assert_eq!(times[5], 300);
        // 锚点与尾部词不受影响
        assert_eq!(times[0], 0);
        assert_eq!(times[1], 100);
        assert_eq!(times[9], 500);
    }

    #[test]
    fn anchor_interpolate_trailing_run_uses_audio_end() {
        let mut times = vec![0, 100, 100, 100, 100, 100];
        let confs = vec![0.9, 0.9, 0.01];
        anchor_interpolate(&mut times, &confs, 400);
        // 词 2 填满左锚词尾 100 与音频末尾 400 之间
        assert_eq!(times[4], 100);
        assert_eq!(times[5], 400);
    }

    #[test]
    fn attribute_by_time_midpoints() {
        let spans = vec![
            TimelineSpan {
                concat_start_sample: 0,
                concat_end_sample: 80000,
                source_start_ms: 0,
                source_end_ms: 5000,
            },
            TimelineSpan {
                concat_start_sample: 80000,
                concat_end_sample: 160000,
                source_start_ms: 13000,
                source_end_ms: 18000,
            },
        ];
        let attr = attribute_by_time(&[0, 4000, 5000, 12000], &spans);
        assert_eq!(attr[0], vec![0, 1]);
        assert_eq!(attr[1], vec![2, 3]);
    }

    #[test]
    fn sequence_health_detects_broken() {
        let mk = |s: u64, e: u64, c: f32| AlignUnit {
            word_idx: 0,
            start_local_ms: s,
            end_local_ms: e,
            confidence: c,
        };
        // 健康
        assert!(sequence_healthy(
            &[mk(0, 100, 0.9), mk(100, 200, 0.8), mk(200, 300, 0.9)],
            0.6
        ));
        // 挤压 (20ms 下限)
        assert!(!sequence_healthy(&[mk(0, 10, 0.9)], 0.6));
        // 逆序 (词尾早于上一词词尾)
        assert!(!sequence_healthy(&[mk(0, 100, 0.9), mk(50, 80, 0.9)], 0.6));
        // 低置信比例超限
        assert!(!sequence_healthy(
            &[mk(0, 100, 0.01), mk(100, 200, 0.01), mk(200, 300, 0.9)],
            0.6
        ));
        // 单个超长词 (超过 6×中位数+500)
        assert!(!sequence_healthy(
            &[mk(0, 100, 0.9), mk(100, 200, 0.9), mk(200, 3000, 0.9)],
            0.6
        ));
    }
}