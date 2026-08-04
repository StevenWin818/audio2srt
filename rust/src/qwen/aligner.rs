use crate::qwen::audio::AudioProcessor;
use crate::qwen::backend::DecoderBackend;
use crate::qwen::error::QwenError;
use llama_cpp_sys_2 as ll;
use ort::session::Session;
use ort::value::Tensor;
use serde::{Deserialize, Serialize};
use std::ffi::{c_char, CString};
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
    use_gpu: bool,
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
        let (frontend, backend) = if fe_path.exists() && be_path.exists() {
            let fe = Session::builder()
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend builder: {}", e)))?
                .with_intra_threads(2)
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend intra_threads: {}", e)))?
                .commit_from_file(&fe_path)
                .map_err(|e| QwenError::OnnxError(format!("aligner frontend commit: {}", e)))?;
            let be = Session::builder()
                .map_err(|e| QwenError::OnnxError(format!("aligner backend builder: {}", e)))?
                .with_intra_threads(2)
                .map_err(|e| QwenError::OnnxError(format!("aligner backend intra_threads: {}", e)))?
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
        let use_gpu = !matches!(decoder_backend, DecoderBackend::Cpu) && supports_offload;

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
        ctx_params.n_ctx = 4096;
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
    pub fn align(
        &mut self,
        samples_16k: &[f32],
        text: &str,
        segment_start_ms: u64,
        segment_end_ms: u64,
        _language: Option<&str>,
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
            });
        }

        if self.frontend.is_none() || self.backend.is_none() {
            return Ok(AlignmentResult {
                units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                elapsed_ms: 0,
            });
        }
        if self.model.is_null() || self.context.is_null() || self.audio_start_id < 0 {
            return Ok(AlignmentResult {
                units: linear_align(trimmed, segment_start_ms, segment_end_ms),
                elapsed_ms: 0,
            });
        }

        // 1. 词表切分 (CJK 逐字, 拉丁按词)
        let words = tokenize_for_align(trimmed);
        if words.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
            });
        }

        // 2-4. ONNX 编码器 (mel -> frontend -> backend), 借用在块结束后释放
        let features = {
            let frontend = self.frontend.as_mut().unwrap();
            let backend = self.backend.as_mut().unwrap();

            // 2. mel (与 encoder 相同的归一化 log-mel; 对齐器帧数 = 采样数/160, 丢弃尾帧)
            let (mel, _n_frames_log) = AudioProcessor::log_mel(samples_16k)?;
            let n_frames = samples_16k.len() / 160;
            if n_frames == 0 {
                return Ok(AlignmentResult {
                    units: vec![],
                    elapsed_ms: 0,
                });
            }

            // 3. frontend: 100 帧/块 -> 13 token/块, 拼接后按有效长度截断
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
                    let src = m * n_frames;
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

            // 4. backend: 全零注意力掩码 (全局注意力, 与参考实现一致)
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
            (features, n_frames, n_audio, n_chunks)
        };
        let (features, n_frames, n_audio, n_chunks) = features;

        // 5. 组装序列: [audio_start] + audio×N + [audio_end] + w1 + ts + ts + w2 + ts + ts ...
        //    ts 标记紧跟词后 (HaujetZhao GGUF 参考实现顺序)
        let mut post_ids: Vec<ll::llama_token> = vec![self.audio_end_id];
        let mut ts_pos_in_post: Vec<usize> = Vec::with_capacity(words.len() * 2);
        let mut post_len = 1usize; // audio_end
        for w in &words {
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

        // 6. llama 推理: 前缀 token + 音频嵌入 (4 轴 M-RoPE) + 后缀 token (ts 位置取 logits)
        let mem = unsafe { ll::llama_get_memory(self.context) };
        unsafe { ll::llama_memory_clear(mem, true) };

        self.submit_token_batch(&[self.audio_start_id], 0)?;
        let pos = 1usize;
        self.submit_embedding_batch(&features, n_audio, pos)?;
        let post_start = 1 + n_audio;
        self.submit_token_batch_logits(&post_ids, post_start, &ts_pos_in_post)?;

        // 7. 读取时间戳 logits: argmax(logits[:4000]) × 80ms
        let n_embd = self.n_embd as usize;
        let mut raw_ts: Vec<f64> = Vec::with_capacity(ts_pos_in_post.len());
        for &i in &ts_pos_in_post {
            let logits_ptr = unsafe { ll::llama_get_logits_ith(self.context, i as i32) };
            if logits_ptr.is_null() {
                return Err(QwenError::DecoderError(
                    "aligner: llama_get_logits_ith returned NULL".into(),
                ));
            }
            let logits =
                unsafe { std::slice::from_raw_parts(logits_ptr, 152064).to_vec() };
            let classes = logits.len().min(TIMESTAMP_CLASSES);
            let argmax = (0..classes)
                .max_by(|&a, &b| {
                    logits[a]
                        .partial_cmp(&logits[b])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            raw_ts.push(argmax as f64 * self.step_ms);
        }

        // 8. LIS 修正 -> 逐词时间
        let fixed = fix_timestamp(&raw_ts);
        let mut items: Vec<AlignedToken> = words
            .iter()
            .enumerate()
            .map(|(i, w)| AlignedToken {
                text: w.clone(),
                start_ms: fixed.get(i * 2).copied().unwrap_or(0) as u64,
                end_ms: fixed.get(i * 2 + 1).copied().unwrap_or(0) as u64,
                confidence: None,
            })
            .collect();

        // 9. reconcile: 从原始文本找回标点/空格, 重组时间戳序列
        items = reconcile(trimmed, items);

        println!(
            "[aligner] aligned {} words ({} frames, {} audio tokens, {} chunks) in {}ms",
            words.len(),
            n_frames,
            n_audio,
            n_chunks,
            start_time.elapsed().as_millis()
        );

        Ok(AlignmentResult {
            units: items,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
        })
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
                        for j in 0..N_POS_PER_EMBD {
                            *batch.pos.add(j * chunk_tokens + i) = p;
                        }
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

/// 音频卷积下采样后的 token 数: 每 100 mel 帧 13 个 (Python `//` 向下取整语义)
fn feat_output_lengths(input_lengths: i64) -> i64 {
    let leave = input_lengths % 100;
    let feat = (leave - 1).div_euclid(2) + 1;
    ((feat - 1).div_euclid(2) + 1 - 1).div_euclid(2) + 1 + (input_lengths / 100) * TOKENS_PER_CHUNK as i64
}

fn is_kept_char(c: char) -> bool {
    if c == '\'' {
        return true;
    }
    c.is_alphabetic() || c.is_numeric()
}

fn is_cjk_char(c: char) -> bool {
    let code = c as u32;
    (0x4E00..=0x9FFF).contains(&code)
        || (0x3400..=0x4DBF).contains(&code)
        || (0x20000..=0x2A6DF).contains(&code)
        || (0x2A700..=0x2B73F).contains(&code)
        || (0x2B740..=0x2B81F).contains(&code)
        || (0x2B820..=0x2CEAF).contains(&code)
        || (0xF900..=0xFAFF).contains(&code)
}

/// 通用分词: 按空白切词, 过滤非字母/数字, CJK 逐字拆出 (中英混排适用)
fn tokenize_for_align(text: &str) -> Vec<String> {
    let mut tokens: Vec<String> = Vec::new();
    for seg in text.split_whitespace() {
        let cleaned: String = seg.chars().filter(|c| is_kept_char(*c)).collect();
        if cleaned.is_empty() {
            continue;
        }
        let mut buf = String::new();
        for ch in cleaned.chars() {
            if is_cjk_char(ch) {
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
fn reconcile(original_text: &str, items: Vec<AlignedToken>) -> Vec<AlignedToken> {
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
                confidence: None,
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

/// 线性字符平分回退 (无模型时)
fn linear_align(text: &str, segment_start_ms: u64, segment_end_ms: u64) -> Vec<AlignedToken> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let duration = segment_end_ms.saturating_sub(segment_start_ms);
    let step = if n > 0 { duration as f64 / n as f64 } else { 0.0 };
    let mut units = Vec::with_capacity(n);
    for (i, ch) in chars.iter().enumerate() {
        let start = segment_start_ms + (i as f64 * step) as u64;
        units.push(AlignedToken {
            text: ch.to_string(),
            start_ms: start,
            end_ms: if i + 1 == n {
                segment_end_ms
            } else {
                segment_start_ms + ((i + 1) as f64 * step) as u64
            },
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
            tokenize_for_align("今天你好 world"),
            vec![
                "今".to_string(),
                "天".to_string(),
                "你".to_string(),
                "好".to_string(),
                "world".to_string()
            ]
        );
        assert_eq!(
            tokenize_for_align("Hello, world!"),
            vec!["Hello".to_string(), "world".to_string()]
        );
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
}