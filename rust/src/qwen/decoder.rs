use crate::qwen::backend::DecoderBackend;
use crate::qwen::encoder::EncoderOutput;
use crate::qwen::error::QwenError;
use std::ffi::CString;
use std::os::raw::c_char;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

// llama-cpp-sys-2 crate 在 Cargo.toml 中显式引入，通过 llama_cpp_sys_2 重导出底层 C API
use llama_cpp_sys_2 as ll;


pub struct DecodeRequest<'a> {
    pub encoder_output: &'a EncoderOutput,
    pub language: Option<&'a str>,
    pub context: Option<&'a str>,
    /// 最大生成 token 数: 按音频时长动态设置, None 时用默认 256
    pub max_new_tokens: Option<usize>,
    /// 采样温度 (0.0 = 贪婪解码)
    pub temperature: f32,
    /// 质量回退时的温度增量 (每次重试 +inc, 上限 1.0)
    pub temperature_inc: f32,
    /// 生成序列尾部香农熵阈值: 低于则视为重复循环, 升温回退重试
    pub entropy_thold: f32,
    /// 平均 token 对数概率下限: 低于则升温回退重试
    pub logprob_thold: f32,
    /// true = 每段解码前清空 KV 缓存 (禁用跨段状态记忆)
    pub clear_kv: bool,
}

/// 单次解码尝试的内部产物 (质量指标用于回退判定, 不对外暴露)
#[derive(Clone)]
struct AttemptOutput {
    text: String,
    detected_language: Option<String>,
    generated_tokens: usize,
    prompt_tokens: usize,
    avg_logprob: f64,
    entropy: f64,
}

#[derive(Debug, Clone)]
pub struct DecodeResult {
    pub text: String,
    pub detected_language: Option<String>,
    pub generated_tokens: usize,
    pub prompt_tokens: usize,
    pub elapsed_ms: u64,
}

pub struct QwenDecoder {
    backend: DecoderBackend,
    model_path: String,
    #[allow(dead_code)]
    model: *mut ll::llama_model,
    context: *mut ll::llama_context,
    vocab: *const ll::llama_vocab,
    n_embd: i32,
    use_gpu: bool,
    /// 实际生效后端 ("CUDA" / "Vulkan" / "CPU")
    pub actual_backend: String,
    /// offload 状态 ("GPU (N/N layers)" / "CPU")
    pub offload_info: String,
    /// KV 缓存已消费的位置: 禁用状态历史时恒为 0, 保留跨段历史时随解码推进
    kv_cache_pos: usize,
    /// 上下文长度上限 (KV 溢出时强制清空历史)
    n_ctx: usize,
}

unsafe impl Send for QwenDecoder {}
unsafe impl Sync for QwenDecoder {}

static BACKEND_INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

/// 把 llama.cpp 的内部日志重定向到 stdout，以便观察 `n_gpu_layers` / `offloaded X/Y layers to GPU`
/// 等关键加载信息 —— 默认 ggml_log_callback 走 stderr，在 Flutter runner 下常被吞掉。
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
        ll::GGML_LOG_LEVEL_ERROR => "[llama:ERROR]",
        ll::GGML_LOG_LEVEL_WARN => "[llama:WARN] ",
        ll::GGML_LOG_LEVEL_INFO => "[llama]     ",
        ll::GGML_LOG_LEVEL_DEBUG => "[llama:DBG] ",
        _ => "[llama]     ",
    };
    // 输出每行加前缀。ggml 的 log 通常自带换行；按行处理避免折断显示。
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

/// 运行时探测是否有可用的 GPU 设备
fn gpu_device_available() -> bool {
    unsafe {
        !ll::ggml_backend_dev_by_type(ll::GGML_BACKEND_DEVICE_TYPE_GPU).is_null()
            || !ll::ggml_backend_dev_by_type(ll::GGML_BACKEND_DEVICE_TYPE_IGPU).is_null()
    }
}

impl QwenDecoder {
    /// 加载解码器 GGUF。
    ///
    /// `decoder_file`: 调用方 (UI 量化选择) 指定的文件名 (如 "decoder.q4_k_m.gguf")。
    /// 为 None 时按优先级自动扫描所有量化命名。
    pub fn load(
        model_dir: &str,
        backend: DecoderBackend,
        decoder_file: Option<&str>,
    ) -> Result<Self, QwenError> {
        ensure_backend_init();

        let dir = Path::new(model_dir);
        let scan_names = [
            "decoder.bf16.gguf",
            "decoder.f16.gguf",
            "decoder.q8_0.gguf",
            "decoder.q6_k.gguf",
            "decoder.q5_k_m.gguf",
            "decoder.q4_k_m.gguf",
            "decoder.q4_k.gguf",
            "decoder.gguf",
            "asr_decoder.q4_k.gguf",
        ];
        let chosen = if let Some(file) = decoder_file {
            let exact = dir.join(file);
            if exact.exists() {
                exact
            } else {
                // 指定量化文件缺失 (旧版本目录命名): 回退到自动扫描
                println!(
                    "[decoder] WARNING: requested decoder file '{}' not found, falling back to auto-scan",
                    file
                );
                scan_names
                    .iter()
                    .map(|f| dir.join(f))
                    .find(|p| p.exists())
                    .ok_or_else(|| {
                        QwenError::ModelNotFound(format!("No decoder GGUF in {}", model_dir))
                    })?
            }
        } else {
            scan_names
                .iter()
                .map(|f| dir.join(f))
                .find(|p| p.exists())
                .ok_or_else(|| {
                    QwenError::ModelNotFound(format!("No decoder GGUF in {}", model_dir))
                })?
        };
        let chosen_str = chosen.to_string_lossy().to_string();
        println!("[decoder] using model: {}", chosen_str);

        // 诊断 GPU 后端可达性 —— 反映在 Vulkan / CUDA 后端是否被编译进当前 cdylib。
        // 如果 `llama_supports_gpu_offload()` 返回 false，说明 build.rs 没启用 GGML_VULKAN
        // (或 GGML_CUDA 等 GPU 后端)，n_gpu_layers=-1 也会被 llama-model.cpp:1268 在
        // `devices.empty()` 时归零为 0，模型实际全部跑在 CPU 上。
        let supports_offload = unsafe { ll::llama_supports_gpu_offload() };
        let gpu_available = gpu_device_available();
        let requested_gpu = !matches!(backend, DecoderBackend::Cpu);
        println!(
            "[decoder] llama_supports_gpu_offload()={}  gpu_device_available()={}  backend={:?}",
            supports_offload, gpu_available, backend
        );
        if requested_gpu && (!supports_offload || !gpu_available) {
            println!("[decoder] **WARNING** backend requested GPU 但运行时无可用 GPU 设备 —— 模型将走 CPU 路径");
        }

        // 实际是否真用 GPU: 需同时满足 请求后端非 CPU + 编译支持 offload + 运行时存在 GPU 设备。
        let use_gpu = requested_gpu && supports_offload && gpu_available;
        let path_c = CString::new(chosen_str.clone())
            .map_err(|e| QwenError::DecoderError(format!("path CString: {}", e)))?;

        // 模型参数：请求时将所有层卸载至 GPU
        let mut model_params = unsafe { ll::llama_model_default_params() };
        model_params.n_gpu_layers = if use_gpu { -1 } else { 0 };
        model_params.use_mmap = true;
        model_params.vocab_only = false;

        let model = unsafe { ll::llama_model_load_from_file(path_c.as_ptr(), model_params) };
        if model.is_null() {
            return Err(QwenError::DecoderError(format!(
                "llama_model_load_from_file returned NULL for {}",
                chosen_str
            )));
        }

        let vocab = unsafe { ll::llama_model_get_vocab(model) };
        let n_embd = unsafe { ll::llama_model_n_embd(model) };
        // 打印 GGUF 内嵌的模型描述 (含量化类型)，便于确认实际加载的是哪个变体
        let mut desc_buf = vec![0u8; 256];
        let desc_len = unsafe {
            ll::llama_model_desc(
                model,
                desc_buf.as_mut_ptr() as *mut c_char,
                desc_buf.len(),
            )
        };
        if desc_len > 0 {
            desc_buf.truncate(desc_len as usize);
            println!("[decoder] model desc: {}", String::from_utf8_lossy(&desc_buf));
        }
        println!(
            "[decoder] loaded, n_embd={} vocab_tokens={}",
            n_embd,
            unsafe { ll::llama_vocab_n_tokens(vocab) }
        );

        // 记录实际生效后端 (供 UI 显示): 按编译的 GGML 后端推断
        let actual_backend = if use_gpu {
            if cfg!(any(feature = "cuda", feature = "qwen-cuda")) {
                "CUDA".to_string()
            } else if cfg!(any(feature = "vulkan", feature = "qwen-vulkan")) {
                "Vulkan".to_string()
            } else {
                "CPU".to_string()
            }
        } else {
            "CPU".to_string()
        };
        let n_layer = unsafe { ll::llama_model_n_layer(model) };
        // llama.cpp offload 计数 = repeating 层 + output 层 (n_layer + 1)
        let offload_info = if use_gpu {
            format!("GPU ({}/{} layers)", n_layer + 1, n_layer + 1)
        } else {
            "CPU".to_string()
        };
        println!(
            "[decoder] actual_backend={} offload={}",
            actual_backend, offload_info
        );

        // 上下文参数: n_ctx 覆盖 45s 块的理论上限 (~585 audio tokens + 前后缀 + 生成 ≤768
        // ≈ 1400), 2048 有充足余量。
        // KV cache 随 n_ctx 线性增长; GPU 模式下 KV 与权重同在显存
        let mut ctx_params = unsafe { ll::llama_context_default_params() };
        ctx_params.n_ctx = 2048;
        ctx_params.n_batch = 1024;
        ctx_params.n_ubatch = 512;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = 8;
        ctx_params.n_threads_batch = 8;
        ctx_params.flash_attn_type = ll::LLAMA_FLASH_ATTN_TYPE_ENABLED as _;
        // 生成生成式使用：无池化、因果注意力机制
        ctx_params.pooling_type = ll::LLAMA_POOLING_TYPE_NONE as _;
        ctx_params.embeddings = false;
        let n_ctx = ctx_params.n_ctx as usize;

        let context = unsafe { ll::llama_init_from_model(model, ctx_params) };
        if context.is_null() {
            unsafe { ll::llama_model_free(model) };
            return Err(QwenError::DecoderError(
                "llama_init_from_model returned NULL".into(),
            ));
        }

        Ok(Self {
            backend,
            model_path: chosen_str,
            model,
            context,
            vocab,
            n_embd,
            use_gpu,
            actual_backend,
            offload_info,
            kv_cache_pos: 0,
            n_ctx,
        })
    }

    pub fn decode(
        &mut self,
        req: &DecodeRequest,
        cancel: &AtomicBool,
    ) -> Result<DecodeResult, QwenError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(QwenError::Cancelled);
        }
        let start_time = std::time::Instant::now();

        // 质量回退循环:
        // 1. avg_logprob < logprob_thold -> 不确信, 升温重试;
        // 2. 生成 token 数 > 32 且尾部序列香农熵 < entropy_thold -> 重复循环, 升温重试。
        let mut temperature = req.temperature.clamp(0.0_f32, 1.0_f32);
        let mut attempt_index = 0u32;
        let out: AttemptOutput = {
            // 内容安全网: 升温回退的最终结果为空时, 退回最近一次非空尝试,
            // 避免"空输出 -> pending 合并 -> 整段音频被丢弃"
            let mut last_non_empty: Option<AttemptOutput> = None;
            let out = loop {
                let out = match self.decode_attempt(req, cancel, temperature, attempt_index) {
                    Ok(o) => o,
                    Err(e) => {
                        // 出错/取消后重置 KV 状态, 防止污染下一次解码
                        self.reset_kv();
                        return Err(e);
                    }
                };
                if !out.text.trim().is_empty() {
                    last_non_empty = Some(out.clone());
                }
                let logprob_fail = out.avg_logprob < req.logprob_thold as f64;
                let entropy_fail =
                    out.generated_tokens > 32 && out.entropy < req.entropy_thold as f64;
                let quality_fail = logprob_fail || entropy_fail;
                let can_retry =
                    req.temperature_inc > 0.0 && temperature + req.temperature_inc < 1.0 + 1e-6;
                if quality_fail {
                    println!(
                        "[decoder] quality fallback: avg_logprob={:.3} entropy={:.3} (thresholds logprob={:.2} entropy={:.2})",
                        out.avg_logprob, out.entropy, req.logprob_thold, req.entropy_thold
                    );
                }
                if !quality_fail || !can_retry {
                    break out;
                }
                temperature += req.temperature_inc.max(0.0);
                attempt_index += 1;
                println!("[decoder] retry with temperature={:.2}", temperature);
            };
            if out.text.trim().is_empty() {
                if let Some(prev) = last_non_empty {
                    println!(
                        "[decoder] final attempt empty, falling back to last non-empty attempt ({} tokens)",
                        prev.generated_tokens
                    );
                    prev
                } else {
                    out
                }
            } else {
                out
            }
        };

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        println!(
            "[decoder] Transcribed {} tokens in {}ms: '{}'",
            out.generated_tokens, elapsed_ms, out.text
        );
        Ok(DecodeResult {
            text: out.text,
            detected_language: out.detected_language,
            generated_tokens: out.generated_tokens,
            prompt_tokens: out.prompt_tokens,
            elapsed_ms,
        })
    }

    /// 单次解码尝试: 构建 prompt、注入音频嵌入、自回归生成并统计质量指标。
    /// KV 缓存策略: 首次尝试且启用状态历史时续用上次 KV (位置继续推进),
    /// 否则 (禁用历史 / 回退重试 / 上下文将溢出) 清空后从 0 开始。
    fn decode_attempt(
        &mut self,
        req: &DecodeRequest,
        cancel: &AtomicBool,
        temperature: f32,
        attempt_index: u32,
    ) -> Result<AttemptOutput, QwenError> {
        let attempt_start = std::time::Instant::now();

        // ---- 1. 维度校验 ----
        let enc_shape = &req.encoder_output.shape;
        let n_embd = self.n_embd as usize;
        let enc_len = req.encoder_output.embeddings.len();

        let last_dim = if enc_shape.len() >= 2 {
            enc_shape[enc_shape.len() - 1]
        } else if n_embd > 0 {
            enc_len / (enc_len / n_embd).max(1)
        } else {
            0
        };

        println!(
            "[decoder] Encoder output shape={:?} last_dim={} decoder n_embd={}",
            enc_shape, last_dim, n_embd
        );

        if last_dim != n_embd {
            println!(
                "[decoder] **ERROR** Encoder feature dimension ({}) ≠ decoder LLM dimension ({}). Audio-LM projection layer is required.",
                last_dim, n_embd
            );
            return Err(QwenError::DecoderError(format!(
                "Encoder feature dimension ({}) does not match decoder LLM dimension ({}). Linear projection layer missing.",
                last_dim, n_embd
            )));
        }

        let n_audio_tokens = if n_embd > 0 { enc_len / n_embd } else { 0 };
        let max_new_tokens = req
            .max_new_tokens
            .unwrap_or(256)
            .clamp(256, 768);

        // ---- 2. 精确匹配 Qwen3-ASR 官方对话 Prompt 模板。----
        // 模型在训练时将音频特征作为完整的 user 消息。
        // 在 <|audio_end|> 之后添加额外的自然语言指令，
        // 会偏离其 ASR 专属输出协议，导致无法稳定触发正常自回归转写。
        let context = req.context.unwrap_or("");
        let sys_prompt = format!(
            "<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n",
            context
        );
        let mut prefix_tokens = self.tokenize_str(&sys_prompt)?;

        let (audio_start_tok, audio_end_tok, _) = find_audio_tokens(self.vocab);
        if audio_start_tok >= 0 {
            prefix_tokens.push(audio_start_tok);
        } else {
            let mut tag_tokens = self.tokenize_str("<|audio_start|>")?;
            prefix_tokens.append(&mut tag_tokens);
        }

        let mut suffix_tokens = Vec::new();
        if audio_end_tok >= 0 {
            suffix_tokens.push(audio_end_tok);
        } else {
            let mut tag_tokens = self.tokenize_str("<|audio_end|>")?;
            suffix_tokens.append(&mut tag_tokens);
        }

        // 对齐官方推理 (qwen_asr/inference/qwen3_asr.py `_build_text_prompt`)：
        // 强制指定语言时，在 assistant 前缀注入 "language X<asr_text>"，
        // 模型会据此只输出纯文本转写，而不是自己生成 "language X<asr_text>" 元数据头。
        let forced_language = req.language.and_then(canonical_qwen_language);
        let mut user_suffix_str = "<|im_end|>\n<|im_start|>assistant\n".to_string();
        if let Some(language) = forced_language {
            user_suffix_str.push_str(&format!("language {language}<asr_text>"));
        }
        let mut user_suffix_tokens = self.tokenize_str(&user_suffix_str)?;
        suffix_tokens.append(&mut user_suffix_tokens);

        // ---- 3. KV 状态决策 (跨段状态历史 / 回退重试 / 上下文溢出) ----
        let keep_history = !req.clear_kv;
        let need_clear = attempt_index > 0 || !keep_history;
        if need_clear {
            self.reset_kv();
        } else {
            let est_len = self.kv_cache_pos
                + prefix_tokens.len()
                + n_audio_tokens
                + suffix_tokens.len()
                + max_new_tokens;
            if est_len > self.n_ctx {
                println!(
                    "[decoder] KV context would overflow ({} tokens needed, n_ctx={}), clearing history",
                    est_len, self.n_ctx
                );
                self.reset_kv();
            }
        }
        let start_pos = self.kv_cache_pos;

        // ---- 4. 提交前缀 Token (System Prompt + <|audio_start|>) ----
        self.submit_token_batch(&prefix_tokens, start_pos)?;
        let mut current_pos = start_pos + prefix_tokens.len();

        // ---- 5. 将声学 Embedding 向量注入 LLM KV Cache ----
        let embeddings = &req.encoder_output.embeddings;
        if n_audio_tokens > 0 {
            println!(
                "[decoder] Ingesting {} audio embedding vectors (dim={}) into llama.cpp GPU KV cache",
                n_audio_tokens, n_embd
            );
            self.submit_embedding_batch(embeddings, n_audio_tokens, current_pos)?;
            current_pos += n_audio_tokens;
        } else {
            println!("[decoder] Warning: No audio embeddings available from ONNX encoder!");
        }

        // ---- 6. 提交后缀 Token (<|audio_end|> + 用户 Prompt + Assistant 标记) ----
        self.submit_token_batch(&suffix_tokens, current_pos)?;
        current_pos += suffix_tokens.len();

        // ---- 7. 自回归生成 (手动 softmax 采样, 同时统计质量指标) ----
        let eos_tok = unsafe { ll::llama_vocab_eos(self.vocab) };
        let eot_tok = unsafe { ll::llama_vocab_eot(self.vocab) };
        let vocab_n = unsafe { ll::llama_vocab_n_tokens(self.vocab) } as usize;

        // Token Piece 是任意字节片段，单个片段不保证是有效的 UTF-8 字符。
        // 将所有片段收集完后再统一转换为 UTF-8 字符串，
        // 避免中日韩等多字节字符因跨 Token 切割而导致无声丢弃和乱码。
        let mut output_bytes = Vec::<u8>::new();
        let mut generated_tokens: Vec<ll::llama_token> = Vec::with_capacity(max_new_tokens);
        let mut sum_logprob = 0.0f64;
        let mut rng_state = new_rng_seed();

        for _ in 0..max_new_tokens {
            if cancel.load(Ordering::Relaxed) {
                return Err(QwenError::Cancelled);
            }

            let logits_ptr = unsafe { ll::llama_get_logits(self.context) };
            let (next, logprob) = if logits_ptr.is_null() || vocab_n == 0 {
                (-1, 0.0)
            } else {
                let logits = unsafe { std::slice::from_raw_parts(logits_ptr, vocab_n) };
                sample_token(logits, temperature, &mut rng_state)
            };
            if next < 0 {
                break;
            }

            sum_logprob += logprob;
            generated_tokens.push(next);

            let is_eog = unsafe { ll::llama_vocab_is_eog(self.vocab, next) };
            if is_eog
                || next == eos_tok
                || next == eot_tok
                || next == 151645 // <|im_end|>
                || next == 151643 // <|endoftext|>
                || next == 151670 // <|audio_end|>
                || next == 128247 // </s>
            {
                let piece = self.token_piece(next).unwrap_or_default();
                println!("[decoder] Break triggered on token ID={} ({:?})", next, String::from_utf8_lossy(&piece));
                break;
            }

            output_bytes.extend_from_slice(&self.token_piece(next)?);

            if output_bytes.ends_with(b"\n\n") {
                break;
            }

            self.submit_token_batch(std::slice::from_ref(&next), current_pos)?;
            current_pos += 1;
        }

        // 记录 KV 位置: 保留历史时下一次解码从此处继续
        self.kv_cache_pos = current_pos;

        let generated = generated_tokens.len();
        let avg_logprob = if generated > 0 {
            sum_logprob / generated as f64
        } else {
            0.0
        };
        let entropy = sequence_entropy(&generated_tokens);

        // ---- 8. 解析输出 ----
        let output = String::from_utf8_lossy(&output_bytes).into_owned();
        let (detected_language, asr_text) =
            parse_qwen_asr_output(&output, forced_language);
        let cleaned = asr_text
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|audio_start|>", "")
            .replace("<|audio_end|>", "")
            .replace("<|nospeech|>", "")
            .replace("🎵", "")
            .replace("[音乐]", "")
            .replace("(音乐)", "")
            .replace("[Music]", "")
            .trim()
            .to_string();

        let prompt_len = prefix_tokens.len() + n_audio_tokens + suffix_tokens.len();
        println!(
            "[decoder] attempt #{}: {} tokens in {}ms, avg_logprob={:.3}, entropy={:.3}",
            attempt_index,
            generated,
            attempt_start.elapsed().as_millis(),
            avg_logprob,
            entropy
        );

        Ok(AttemptOutput {
            text: cleaned,
            detected_language,
            generated_tokens: generated,
            prompt_tokens: prompt_len,
            avg_logprob,
            entropy,
        })
    }

    /// 出错/取消后重置 KV 缓存与位置, 防止污染下一次解码
    fn reset_kv(&mut self) {
        unsafe {
            if !self.context.is_null() {
                let mem = ll::llama_get_memory(self.context);
                if !mem.is_null() {
                    ll::llama_memory_clear(mem, true);
                }
            }
        }
        self.kv_cache_pos = 0;
    }

    /// 供 ForcedAligner 复用: 用当前 GGUF 词表对单个词做 BPE 编码。
    pub fn tokenize_str(&self, text: &str) -> Result<Vec<ll::llama_token>, QwenError> {
        let text_c = CString::new(text)
            .map_err(|e| QwenError::DecoderError(format!("tokenize CString: {}", e)))?;
        let bytes = text.as_bytes();
        let len_i32 = bytes.len() as i32;

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
        if rc < 0 {
            return Err(QwenError::DecoderError(format!(
                "llama_tokenize write failed (rc={})",
                rc
            )));
        }
        tokens.truncate(rc as usize);
        Ok(tokens)
    }

    fn token_piece(&self, token: ll::llama_token) -> Result<Vec<u8>, QwenError> {
        let mut buf = vec![0u8; 64];
        let mut n = unsafe {
            ll::llama_token_to_piece(
                self.vocab,
                token,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as i32,
                0,
                false,
            )
        };

        if n < 0 {
            buf.resize((-n) as usize, 0);
            n = unsafe {
                ll::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
        }
        if n < 0 {
            return Err(QwenError::DecoderError(format!(
                "llama_token_to_piece buffer sizing failed for token {} (rc={})",
                token, n
            )));
        }
        buf.truncate(n as usize);
        Ok(buf)
    }

    /// 向 llama_decode 提交 Token ID 序列，位置递增
    fn submit_token_batch(&mut self, tokens: &[ll::llama_token], start_pos: usize) -> Result<(), QwenError> {
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
                    *batch.logits.add(i) = if i + 1 == tokens.len() { 1 } else { 0 };
                }
            }
        }
        batch.n_tokens = n;

        let rc = unsafe { ll::llama_decode(self.context, batch) };
        unsafe { ll::llama_batch_free(batch) };

        if rc != 0 {
            return Err(QwenError::DecoderError(format!(
                "llama_decode rc={}",
                rc
            )));
        }
        Ok(())
    }

    /// 将声学 Embedding 向量分块注入 llama，并赋予 <|audio_pad|> Token 身份
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
                "embeddings len ({}) < n_tokens ({}) * n_embd ({})",
                embeddings.len(),
                n_tokens,
                n_embd
            )));
        }

        let non_finite = embeddings
            .iter()
            .take(n_tokens * n_embd)
            .filter(|v| !v.is_finite())
            .count();
        if non_finite != 0 {
            return Err(QwenError::DecoderError(format!(
                "encoder produced {} non-finite embedding values",
                non_finite
            )));
        }
        let (min_embedding, max_embedding) = embeddings
            .iter()
            .take(n_tokens * n_embd)
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &value| {
                (min.min(value), max.max(value))
            });

        let physical_batch = 256.min(n_tokens);
        println!(
            "[decoder] Audio embedding range=[{:.4}, {:.4}], physical batch={}{}",
            min_embedding,
            max_embedding,
            physical_batch,
            if self.use_gpu {
                " (Vulkan-safe)"
            } else {
                ""
            }
        );

        // Qwen3-ASR 解码器继承自 Qwen3-Omni/Qwen3-VL 架构，使用交织多模态 RoPE
        // (interleaved M-RoPE, rope_type=IMROPE, mrope_section=[24,20,20])。
        // llama.cpp 0.1.152 在 MRoPE 模型下，对 **embedding-only** 批次不做 1D-
        // 位置自动广播 —— `ubatch_add` (`llama-batch.cpp:713-720`) 对每条 RoPE
        // 轴 (j ∈ {0,1,2,3}) 都从 `batch.pos[j*n_tokens + idxs[i]]` 取位置。
        // 仅 **text-token 批次** 例外：`llama-graph.cpp:146-156` 会把 T/H/W 三轴
        // 全部赋成 `pos[i]`、E 轴赋成 0，形成"3 axis 同位置"。
        //
        // 官方 Qwen3-ASR 推理 (`qwen_asr/.../modeling_qwen3_asr.py` 中
        // `get_rope_index`、以及 llama.cpp 多模态 `mtmd-helper.cpp:184-198`
        // 的 `set_position_mrope_1d`) 对音频段就是 **T=H=W=T 全同位置**的共享
        // 排布。
        //
        // 因此嵌入批次提交时，必须把 `batch.pos` 显式按 M-RoPE 轴交织布局填
        // 满 (长度 `n_pos_per_embd * n_tokens`)，否则 llama.cpp 会读跨边界的
        // 堆内存，导致 RoPE 旋转角任意漂移 —— 输出乱码、且因 LLM 倾向低熵
        // "安全"token 而出现大量空格 / 中日乱字。
        //
        // `llama_batch_init(n_alloc, embd_dim, n_seq)` 默认 `pos` 仅 `n_alloc`
        // 长，无法承载 4*n_tokens 的位置数据；以 `chunk_tokens * 4` 为分配
        // 长度即可 (embd/logits/seq_id 等 buffer 同样 N 倍扩容，只使用前 chunk
        // 个元素即可)。
        const N_POS_PER_EMBD: usize = 4;

        for token_offset in (0..n_tokens).step_by(physical_batch) {
            let chunk_tokens = (n_tokens - token_offset).min(physical_batch);
            let n_alloc = chunk_tokens * N_POS_PER_EMBD;
            let mut batch =
                unsafe { ll::llama_batch_init(n_alloc as i32, self.n_embd, 1) };

            unsafe {
                if batch.embd.is_null() {
                    ll::llama_batch_free(batch);
                    return Err(QwenError::DecoderError(
                        "llama_batch_init returned a null embedding buffer".into(),
                    ));
                }
                std::ptr::copy_nonoverlapping(
                    embeddings.as_ptr().add(token_offset * n_embd),
                    batch.embd,
                    chunk_tokens * n_embd,
                );

                // C 堆内存 malloc 分配的 n_seq_id、seq_id、logits 包含随机未初始化数据，
                // 必须显式填充合法初始值 (n_seq_id=1, seq_id[i][0]=0, logits=0)，
                // 避免 llama_decode 报 init: invalid seq_id[-2] 错误。
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

                // 按 mtmd-helper.cpp:184-198 `set_position_mrope_1d` 方式填
                // 所有 4 个 RoPE 轴：T = H = W = E = (start_pos + token_offset + i)，
                // 这是 Qwen3-ASR 纯 1D 位置在 MRoPE 上的正确等效。
                // (rope_sections[3]==0 时 E 轴不会被真正消费，但写入也无害。)
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
                    "llama_decode embedding batch rc={} at audio token {}",
                    rc, token_offset
                )));
            }
        }
        Ok(())
    }

    pub fn backend(&self) -> DecoderBackend {
        self.backend
    }

    pub fn model_path(&self) -> &str {
        &self.model_path
    }

    pub fn n_embd(&self) -> i32 {
        self.n_embd
    }

    pub fn use_gpu(&self) -> bool {
        self.use_gpu
    }
}

// ---------------------------------------------------------------------------
// 采样与质量指标辅助函数
// ---------------------------------------------------------------------------

/// log-sum-exp 截断阈值: 低于 max_l - TRUNC 的项每项贡献 ≤ e^-20 ≈ 2.1e-9,
/// 15 万词表全部排除项合计误差 < 4e-4 (相对 sum ≥ 1), 对 -1.0 量级的
/// avg_logprob 阈值判定完全无感, 但避免了每个生成 token 都做 15 万次 exp
/// (全量 f64 exp 约 3-5ms/token, 占解码阶段 15%+; 截断后仅对接近 max 的少量项做 exp)。
const LOGSUM_EXP_TRUNC: f32 = 20.0;

/// 手动 softmax 采样 (不依赖 llama.cpp sampler, 便于同时统计对数概率指标):
/// temperature <= 0 时取 argmax (贪婪), 否则按 softmax(logits/T) 多项分布采样。
/// 返回 (token, log_prob): log_prob 为**未缩放** softmax 下的自然对数概率
/// ln P(t) = (logits[t] - max_l) - ln(Σ_j exp(logits[j] - max_l))。
fn sample_token(logits: &[f32], temperature: f32, rng: &mut u64) -> (ll::llama_token, f64) {
    let mut max_l = f32::NEG_INFINITY;
    let mut max_idx = 0usize;
    for (i, &v) in logits.iter().enumerate() {
        if v > max_l {
            max_l = v;
            max_idx = i;
        }
    }

    if temperature <= 0.0 {
        // 贪婪: 截断 log-sum-exp 求 ln P(argmax) = -ln(Σ exp(l - max_l))
        let mut sum_unscaled = 0.0f64;
        for &v in logits.iter() {
            let d = v - max_l;
            if d >= -LOGSUM_EXP_TRUNC {
                sum_unscaled += d.exp() as f64; // f32 exp, 仅近 max 的少量项
            }
        }
        let lp = -sum_unscaled.ln();
        return (max_idx as ll::llama_token, lp);
    }

    let inv_t = 1.0 / temperature;
    let mut sum_scaled = 0.0f64;
    let mut sum_unscaled = 0.0f64;
    for &v in logits.iter() {
        let d = v - max_l;
        let s = d * inv_t;
        if s >= -LOGSUM_EXP_TRUNC {
            sum_scaled += s.exp() as f64;
        }
        if d >= -LOGSUM_EXP_TRUNC {
            sum_unscaled += d.exp() as f64;
        }
    }

    let target = next_uniform(rng) * sum_scaled;
    let mut cum = 0.0f64;
    let mut chosen = logits.len() - 1;
    for (i, &v) in logits.iter().enumerate() {
        let s = (v - max_l) * inv_t;
        if s >= -LOGSUM_EXP_TRUNC {
            cum += s.exp() as f64;
            if cum >= target {
                chosen = i;
                break;
            }
        }
    }
    let lp = (logits[chosen] - max_l) as f64 - sum_unscaled.ln();
    (chosen as ll::llama_token, lp)
}

/// 生成序列尾部 32 个 token 的香农熵 (自然对数):
/// 重复循环 (如 "你好你好你好") 时熵显著降低, 用于触发升温回退。
fn sequence_entropy(tokens: &[ll::llama_token]) -> f64 {
    const WINDOW: usize = 32;
    if tokens.is_empty() {
        return 0.0;
    }
    let start = tokens.len().saturating_sub(WINDOW);
    let window = &tokens[start..];
    let mut counts = std::collections::HashMap::new();
    for &t in window {
        *counts.entry(t).or_insert(0usize) += 1;
    }
    let n = window.len() as f64;
    let mut entropy = 0.0f64;
    for &c in counts.values() {
        let p = c as f64 / n;
        entropy -= p * p.ln();
    }
    entropy
}

/// SplitMix64: 无外部依赖的小型伪随机数发生器 (温度采样用)
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn next_uniform(rng: &mut u64) -> f64 {
    (splitmix64(rng) >> 11) as f64 / ((1u64 << 53) as f64)
}

fn new_rng_seed() -> u64 {
    use std::sync::atomic::AtomicU64;
    use std::time::{SystemTime, UNIX_EPOCH};
    static COUNTER: AtomicU64 = AtomicU64::new(0x9E37_79B9_7F4A_7C15);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let salt = COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Ordering::Relaxed);
    nanos ^ salt.rotate_left(17)
}

impl Drop for QwenDecoder {
    fn drop(&mut self) {
        println!("[decoder] Dropping QwenDecoder: releasing Vulkan GPU memory & llama.cpp context");
        unsafe {
            if !self.context.is_null() {
                ll::llama_free(self.context);
                self.context = std::ptr::null_mut();
            }
            if !self.model.is_null() {
                ll::llama_model_free(self.model);
                self.model = std::ptr::null_mut();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 辅助函数：从 llama.cpp 词汇表中检测音频控制 Token ID
// ---------------------------------------------------------------------------

/// 使用特殊的解析模式分词字符串 "<|audio_start|>", "<|audio_end|>" 和 "<|audio_pad|>"。
fn find_audio_tokens(
    vocab: *const ll::llama_vocab,
) -> (ll::llama_token, ll::llama_token, ll::llama_token) {
    let probe_start = CString::new("<|audio_start|>").unwrap();
    let probe_end = CString::new("<|audio_end|>").unwrap();
    let probe_pad = CString::new("<|audio_pad|>").unwrap();

    let mut start_buf = [-1_i32; 4];
    let mut end_buf = [-1_i32; 4];
    let mut pad_buf = [-1_i32; 4];

    let rc_start = unsafe {
        ll::llama_tokenize(
            vocab,
            probe_start.as_ptr(),
            probe_start.to_bytes().len() as i32,
            start_buf.as_mut_ptr(),
            start_buf.len() as i32,
            false,
            true,
        )
    };
    let rc_end = unsafe {
        ll::llama_tokenize(
            vocab,
            probe_end.as_ptr(),
            probe_end.to_bytes().len() as i32,
            end_buf.as_mut_ptr(),
            end_buf.len() as i32,
            false,
            true,
        )
    };
    let rc_pad = unsafe {
        ll::llama_tokenize(
            vocab,
            probe_pad.as_ptr(),
            probe_pad.to_bytes().len() as i32,
            pad_buf.as_mut_ptr(),
            pad_buf.len() as i32,
            false,
            true,
        )
    };

    let sid = if rc_start == 1 { start_buf[0] } else { -1 };
    let eid = if rc_end == 1 { end_buf[0] } else { -1 };
    let pid = if rc_pad == 1 { pad_buf[0] } else { 151676 };

    println!(
        "[decoder] Found audio token IDs: <|audio_start|>={} <|audio_end|>={} <|audio_pad|>={}",
        sid, eid, pid
    );
    (sid, eid, pid)
}

fn canonical_qwen_language(language: &str) -> Option<&'static str> {
    match language.trim().to_ascii_lowercase().as_str() {
        "" | "auto" => None,
        "zh" | "zh-cn" | "chinese" => Some("Chinese"),
        "en" | "english" => Some("English"),
        "ja" | "jp" | "japanese" => Some("Japanese"),
        "ko" | "kr" | "korean" => Some("Korean"),
        "yue" | "cantonese" => Some("Cantonese"),
        _ => None,
    }
}

fn parse_qwen_asr_output(
    output: &str,
    forced_language: Option<&str>,
) -> (Option<String>, String) {
    if let Some(language) = forced_language {
        // 已把 "language X<asr_text>" 注入 assistant 前缀，预期输出为纯文本。
        // 若模型未被前缀引导仍自带头部，则剥离，避免元数据混入字幕正文。
        let text = strip_asr_language_header(output);
        return (Some(language.to_string()), text);
    }

    let Some((metadata, text)) = output.split_once("<asr_text>") else {
        return (None, output.to_string());
    };
    let metadata = metadata.trim();
    let language = metadata
        .to_ascii_lowercase()
        .find("language ")
        .map(|idx| metadata[idx + "language ".len()..].trim().to_string())
        .filter(|value| !value.is_empty() && !value.eq_ignore_ascii_case("none"));
    (language, text.to_string())
}

fn strip_asr_language_header(output: &str) -> String {
    let s = output.trim();
    if let Some((_, text)) = s.split_once("<asr_text>") {
        return text.trim().to_string();
    }
    s.to_string()
}

#[cfg(test)]
mod tests {
    use super::{
        canonical_qwen_language, next_uniform, parse_qwen_asr_output, sample_token,
        sequence_entropy, LOGSUM_EXP_TRUNC,
    };

    #[test]
    fn maps_ui_language_codes_to_qwen_names() {
        assert_eq!(canonical_qwen_language("zh"), Some("Chinese"));
        assert_eq!(canonical_qwen_language("en"), Some("English"));
        assert_eq!(canonical_qwen_language("auto"), None);
    }

    #[test]
    fn parses_auto_language_output() {
        let (language, text) =
            parse_qwen_asr_output("language Chinese<asr_text>你好世界", None);
        assert_eq!(language.as_deref(), Some("Chinese"));
        assert_eq!(text, "你好世界");
    }

    #[test]
    fn forced_language_output_is_text_only() {
        let (language, text) = parse_qwen_asr_output("hello", Some("English"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "hello");
    }

    #[test]
    fn forced_language_strips_residual_header() {
        let (language, text) =
            parse_qwen_asr_output("language English<asr_text>hello world", Some("English"));
        assert_eq!(language.as_deref(), Some("English"));
        assert_eq!(text, "hello world");
    }

    #[test]
    fn sequence_entropy_detects_repetition() {
        // 重复 token 序列熵显著低于多样序列
        let repetitive = vec![1_i32; 40];
        let diverse: Vec<i32> = (0..40).collect();
        assert!(sequence_entropy(&repetitive) < 0.5);
        assert!(sequence_entropy(&repetitive) < sequence_entropy(&diverse));
        // 空序列与单 token 序列熵为 0
        assert_eq!(sequence_entropy(&[]), 0.0);
        assert_eq!(sequence_entropy(&[1]), 0.0);
    }

    #[test]
    fn sample_token_greedy_and_temperature() {
        let logits = vec![0.0f32, 0.1, 0.2, 0.3, -5.0];
        let mut rng = 42u64;
        // 贪婪: 恒选 argmax, logprob 为未缩放 softmax 对数概率
        let (tok, lp) = sample_token(&logits, 0.0, &mut rng);
        assert_eq!(tok, 3);
        assert!(lp < 0.0 && lp > -10.0);
        // 温度采样: 返回合法 token id 与负对数概率
        let (tok2, lp2) = sample_token(&logits, 1.0, &mut rng);
        assert!((0..5).contains(&tok2));
        assert!(lp2 < 0.0);
    }

    #[test]
    fn sample_token_logprob_matches_softmax() {
        // 回归测试: 此前 log_sum 把 max_l 扣了两次, 导致 lp ≈ -max_l ≈ -40,
        // 正常文本被误判为低质量并触发 6 次升温回退, 最终用 temperature=1.0
        // 的随机结果覆盖正确转写 (并因空输出触发 pending 合并丢弃整段音频)。
        // 峰值分布: 贪婪 token 的 logprob 应接近 0
        let mut peaked = vec![-100.0f32; 128];
        peaked[7] = 40.0;
        let mut rng = 42u64;
        let (tok, lp) = sample_token(&peaked, 0.0, &mut rng);
        assert_eq!(tok, 7);
        assert!(
            lp > -1.0,
            "greedy logprob should be near 0 for peaked logits, got {lp}"
        );
        // 平坦分布: logprob 不可能低于 -ln(词表大小)
        let flat = vec![0.0f32; 1024];
        let (_, lp2) = sample_token(&flat, 0.0, &mut rng);
        let floor = -(1024.0f64).ln() - 1e-3;
        assert!(
            lp2 >= floor,
            "logprob {lp2} below theoretical floor {floor}"
        );
    }

    #[test]
    fn truncated_logsumexp_error_is_negligible() {
        // 截断求和 (d >= -20) 与全量精确求和的相对误差必须 < 1e-3,
        // 保证优化不会影响 avg_logprob 阈值判定的语义
        let mut rng_state = 12345u64;
        for case in 0..20 {
            let logits: Vec<f32> = (0..8192)
                .map(|_| (next_uniform(&mut rng_state) as f32 * 80.0 - 40.0) as f32)
                .collect();
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exact_sum: f64 = logits
                .iter()
                .map(|&v| ((v - max_l) as f64).exp())
                .sum();
            let trunc_sum: f64 = logits
                .iter()
                .map(|&v| {
                    let d = v - max_l;
                    if d >= -LOGSUM_EXP_TRUNC {
                        d.exp() as f64
                    } else {
                        0.0
                    }
                })
                .sum();
            let err = ((exact_sum - trunc_sum).abs() / exact_sum).max(0.0);
            assert!(
                err < 1e-3,
                "case {case}: truncation relative error {err} too large"
            );
        }
    }
}
