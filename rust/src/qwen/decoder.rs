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
        println!(
            "[decoder] llama_supports_gpu_offload()={}  backend={:?}  use_gpu={}",
            supports_offload, backend, !matches!(backend, DecoderBackend::Cpu)
        );
        if !supports_offload && !matches!(backend, DecoderBackend::Cpu) {
            println!("[decoder] **WARNING** backend requested GPU 但 ggml 未编译进 GPU 后端 —— 模型将走 CPU 路径");
        }

        let use_gpu = !matches!(backend, DecoderBackend::Cpu) && supports_offload;
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

        // 上下文参数
        let mut ctx_params = unsafe { ll::llama_context_default_params() };
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 1024;
        ctx_params.n_ubatch = 512;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = 8;
        ctx_params.n_threads_batch = 8;
        ctx_params.flash_attn_type = ll::LLAMA_FLASH_ATTN_TYPE_ENABLED as _;
        // 生成生成式使用：无池化、因果注意力机制
        ctx_params.pooling_type = ll::LLAMA_POOLING_TYPE_NONE as _;
        ctx_params.embeddings = false;

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

        // 清理 KV Cache
        let mem = unsafe { ll::llama_get_memory(self.context) };
        unsafe { ll::llama_memory_clear(mem, true) };

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

        // 精确匹配 Qwen3-ASR 官方对话 Prompt 模板。
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

        // 1) 提交前缀 Token (System Prompt + <|audio_start|>)
        self.submit_token_batch(&prefix_tokens, 0)?;
        let mut current_pos = prefix_tokens.len();

        // 2) 将声学 Embedding 向量注入 LLM KV Cache
        let embeddings = &req.encoder_output.embeddings;
        let n_audio_tokens = if n_embd > 0 { embeddings.len() / n_embd } else { 0 };

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

        // 3) 准备并提交后缀 Token (<|audio_end|> + 用户 Prompt + Assistant 标记)
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

        self.submit_token_batch(&suffix_tokens, current_pos)?;
        current_pos += suffix_tokens.len();

        // 4) Qwen3-ASR 官方推理使用 temperature=0.0。
        let chain_params = unsafe { ll::llama_sampler_chain_default_params() };
        let sampler = unsafe { ll::llama_sampler_chain_init(chain_params) };
        if sampler.is_null() {
            return Err(QwenError::DecoderError(
                "llama_sampler_chain_init returned NULL".into(),
            ));
        }

        let greedy_sampler = unsafe { ll::llama_sampler_init_greedy() };
        if greedy_sampler.is_null() {
            unsafe { ll::llama_sampler_free(sampler) };
            return Err(QwenError::DecoderError(
                "llama_sampler_init_greedy returned NULL".into(),
            ));
        }
        unsafe { ll::llama_sampler_chain_add(sampler, greedy_sampler) };

        let eos_tok = unsafe { ll::llama_vocab_eos(self.vocab) };
        let eot_tok = unsafe { ll::llama_vocab_eot(self.vocab) };

        const MAX_NEW_TOKENS: usize = 256;
        // Token Piece 是任意字节片段，单个片段不保证是有效的 UTF-8 字符。
        // 将所有片段收集完后再统一转换为 UTF-8 字符串，
        // 避免中日韩等多字节字符因跨 Token 切割而导致无声丢弃和乱码。
        let mut output_bytes = Vec::<u8>::new();
        let mut generated = 0usize;

        for _ in 0..MAX_NEW_TOKENS {
            if cancel.load(Ordering::Relaxed) {
                unsafe { ll::llama_sampler_free(sampler) };
                return Err(QwenError::Cancelled);
            }

            let next = unsafe { ll::llama_sampler_sample(sampler, self.context, -1) };
            if next < 0 {
                break;
            }

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

            generated += 1;

            if output_bytes.ends_with(b"\n\n") {
                break;
            }

            self.submit_token_batch(std::slice::from_ref(&next), current_pos)?;
            current_pos += 1;
        }

        unsafe { ll::llama_sampler_free(sampler) };

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

        let elapsed_ms = start_time.elapsed().as_millis() as u64;
        println!("[decoder] Transcribed {} tokens in {}ms: '{}'", generated, elapsed_ms, cleaned);

        let prompt_len = prefix_tokens.len() + n_audio_tokens + suffix_tokens.len();

        Ok(DecodeResult {
            text: cleaned,
            detected_language,
            generated_tokens: generated,
            prompt_tokens: prompt_len,
            elapsed_ms,
        })
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
    use super::{canonical_qwen_language, parse_qwen_asr_output};

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
}
