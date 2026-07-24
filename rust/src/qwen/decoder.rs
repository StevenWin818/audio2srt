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
}

unsafe impl Send for QwenDecoder {}
unsafe impl Sync for QwenDecoder {}

static BACKEND_INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn ensure_backend_init() {
    BACKEND_INITIALIZED.get_or_init(|| unsafe {
        ll::llama_backend_init();
    });
}

impl QwenDecoder {
    pub fn load(model_dir: &str, backend: DecoderBackend) -> Result<Self, QwenError> {
        ensure_backend_init();

        let dir = Path::new(model_dir);
        let candidates = [
            dir.join("asr_decoder.q4_k.gguf"),
            dir.join("decoder.q4_k.gguf"),
            dir.join("decoder.gguf"),
        ];
        let chosen = candidates
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| QwenError::ModelNotFound(format!("No decoder.gguf in {}", model_dir)))?;
        let chosen_str = chosen.to_string_lossy().to_string();
        println!("[decoder] using model: {}", chosen_str);

        let use_gpu = !matches!(backend, DecoderBackend::Cpu);

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
        println!(
            "[decoder] loaded, n_embd={} vocab_tokens={}",
            n_embd,
            unsafe { ll::llama_vocab_n_tokens(vocab) }
        );

        // 上下文参数
        let mut ctx_params = unsafe { ll::llama_context_default_params() };
        ctx_params.n_ctx = 4096;
        ctx_params.n_batch = 1024;
        ctx_params.n_ubatch = 256;
        ctx_params.n_seq_max = 1;
        ctx_params.n_threads = 4;
        ctx_params.n_threads_batch = 4;
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

        let lang_hint = match req.language.as_deref() {
            Some(l) if !l.is_empty() && l != "auto" => format!(" (in {})", l),
            _ => String::new(),
        };

        let sys_prompt = format!(
            "<|im_start|>system\nYou are a speech recognition assistant. Transcribe the user's audio verbatim{}.<|im_end|>\n<|im_start|>user\n",
            lang_hint
        );
        let mut prefix_tokens = self.tokenize_str(&sys_prompt)?;

        let (audio_start_tok, audio_end_tok) = find_audio_tokens(self.vocab);
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

        let user_suffix_str = "\nPlease transcribe the speech above.<|im_end|>\n<|im_start|>assistant\n";
        let mut user_suffix_tokens = self.tokenize_str(user_suffix_str)?;
        suffix_tokens.append(&mut user_suffix_tokens);

        self.submit_token_batch(&suffix_tokens, current_pos)?;
        current_pos += suffix_tokens.len();

        // 4) 使用 Sampler 采样链（重复惩罚 + 贪婪采样）自回归生成 Token
        let chain_params = unsafe { ll::llama_sampler_chain_default_params() };
        let sampler = unsafe { ll::llama_sampler_chain_init(chain_params) };
        if sampler.is_null() {
            return Err(QwenError::DecoderError(
                "llama_sampler_chain_init returned NULL".into(),
            ));
        }

        let pen_sampler = unsafe {
            ll::llama_sampler_init_penalties(
                64,    // repeat_last_n
                1.20,  // penalty_repeat
                0.20,  // penalty_freq
                0.20,  // penalty_present
            )
        };
        if !pen_sampler.is_null() {
            unsafe { ll::llama_sampler_chain_add(sampler, pen_sampler) };
        }
        let greedy_sampler = unsafe { ll::llama_sampler_init_greedy() };
        if !greedy_sampler.is_null() {
            unsafe { ll::llama_sampler_chain_add(sampler, greedy_sampler) };
        }

        let eos_tok = unsafe { ll::llama_vocab_eos(self.vocab) };
        let eot_tok = unsafe { ll::llama_vocab_eot(self.vocab) };

        const MAX_NEW_TOKENS: usize = 256;
        let mut output = String::new();
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
            if is_eog || next == eos_tok || next == eot_tok || next == 151645 || next == 151643 || next == 128247 {
                break;
            }

            let mut buf = [0u8; 64];
            let n = unsafe {
                ll::llama_token_to_piece(
                    self.vocab,
                    next,
                    buf.as_mut_ptr() as *mut c_char,
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            if n > 0 {
                let take = (n as usize).min(buf.len());
                if let Ok(s) = std::str::from_utf8(&buf[..take]) {
                    output.push_str(s);
                }
            }

            generated += 1;

            if output.ends_with("\n\n\n") {
                break;
            }

            self.submit_token_batch(std::slice::from_ref(&next), current_pos)?;
            current_pos += 1;
        }

        unsafe { ll::llama_sampler_free(sampler) };

        let cleaned = output
            .replace("<|im_end|>", "")
            .replace("<|im_start|>", "")
            .replace("<|endoftext|>", "")
            .replace("<|audio_start|>", "")
            .replace("<|audio_end|>", "")
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
            detected_language: req.language.map(|s| s.to_string()),
            generated_tokens: generated,
            prompt_tokens: prompt_len,
            elapsed_ms,
        })
    }

    fn tokenize_str(&self, text: &str) -> Result<Vec<ll::llama_token>, QwenError> {
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
                true,
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
                true,
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

        if rc < 0 {
            return Err(QwenError::DecoderError(format!(
                "llama_decode rc={}",
                rc
            )));
        }
        Ok(())
    }

    /// 将声学 Embedding 向量批次直接注入 llama
    /// **当前未启用** — Encoder 输出维度 ≠ LLM n_embd，且 decoder.gguf 缺乏音频投影层。保留以备后续集成。
    #[allow(dead_code)]
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

        let mut batch = unsafe { ll::llama_batch_init(n_tokens as i32, self.n_embd, 1) };

        unsafe {
            if !batch.embd.is_null() {
                std::ptr::copy_nonoverlapping(
                    embeddings.as_ptr(),
                    batch.embd,
                    n_tokens * n_embd,
                );
            }
            for i in 0..n_tokens {
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
                    *batch.logits.add(i) = 0;
                }
            }
            batch.n_tokens = n_tokens as i32;
        }

        let rc = unsafe { ll::llama_decode(self.context, batch) };
        unsafe { ll::llama_batch_free(batch) };

        if rc < 0 {
            return Err(QwenError::DecoderError(format!(
                "llama_decode embedding batch rc={}",
                rc
            )));
        }
        Ok(())
    }

    pub fn load_test_stub() -> Self {
        unimplemented!("QwenDecoder::load_test_stub is not supported after migration")
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
// 辅助函数：从 llama.cpp 词汇表中检测音频控制 Token ID
// ---------------------------------------------------------------------------

/// 使用特殊的解析模式分词字符串 "<|audio_start|>" 和 "<|audio_end|>"。
/// 如果词汇表中不包含专用的 Token ID，则返回 (-1, -1)。
fn find_audio_tokens(vocab: *const ll::llama_vocab) -> (i32, i32) {
    let probe_start = "<|audio_start|>";
    let probe_end = "<|audio_end|>";
    let start_c = match CString::new(probe_start) {
        Ok(s) => s,
        Err(_) => return (-1, -1),
    };
    let end_c = match CString::new(probe_end) {
        Ok(s) => s,
        Err(_) => return (-1, -1),
    };

    let start_needed = unsafe {
        ll::llama_tokenize(
            vocab,
            start_c.as_ptr(),
            probe_start.len() as i32,
            std::ptr::null_mut(),
            0,
            true,
            true,
        )
    };
    let end_needed = unsafe {
        ll::llama_tokenize(
            vocab,
            end_c.as_ptr(),
            probe_end.len() as i32,
            std::ptr::null_mut(),
            0,
            true,
            true,
        )
    };

    let n_start = start_needed.abs() as usize;
    let n_end = end_needed.abs() as usize;

    if n_start == 0 || n_end == 0 {
        println!("[decoder] Audio token markers NOT found in vocab");
        return (-1, -1);
    }

    let mut start_buf = vec![0_i32; n_start + 4];
    let mut end_buf = vec![0_i32; n_end + 4];
    let rc_start = unsafe {
        ll::llama_tokenize(
            vocab,
            start_c.as_ptr(),
            probe_start.len() as i32,
            start_buf.as_mut_ptr(),
            start_buf.len() as i32,
            true,
            true,
        )
    };
    let rc_end = unsafe {
        ll::llama_tokenize(
            vocab,
            end_c.as_ptr(),
            probe_end.len() as i32,
            end_buf.as_mut_ptr(),
            end_buf.len() as i32,
            true,
            true,
        )
    };

    if rc_start <= 0 || rc_end <= 0 {
        return (-1, -1);
    }
    let sid = start_buf[0];
    let eid = end_buf[0];
    println!("[decoder] Found audio token IDs: <|audio_start|>={} <|audio_end|>={}", sid, eid);
    (sid, eid)
}
