use crate::qwen::aligner::{AlignmentResult, QwenAligner};
use crate::qwen::backend::{DecoderBackend, EncoderBackend};
use crate::qwen::decoder::{DecodeRequest, DecodeResult, QwenDecoder};
use crate::qwen::encoder::{EncoderOutput, QwenEncoder};
use crate::qwen::error::QwenError;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct QwenRuntime {
    /// encoder (单实例; 编码线程池各自加载独立 Session 以支持 CPU 并行)
    pub asr_encoder: parking_lot::Mutex<QwenEncoder>,
    pub asr_decoder: parking_lot::Mutex<QwenDecoder>,
    pub aligner: Option<parking_lot::Mutex<QwenAligner>>,
    pub cancel: std::sync::Arc<AtomicBool>,
    /// 模型目录 (供 UI 显示)
    pub model_dir: String,
    /// decoder GGUF 文件名 (量化信息)
    pub decoder_file: String,
    /// encoder 实际执行提供程序 ("CUDA" / "DirectML" / "CPU")
    pub encoder_ep: String,
    /// decoder 实际后端 ("CUDA" / "Vulkan" / "CPU")
    pub decoder_backend: String,
    /// decoder offload 状态
    pub decoder_offload: String,
}

impl QwenRuntime {
    pub fn load(
        asr_model_dir: &str,
        aligner_model_dir: Option<&str>,
        encoder_backend: EncoderBackend,
        decoder_backend: DecoderBackend,
        decoder_file: Option<&str>,
    ) -> Result<Self, QwenError> {
        println!(
            "[runtime] loading Qwen runtime\n  asr_dir={}\n  aligner_dir={:?}\n  enc_backend={:?}\n  dec_backend={:?}\n  decoder_file={:?}",
            asr_model_dir, aligner_model_dir, encoder_backend, decoder_backend, decoder_file
        );
        let encoder = QwenEncoder::load(asr_model_dir, encoder_backend)?;
        let decoder = QwenDecoder::load(asr_model_dir, decoder_backend, decoder_file)?;
        let aligner = if let Some(align_dir) = aligner_model_dir {
            match QwenAligner::load(align_dir, decoder_backend) {
                Ok(a) => Some(parking_lot::Mutex::new(a)),
                Err(e) => {
                    println!("[runtime] aligner load failed: {:?}", e);
                    None
                }
            }
        } else {
            None
        };
        let encoder_ep = encoder.actual_ep.clone();
        let decoder_backend = decoder.actual_backend.clone();
        let decoder_offload = decoder.offload_info.clone();
        let decoder_file = std::path::Path::new(&decoder.model_path())
            .file_name()
            .map(|f| f.to_string_lossy().into_owned())
            .unwrap_or_default();
        Ok(Self {
            asr_encoder: parking_lot::Mutex::new(encoder),
            asr_decoder: parking_lot::Mutex::new(decoder),
            aligner,
            cancel: Arc::new(AtomicBool::new(false)),
            model_dir: asr_model_dir.to_string(),
            decoder_file,
            encoder_ep,
            decoder_backend,
            decoder_offload,
        })
    }

    pub fn transcribe_segment(
        &self,
        samples_16k: &[f32],
        language: Option<&str>,
        context_prompt: Option<&str>,
    ) -> Result<DecodeResult, QwenError> {
        let enc_out = self.encode_segment(samples_16k)?;
        self.decode_segment(&enc_out, language, context_prompt)
    }

    /// 仅执行 ONNX 编码器前向传播。与 `decode_segment` 拆分后，
    /// 编码线程和解码线程可在不同阶段重叠运行（段 N 解码时，段 N+1 编码）。
    pub fn encode_segment(&self, samples_16k: &[f32]) -> Result<EncoderOutput, QwenError> {
        let mut enc = self.asr_encoder.lock();
        enc.encode(samples_16k, &self.cancel)
    }

    /// encoder 的模型目录 (供编码线程池加载独立 Session；QwenEncoder::load 接收目录)
    pub fn encoder_model_path(&self) -> String {
        let p = self.asr_encoder.lock().frontend_path().to_string();
        std::path::Path::new(&p)
            .parent()
            .map(|d| d.to_string_lossy().into_owned())
            .unwrap_or(p)
    }

    /// encoder 的请求后端 (供编码线程池使用相同 EP)
    pub fn encoder_backend(&self) -> EncoderBackend {
        self.asr_encoder.lock().backend()
    }

    /// encoder 实际生效的 EP ("CUDA" / "DirectML" / "CPU")
    pub fn encoder_actual_ep(&self) -> String {
        self.asr_encoder.lock().actual_ep.clone()
    }

    /// 仅执行 GGUF/llama.cpp 解码器前向传播。
    pub fn decode_segment(
        &self,
        encoder_output: &EncoderOutput,
        language: Option<&str>,
        context_prompt: Option<&str>,
    ) -> Result<DecodeResult, QwenError> {
        let req = DecodeRequest {
            encoder_output,
            language,
            context: context_prompt,
        };
        let mut dec = self.asr_decoder.lock();
        dec.decode(&req, &self.cancel)
    }

    pub fn align_segment(
        &self,
        samples_16k: &[f32],
        text: &str,
        segment_start_ms: u64,
        segment_end_ms: u64,
        language: &Option<String>,
    ) -> Result<AlignmentResult, QwenError> {
        if let Some(aligner_mutex) = &self.aligner {
            let mut aligner = aligner_mutex.lock();
            aligner.align(
                samples_16k,
                text,
                segment_start_ms,
                segment_end_ms,
                language.as_deref(),
                &self.cancel,
            )
        } else {
            let start_time = std::time::Instant::now();
            let trimmed = text.trim();
            let chars: Vec<char> = trimmed.chars().collect();
            let n_chars = chars.len();
            let duration = segment_end_ms.saturating_sub(segment_start_ms);
            let step = if n_chars > 0 {
                duration as f64 / n_chars as f64
            } else {
                0.0
            };
            let units = chars
                .into_iter()
                .enumerate()
                .map(|(i, ch)| {
                    let start = segment_start_ms + (i as f64 * step) as u64;
                    let end = if i + 1 == n_chars {
                        segment_end_ms
                    } else {
                        segment_start_ms + ((i + 1) as f64 * step) as u64
                    };
                    crate::qwen::aligner::AlignedToken {
                        text: ch.to_string(),
                        start_ms: start.min(segment_end_ms),
                        end_ms: end.min(segment_end_ms),
                        confidence: Some(1.0),
                    }
                })
                .collect();
            let elapsed_ms = start_time.elapsed().as_millis() as u64;
            Ok(AlignmentResult { units, elapsed_ms })
        }
    }

    pub fn cancel(&self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn reset_cancel(&self) {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen::backend::{DecoderBackend, EncoderBackend};

    /// 完全复刻 app 的运行时加载路径 (Auto 后端 + 实际模型目录)，
    /// 验证 encoder 在真实运行时中是否占用 GPU 显存。
    /// 运行: QWEN_MODEL_DIR=... cargo test --lib runtime_load_and_encode_auto -- --nocapture
    #[test]
    fn runtime_load_and_encode_auto() {
        fn gpu_mem(label: &str) {
            if let Ok(out) = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader"])
                .output()
            {
                println!("[test] GPU mem {}: {}", label, String::from_utf8_lossy(&out.stdout).trim());
            }
        }
        let model_dir = std::env::var("QWEN_MODEL_DIR").unwrap_or_else(|_| {
            let appdata = std::env::var("APPDATA").expect("APPDATA");
            format!("{}\\com.audio2srt\\audio2srt\\models\\qwen3-asr-1.7b", appdata)
        });
        assert!(std::path::Path::new(&model_dir).exists(), "model dir not found: {}", model_dir);

        gpu_mem("before load");
        let runtime = QwenRuntime::load(&model_dir, None, EncoderBackend::Auto, DecoderBackend::Auto, None)
            .expect("runtime load failed");
        gpu_mem("after load (pre-encode)");

        // 多次编码取稳定耗时 (首次含权重上传/warmup)
        let samples = vec![0.0f32; 160000];
        for i in 0..3 {
            let t0 = std::time::Instant::now();
            let out = runtime.encode_segment(&samples).expect("encode failed");
            let dt = t0.elapsed();
            println!(
                "[test] encode #{i}: 10s audio in {:.2}s, {} embeddings",
                dt.as_secs_f64(),
                out.embeddings.len()
            );
        }
        gpu_mem("after encode");
        assert!(true);
    }
}
