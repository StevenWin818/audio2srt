use crate::qwen::aligner::{AlignmentResult, QwenAligner};
use crate::qwen::backend::{DecoderBackend, EncoderBackend};
use crate::qwen::decoder::{DecodeRequest, DecodeResult, QwenDecoder};
use crate::qwen::encoder::QwenEncoder;
use crate::qwen::error::QwenError;
use std::sync::atomic::AtomicBool;
use std::sync::Arc;

pub struct QwenRuntime {
    pub asr_encoder: parking_lot::Mutex<QwenEncoder>,
    pub asr_decoder: parking_lot::Mutex<QwenDecoder>,
    pub aligner: Option<parking_lot::Mutex<QwenAligner>>,
    pub cancel: std::sync::Arc<AtomicBool>,
}

impl QwenRuntime {
    pub fn load(
        asr_model_dir: &str,
        aligner_model_dir: Option<&str>,
        encoder_backend: EncoderBackend,
        decoder_backend: DecoderBackend,
    ) -> Result<Self, QwenError> {
        println!(
            "[runtime] loading Qwen runtime\n  asr_dir={}\n  aligner_dir={:?}\n  enc_backend={:?}\n  dec_backend={:?}",
            asr_model_dir, aligner_model_dir, encoder_backend, decoder_backend
        );
        let encoder = QwenEncoder::load(asr_model_dir, encoder_backend)?;
        let decoder = QwenDecoder::load(asr_model_dir, decoder_backend)?;
        let aligner = if let Some(align_dir) = aligner_model_dir {
            match QwenAligner::load(align_dir) {
                Ok(a) => Some(parking_lot::Mutex::new(a)),
                Err(e) => {
                    println!("[runtime] aligner load failed: {:?}", e);
                    None
                }
            }
        } else {
            None
        };
        Ok(Self {
            asr_encoder: parking_lot::Mutex::new(encoder),
            asr_decoder: parking_lot::Mutex::new(decoder),
            aligner,
            cancel: Arc::new(AtomicBool::new(false)),
        })
    }

    pub fn transcribe_segment(
        &self,
        samples_16k: &[f32],
        language: Option<&str>,
        context_prompt: Option<&str>,
    ) -> Result<DecodeResult, QwenError> {
        // 1) ONNX 编码器前向传播 — 生成声学 Embedding / 特征向量。
        let enc_out = {
            let mut enc = self.asr_encoder.lock();
            enc.encode(samples_16k, &self.cancel)?
        };

        // 2) GGUF/llama.cpp 解码器前向传播 — 从 Logits 采样自回归生成文本。
        let req = DecodeRequest {
            encoder_output: &enc_out,
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
    ) -> Result<AlignmentResult, QwenError> {
        if let Some(aligner_mutex) = &self.aligner {
            let mut aligner = aligner_mutex.lock();
            aligner.align(
                samples_16k,
                text,
                segment_start_ms,
                segment_end_ms,
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

    pub fn cancel(&mut self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Relaxed);
    }

    pub fn reset_cancel(&mut self) {
        self.cancel.store(false, std::sync::atomic::Ordering::Relaxed);
    }
}
