use crate::qwen::error::QwenError;
use realfft::RealFftPlanner;
use std::sync::OnceLock;

pub const SAMPLE_RATE: u32 = 16000;
pub const N_FFT: usize = 400;
pub const HOP_LENGTH: usize = 160;
pub const N_MELS: usize = 128;
pub const PREEMPH: f32 = 0.97;

/// 隐藏缓存以避免每次调用时重建
struct MelPlan {
    fft: std::sync::Arc<dyn realfft::RealToComplex<f32>>,
    window: Vec<f32>,
    mel: Vec<f32>, // row-major [N_MELS x (N_FFT/2 + 1)]
}

static MEL_PLAN: OnceLock<MelPlan> = OnceLock::new();

fn mel_plan() -> &'static MelPlan {
    MEL_PLAN.get_or_init(|| {
        let mut planner = RealFftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(N_FFT);

        // 周期性 Hann 窗（torch.hann_window(periodic=True) 等效）。
        let mut window = Vec::with_capacity(N_FFT);
        for n in 0..N_FFT {
            let v = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * n as f32 / N_FFT as f32).cos());
            window.push(v);
        }

        // Slaney 风格的 mel 滤波器组，normal=false（与 ASR 的 torchaudio 默认值匹配）
        let f_min = 0.0_f64;
        let f_max = SAMPLE_RATE as f64 / 2.0;
        let mel_max = 2595.0 * (1.0 + f_max / 700.0).log10();
        let mel_min = 2595.0 * (1.0 + f_min / 700.0).log10();
        let mel_points: Vec<f64> = (0..=N_MELS + 1)
            .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (N_MELS + 1) as f64)
            .collect();
        let hz_points: Vec<f64> = mel_points
            .iter()
            .map(|m| 700.0 * (10f64.powf(*m / 2595.0) - 1.0))
            .collect();
        let n_freqs = N_FFT / 2 + 1;
        let fft_freqs: Vec<f64> =
            (0..n_freqs).map(|i| i as f64 * SAMPLE_RATE as f64 / N_FFT as f64).collect();

        let mut mel = vec![0.0f32; N_MELS * n_freqs];
        for m in 0..N_MELS {
            let f_left = hz_points[m];
            let f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];
            for (k, &f) in fft_freqs.iter().enumerate() {
                let w = if f >= f_left && f <= f_center {
                    (f - f_left) / (f_center - f_left)
                } else if f > f_center && f <= f_right {
                    (f_right - f) / (f_right - f_center)
                } else {
                    0.0
                };
                if w > 0.0 {
                    mel[m * n_freqs + k] = w as f32;
                }
            }
        }

        MelPlan {
            fft,
            window,
            mel,
        }
    })
}

pub struct AudioProcessor;

impl AudioProcessor {
    /// 在范围 [-1.0, 1.0] 内将样本重新采样或验证为 16 kHz 单声道 f32
    pub fn prepare_samples(samples: &[f32], input_sample_rate: u32) -> Result<Vec<f32>, QwenError> {
        if samples.is_empty() {
            return Err(QwenError::AudioError("Empty audio samples".into()));
        }
        if input_sample_rate == SAMPLE_RATE {
            Ok(samples.to_vec())
        } else {
            let ratio = SAMPLE_RATE as f64 / input_sample_rate as f64;
            let output_len = (samples.len() as f64 * ratio) as usize;
            let mut output = Vec::with_capacity(output_len);
            // 简单的线性重采样器；encoder 期望为 16 kHz
            for i in 0..output_len {
                let idx = i as f64 / ratio;
                let idx_floor = idx.floor() as usize;
                let idx_ceil = (idx_floor + 1).min(samples.len() - 1);
                let frac = (idx - idx_floor as f64) as f32;
                let sample = samples[idx_floor] * (1.0 - frac) + samples[idx_ceil] * frac;
                output.push(sample);
            }
            Ok(output)
        }
    }

    /// NeMo 风格的 log-mel 频谱图，带有可选的每话语 CMVN。
    /// 返回形状为 [N_MELS, T_frames] 的平面行优先 `Vec<f32>`。
    pub fn log_mel(samples: &[f32]) -> Result<(Vec<f32>, usize), QwenError> {
        if samples.len() < N_FFT {
            return Err(QwenError::AudioError(format!(
                "Audio too short for STFT (got {}, need {})",
                samples.len(),
                N_FFT
            )));
        }
        let mut x = vec![0.0f32; samples.len()];
        // Pre-emphasis.
        x[0] = samples[0];
        for i in 1..samples.len() {
            x[i] = samples[i] - PREEMPH * samples[i - 1];
        }

        let plan = mel_plan();
        let mut inner = plan.fft.make_input_vec();
        let mut spectrum = plan.fft.make_output_vec();
        let n_freqs = N_FFT / 2 + 1;
        let n_frames = (x.len() - N_FFT) / HOP_LENGTH + 1;
        let mut feats = vec![0.0f32; N_MELS * n_frames];

        for frame in 0..n_frames {
            let start = frame * HOP_LENGTH;
            // 阶段输入：窗口帧，填充为 N_FFT。
            for i in 0..N_FFT {
                inner[i] = x[start + i] * plan.window[i];
            }
            plan.fft
                .process(&mut inner, &mut spectrum)
                .map_err(|e| QwenError::AudioError(format!("FFT failed: {}", e)))?;

            // Power spectrum -> log-mel.
            for m in 0..N_MELS {
                let mut acc = 0.0f32;
                let row = m * n_freqs;
                for k in 0..n_freqs {
                    let c = spectrum[k];
                    let power = c.re * c.re + c.im * c.im;
                    acc += plan.mel[row + k] * power;
                }
                // Natural log, clamped to a small floor (long-form std).
                let v = acc.max(1e-10).ln();
                feats[m * n_frames + frame] = v;
            }
        }

        // 每个话语 CMVN（随时间变化的均值归一化）— NeMo 评估默认值。
        let mut mean = vec![0.0f64; N_MELS];
        for m in 0..N_MELS {
            let mut sum = 0.0f64;
            for f in 0..n_frames {
                sum += feats[m * n_frames + f] as f64;
            }
            mean[m] = sum / n_frames as f64;
        }
        let mut std = vec![1.0f64; N_MELS];
        for m in 0..N_MELS {
            let mut acc = 0.0f64;
            for f in 0..n_frames {
                let d = feats[m * n_frames + f] as f64 - mean[m];
                acc += d * d;
            }
            let var = (acc / n_frames as f64).max(1e-5);
            std[m] = var.sqrt();
        }
        for m in 0..N_MELS {
            for f in 0..n_frames {
                let v = (feats[m * n_frames + f] as f64 - mean[m]) / std[m];
                feats[m * n_frames + f] = v as f32;
            }
        }

        Ok((feats, n_frames))
    }
}
