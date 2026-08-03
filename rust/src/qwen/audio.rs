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
    /// 每个 mel 滤波器非零频点区间 [k_start, k_end] (闭区间)，
    /// 利用三角滤波器的稀疏性避免全频点遍历
    bands: Vec<(usize, usize)>,
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
        let mut bands = vec![(0usize, 0usize); N_MELS];
        for m in 0..N_MELS {
            let f_left = hz_points[m];
            let f_center = hz_points[m + 1];
            let f_right = hz_points[m + 2];
            let mut k_start = usize::MAX;
            let mut k_end = 0usize;
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
                    if k < k_start {
                        k_start = k;
                    }
                    if k > k_end {
                        k_end = k;
                    }
                }
            }
            bands[m] = if k_start <= k_end {
                (k_start, k_end)
            } else {
                (0, 0)
            };
        }

        MelPlan {
            fft,
            window,
            mel,
            bands,
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
        // 与参考实现对齐：
        // 先做左右各 n_fft/2 采样点的 reflect padding，再以 center=False 做 STFT。
        // 无 padding 会少 2~3 帧，且边界帧与训练分布不一致。
        let pad = N_FFT / 2;
        let n_total = samples.len() + 2 * pad;
        let mut x = vec![0.0f32; n_total];
        x[pad..pad + samples.len()].copy_from_slice(samples);
        for i in 0..pad {
            x[i] = samples[pad - i];
            x[pad + samples.len() + i] = samples[samples.len() - 2 - i];
        }

        let plan = mel_plan();
        let mut inner = plan.fft.make_input_vec();
        let mut spectrum = plan.fft.make_output_vec();
        let n_freqs = N_FFT / 2 + 1;
        let n_frames = (n_total - N_FFT) / HOP_LENGTH + 1;
        if n_frames == 0 {
            return Err(QwenError::AudioError("audio too short for stft".into()));
        }
        let mut feats = vec![0.0f32; N_MELS * n_frames];

        for frame in 0..n_frames {
            let start = frame * HOP_LENGTH;
            for i in 0..N_FFT {
                inner[i] = x[start + i] * plan.window[i];
            }
            plan.fft
                .process(&mut inner, &mut spectrum)
                .map_err(|e| QwenError::AudioError(format!("FFT failed: {}", e)))?;

            // 功率谱 -> Whisper log-mel（以 10 为底的对数）
            // 利用三角滤波器的稀疏性: 每个滤波器只遍历其非零频点区间
            for m in 0..N_MELS {
                let (k_start, k_end) = plan.bands[m];
                let mut acc = 0.0f32;
                let row = m * n_freqs;
                for k in k_start..=k_end {
                    let c = spectrum[k];
                    let power = c.re * c.re + c.im * c.im;
                    acc += plan.mel[row + k] * power;
                }
                let v = acc.max(1e-10).log10();
                feats[m * n_frames + frame] = v;
            }
        }

        // 标准 Whisper Feature Extractor 归一化：
        // 以整段频谱的最大值截断 log10 功率于 (max - 8.0)，随后映射至 [-1.0, 1.0]。
        // (参考实现为相对截断 log_spec.max() - 8.0，而非固定的 -8.0)
        let max_f = feats.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let floor = max_f - 8.0;
        for v in feats.iter_mut() {
            let clamped = (*v).max(floor);
            *v = (clamped + 4.0) / 4.0;
        }

        let min_f = feats.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_f2 = feats.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mean_f: f32 = feats.iter().sum::<f32>() / feats.len() as f32;
        println!("[audio] Mel features: n_frames={} min={:.4} max={:.4} mean={:.4}", n_frames, min_f, max_f2, mean_f);

        Ok((feats, n_frames))
    }
}
