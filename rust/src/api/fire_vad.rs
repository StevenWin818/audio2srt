//! FireRedVAD 非流式 VAD 引擎 (DFSMN, 16kHz 80 维 Kaldi fbank + 全局 CMVN)。
//!
//! 替代 whisper.cpp 内置 Silero VAD: FLEURS-VAD-102 上 F1 97.57 (Silero 95.95),
//! 误报率 2.69% (Silero 9.41%), 显著降低背景音乐被误判为语音导致的字幕偏移。
//!
//! 模型文件: `fireredvad_vad.onnx` (输入 `feat` [batch, time, 80] -> 输出 `probs` [batch, time, 1],
//! 官方 export_onnx.py 从 model.pth.tar 导出), 同目录下需存在 `cmvn.ark` (Kaldi 二进制矩阵)。

use std::path::Path;

use ort::inputs;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;

const SAMPLE_RATE: usize = 16000;
/// fbank 帧长 25ms / 帧移 10ms
const FRAME_LENGTH: usize = 400;
const FRAME_SHIFT: usize = 160;
const FFT_SIZE: usize = 512;
const NUM_BINS: usize = 80;
/// 模型输入维度
const FEAT_DIM: usize = 80;
/// 单条语音段最长 20s (超长语音在概率最低处拆分)
const DEFAULT_MAX_SPEECH_FRAME: usize = 2000;

/// VAD 输出的语音段, start/end 为 10ms 帧单位 (与 whisper.cpp Silero 段语义一致)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VadSegment {
    pub start: u32,
    pub end: u32,
}

#[derive(Debug, Clone)]
pub struct FireVadConfig {
    pub smooth_window_size: usize,
    pub threshold: f32,
    pub min_speech_frame: usize,
    pub max_speech_frame: usize,
    pub min_silence_frame: usize,
    pub merge_silence_frame: usize,
    pub extend_speech_frame: usize,
}

impl Default for FireVadConfig {
    fn default() -> Self {
        Self {
            smooth_window_size: 5,
            threshold: 0.4,
            min_speech_frame: 20,
            max_speech_frame: DEFAULT_MAX_SPEECH_FRAME,
            min_silence_frame: 20,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
        }
    }
}

/// Kaldi 风格 fbank: 80 mel 滤波器, 25ms/10ms, snip_edges, povey 窗,
/// 去 DC + 预加重 0.97。与官方 kaldi-native-fbank / torchaudio compliance.kaldi 对齐。
struct KaldiFbank {
    window: Vec<f32>,
    /// 每个 mel 滤波器的 (起始 fft bin, 权重)
    mel_filters: Vec<(usize, Vec<f32>)>,
    fft_plan: realfft::RealFftPlanner<f32>,
}

impl KaldiFbank {
    fn new() -> Self {
        let window: Vec<f32> = (0..FRAME_LENGTH)
            .map(|i| {
                let a = 2.0 * std::f32::consts::PI / (FRAME_LENGTH as f32 - 1.0);
                (0.5 - 0.5 * (a * i as f32).cos()).powf(0.85)
            })
            .collect();

        let fft_bin_width = SAMPLE_RATE as f32 / FFT_SIZE as f32;
        let mel_low = mel_scale(20.0);
        let mel_high = mel_scale(SAMPLE_RATE as f32 / 2.0);
        let mel_delta = (mel_high - mel_low) / (NUM_BINS as f32 + 1.0);
        let mut mel_filters = Vec::with_capacity(NUM_BINS);
        for bin in 0..NUM_BINS {
            let left_mel = mel_low + bin as f32 * mel_delta;
            let center_mel = mel_low + (bin as f32 + 1.0) * mel_delta;
            let right_mel = mel_low + (bin as f32 + 2.0) * mel_delta;
            let mut first = None;
            let mut last = None;
            let mut weights = vec![0.0f32; FFT_SIZE / 2];
            for i in 0..FFT_SIZE / 2 {
                let freq = fft_bin_width * i as f32;
                let m = mel_scale(freq);
                if m > left_mel && m < right_mel {
                    let w = if m <= center_mel {
                        (m - left_mel) / (center_mel - left_mel)
                    } else {
                        (right_mel - m) / (right_mel - center_mel)
                    };
                    weights[i] = w;
                    if first.is_none() {
                        first = Some(i);
                    }
                    last = Some(i);
                }
            }
            let (f, l) = (first.unwrap(), last.unwrap());
            mel_filters.push((f, weights[f..=l].to_vec()));
        }

        Self {
            window,
            mel_filters,
            fft_plan: realfft::RealFftPlanner::new(),
        }
    }

    /// 计算 log-fbank, 自动检测并将 [-1.0, 1.0] 归一化浮点采样放大 32768.0 倍以匹配 Kaldi 16-bit PCM 刻度。
    fn compute(&mut self, samples: &[f32]) -> Vec<f32> {
        let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 && max_abs <= 2.0 { 32768.0 } else { 1.0 };
        self.compute_with_scale(samples, scale)
    }

    /// 底层原始计算 (用于单元测试对比未缩放参考特征)。
    #[cfg(test)]
    fn compute_raw(&mut self, samples: &[f32]) -> Vec<f32> {
        self.compute_with_scale(samples, 1.0)
    }

    fn compute_with_scale(&mut self, samples: &[f32], scale: f32) -> Vec<f32> {
        if samples.len() < FRAME_LENGTH {
            return Vec::new();
        }
        let num_frames = 1 + (samples.len() - FRAME_LENGTH) / FRAME_SHIFT;
        let mut feat = vec![0.0f32; num_frames * NUM_BINS];
        if num_frames == 0 {
            return feat;
        }

        let fft = self.fft_plan.plan_fft_forward(FFT_SIZE);
        let mut fft_real = vec![0.0f32; FFT_SIZE];
        let mut spectrum =
            vec![realfft::num_complex::Complex::new(0.0f32, 0.0f32); FFT_SIZE / 2 + 1];

        for i in 0..num_frames {
            let frame = &samples[i * FRAME_SHIFT..i * FRAME_SHIFT + FRAME_LENGTH];
            // 1. 去直流并按指定比例缩放
            let mean = frame.iter().sum::<f32>() / FRAME_LENGTH as f32;
            for (j, s) in frame.iter().enumerate() {
                fft_real[j] = (s - mean) * scale;
            }
            // 2. 预加重 0.97 (与 kaldi-native-fbank 一致: 帧内首样本同样按
            //    x'[0] = x[0] - 0.97*x[0] 处理, 等价于 replicate 边界)
            for j in (1..FRAME_LENGTH).rev() {
                fft_real[j] -= 0.97 * fft_real[j - 1];
            }
            fft_real[0] -= 0.97 * fft_real[0];
            // 3. povey 窗
            for j in 0..FRAME_LENGTH {
                fft_real[j] *= self.window[j];
            }
            for j in FRAME_LENGTH..FFT_SIZE {
                fft_real[j] = 0.0;
            }
            // 4. FFT + 功率谱
            let _ = fft.process(&mut fft_real, &mut spectrum);
            // 5. mel 滤波 + log
            for (m, (start, weights)) in self.mel_filters.iter().enumerate() {
                let mut energy = 0.0f32;
                for (k, w) in weights.iter().enumerate() {
                    let idx = start + k;
                    energy += w * (spectrum[idx].re * spectrum[idx].re + spectrum[idx].im * spectrum[idx].im);
                }
                if energy < 1e-20 {
                    energy = 1e-20;
                }
                feat[i * NUM_BINS + m] = energy.ln();
            }
        }
        feat
    }
}

fn mel_scale(freq: f32) -> f32 {
    1127.0 * (1.0 + freq / 700.0).ln()
}

/// 解析 Kaldi 二进制 ark 中的 CMVN 统计量矩阵, 返回 (means, inverse_std_variances)。
/// 格式 (kaldiio 兼容): `\0B` + 类型 token ("DM"/"FM") + `\x04` + rows + `\x04` + cols
/// + rows*cols 个 float64(float32) 原始数据。
fn parse_kaldi_cmvn(path: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("读取 cmvn.ark 失败 ({}): {}", path.display(), e))?;
    if data.len() < 15 || data[0] != 0 || data[1] != b'B' {
        return Err(format!("cmvn.ark 不是 Kaldi 二进制矩阵: {}", path.display()));
    }
    let mut pos = 2usize;
    // 类型 token: 读到空白字符为止
    let token_start = pos;
    while pos < data.len() && !data[pos].is_ascii_whitespace() {
        pos += 1;
    }
    let token = std::str::from_utf8(&data[token_start..pos])
        .map_err(|_| format!("cmvn.ark token 无效: {}", path.display()))?;
    let float_size = match token {
        "DM" => 8usize,
        "FM" => 4usize,
        other => {
            return Err(format!("cmvn.ark 类型不支持: {:?}", other));
        }
    };
    pos += 1; // 跳过空白

    let read_i32 = |data: &[u8], pos: &mut usize| -> Result<i32, String> {
        if data.get(*pos) != Some(&4) {
            return Err("cmvn.ark 头部格式无效".to_string());
        }
        *pos += 1;
        let bytes = data
            .get(*pos..*pos + 4)
            .ok_or_else(|| "cmvn.ark 头部截断".to_string())?;
        let v = i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        *pos += 4;
        Ok(v)
    };

    let rows = read_i32(&data, &mut pos)? as usize;
    let cols = read_i32(&data, &mut pos)? as usize;
    if rows != 2 || cols < 2 || cols - 1 != FEAT_DIM {
        return Err(format!(
            "cmvn.ark 维度不匹配: 期望 (2, {}), 实际 ({}, {})",
            FEAT_DIM + 1,
            rows,
            cols
        ));
    }
    let count_elem = rows * cols;
    let need = pos + count_elem * float_size;
    if need > data.len() {
        return Err(format!("cmvn.ark 数据截断: 需要 {} 字节, 实际 {}", need, data.len()));
    }
    let mut stats = vec![0.0f64; count_elem];
    if float_size == 8 {
        for (i, chunk) in data[pos..need].chunks_exact(8).enumerate() {
            stats[i] = f64::from_le_bytes(chunk.try_into().unwrap());
        }
    } else {
        for (i, chunk) in data[pos..need].chunks_exact(4).enumerate() {
            stats[i] = f32::from_le_bytes(chunk.try_into().unwrap()) as f64;
        }
    }

    let count = stats[cols - 1];
    if count < 1.0 {
        return Err(format!("cmvn.ark count 无效: {}", count));
    }
    let mut means = vec![0.0f32; FEAT_DIM];
    let mut istd = vec![0.0f32; FEAT_DIM];
    for d in 0..FEAT_DIM {
        let mean = stats[d] / count;
        let variance = (stats[cols + d] / count) - mean * mean;
        let variance = if variance < 1e-20 { 1e-20 } else { variance };
        means[d] = mean as f32;
        istd[d] = (1.0 / variance.sqrt()) as f32;
    }
    Ok((means, istd))
}

#[derive(Clone, Copy, PartialEq)]
enum VadState {
    Silence,
    PossibleSpeech,
    Speech,
    PossibleSilence,
}

/// 与官方 fireredvad VadPostprocessor 逐帧等价的 Rust 移植。
struct VadPostprocessor {
    cfg: FireVadConfig,
}

impl VadPostprocessor {
    fn smooth_prob(&self, probs: &[f32]) -> Vec<f32> {
        let w = self.cfg.smooth_window_size.max(1);
        if w <= 1 {
            return probs.to_vec();
        }
        let mut smoothed = vec![0.0f32; probs.len()];
        let mut window_sum = 0.0f32;
        for (i, p) in probs.iter().enumerate() {
            window_sum += p;
            if i >= w {
                window_sum -= probs[i - w];
            }
            if i < w - 1 {
                // 边界: 用累积平均 (与官方 np.convolve + 前缘覆写一致)
                smoothed[i] = window_sum / (i as f32 + 1.0);
            } else {
                smoothed[i] = window_sum / w as f32;
            }
        }
        smoothed
    }

    /// 状态机平滑: min_speech_frame / min_silence_frame 约束下的状态迁移。
    fn smooth_preds_with_state_machine(&self, binary: &[i32]) -> Vec<i32> {
        if self.cfg.min_speech_frame <= 0 && self.cfg.min_silence_frame <= 0 {
            return binary.to_vec();
        }
        let n = binary.len();
        let mut decisions = vec![0i32; n];
        let mut state = VadState::Silence;
        let mut speech_start = -1i64;
        let mut silence_start = -1i64;
        for (t, is_speech) in binary.iter().enumerate() {
            let t = t as i64;
            match state {
                VadState::Silence => {
                    if *is_speech == 1 {
                        state = VadState::PossibleSpeech;
                        speech_start = t;
                    }
                }
                VadState::PossibleSpeech => {
                    if *is_speech == 1 {
                        if t - speech_start >= self.cfg.min_speech_frame as i64 {
                            state = VadState::Speech;
                            for d in (speech_start as usize)..(t as usize) {
                                decisions[d] = 1;
                            }
                        }
                    } else {
                        state = VadState::Silence;
                        speech_start = -1;
                    }
                }
                VadState::Speech => {
                    if *is_speech == 0 {
                        state = VadState::PossibleSilence;
                        silence_start = t;
                    }
                }
                VadState::PossibleSilence => {
                    if *is_speech == 0 {
                        if t - silence_start >= self.cfg.min_silence_frame as i64 {
                            state = VadState::Silence;
                            speech_start = -1;
                        }
                    } else {
                        state = VadState::Speech;
                        silence_start = -1;
                    }
                }
            }
            let decision = match state {
                VadState::Speech | VadState::PossibleSilence => 1,
                VadState::Silence | VadState::PossibleSpeech => 0,
            };
            decisions[t as usize] = decision;
        }
        decisions
    }

    /// 语音段起点回拨: 段起点前 smooth_window_size 帧标记为语音 (平滑窗起点修正)。
    fn fix_smooth_window_start(&self, decisions: &[i32]) -> Vec<i32> {
        let w = self.cfg.smooth_window_size.max(1);
        let mut new_decisions = decisions.to_vec();
        for t in 1..decisions.len() {
            if decisions[t - 1] == 0 && decisions[t] == 1 {
                let start = t.saturating_sub(w);
                for d in start..t {
                    new_decisions[d] = 1;
                }
            }
        }
        new_decisions
    }

    /// 合并短静音段 (merge_silence_frame > 0 时生效)。
    fn merge_short_silence(&self, decisions: &[i32]) -> Vec<i32> {
        if self.cfg.merge_silence_frame <= 0 {
            return decisions.to_vec();
        }
        let mut new_decisions = decisions.to_vec();
        let mut silence_start: Option<usize> = None;
        for t in 1..decisions.len() {
            if decisions[t - 1] == 1 && decisions[t] == 0 && silence_start.is_none() {
                silence_start = Some(t);
            } else if decisions[t - 1] == 0 && decisions[t] == 1 && silence_start.is_some() {
                let start = silence_start.take().unwrap();
                if t - start < self.cfg.merge_silence_frame {
                    for d in start..t {
                        new_decisions[d] = 1;
                    }
                }
            }
        }
        new_decisions
    }

    /// 语音段前后各扩展 extend_speech_frame 帧。
    fn extend_speech(&self, decisions: &[i32]) -> Vec<i32> {
        if self.cfg.extend_speech_frame <= 0 {
            return decisions.to_vec();
        }
        let n = decisions.len();
        let mut new_decisions = vec![0i32; n];
        for t in 0..n {
            if decisions[t] == 1 {
                let start = t.saturating_sub(self.cfg.extend_speech_frame);
                let end = (t + self.cfg.extend_speech_frame + 1).min(n);
                for d in start..end {
                    new_decisions[d] = 1;
                }
            }
        }
        new_decisions
    }

    /// 超长语音段 (max_speech_frame) 在窗口内概率最低处拆分。
    /// 与官方 _find_split_points 一致: 以拆前的段为单位, 窗口为
    /// [split_cursor + max/2, split_cursor + max), 取 argmin 作为拆分点。
    fn split_long_speech(&self, decisions: &[i32], probs: &[f32]) -> Vec<i32> {
        if self.cfg.max_speech_frame <= 0 {
            return decisions.to_vec();
        }
        let mut new_decisions = decisions.to_vec();
        let segments = self.decision_to_segment(decisions, 1e9);
        for seg in &segments {
            let start_frame = seg.start as usize;
            let end_frame = seg.end as usize;
            if end_frame.saturating_sub(start_frame) <= self.cfg.max_speech_frame {
                continue;
            }
            let segment_probs = &probs[start_frame..end_frame.min(probs.len())];
            let length = segment_probs.len();
            let mut cursor = 0usize;
            while length.saturating_sub(cursor) > self.cfg.max_speech_frame {
                let win_start = cursor + self.cfg.max_speech_frame / 2;
                let win_end = cursor + self.cfg.max_speech_frame;
                if win_start >= length || win_end > length {
                    break;
                }
                let window = &segment_probs[win_start..win_end];
                let (min_idx, _) = window
                    .iter()
                    .enumerate()
                    .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap();
                let split_frame = start_frame + win_start + min_idx;
                new_decisions[split_frame] = 0;
                cursor = win_start + min_idx + 1;
            }
        }
        new_decisions
    }

    /// 完整后处理: 返回逐帧决策 (1=语音)。
    fn process(&self, raw_probs: &[f32]) -> Vec<i32> {
        if raw_probs.is_empty() {
            return Vec::new();
        }
        let smoothed = self.smooth_prob(raw_probs);
        let binary: Vec<i32> = smoothed.iter().map(|p| if *p >= self.cfg.threshold { 1 } else { 0 }).collect();
        let decisions = self.smooth_preds_with_state_machine(&binary);
        let fixed = self.fix_smooth_window_start(&decisions);
        let merged = self.merge_short_silence(&fixed);
        let extended = self.extend_speech(&merged);
        self.split_long_speech(&extended, raw_probs)
    }

    /// 决策序列 -> 语音段 (10ms 帧单位)。末尾语音段结束时间按 wav 时长截断。
    fn decision_to_segment(&self, decisions: &[i32], wav_dur_sec: f32) -> Vec<VadSegment> {
        let mut segments = Vec::new();
        let mut speech_start: Option<usize> = None;
        for (t, d) in decisions.iter().enumerate() {
            if *d == 1 && speech_start.is_none() {
                speech_start = Some(t);
            } else if *d == 0 && speech_start.is_some() {
                let s = speech_start.take().unwrap();
                segments.push(VadSegment {
                    start: s as u32,
                    end: t as u32,
                });
            }
        }
        if let Some(s) = speech_start {
            let end_s = ((decisions.len() as f32 * 0.01) + 0.025).min(wav_dur_sec);
            segments.push(VadSegment {
                start: s as u32,
                end: (end_s * 100.0).round() as u32,
            });
        }
        segments
    }
}

/// FireRedVAD 非流式语音活动检测引擎。
pub struct FireVadEngine {
    session: Session,
    means: Vec<f32>,
    istd: Vec<f32>,
    fbank: KaldiFbank,
    postprocessor: VadPostprocessor,
}

impl FireVadEngine {
    /// 加载 ONNX 模型 (CPU 推理) 及其同目录下的 cmvn.ark。
    pub fn load(model_path: &str, cfg: FireVadConfig) -> Result<Self, String> {
        let model_path = Path::new(model_path);
        let cmvn_path = model_path
            .parent()
            .map(|p| p.join("cmvn.ark"))
            .unwrap_or_else(|| Path::new("cmvn.ark").to_path_buf());
        let (means, istd) = parse_kaldi_cmvn(&cmvn_path)?;

        let mut builder = Session::builder()
            .map_err(|e| format!("Session::builder: {}", e))?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("opt_level: {}", e))?;
        // VAD 线程已绑定 E 核: 单线程推理, 避免 ORT 线程池逃逸抢占 P 核;
        // 模型仅 ~2.4MB, 30s 窗口推理约 10ms, 单线程足够。
        builder = builder
            .with_intra_threads(1)
            .map_err(|e| format!("intra_threads: {}", e))?;
        let cpu_ep = ort::ep::CPU::default().with_arena_allocator(false).build();
        let session = builder
            .with_execution_providers([cpu_ep])
            .map_err(|e| format!("CPU EP register: {}", e))?
            .commit_from_file(model_path)
            .map_err(|e| format!("commit_from_file ({}): {}", model_path.display(), e))?;

        Ok(Self {
            session,
            means,
            istd,
            fbank: KaldiFbank::new(),
            postprocessor: VadPostprocessor { cfg },
        })
    }

    /// 对一段 16kHz 单声道音频执行语音活动检测。
    /// 返回语音段列表, 段起止为相对输入起始的 10ms 帧索引。
    /// 推理失败时上抛错误: 调用方不得把推理失败当成"无语音"。
    pub fn detect(&mut self, samples: &[f32]) -> Result<Vec<VadSegment>, String> {
        let mut feat = self.fbank.compute(samples);
        if feat.is_empty() {
            return Ok(Vec::new());
        }
        let t = feat.len() / FEAT_DIM;
        for i in 0..feat.len() {
            let d = i % FEAT_DIM;
            feat[i] = (feat[i] - self.means[d]) * self.istd[d];
        }

        let tensor = Tensor::<f32>::from_array((
            vec![1_i64, t as i64, FEAT_DIM as i64],
            feat.into_boxed_slice(),
        ))
        .map_err(|e| format!("Tensor::from_array: {}", e))?;
        let outputs = self
            .session
            .run(inputs!["feat" => tensor])
            .map_err(|e| format!("FireRedVAD inference error: {:?}", e))?;
        let (_, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| format!("extract probs: {}", e))?;
        
        // 自动适配 1-class (普通 VAD) 与 3-class (AED: [speech, singing, music])
        let probs: Vec<f32> = if data.len() == t * 3 {
            // 提取第一通道: 纯人声 (speech) 概率通道，彻底排除纯背景音乐 (music) 与歌唱干扰
            (0..t).map(|i| data[i * 3 + 0]).collect()
        } else {
            data.to_vec()
        };

        let decisions = self.postprocessor.process(&probs);
        let wav_dur = samples.len() as f32 / SAMPLE_RATE as f32;
        Ok(self.postprocessor.decision_to_segment(&decisions, wav_dur))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 读取 .npy 文件 (float32/float64, 小端), 用于加载参考向量。
    fn read_npy(path: &str) -> (Vec<usize>, Vec<f32>) {
        let data = std::fs::read(path).unwrap();
        assert_eq!(&data[0..6], b"\x93NUMPY");
        let major = data[6];
        let header_len = if major == 1 {
            u16::from_le_bytes([data[8], data[9]]) as usize
        } else {
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
        };
        let header_start = if major == 1 { 10 } else { 12 };
        let header = std::str::from_utf8(&data[header_start..header_start + header_len]).unwrap();
        let shape_start = header.find('(').unwrap();
        let shape_end = header.find(')').unwrap();
        let dims: Vec<usize> = header[shape_start + 1..shape_end]
            .split(',')
            .filter_map(|s| s.trim().parse().ok())
            .collect();
        let descr_pos = header.find("descr").unwrap();
        let descr = &header[descr_pos..descr_pos + 30];
        let dtype = if descr.contains("<f4") { 4 } else if descr.contains("<f8") { 8 } else { panic!("dtype: {}", descr) };
        let body = &data[header_start + header_len..];
        let mut vals = Vec::with_capacity(body.len() / dtype);
        if dtype == 4 {
            for c in body.chunks_exact(4) {
                vals.push(f32::from_le_bytes(c.try_into().unwrap()));
            }
        } else {
            for c in body.chunks_exact(8) {
                vals.push(f64::from_le_bytes(c.try_into().unwrap()) as f32);
            }
        }
        (dims, vals)
    }

    fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_fbank_matches_reference() {
        let wav_path = r"C:\Temp\kilo\test_wav.npy";
        let ref_path = r"C:\Temp\kilo\test_fbank_ref.npy";
        if !Path::new(wav_path).exists() || !Path::new(ref_path).exists() {
            println!("Test npy reference files not found, skipping test_fbank_matches_reference");
            return;
        }
        let (dims, wav) = read_npy(wav_path);
        assert_eq!(dims, vec![64000]);
        let (fdims, ref_feat) = read_npy(ref_path);
        assert_eq!(fdims, vec![398, 80]);

        let mut fbank = KaldiFbank::new();
        let feat = fbank.compute_raw(&wav);
        assert_eq!(feat.len(), 398 * 80);
        let diff = max_abs_diff(&feat, &ref_feat);
        println!("fbank max diff vs torchaudio: {}", diff);
        assert!(diff < 2e-4, "fbank mismatch: {}", diff);
    }

    #[test]
    fn test_cmvn_and_inference_matches_reference() {
        let wav_path = r"C:\Temp\kilo\test_wav.npy";
        let model = r"C:\Temp\kilo\onnx_export\fireredvad_vad.onnx";
        let cmvn_ark = r"C:\Projects\audio2srt\FireRedVAD\VAD\cmvn.ark";
        if !Path::new(wav_path).exists() || !Path::new(model).exists() || !Path::new(cmvn_ark).exists() {
            println!("Test reference model/data files not found, skipping test_cmvn_and_inference_matches_reference");
            return;
        }

        let (dims, wav) = read_npy(wav_path);
        assert_eq!(dims, vec![64000]);
        let (pdims, probs_ref) = read_npy(r"C:\Temp\kilo\test_probs_ref.npy");
        assert_eq!(pdims, vec![398, 1]);

        let (means, istd) = parse_kaldi_cmvn(Path::new(cmvn_ark)).unwrap();
        let (mdims, means_ref) = read_npy(r"C:\Temp\kilo\cmvn_means.npy");
        let (idims, istd_ref) = read_npy(r"C:\Temp\kilo\cmvn_istd.npy");
        assert_eq!(mdims, vec![80]);
        assert_eq!(idims, vec![80]);
        assert!(max_abs_diff(&means, &means_ref) < 1e-5, "means mismatch: {}", max_abs_diff(&means, &means_ref));
        assert!(max_abs_diff(&istd, &istd_ref) < 1e-5, "istd mismatch: {}", max_abs_diff(&istd, &istd_ref));

        // 使用宽松阈值确保输出段非空, 重点验证模型输出概率
        let mut engine = FireVadEngine::load(
            model,
            FireVadConfig {
                threshold: 0.001,
                ..Default::default()
            },
        )
        .unwrap();
        let segs = engine.detect(&wav).unwrap();
        assert!(!segs.is_empty(), "expected speech segments");

        // 直接复算概率并与 onnxruntime 参考对比 (使用未缩放特征对比参考张量)
        let mut feat = engine.fbank.compute_raw(&wav);
        for i in 0..feat.len() {
            let d = i % FEAT_DIM;
            feat[i] = (feat[i] - engine.means[d]) * engine.istd[d];
        }
        let t = feat.len() / FEAT_DIM;
        let tensor = Tensor::<f32>::from_array((
            vec![1_i64, t as i64, FEAT_DIM as i64],
            feat.into_boxed_slice(),
        ))
        .unwrap();
        let outputs = engine.session.run(inputs!["feat" => tensor]).unwrap();
        let (_, data) = outputs[0].try_extract_tensor::<f32>().unwrap();
        let probs: Vec<f32> = data.to_vec();
        let ref_probs: Vec<f32> = probs_ref.clone();
        assert_eq!(probs.len(), ref_probs.len());
        let diff = max_abs_diff(&probs, &ref_probs);
        println!("probs max diff vs onnxruntime: {}", diff);
        assert!(diff < 1e-5, "probs mismatch: {}", diff);
    }

    #[test]
    fn test_postprocessor_state_machine() {
        let cfg = FireVadConfig {
            smooth_window_size: 1,
            threshold: 0.5,
            min_speech_frame: 5,
            max_speech_frame: 1000,
            min_silence_frame: 4,
            merge_silence_frame: 0,
            extend_speech_frame: 0,
        };
        let pp = VadPostprocessor { cfg };
        // 3 帧语音不够 min_speech_frame -> 无段
        let d = pp.process(&[0.1, 0.9, 0.9, 0.9, 0.1, 0.1]);
        assert_eq!(d, vec![0, 0, 0, 0, 0, 0]);
        // 足够长的语音 + 短静音不打断
        let mut probs = vec![0.1f32; 30];
        for p in probs.iter_mut().skip(4).take(20) {
            *p = 0.9;
        }
        let d = pp.process(&probs);
        let segs = pp.decision_to_segment(&d, 3.0);
        assert_eq!(segs.len(), 1);
        assert!(segs[0].start <= 4 && segs[0].end >= 24);
    }

    #[test]
    fn test_cmvn_parse_error_paths() {
        let err = parse_kaldi_cmvn(Path::new(r"C:\Temp\kilo\onnx_export\fireredvad_vad.onnx"));
        assert!(err.is_err());
    }

    #[test]
    fn test_real_audio_detect() {
        let wav_path = r"C:\Temp\kilo\FireRedVAD-code\assets\hello_zh.wav";
        let model_path = r"C:\Projects\audio2srt\assets\models\fireredvad_vad.onnx";
        if !Path::new(wav_path).exists() || !Path::new(model_path).exists() {
            println!("Test wav or model not found, skipping test_real_audio_detect");
            return;
        }

        // 读取 16-bit PCM WAV 数据转为 [-1.0, 1.0] 的浮点采样
        let bytes = std::fs::read(wav_path).unwrap();
        // WAV 头部 44 字节后为 PCM 采样
        let pcm_bytes = &bytes[44..];
        let mut samples = Vec::with_capacity(pcm_bytes.len() / 2);
        for chunk in pcm_bytes.chunks_exact(2) {
            let val = i16::from_le_bytes([chunk[0], chunk[1]]);
            samples.push(val as f32 / 32768.0);
        }

        let mut engine = FireVadEngine::load(model_path, FireVadConfig::default()).unwrap();
        let segs = engine.detect(&samples).unwrap();
        println!("Detected segments on hello_zh.wav: {:?}", segs);
        assert!(!segs.is_empty(), "hello_zh.wav should have detected speech segments!");
    }
}
