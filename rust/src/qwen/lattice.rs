//! LattifAI Lattice-1 对齐器。
//!
//! 纯本地强制对齐核心资产:
//!   1. `acoustic_opt.onnx`     声学模型: 原始波形 (1, T) @16k -> emission (1, F, V), 帧移 20ms
//!   2. `phoneme_symbols.json`  213 个 IPA 音素符号
//!   3. `topology.json`         transition.bin 的 (token, ctx) -> pdf 映射; 含 num_pdfids (= 7373)
//!   4. `g2pp_encoder.onnx`     官方神经网络 G2P 编码器 (DeepPhonemizer 4 层 Transformer)
//!   5. `g2pp_decoder.onnx`     官方神经网络 G2P 自回归解码器
//!   6. `text_symbols.json`     官方 5635 字符输入字典映射
//!
//! 对齐算法: 音素级单调 Viterbi (词间插入可零帧跳过的静音 Gap), 每帧对每个音素
//! 取其 pdf 集合的 logsumexp —— 对上下文绑定态做边际化近似。
//! 词置信度 = 该词各帧上所属音素集合的概率质量均值。
//!
//! 资产目录约定 (任一命中即启用): 环境变量 AUDIO2SRT_LATTICE_DIR /
//! <qwen_dir>/../Lattice-1 / %APPDATA%/com.audio2srt/audio2srt/models/Lattice-1。
//! 加载失败或对齐不健康时上层自动回退 Qwen3-ForcedAligner, 不影响既有链路。

use crate::qwen::aligner::{map_to_media, tokenize_for_align, AlignedToken, AlignmentResult, TimelineSpan};
use crate::qwen::error::QwenError;
use crate::qwen::g2p::G2pEngine;
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

/// 音素 token id 起始偏移: transition.bin 中 token id 0..=216 = 特殊符 + 213 音素。
/// 特殊符占用 0..=2 (pdf_id 有 0->0/1->1/2->2 的恒等映射), 音素从 3 开始按
/// phoneme_symbols.json 顺序排列。
const PHONE_TOKEN_OFFSET: u16 = 3;
/// emission 帧移 (ms): config.json frame_shift=10ms x subsampling_factor=2
const FRAME_SHIFT_MS: i64 = 20;
/// 静音/Gap 先验对数加成 (消除音素~35个PDF vs Gap 3个PDF在底噪下的维度失衡, 避免静音段贪婪跳入下词首音素)
const GAP_PRIOR_BONUS: f32 = 2.4;
/// 声学模型单次推理上限 (采样数): 与官方 worker 一致按 60s 分块
const CHUNK_SAMPLES: usize = 60 * 16000;
/// 输入过短时右侧补零到 320 采样 (官方 worker 同款处理)
const MIN_SAMPLES_PAD: usize = 320;
/// 词项 OOV 比例上限: 超过说明词典覆盖不足, 对齐结果不可信
const MAX_OOV_RATIO: f32 = 0.4;
/// 词时长下限 (ms), 低于视为挤压错位
const WORD_DUR_MIN_MS: i64 = 20;
/// 低置信词阈值 (音素集合概率质量, 7373类下0.005已达随机先验37倍)
const CONF_LOW: f32 = 0.005;
/// 低置信词比例上限 (放宽至 0.90，让有强 BGM 场景也能稳定输出单调对齐，不随意回退)
const LOW_CONF_RATIO: f32 = 0.90;
/// 词时长总和 / 音频时长 合理区间 (VAD 切块语音占比区间)
const COVERAGE_MIN: f32 = 0.05;
const COVERAGE_MAX: f32 = 1.8;

/// 运行时探测 Lattice-1 资产目录; 找不到返回 None (静默, 属正常配置)。
pub fn probe_lattice_dir(qwen_dir: &str) -> Option<String> {
    if let Ok(env_dir) = std::env::var("AUDIO2SRT_LATTICE_DIR") {
        if is_lattice_dir(&env_dir) {
            return Some(env_dir);
        }
    }
    // <qwen_dir>/../Lattice-1
    if let Some(parent) = Path::new(qwen_dir).parent() {
        let sibling = parent.join("Lattice-1");
        if is_lattice_dir(&sibling.to_string_lossy()) {
            return Some(sibling.to_string_lossy().into_owned());
        }
    }
    // %APPDATA%/com.audio2srt/audio2srt/models/Lattice-1
    if let Ok(appdata) = std::env::var("APPDATA") {
        let default = Path::new(&appdata)
            .join("com.audio2srt")
            .join("audio2srt")
            .join("models")
            .join("Lattice-1");
        if is_lattice_dir(&default.to_string_lossy()) {
            return Some(default.to_string_lossy().into_owned());
        }
    }
    None
}

fn is_lattice_dir(dir: &str) -> bool {
    let p = Path::new(dir);
    p.join("acoustic_opt.onnx").is_file()
        && p.join("phoneme_symbols.json").is_file()
        && p.join("topology.json").is_file()
        && p.join("g2pp_encoder.onnx").is_file()
        && p.join("g2pp_decoder.onnx").is_file()
        && p.join("text_symbols.json").is_file()
}

/// 该语言是否可路由到 Lattice-1 (官方支持 EN/ZH/DE; 粤语声调体系不同不路由)
pub fn language_supported(language: Option<&str>) -> bool {
    match language.map(|s| s.trim().to_lowercase()) {
        Some(l) => matches!(
            l.as_str(),
            "en" | "eng" | "english" | "zh" | "zh-cn" | "cmn" | "chinese" | "de" | "deu" | "german"
        ),
        None => false,
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum NodeKind {
    Phone,
    /// 词间静音: 只落在 blank/special pdf 集合上, 可消耗零帧
    Gap,
}

struct SeqNode {
    kind: NodeKind,
    /// Phone: 音素 token id; Gap 未用
    token: u16,
    /// 所属词索引 (Gap 为 usize::MAX)
    word_idx: usize,
}

pub struct LatticeAligner {
    session: Session,
    /// IPA 符号表 (索引 i -> token id PHONE_TOKEN_OFFSET + i)
    #[allow(dead_code)]
    vocab: Vec<String>,
    #[allow(dead_code)]
    sym_to_tok: HashMap<String, u16>,
    /// token id -> 去重升序 pdf 集合 (来自 topology.json)
    phone_pdfs: Vec<Vec<u32>>,
    /// 特殊 token (0..PHONE_TOKEN_OFFSET) pdf 并集: Gap 帧打分集合
    gap_pdfs: Vec<u32>,
    /// 官方原生神经网络 G2P 引擎 (带 LRU 内存缓存)
    g2p: G2pEngine,
    unk_pron: Option<Vec<u16>>,
    /// emission 输出维度 (= topology.json num_pdfids, Lattice-1 为 7373)
    vocab_size: usize,
}

impl LatticeAligner {
    pub fn load(dir: &str) -> Result<Self, String> {
        let root = Path::new(dir);
        let vocab_bytes = std::fs::read(root.join("phoneme_symbols.json"))
            .map_err(|e| format!("read phoneme_symbols.json: {e}"))?;
        let vocab: Vec<String> = serde_json::from_slice(&vocab_bytes)
            .map_err(|e| format!("parse phoneme_symbols.json: {e}"))?;
        if vocab.is_empty() || vocab.len() > 1000 {
            return Err(format!("unexpected phoneme_symbols size {}", vocab.len()));
        }

        let topo_bytes =
            std::fs::read(root.join("topology.json")).map_err(|e| format!("read topology.json: {e}"))?;
        #[derive(serde::Deserialize)]
        struct Topology {
            numpdfids: usize,
            phonepdfs: HashMap<String, Vec<u32>>,
        }
        let topo: Topology = serde_json::from_slice(&topo_bytes).map_err(|e| format!("parse topology.json: {e}"))?;
        if topo.numpdfids == 0 || topo.numpdfids > 100_000 {
            return Err(format!("unexpected num_pdfids {}", topo.numpdfids));
        }
        let mut phone_pdfs: Vec<Vec<u32>> = vec![Vec::new(); 256];
        // 特殊 token 0, 1, 2 分别映射到 Blank(0), Silence(1), Disambiguation(2)
        phone_pdfs[0] = vec![0];
        phone_pdfs[1] = vec![1];
        phone_pdfs[2] = vec![2];

        for (k, v) in &topo.phonepdfs {
            let tok: u16 = k.parse().map_err(|_| format!("bad topology key {k}"))?;
            if tok as usize >= phone_pdfs.len() {
                return Err(format!("topology token id {tok} out of range"));
            }
            let mut set = v.clone();
            set.sort_unstable();
            set.dedup();
            phone_pdfs[tok as usize] = set;
        }
        // 一致性检查: 全部音素 token 必须有非空 pdf 集合, 否则偏移假设错误
        for (i, sym) in vocab.iter().enumerate() {
            let tok = PHONE_TOKEN_OFFSET as usize + i;
            if tok >= phone_pdfs.len() || phone_pdfs[tok].is_empty() {
                return Err(format!(
                    "phone '{sym}' (token {}) has empty pdf set - token offset assumption wrong",
                    PHONE_TOKEN_OFFSET as usize + i
                ));
            }
        }
        let mut gap_set: Vec<u32> = phone_pdfs[..PHONE_TOKEN_OFFSET as usize]
            .iter()
            .flat_map(|s| s.iter().copied())
            .collect();
        if gap_set.is_empty() {
            gap_set = vec![0, 1, 2];
        }
        gap_set.sort_unstable();
        gap_set.dedup();

        let sym_to_tok: HashMap<String, u16> = vocab
            .iter()
            .enumerate()
            .map(|(i, s)| (s.clone(), PHONE_TOKEN_OFFSET + i as u16))
            .collect();
        let spn_tok = sym_to_tok.get("spn").copied();

        // 载入官方神经网络 G2P 引擎
        let g2p = G2pEngine::load(root).map_err(|e| format!("load g2p engine: {e}"))?;

        println!(
            "[lattice] assets loaded: dir={dir}, phonemes={}, num_pdfids={}, gap_pdfs={}",
            vocab.len(),
            topo.numpdfids,
            gap_set.len()
        );

        // 声学模型 Session (CPU EP; 与官方 worker 相同的 CPU 推理路径)
        let model_path = root.join("acoustic_opt.onnx");
        let session = Session::builder()
            .map_err(|e| format!("Session::builder: {e}"))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| format!("opt_level: {e}"))?
            .with_intra_threads(8)
            .map_err(|e| format!("intra_threads: {e}"))?
            .commit_from_file(model_path.to_string_lossy().as_ref())
            .map_err(|e| format!("commit acoustic_opt.onnx: {e}"))?;

        Ok(Self {
            session,
            vocab,
            sym_to_tok,
            phone_pdfs,
            gap_pdfs: gap_set,
            g2p,
            unk_pron: spn_tok.map(|t| vec![t]),
            vocab_size: topo.numpdfids,
        })
    }

    /// 动态多语种 G2P: 原生官方神经网络 G2P 推理 (支持中/英/德及语境转换)
    pub fn get_pronunciation(&self, word: &str) -> Option<Vec<u16>> {
        let key = word.trim();
        if key.is_empty() {
            return None;
        }
        if let Some(pron) = self.g2p.get_pronunciation(key) {
            return Some(pron);
        }
        // 清理标点后再次重试 (如 "we're", "sunday.", "你好！")
        let clean = key.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'');
        if clean != key && !clean.is_empty() {
            if let Some(pron) = self.g2p.get_pronunciation(clean) {
                return Some(pron);
            }
        }
        None
    }

    /// 强制对齐: 与 QwenAligner::align 相同的签名语义。
    /// 任何失败 (资产/覆盖/健康度/取消) 返回 Err, 由 runtime 回退 Qwen 对齐器。
    pub fn align(
        &mut self,
        samples_16k: &[f32],
        text: &str,
        segment_start_ms: u64,
        segment_end_ms: u64,
        language: Option<&str>,
        timeline: &[TimelineSpan],
        cancel: &AtomicBool,
    ) -> Result<AlignmentResult, QwenError> {
        let start_time = std::time::Instant::now();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
                align_quality: "LatticeAligned".into(),
            });
        }

        // 1. 分词 + 查发音词典 (官方神经网络 G2P 得到精准音素)
        let words = tokenize_for_align(trimmed, language);
        if words.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
                align_quality: "LatticeAligned".into(),
            });
        }
        let mut prons: Vec<Vec<u16>> = Vec::with_capacity(words.len());
        let mut oov = 0usize;
        for w in &words {
            match self.get_pronunciation(w).or_else(|| self.unk_pron.clone()) {
                Some(p) => prons.push(p),
                None => {
                    oov += 1;
                    prons.push(self.unk_pron.clone().unwrap_or_else(|| vec![0]));
                }
            }
        }
        let oov_ratio = oov as f32 / words.len() as f32;
        if oov_ratio > MAX_OOV_RATIO {
            return Err(QwenError::BackendUnavailable(format!(
                "lattice: excessive oov ratio {oov_ratio:.2} (threshold {MAX_OOV_RATIO})"
            )));
        }

        // 2. 线性拓扑: Gap -> Phone1 -> ... -> PhoneN -> Gap ...
        let mut nodes: Vec<SeqNode> = Vec::new();
        nodes.push(SeqNode {
            kind: NodeKind::Gap,
            token: 0,
            word_idx: usize::MAX,
        });
        for (wi, pron) in prons.iter().enumerate() {
            for &tok in pron {
                nodes.push(SeqNode {
                    kind: NodeKind::Phone,
                    token: tok,
                    word_idx: wi,
                });
            }
            nodes.push(SeqNode {
                kind: NodeKind::Gap,
                token: 0,
                word_idx: usize::MAX,
            });
        }
        let n_states = nodes.len();

        // 3. 声学模型 emission: [n_frames x vocab_size] (展平), 帧移 20ms
        let emissions = self.compute_emission(samples_16k, cancel)?;
        let n_frames = emissions.len() / self.vocab_size;
        if n_frames == 0 {
            return Err(QwenError::BackendUnavailable("lattice: 0 emission frames".into()));
        }

        // 4. 音素/Gap 帧打分缓存: 提取本序列用到的所有唯一 token 的 logsumexp (以及 Gap 的)
        let mut unique_tokens: Vec<u16> = prons.iter().flatten().copied().collect();
        unique_tokens.sort_unstable();
        unique_tokens.dedup();

        let mut slot_of: HashMap<u16, usize> = HashMap::with_capacity(unique_tokens.len());
        for (slot, &tok) in unique_tokens.iter().enumerate() {
            slot_of.insert(tok, slot);
        }
        let n_slots = unique_tokens.len();
        let mut scores = vec![f32::NEG_INFINITY; n_frames * n_slots];
        let mut gap_scores = vec![f32::NEG_INFINITY; n_frames];

        for t in 0..n_frames {
            let frame = &emissions[t * self.vocab_size..(t + 1) * self.vocab_size];

            // Gap 分数 (在特殊符集合上边际化) + 先验加成 (一次扫描找 max，单次 ln 计算)
            let mut gap_max = f32::NEG_INFINITY;
            for &pdf in &self.gap_pdfs {
                let p = pdf as usize;
                if p < frame.len() && frame[p] > gap_max {
                    gap_max = frame[p];
                }
            }
            if gap_max > f32::NEG_INFINITY {
                let mut sum = 0f32;
                for &pdf in &self.gap_pdfs {
                    let p = pdf as usize;
                    if p < frame.len() {
                        sum += (frame[p] - gap_max).exp();
                    }
                }
                gap_scores[t] = gap_max + sum.ln() + GAP_PRIOR_BONUS;
            } else {
                gap_scores[t] = f32::NEG_INFINITY;
            }

            // 各音素分数 (在其 pdf 集合上边际化，避免 pairwise logaddexp 的数百万次 exp/ln 开销)
            for (slot, &tok) in unique_tokens.iter().enumerate() {
                let pdfs = &self.phone_pdfs[tok as usize];
                let mut p_max = f32::NEG_INFINITY;
                for &pdf in pdfs {
                    let p = pdf as usize;
                    if p < frame.len() && frame[p] > p_max {
                        p_max = frame[p];
                    }
                }
                if p_max > f32::NEG_INFINITY {
                    let mut sum = 0f32;
                    for &pdf in pdfs {
                        let p = pdf as usize;
                        if p < frame.len() {
                            sum += (frame[p] - p_max).exp();
                        }
                    }
                    scores[t * n_slots + slot] = p_max + sum.ln();
                } else {
                    scores[t * n_slots + slot] = f32::NEG_INFINITY;
                }
            }
        }

        // 5. Viterbi 动态规划: 单调沿线性拓扑向前推进
        let mut dp = vec![f32::NEG_INFINITY; n_states];
        let mut new_dp = vec![f32::NEG_INFINITY; n_states];
        // bp[t * n_states + s] 记录在帧 t 时刻处于状态 s 的前驱状态 (t 从 1 开始)
        let mut bp = vec![0usize; n_frames * n_states];

        // 初始帧 t=0 赋值
        dp[0] = gap_scores[0];
        if n_states > 1 {
            dp[1] = scores[slot_of[&nodes[1].token]];
        }

        for t in 1..n_frames {
            if cancel.load(Ordering::Relaxed) {
                return Err(QwenError::Cancelled);
            }
            new_dp.fill(f32::NEG_INFINITY);

            for s in 0..n_states {
                let emit = match nodes[s].kind {
                    NodeKind::Gap => gap_scores[t],
                    NodeKind::Phone => scores[t * n_slots + slot_of[&nodes[s].token]],
                };

                let mut best_score = f32::NEG_INFINITY;
                let mut best_prev = usize::MAX;

                // 1. 自环 (停留在状态 s，消耗帧 t)
                if dp[s] > best_score {
                    best_score = dp[s];
                    best_prev = s;
                }
                // 2. 从上一状态推进 (s - 1)
                if s >= 1 && dp[s - 1] > best_score {
                    best_score = dp[s - 1];
                    best_prev = s - 1;
                }
                // 3. 跨过 Gap 静音状态推进 (s - 2，当前一状态为 Gap 时可直接从上一音素进入本音素)
                if s >= 2 && nodes[s - 1].kind == NodeKind::Gap && dp[s - 2] > best_score {
                    best_score = dp[s - 2];
                    best_prev = s - 2;
                }

                if best_score != f32::NEG_INFINITY {
                    new_dp[s] = best_score + emit;
                    bp[t * n_states + s] = best_prev;
                }
            }

            dp.copy_from_slice(&new_dp);
        }

        // 终态回溯起点: 尾 Gap (n_states-1) 或尾音素 (n_states-2, 尾 Gap 零帧)
        let last_gap = dp[n_states - 1];
        let last_phone = if n_states >= 2 {
            dp[n_states - 2]
        } else {
            f32::NEG_INFINITY
        };
        let end_state = if last_gap >= last_phone {
            n_states - 1
        } else {
            n_states - 2
        };
        if dp[end_state] == f32::NEG_INFINITY {
            return Err(QwenError::BackendUnavailable(
                "lattice: viterbi failed to reach end state".into(),
            ));
        }

        // 6. 回溯: 帧同步状态序列映射 (彻底杜绝 t=0 截断丢失帧的零帧 Bug)
        let mut state_of_frame = vec![0usize; n_frames];
        let mut cur_s = end_state;
        state_of_frame[n_frames - 1] = cur_s;
        for t in (1..n_frames).rev() {
            cur_s = bp[t * n_states + cur_s];
            state_of_frame[t - 1] = cur_s;
        }

        // 7. 词级聚合: 帧区间 + 置信度
        let mut word_first = vec![usize::MAX; words.len()];
        let mut word_last = vec![0usize; words.len()];
        let mut word_conf_sum = vec![0f32; words.len()];
        let mut word_frames_n = vec![0usize; words.len()];
        for (t, &s) in state_of_frame.iter().enumerate() {
            let node = &nodes[s];
            if node.kind != NodeKind::Phone {
                continue;
            }
            let wi = node.word_idx;
            let slot = slot_of[&node.token];
            let conf = scores[t * n_slots + slot].exp().min(1.0);

            word_first[wi] = word_first[wi].min(t);
            word_last[wi] = word_last[wi].max(t + 1);
            word_conf_sum[wi] += conf;
            word_frames_n[wi] += 1;
        }

        // 8. 组装 + 健康门控 (不过关整体回退 Qwen 对齐器)
        let audio_dur_ms = ((samples_16k.len() as i64) * 1000 / 16000).max(1);
        let mut units: Vec<AlignedToken> = Vec::with_capacity(words.len());
        let mut total_word_ms = 0i64;
        let mut low_conf_words = 0usize;
        for (wi, wtext) in words.iter().enumerate() {
            let (fa, fb) = (word_first[wi], word_last[wi]);
            if fa == usize::MAX || fb <= fa {
                return Err(QwenError::BackendUnavailable(format!(
                    "lattice: word '{wtext}' got zero frames"
                )));
            }
            let start_local = fa as i64 * FRAME_SHIFT_MS;
            let end_local = (fb as i64 * FRAME_SHIFT_MS).min(audio_dur_ms);
            if end_local - start_local < WORD_DUR_MIN_MS {
                return Err(QwenError::BackendUnavailable(format!(
                    "lattice: word '{wtext}' squeezed to {}ms",
                    end_local - start_local
                )));
            }
            total_word_ms += end_local - start_local;

            let start_ms = if timeline.is_empty() {
                segment_start_ms + start_local.max(0) as u64
            } else {
                map_to_media(timeline, start_local.max(0) as u64)
            };
            let end_ms = if timeline.is_empty() {
                segment_start_ms + end_local.max(0) as u64
            } else {
                map_to_media(timeline, end_local.max(0) as u64)
            };
            let conf = if word_frames_n[wi] > 0 {
                word_conf_sum[wi] / word_frames_n[wi] as f32
            } else {
                0.0
            };
            if conf < CONF_LOW {
                low_conf_words += 1;
            }
            units.push(AlignedToken {
                text: wtext.clone(),
                start_ms,
                end_ms: end_ms.min(segment_end_ms),
                confidence: Some(conf),
            });
        }
        let coverage = total_word_ms as f32 / audio_dur_ms as f32;
        let low_ratio = low_conf_words as f32 / words.len() as f32;
        if !(COVERAGE_MIN..=COVERAGE_MAX).contains(&coverage) || low_ratio > LOW_CONF_RATIO {
            return Err(QwenError::BackendUnavailable(format!(
                "lattice: unhealthy result (coverage={coverage:.2}, low_conf_ratio={low_ratio:.2})"
            )));
        }

        println!(
            "[lattice] aligned {} words ({} states, {} frames, oov={oov}) in {}ms",
            words.len(),
            n_states,
            n_frames,
            start_time.elapsed().as_millis()
        );
        let reconciled_units = crate::qwen::aligner::reconcile(text, units);
        Ok(AlignmentResult {
            units: reconciled_units,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
            align_quality: "LatticeAligned".into(),
        })
    }

    /// ONNX emission 推理: 波形 -> log-softmax 展平 [F x vocab]
    fn compute_emission(&mut self, samples: &[f32], cancel: &AtomicBool) -> Result<Vec<f32>, QwenError> {
        let mut all: Vec<f32> = Vec::new();
        if samples.is_empty() {
            return Ok(all);
        }
        let mut offset = 0usize;
        loop {
            if cancel.load(Ordering::Relaxed) {
                return Err(QwenError::Cancelled);
            }
            let end = (offset + CHUNK_SAMPLES).min(samples.len());
            let mut chunk = samples[offset..end].to_vec();
            if chunk.len() < MIN_SAMPLES_PAD {
                chunk.resize(MIN_SAMPLES_PAD, 0.0);
            }
            let tensor = Tensor::<f32>::from_array((vec![1i64, chunk.len() as i64], chunk.into_boxed_slice()))
                .map_err(|e| QwenError::OnnxError(format!("lattice tensor: {e}")))?;
            let outputs = self
                .session
                .run(inputs!["audios" => tensor])
                .map_err(|e| QwenError::OnnxError(format!("lattice run: {e}")))?;
            let (_out_name, first) = outputs
                .iter()
                .next()
                .ok_or_else(|| QwenError::OnnxError("lattice: no outputs".into()))?;
            let (_shape, data) = first
                .try_extract_tensor::<f32>()
                .map_err(|e| QwenError::OnnxError(format!("lattice extract: {e}")))?;
            all.extend_from_slice(data);
            if end >= samples.len() {
                break;
            }
            offset = end;
        }
        Ok(all)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 语言路由纯函数
    #[test]
    fn language_routing() {
        assert!(language_supported(Some("English")));
        assert!(language_supported(Some("zh")));
        assert!(language_supported(Some("German")));
        assert!(language_supported(Some("chinese")));
        assert!(!language_supported(Some("Japanese")));
        assert!(!language_supported(Some("Korean")));
        assert!(!language_supported(Some("Cantonese")));
        assert!(!language_supported(None));
    }


}
