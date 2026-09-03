//! 官方 LattifAI G2P (DeepPhonemizer) 本地 ONNX 神经音素转换引擎。
//!
//! 资产:
//!   - `g2pp_encoder.onnx`: 自回归 Transformer 编码器 [seq_len, 1] -> [seq_len, 1, 512]
//!   - `g2pp_decoder.onnx`: 自回归 Transformer 解码器 [tgt_len, 1] + memory + causal_mask -> [1, 216]
//!   - `text_symbols.json`: 5635 个字符映射表
//!
//! 输出 Token ID:
//!   0: pad `_`, 1: start `<omni>`, 2: end `<end>`
//!   3..=215: 对应 `phoneme_symbols.json` 中 213 个音素符号, 与 `transition.bin` 和 `acoustic_opt.onnx`
//!   的音素 token 完全一致, 无需额外映射。

use crate::qwen::error::QwenError;
use ort::inputs;
use ort::session::{builder::GraphOptimizationLevel, Session};
use ort::value::Tensor;
use parking_lot::Mutex;
use std::collections::HashMap;
use std::path::Path;

pub struct G2pEngine {
    encoder_session: Mutex<Session>,
    decoder_session: Mutex<Session>,
    char_to_id: HashMap<char, i64>,
    cache: Mutex<HashMap<String, Vec<u16>>>,
}

impl G2pEngine {
    /// 从模型目录载入 G2P 神经推理引擎
    pub fn load(dir: &Path) -> Result<Self, QwenError> {
        let symbols_path = dir.join("text_symbols.json");
        let symbols_str = std::fs::read_to_string(&symbols_path)
            .map_err(|e| QwenError::BackendUnavailable(format!("read text_symbols.json: {e}")))?;
        let symbols: Vec<String> = serde_json::from_str(&symbols_str)
            .map_err(|e| QwenError::BackendUnavailable(format!("parse text_symbols.json: {e}")))?;

        let mut char_to_id = HashMap::with_capacity(symbols.len() + 8);
        for (i, sym) in symbols.iter().enumerate() {
            if let Some(ch) = sym.chars().next() {
                // 特殊符占 0, 1, 2; text_symbols 从 3 开始
                char_to_id.insert(ch, (i + 3) as i64);
            }
        }

        let enc_path = dir.join("g2pp_encoder.onnx");
        let enc_session = Session::builder()
            .map_err(|e| QwenError::OnnxError(format!("Session::builder enc: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| QwenError::OnnxError(format!("opt_level enc: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| QwenError::OnnxError(format!("intra_threads enc: {e}")))?
            .commit_from_file(enc_path.to_string_lossy().as_ref())
            .map_err(|e| QwenError::OnnxError(format!("load g2pp_encoder.onnx: {e}")))?;

        let dec_path = dir.join("g2pp_decoder.onnx");
        let dec_session = Session::builder()
            .map_err(|e| QwenError::OnnxError(format!("Session::builder dec: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| QwenError::OnnxError(format!("opt_level dec: {e}")))?
            .with_intra_threads(4)
            .map_err(|e| QwenError::OnnxError(format!("intra_threads dec: {e}")))?
            .commit_from_file(dec_path.to_string_lossy().as_ref())
            .map_err(|e| QwenError::OnnxError(format!("load g2pp_decoder.onnx: {e}")))?;

        let lexicon_path = dir.join("lexicon.json");
        let initial_cache: HashMap<String, Vec<u16>> = if lexicon_path.is_file() {
            match std::fs::read_to_string(&lexicon_path) {
                Ok(s) => match serde_json::from_str::<HashMap<String, Vec<u16>>>(&s) {
                    Ok(m) => {
                        println!("[lattice-g2p] loaded offline lexicon: {} words", m.len());
                        m
                    }
                    Err(e) => {
                        println!("[lattice-g2p] parse lexicon.json warning: {e}");
                        HashMap::with_capacity(4096)
                    }
                },
                Err(e) => {
                    println!("[lattice-g2p] read lexicon.json warning: {e}");
                    HashMap::with_capacity(4096)
                }
            }
        } else {
            HashMap::with_capacity(4096)
        };

        println!(
            "[lattice-g2p] loaded G2P neural models: chars={}, cache={}",
            char_to_id.len(),
            initial_cache.len()
        );

        Ok(Self {
            encoder_session: Mutex::new(enc_session),
            decoder_session: Mutex::new(dec_session),
            char_to_id,
            cache: Mutex::new(initial_cache),
        })
    }

    /// 预测指定单字、单词或词组的发音音素序列 (直接对应 Lattice 的 token id >= 3)
    pub fn get_pronunciation(&self, text: &str) -> Option<Vec<u16>> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return None;
        }

        // 1. 优先查内存发音缓存
        {
            let cache = self.cache.lock();
            if let Some(phones) = cache.get(trimmed) {
                return if phones.is_empty() { None } else { Some(phones.clone()) };
            }
        }

        // 2. 神经推理
        match self.predict_uncached(trimmed) {
            Ok(phones) => {
                let mut cache = self.cache.lock();
                if cache.len() >= 50000 {
                    cache.clear();
                }
                cache.insert(trimmed.to_string(), phones.clone());
                if phones.is_empty() {
                    None
                } else {
                    Some(phones)
                }
            }
            Err(e) => {
                eprintln!("[lattice-g2p] predict error for '{trimmed}': {e}");
                None
            }
        }
    }

    fn predict_uncached(&self, text: &str) -> Result<Vec<u16>, QwenError> {
        // 构建输入字符 token 序列: [1 (<omni>), c1, c2, ..., cn, 2 (<end>)]
        let mut token_ids: Vec<i64> = vec![1];
        for ch in text.chars() {
            for lower_ch in ch.to_lowercase() {
                if let Some(&id) = self.char_to_id.get(&lower_ch) {
                    token_ids.push(id);
                }
            }
        }
        if token_ids.len() <= 1 {
            return Ok(Vec::new());
        }
        token_ids.push(2);

        let seq_len = token_ids.len();
        let enc_in = Tensor::<i64>::from_array((
            vec![seq_len as i64, 1i64],
            token_ids.into_boxed_slice(),
        ))
        .map_err(|e| QwenError::OnnxError(format!("g2p enc input: {e}")))?;

        let mem_data: Vec<f32> = {
            let mut enc_session = self.encoder_session.lock();
            let enc_out = enc_session
                .run(inputs!["text_ids" => enc_in])
                .map_err(|e| QwenError::OnnxError(format!("g2p enc run: {e}")))?;

            let (_mem_name, mem_val) = enc_out
                .iter()
                .next()
                .ok_or_else(|| QwenError::OnnxError("g2p enc: no outputs".into()))?;
            let (_mem_shape, tensor_data) = mem_val
                .try_extract_tensor::<f32>()
                .map_err(|e| QwenError::OnnxError(format!("g2p mem extract: {e}")))?;
            tensor_data.to_vec()
        };

        // 自回归贪婪解码
        let mut tgt_list: Vec<i64> = vec![1]; // start with <omni>
        const MAX_STEPS: usize = 50;

        let mut dec_session = self.decoder_session.lock();
        for _ in 0..MAX_STEPS {
            let tgt_len = tgt_list.len();
            let mut mask = vec![0.0f32; tgt_len * tgt_len];
            for r in 0..tgt_len {
                for c in (r + 1)..tgt_len {
                    mask[r * tgt_len + c] = f32::NEG_INFINITY;
                }
            }

            let tgt_tensor = Tensor::<i64>::from_array((
                vec![tgt_len as i64, 1i64],
                tgt_list.clone().into_boxed_slice(),
            ))
            .map_err(|e| QwenError::OnnxError(format!("g2p dec tgt: {e}")))?;

            let mem_tensor = Tensor::<f32>::from_array((
                vec![seq_len as i64, 1i64, 512i64],
                mem_data.to_vec().into_boxed_slice(),
            ))
            .map_err(|e| QwenError::OnnxError(format!("g2p dec mem: {e}")))?;

            let mask_tensor = Tensor::<f32>::from_array((
                vec![tgt_len as i64, tgt_len as i64],
                mask.into_boxed_slice(),
            ))
            .map_err(|e| QwenError::OnnxError(format!("g2p dec mask: {e}")))?;

            let dec_out = dec_session
                .run(inputs![
                    "tgt_ids" => tgt_tensor,
                    "memory" => mem_tensor,
                    "tgt_mask" => mask_tensor
                ])
                .map_err(|e| QwenError::OnnxError(format!("g2p dec run: {e}")))?;

            let (_log_name, log_val) = dec_out
                .iter()
                .next()
                .ok_or_else(|| QwenError::OnnxError("g2p dec: no outputs".into()))?;
            let (_log_shape, logits) = log_val
                .try_extract_tensor::<f32>()
                .map_err(|e| QwenError::OnnxError(format!("g2p log extract: {e}")))?;

            let mut best_idx = 0usize;
            let mut best_val = f32::NEG_INFINITY;
            for (i, &v) in logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_idx = i;
                }
            }

            // 2 为 <end> 标识符
            if best_idx == 2 {
                break;
            }
            tgt_list.push(best_idx as i64);
        }

        // 过滤保留有效音素 token (>= 3)
        let phones: Vec<u16> = tgt_list
            .into_iter()
            .filter(|&t| t >= 3)
            .map(|t| t as u16)
            .collect();

        Ok(phones)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn g2p_inference_and_cache_acceleration() {
        let Some(dir) = crate::qwen::lattice::probe_lattice_dir("") else {
            println!("[test] Lattice-1 assets not present, skipping G2P test");
            return;
        };
        let p = Path::new(&dir);
        let engine = G2pEngine::load(p).expect("Failed to load G2pEngine");

        // 1. 英文转写
        let hello = engine.get_pronunciation("hello").expect("hello failed");
        assert!(!hello.is_empty());

        // 2. 缓存命中速度验证 (第二次应瞬时返回)
        let t0 = std::time::Instant::now();
        let hello2 = engine.get_pronunciation("hello").expect("cache hit failed");
        let cache_elapsed = t0.elapsed();
        assert_eq!(hello, hello2);
        println!("[test] G2P cache hit elapsed: {:?}", cache_elapsed);
        assert!(cache_elapsed.as_millis() < 5, "Cache hit must be < 5ms");

        // 3. 中文多字与多音词测试
        let chongqing = engine.get_pronunciation("重庆").expect("重庆 failed");
        assert!(!chongqing.is_empty());
        println!("[test] 重庆 -> {:?}", chongqing);

        let kuaiji = engine.get_pronunciation("会计").expect("会计 failed");
        assert!(!kuaiji.is_empty());
        println!("[test] 会计 -> {:?}", kuaiji);

        // 4. 德语测试
        let guten = engine.get_pronunciation("Guten").expect("Guten failed");
        assert!(!guten.is_empty());
        println!("[test] Guten -> {:?}", guten);
    }
}
