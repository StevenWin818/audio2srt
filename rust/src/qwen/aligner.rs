use crate::qwen::error::QwenError;
use ort::session::Session;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignedToken {
    pub text: String,
    pub start_ms: u64,
    pub end_ms: u64,
    pub confidence: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlignmentResult {
    pub units: Vec<AlignedToken>,
    pub elapsed_ms: u64,
}

pub struct QwenAligner {
    model_path: Option<String>,
    session: Option<Session>,
    #[allow(dead_code)]
    input_name: String,
}

impl QwenAligner {
    pub fn load(model_dir: &str) -> Result<Self, QwenError> {
        let dir = Path::new(model_dir);
        let candidates = [
            dir.join("aligner_encoder_frontend.int4.onnx"),
            dir.join("aligner.int4.onnx"),
            dir.join("aligner.onnx"),
        ];
        let Some(chosen) = candidates.into_iter().find(|p| p.exists()) else {
            // 未找到对齐器 ONNX 模型文件 ── 返回一个可工作的对齐器，回退至先前使用的线性字符启发式算法。
            // `align()` 会检测到缺少 `session` 并自动路由至该路径。
            return Ok(Self {
                model_path: None,
                session: None,
                input_name: String::new(),
            });
        };
        let path_str = chosen.to_string_lossy().to_string();
        let mut builder = Session::builder()
            .map_err(|e| QwenError::OnnxError(format!("aligner Session::builder: {}", e)))?;
        builder = builder
            .with_intra_threads(2)
            .map_err(|e| QwenError::OnnxError(format!("aligner intra_threads: {}", e)))?;
        #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
        {
            use ort::ep::CUDA;
            if let Ok(b) = builder.clone().with_execution_providers([CUDA::default().build()]) {
                builder = b;
            }
        }
        #[cfg(all(target_os = "windows", any(feature = "vulkan", feature = "qwen-dml", feature = "qwen-dml-win")))]
        {
            use ort::ep::DirectML;
            if let Ok(b) = builder.clone().with_execution_providers([DirectML::default().build()]) {
                builder = b;
            }
        }
        let session = builder
            .commit_from_file(&path_str)
            .map_err(|e| QwenError::OnnxError(format!("aligner commit_from_file: {}", e)))?;

        // 提取第一个输入的名称作为 feed key
        let inputs = session.inputs();
        let input_name = inputs
            .first()
            .map(|o| o.name().to_string())
            .unwrap_or_else(|| "input".into());
        println!("[aligner] loaded {} input=\"{}\"", path_str, input_name);

        Ok(Self {
            model_path: Some(path_str),
            session: Some(session),
            input_name,
        })
    }

    pub fn model_path(&self) -> &str {
        self.model_path.as_deref().unwrap_or("")
    }

    pub fn align(
        &mut self,
        _samples_16k: &[f32],
        text: &str,
        segment_start_ms: u64,
        segment_end_ms: u64,
        cancel: &AtomicBool,
    ) -> Result<AlignmentResult, QwenError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(QwenError::Cancelled);
        }
        let start_time = std::time::Instant::now();
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(AlignmentResult {
                units: vec![],
                elapsed_ms: 0,
            });
        }

        let units = match self.session.as_mut() {
            Some(session) => {
                // TODO: 当获取到精确的模型图规范后，实现对齐器 ONNX 的真实前向传播。
                // 目前保留现有的线性字符平分回退算法 — 模型文件已加载并注册至 ONNX Runtime，
                // 后续扩展只需构建相应的 Input Tensor 并解码输出概率矩阵即可。
                let _ = session;
                linear_align(trimmed, segment_start_ms, segment_end_ms)
            }
            None => linear_align(trimmed, segment_start_ms, segment_end_ms),
        };
        Ok(AlignmentResult {
            units,
            elapsed_ms: start_time.elapsed().as_millis() as u64,
        })
    }
}

fn linear_align(
    text: &str,
    segment_start_ms: u64,
    segment_end_ms: u64,
) -> Vec<AlignedToken> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let duration = segment_end_ms.saturating_sub(segment_start_ms);
    let step = if n > 0 { duration as f64 / n as f64 } else { 0.0 };

    let mut units = Vec::with_capacity(n);
    for (i, ch) in chars.iter().enumerate() {
        let start = segment_start_ms + (i as f64 * step) as u64;
        let end = if i + 1 == n {
            segment_end_ms
        } else {
            segment_start_ms + ((i + 1) as f64 * step) as u64
        };
        units.push(AlignedToken {
            text: ch.to_string(),
            start_ms: start.min(segment_end_ms),
            end_ms: end.min(segment_end_ms),
            confidence: Some(1.0),
        });
    }
    units
}
