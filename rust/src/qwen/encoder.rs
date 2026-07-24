use crate::qwen::audio::{AudioProcessor, N_MELS, SAMPLE_RATE};
use crate::qwen::backend::EncoderBackend;
use crate::qwen::error::QwenError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use ort::{ep::DirectML, inputs};
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

#[derive(Debug, Clone)]
pub struct EncoderOutput {
    pub embeddings: Vec<f32>,
    pub samples_16k: Vec<f32>,
    pub shape: Vec<usize>,
    pub audio_duration_ms: u64,
}

enum InputLayout {
    Mel,
    Raw,
    Unknown,
}

pub struct QwenEncoder {
    session: Session,
    backend: EncoderBackend,
    model_path: String,
    input_name: String,
    backend_path: String,
    layout: InputLayout,
}

impl QwenEncoder {
    pub fn load(model_dir: &str, backend: EncoderBackend) -> Result<Self, QwenError> {
        ensure_onnxruntime_loaded();

        let dir = Path::new(model_dir);
        let candidates = [
            dir.join("asr_encoder_frontend.int4.onnx"),
            dir.join("encoder.int4.onnx"),
            dir.join("encoder.onnx"),
        ];
        let chosen = candidates
            .iter()
            .find(|p| p.exists())
            .ok_or_else(|| QwenError::ModelNotFound(format!("No encoder.onnx in {}", model_dir)))?;
        let chosen_str = chosen.to_string_lossy().to_string();
        println!("[encoder] using model: {}", chosen_str);

        let mut builder = Session::builder()
            .map_err(|e| QwenError::OnnxError(format!("Session::builder: {}", e)))?;
        builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| QwenError::OnnxError(format!("opt_level: {}", e)))?;
        builder = builder
            .with_intra_threads(2)
            .map_err(|e| QwenError::OnnxError(format!("intra_threads: {}", e)))?;

        // Windows 系统下配置 DirectML 执行提供程序 (EP)
        #[cfg(all(target_os = "windows", feature = "qwen-dml"))]
        {
            if matches!(backend, EncoderBackend::DirectMl | EncoderBackend::Auto) {
                match builder.clone().with_execution_providers([DirectML::default().build()]) {
                    Ok(b) => {
                        builder = b;
                        println!("[encoder] DirectML EP 注册成功");
                    }
                    Err(e) => {
                        println!("[encoder] DirectML EP 注册失败 ({:?}); 降级至 CPU", e);
                    }
                }
            }
        }

        let session = builder
            .commit_from_file(&chosen_str)
            .map_err(|e| QwenError::OnnxError(format!("commit_from_file({}): {}", chosen_str, e)))?;

        // 从输出节点名称/形状自动检测输入布局
        let (input_name, layout) = detect_input_layout(&session);

        println!(
            "[encoder] input_name=\"{}\" layout={:?}",
            input_name,
            match layout {
                InputLayout::Mel => "mel",
                InputLayout::Raw => "raw",
                InputLayout::Unknown => "unknown(default->mel)",
            }
        );

        Ok(Self {
            session,
            backend,
            model_path: chosen_str,
            input_name,
            backend_path: String::new(),
            layout,
        })
    }

    pub fn encode(
        &mut self,
        samples: &[f32],
        cancel: &AtomicBool,
    ) -> Result<EncoderOutput, QwenError> {
        if cancel.load(Ordering::Relaxed) {
            return Err(QwenError::Cancelled);
        }
        if samples.is_empty() {
            return Err(QwenError::AudioError("empty audio".into()));
        }

        let audio_duration_ms =
            ((samples.len() as f64 / SAMPLE_RATE as f64) * 1000.0) as u64;

        // 构建输入 Tensor 降级策略：优先尝试 *Raw 原始音频*，失败后再重试 Mel 频谱。
        // Qwen3-ASR 的 encoder.onnx 通常要求原始音频，因为其内部图自带 Mel FBank 计算。
        let try_order: [(bool, Vec<f32>, Vec<i64>); 2] = [
            // 1) Raw 原始音频 ── 大多数 ONNX 编码器所期望的输入
            (
                !matches!(self.layout, InputLayout::Mel),
                samples.to_vec(),
                vec![1_i64, samples.len() as i64],
            ),
            // 2) Mel 频谱 ── 针对无内部 FBank 编码器的备用方案
            {
                let maybe_mel = AudioProcessor::log_mel(samples);
                match maybe_mel {
                    Ok((f, nf)) => (true, f, vec![1_i64, N_MELS as i64, nf as i64]),
                    Err(_) => (false, Vec::new(), Vec::new()),
                }
            },
        ];

        let mut last_err: Option<QwenError> = None;
        let mut embeddings = Vec::new();
        let mut shape_vec = Vec::new();

        for (enabled, data, shape) in &try_order {
            if !enabled || data.is_empty() {
                continue;
            }
            let tensor = match Tensor::<f32>::from_array((
                shape.clone(),
                data.clone().into_boxed_slice(),
            )) {
                Ok(t) => t,
                Err(e) => {
                    last_err = Some(QwenError::OnnxError(format!("Tensor::from_array: {}", e)));
                    continue;
                }
            };
            match self
                .session
                .run(inputs![self.input_name.as_str() => tensor])
            {
                Ok(outputs) => {
                    let res = extract_embeddings(outputs);
                    match res {
                        Ok((e, s)) => {
                            embeddings = e;
                            shape_vec = s;
                            last_err = None;
                            break;
                        }
                        Err(e) => {
                            last_err = Some(e);
                            continue;
                        }
                    }
                }
                Err(e) => {
                    last_err =
                        Some(QwenError::OnnxError(format!("session.run: {:?}", e)));
                    continue;
                }
            }
        }

        if embeddings.is_empty() {
            let msg = match last_err {
                Some(e) => format!("{:?}", e),
                None => "encoder: no input shape succeeded".into(),
            };
            return Err(QwenError::OnnxError(msg));
        }

        Ok(EncoderOutput {
            embeddings,
            samples_16k: samples.to_vec(),
            shape: shape_vec,
            audio_duration_ms,
        })
    }

    pub fn load_test_stub() -> Self {
        // 无模型文件无法构建真实 Session
        unimplemented!("QwenEncoder::load_test_stub is not supported after migration")
    }

    pub fn backend(&self) -> EncoderBackend {
        self.backend
    }

    pub fn frontend_path(&self) -> &str {
        &self.model_path
    }

    pub fn backend_path(&self) -> &str {
        &self.backend_path
    }
}

fn detect_input_layout(session: &Session) -> (String, InputLayout) {
    let inputs = session.inputs();
    if inputs.is_empty() {
        return ("input".to_string(), InputLayout::Unknown);
    }
    // 选择匹配已知名称的第一个输入节点；否则使用第一个输出节点
    let outlet = inputs
        .iter()
        .find(|o| {
            let n = o.name().to_lowercase();
            n.contains("feat")
                || n.contains("mel")
                || n.contains("audio")
                || n.contains("waveform")
                || n.contains("signal")
                || n.contains("input")
                || n.contains("log_mel")
        })
        .unwrap_or(&inputs[0]);

    let name = outlet.name().to_string();
    let lname = name.to_lowercase();
    let dtype = outlet.dtype();

    // --- 启发式判定规则 ---
    //
    // Qwen3-ASR 的 encoder.onnx 接收 **原始 16 kHz 音频** (rank-2)，因为其在计算图内部计算 Mel 特征。
    // 如果模型输入名称明确写有 “feats” / “mel”，仍保留 Mel 启发式判定。

    // 按名称启发式判定
    if lname.contains("mel") || lname.contains("feat") || lname.contains("log_mel") {
        return (name, InputLayout::Mel);
    }
    if lname.contains("audio") || lname.contains("waveform") || lname.contains("signal") {
        return (name, InputLayout::Raw);
    }

    // 按 Shape 启发式判定
    if let Some(shape) = dtype.tensor_shape() {
        let n = shape.len();
        if n >= 3 {
            // rank-3 且 dim[1] ∈ {80, 128} 通常为 Mel 频谱输入
            let m = shape.iter().nth(1).copied().unwrap_or(0);
            if m == 80 || m == 128 {
                return (name, InputLayout::Mel);
            }
        } else if n == 2 {
            return (name, InputLayout::Raw);
        }
    }

    // 未知输入的默认策略：原始音频 Raw
    println!("[encoder] Unrecognised input layout, defaulting to Raw audio");
    (name, InputLayout::Raw)
}

fn extract_embeddings(
    outputs: ort::session::SessionOutputs,
) -> Result<(Vec<f32>, Vec<usize>), QwenError> {
    let mut output_iter = outputs.iter();
    let (out_name, first_value) = output_iter
        .next()
        .ok_or_else(|| QwenError::OnnxError("no outputs from encoder".into()))?;
    let (shape_ref, data_slice) = first_value
        .try_extract_tensor::<f32>()
        .map_err(|e| QwenError::OnnxError(format!("try_extract_tensor: {}", e)))?;

    let shape_vec: Vec<usize> = shape_ref.iter().map(|d| *d as usize).collect();
    let mut embeddings = data_slice.to_vec();
    let total_elems = embeddings.len();

    println!(
        "[encoder] Output tensor '{}': shape={:?} total elements={}",
        out_name, shape_vec, total_elems
    );
    // 详细记录前两个维度的 Shape 信息
    if shape_vec.len() >= 2 {
        let first_two: Vec<usize> = shape_vec.iter().take(2).copied().collect();
        println!(
            "[encoder] First dimensions: {:?} (B={} D_or_T={})",
            first_two, shape_vec[0], shape_vec[1]
        );
    }

    // 如果 Shape 为 [B, D, T] (Conv1D Channel-First: 如 [1, 1024, T] 或 [1, 768, T])，
    // 则转置为 [B, T, D] 以满足 LLM 序列注入格式 [T, D]。
    if shape_vec.len() == 3 {
        let d = shape_vec[1];
        let t = shape_vec[2];
        if (d == 1024 || d == 896 || d == 768 || d == 512 || d == 1280) && t != d {
            println!("[encoder] Transposing Conv1D output [1, D={}, T={}] -> [1, T={}, D={}]", d, t, t, d);
            let mut transposed = vec![0.0f32; d * t];
            for id in 0..d {
                for it in 0..t {
                    transposed[it * d + id] = embeddings[id * t + it];
                }
            }
            embeddings = transposed;
            return Ok((embeddings, vec![shape_vec[0], t, d]));
        }
    }

    Ok((embeddings, shape_vec))
}

static ORT_INITIALIZED: std::sync::OnceLock<()> = std::sync::OnceLock::new();

fn ensure_onnxruntime_loaded() {
    ORT_INITIALIZED.get_or_init(|| {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                let target_dll = exe_dir.join("onnxruntime.dll");
                let target_dml = exe_dir.join("DirectML.dll");

                println!("[encoder] Checking runtime DLLs in exe_dir: {:?}", exe_dir);

                let mut found_dll = None;
                let mut found_dml = None;

                if let Ok(user_profile) = std::env::var("USERPROFILE") {
                    let cargo_checkouts = Path::new(&user_profile).join(".cargo").join("git").join("checkouts");
                    if cargo_checkouts.exists() {
                        find_ort_dlls(&cargo_checkouts, &mut found_dll, &mut found_dml);
                    }
                }

                if let Some(src_dll) = found_dll {
                    if !target_dll.exists() {
                        println!("[encoder] Copying ONNX Runtime DLL {:?} -> {:?}", src_dll, target_dll);
                        let _ = std::fs::copy(&src_dll, &target_dll);
                    }
                }
                if let Some(src_dml) = found_dml {
                    if !target_dml.exists() {
                        println!("[encoder] Copying DirectML DLL {:?} -> {:?}", src_dml, target_dml);
                        let _ = std::fs::copy(&src_dml, &target_dml);
                    }
                }

                if target_dll.exists() {
                    let path_str = target_dll.to_string_lossy().to_string();
                    let _ = ort::init_from(&path_str);
                    println!("[encoder] Called ort::init_from({:?})", path_str);
                }
            }
        }
    });
}

fn find_ort_dlls(dir: &Path, found_dll: &mut Option<std::path::PathBuf>, found_dml: &mut Option<std::path::PathBuf>) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.eq_ignore_ascii_case("onnxruntime.dll") && found_dll.is_none() {
                        *found_dll = Some(p.clone());
                    } else if name.eq_ignore_ascii_case("DirectML.dll") && found_dml.is_none() {
                        *found_dml = Some(p.clone());
                    }
                }
            } else if p.is_dir() {
                find_ort_dlls(&p, found_dll, found_dml);
            }
        }
    }
}
