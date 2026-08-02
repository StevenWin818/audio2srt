use crate::qwen::audio::{AudioProcessor, N_MELS, SAMPLE_RATE};
use crate::qwen::backend::EncoderBackend;
use crate::qwen::error::QwenError;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::Tensor;
use ort::inputs;
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
    pub session: Session,
    backend: EncoderBackend,
    model_path: String,
    input_name: String,
    backend_path: String,
    #[allow(dead_code)]
    layout: InputLayout,
}

impl QwenEncoder {
    pub fn load(model_dir: &str, backend: EncoderBackend) -> Result<Self, QwenError> {
        ensure_onnxruntime_loaded(backend);

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
            .with_intra_threads(4)
            .map_err(|e| QwenError::OnnxError(format!("intra_threads: {}", e)))?;
        
        builder = builder
            .with_config_entry("session.use_device_allocator_for_initializers", "1")
            .map_err(|e| QwenError::OnnxError(format!("config_entry: {}", e)))?;

        // EP 选择 (Auto 模式):
        //   Windows + cuDNN 可用:  CUDA (onnxruntime 1.28 GPU, CUDA 13) -> CPU
        //   Windows 无 cuDNN:       DirectML (DX12) -> CPU
        //   其他平台:               CUDA -> CPU
        // 注意: CUDA EP 的 with_execution_providers 注册可能返回 Ok，
        // 但缺少 cuDNN/CUDA 运行时(如 CUDA 13 机器)时实际执行会静默回退 CPU，
        // 因此部署时已按 cuDNN 可用性选择对应的 onnxruntime 版本。
        let mut ep_ok = false;
        let mut ep_errors: Vec<String> = Vec::new();

        let cuda_usable = match backend {
            EncoderBackend::Cuda => true,
            EncoderBackend::DirectMl => false,
            EncoderBackend::Auto => cudnn_available(),
            EncoderBackend::Cpu => false,
        };

        // 1. CUDA 后端分支
        #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
        {
            if cuda_usable {
                use ort::ep::CUDA;
                match builder.clone().with_execution_providers([CUDA::default().build()]) {
                    Ok(b) => {
                        builder = b;
                        ep_ok = true;
                        println!("[encoder] CUDA EP 注册成功 (onnxruntime 1.28 GPU / CUDA 13)");
                    }
                    Err(e) => {
                        let msg = format!("{:?}", e);
                        println!("[encoder] CUDA EP 注册失败 ({:?}); 降级处理", e);
                        ep_errors.push(format!("CUDA EP: {}", msg));
                    }
                }
            }
        }

        // 2. DirectML 后端分支 (Windows; Auto 且无 cuDNN 时, 或显式选择)
        #[cfg(all(target_os = "windows", any(feature = "vulkan", feature = "qwen-dml", feature = "qwen-dml-win")))]
        {
            let want_dml = matches!(backend, EncoderBackend::DirectMl)
                || (matches!(backend, EncoderBackend::Auto) && !cuda_usable);
            if want_dml {
                use ort::ep::DirectML;
                match builder.clone().with_execution_providers([DirectML::default().build()]) {
                    Ok(b) => {
                        builder = b;
                        ep_ok = true;
                        println!("[encoder] Windows DirectML EP 注册成功 (DX12, 无 cuDNN 依赖)");
                    }
                    Err(e) => {
                        let msg = format!("{:?}", e);
                        println!("[encoder] DirectML EP 注册失败 ({:?}); 降级处理", e);
                        ep_errors.push(format!("DirectML EP: {}", msg));
                    }
                }
            }
        }

        if ep_ok {
            println!("[encoder] GPU 加速执行提供程序已启用");
        } else {
            // ===== 明确的 CPU 回退警告 =====
            println!("[encoder] ================================================================");
            println!("[encoder] ⚠️  警告: encoder 未启用任何 GPU 执行提供程序, 将使用 CPU 推理!");
            println!("[encoder]     这会导致编码速度显著下降 (约为 GPU 的 3~5 倍耗时)。");
            if !ep_errors.is_empty() {
                for e in &ep_errors {
                    println!("[encoder]     注册失败详情: {}", e);
                }
            }
            match backend {
                EncoderBackend::Cuda => println!("[encoder]     排查建议: 检查 cuDNN 9 (C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\13.3\\x64) 与 CUDA 13 运行时是否完整; 或改用 DirectML 后端 (设置中关闭 CUDA)。"),
                EncoderBackend::DirectMl => println!("[encoder]     排查建议: 检查 DirectX 12 / DirectML.dll (系统组件) 是否可用。"),
                EncoderBackend::Auto => println!("[encoder]     排查建议: 若安装了 cuDNN 9 请确认路径正确; 否则将自动使用 DirectML 或 CPU。"),
                EncoderBackend::Cpu => println!("[encoder]     当前为显式 CPU 后端 (设置中未启用 GPU)。"),
            }
            println!("[encoder] ================================================================");
        }

        // 3. 非 Windows 平台下的 Vulkan 后端分支 (GGUF-Vulkan + ort-CPU)
        #[cfg(all(not(target_os = "windows"), feature = "vulkan"))]
        {
            println!("[encoder] 非 Windows 平台 (Vulkan 方案): ONNX Runtime 使用 CPU，解码器使用 Vulkan");
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

        // 根据在 load() 时探测到的 ONNX 输入 layout 选择对应输入特征。
        // Qwen3-ASR 的 encoder.onnx 输入名为 `mel`，声明 shape `[1, 128, time]`，
        // 期望外部预计算好的 log-mel 特征图
        // secondary 仅在 layout 不明或可同时支持两种输入时作为最佳排序使用。
        let (primary, fallback) = match self.layout {
            InputLayout::Mel => {
                let mel = AudioProcessor::log_mel(samples)
                    .map(|(f, nf)| (f, vec![1_i64, N_MELS as i64, nf as i64]))
                    .map_err(|e| QwenError::AudioError(format!("log_mel: {:?}", e)))?;
                let mel = Some(mel);
                let raw = None;
                (mel, raw)
            }
            InputLayout::Raw | InputLayout::Unknown => {
                let raw = Some((samples.to_vec(), vec![1_i64, samples.len() as i64]));
                let mel = AudioProcessor::log_mel(samples)
                    .ok()
                    .map(|(f, nf)| (f, vec![1_i64, N_MELS as i64, nf as i64]));
                (raw, mel)
            }
        };

        // 排序：layout 匹配的特征优先；另一特征仅在主路径失败时作为兜底。
        let try_order: [Option<(Vec<f32>, Vec<i64>)>; 2] = [primary, fallback];

        let mut last_err: Option<QwenError> = None;
        let mut embeddings = Vec::new();
        let mut shape_vec = Vec::new();

        for entry in &try_order {
            let Some((data, shape)) = entry else { continue };
            if data.is_empty() {
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
                    println!("[encoder] ONNX inference succeeded with input shape {:?}", shape);
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

        println!(
            "[ONNX Encoder] ONNX 前向传播计算成功完成: 音频时长 {:.2}s -> 生成声学 Embedding 形状 {:?}",
            samples.len() as f64 / SAMPLE_RATE as f64,
            shape_vec
        );

        Ok(EncoderOutput {
            embeddings,
            samples_16k: samples.to_vec(),
            shape: shape_vec,
            audio_duration_ms,
        })
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

/// 打印 ONNX Runtime 构建信息 (含内置的 Execution Provider 列表)
pub fn ort_build_info() -> String {
    std::panic::catch_unwind(|| ort::info().to_string()).unwrap_or_else(|_| "n/a".into())
}

/// 探测 cuDNN 9 安装目录 (CUDA 13.3 变体)。
/// 常见安装位置: C:\Program Files\NVIDIA\CUDNN\v9.x\bin\13.3\x64
fn probe_cudnn_dir() -> Option<std::path::PathBuf> {
    let base = Path::new("C:/Program Files/NVIDIA/CUDNN");
    if let Ok(entries) = std::fs::read_dir(base) {
        let mut dirs: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs.into_iter().rev() {
            for sub in ["bin/13.3/x64", "bin/13.0/x64"] {
                let cand = d.join(sub);
                if cand.join("cudnn64_9.dll").exists() {
                    return Some(cand);
                }
            }
        }
    }
    None
}

/// 探测 CUDA Toolkit bin\x64 目录 (cudart64_13.dll)
fn probe_cuda_bin() -> Option<std::path::PathBuf> {
    let base = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    if let Ok(entries) = std::fs::read_dir(base) {
        let mut dirs: Vec<std::path::PathBuf> = entries
            .flatten()
            .map(|e| e.path())
            .filter(|p| p.is_dir())
            .collect();
        dirs.sort();
        for d in dirs.into_iter().rev() {
            let cand = d.join("bin/x64");
            if cand.join("cudart64_13.dll").exists() {
                return Some(cand);
            }
        }
    }
    None
}

/// 判断 cuDNN 当前是否可加载 (决定 Auto 后端走 CUDA 还是 DirectML)。
/// 检查 exe 目录 / PATH / 常见安装目录。
fn cudnn_available() -> bool {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            if dir.join("cudnn64_9.dll").exists() {
                return true;
            }
        }
    }
    if probe_cudnn_dir().is_some() {
        return true;
    }
    if let Ok(path) = std::env::var("PATH") {
        for d in path.split(';') {
            if !d.is_empty() && Path::new(d).join("cudnn64_9.dll").exists() {
                return true;
            }
        }
    }
    false
}

/// 部署 CUDA 版 ONNX Runtime 到 exe 目录:
/// onnxruntime.dll + onnxruntime_providers_cuda.dll + providers_shared
/// (来自项目内 bundled 目录)，以及 cuDNN 9 / cudart / cublas / cublasLt
/// (来自系统安装，探测失败时跳过并告警)。
fn deploy_cuda_bundle(exe_dir: &Path) {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("third_party")
        .join("onnxruntime-cuda");
    let core_files = [
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
    ];

    // 已部署完整 (核心 DLL + cuDNN) 则跳过，避免每次启动重复拷贝 1.5GB+
    let already = core_files.iter().all(|f| exe_dir.join(f).exists())
        && exe_dir.join("cudnn64_9.dll").exists();
    if already {
        println!("[encoder] CUDA ONNX Runtime already deployed in exe_dir, skip copy");
        return;
    }

    for f in core_files {
        let src = bundled.join(f);
        if src.exists() {
            let dst = exe_dir.join(f);
            let _ = std::fs::remove_file(&dst);
            match std::fs::copy(&src, &dst) {
                Ok(_) => println!("[encoder] Copied {} ({} bytes) -> exe_dir", f, src.metadata().map(|m| m.len()).unwrap_or(0)),
                Err(e) => println!("[encoder] ERROR copying {}: {}", f, e),
            }
        } else {
            println!(
                "[encoder] WARNING: bundled CUDA onnxruntime missing ({});\n[encoder]           run: powershell -File rust/third_party/download_ort_binaries.ps1",
                bundled.join("onnxruntime.dll").display()
            );
        }
    }

    // cuDNN 9 (CUDA 13.3 变体)
    match probe_cudnn_dir() {
        Some(dir) => {
            if let Ok(entries) = std::fs::read_dir(&dir) {
                for e in entries.flatten() {
                    let p = e.path();
                    if p.extension().and_then(|x| x.to_str()) == Some("dll")
                        && p.file_name().and_then(|n| n.to_str()).map(|n| n.starts_with("cudnn")).unwrap_or(false)
                    {
                        let dst = exe_dir.join(p.file_name().unwrap());
                        let _ = std::fs::remove_file(&dst);
                        let _ = std::fs::copy(&p, &dst);
                    }
                }
                println!("[encoder] Copied cuDNN runtime from {}", dir.display());
            }
        }
        None => println!("[encoder] WARNING: cuDNN 9 install not found (C:\\Program Files\\NVIDIA\\CUDNN\\v9.x\\bin\\13.3\\x64). CUDA EP will fail!"),
    }

    // CUDA 运行时: cudart / cublas / cublasLt (PATH 可能已含，但拷贝到 exe 目录更稳妥)
    match probe_cuda_bin() {
        Some(dir) => {
            for f in ["cudart64_13.dll", "cublas64_13.dll", "cublasLt64_13.dll"] {
                let src = dir.join(f);
                if src.exists() {
                    let dst = exe_dir.join(f);
                    let _ = std::fs::remove_file(&dst);
                    let _ = std::fs::copy(&src, &dst);
                }
            }
            println!("[encoder] Copied CUDA runtime DLLs from {}", dir.display());
        }
        None => println!("[encoder] WARNING: CUDA 13 toolkit bin not found; cudart/cublas must be on PATH"),
    }
}

/// 确保 exe 目录中存在正确的 ONNX Runtime DLL，并让 ort (load-dynamic 模式) 加载它。
///
/// 重要背景：deep_filter 依赖启用了 ort 的 `load-dynamic`，导致 ort-sys 构建期
/// `disable-linking`，不会下载/静态链接任何 onnxruntime 二进制；ort 在运行时
/// 动态加载 exe 旁的 onnxruntime.dll。此前这里从 `~/.cargo/git/checkouts` 递归
/// 搜索，找到的是 deepfilter-rt 捆绑的 **CPU 版** onnxruntime.dll (1.23.2, 13.5MB)，
/// 导致 CUDA/DirectML EP 全部无法注册，encoder 只能跑 CPU。
///
/// 修复：根据后端选择部署项目内捆绑的 GPU 版 onnxruntime：
/// - Cuda / Auto(有 cuDNN): onnxruntime 1.28.0 GPU (CUDA 13 支持) + cuDNN 9.24 + cudart/cublas
/// - DirectMl / Auto(无 cuDNN): DirectML 版 onnxruntime (DX12 通用, 无需 cuDNN)
/// - Cpu: DirectML 版 (含 CPU EP) 兜底
fn ensure_onnxruntime_loaded(backend: EncoderBackend) {
    ORT_INITIALIZED.get_or_init(|| {
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                println!("[encoder] Checking runtime DLLs in exe_dir: {:?}", exe_dir);

                let want_cuda = match backend {
                    EncoderBackend::Cuda => true,
                    EncoderBackend::DirectMl => false,
                    EncoderBackend::Auto => cudnn_available(),
                    EncoderBackend::Cpu => false,
                };
                println!(
                    "[encoder] backend={:?} want_cuda={} cudnn_available={}",
                    backend,
                    want_cuda,
                    cudnn_available()
                );

                if want_cuda {
                    deploy_cuda_bundle(&exe_dir);
                } else {
                    // DirectML / CPU: 部署项目内捆绑的 DirectML 版
                    let bundled_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("third_party")
                        .join("onnxruntime-dml");
                    let bundled_dll = bundled_dir.join("onnxruntime.dll");
                    let target_dll = exe_dir.join("onnxruntime.dll");
                    let target_shared = exe_dir.join("onnxruntime_providers_shared.dll");
                    if bundled_dll.exists() {
                        // 覆盖旧拷贝 (可能是 CPU 版)，保证使用 DML 版
                        let _ = std::fs::remove_file(&target_dll);
                        if let Err(e) = std::fs::copy(&bundled_dll, &target_dll) {
                            println!("[encoder] Failed to copy bundled DirectML onnxruntime.dll: {}", e);
                        } else {
                            println!(
                                "[encoder] Copied bundled DirectML onnxruntime.dll ({} bytes) -> {:?}",
                                bundled_dll.metadata().map(|m| m.len()).unwrap_or(0),
                                target_dll
                            );
                        }
                        let bundled_shared = bundled_dir.join("onnxruntime_providers_shared.dll");
                        if bundled_shared.exists() {
                            let _ = std::fs::remove_file(&target_shared);
                            let _ = std::fs::copy(&bundled_shared, &target_shared);
                        }
                    } else {
                        println!(
                            "[encoder] WARNING: bundled DirectML onnxruntime.dll missing at {};\n[encoder]           run: powershell -File rust/third_party/download_ort_binaries.ps1",
                            bundled_dll.display()
                        );
                        // 兜底: 从 git checkouts 搜索 (其他环境)
                        let mut found_dll = None;
                        let mut found_dml = None;
                        if let Ok(user_profile) = std::env::var("USERPROFILE") {
                            let cargo_checkouts =
                                Path::new(&user_profile).join(".cargo").join("git").join("checkouts");
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
                            let target_dml = exe_dir.join("DirectML.dll");
                            if !target_dml.exists() {
                                println!("[encoder] Copying DirectML DLL {:?} -> {:?}", src_dml, target_dml);
                                let _ = std::fs::copy(&src_dml, &target_dml);
                            }
                        }
                    }
                }

                let target_dll = exe_dir.join("onnxruntime.dll");
                if target_dll.exists() {
                    let path_str = target_dll.to_string_lossy().to_string();
                    let _ = ort::init_from(&path_str);
                    println!("[encoder] Called ort::init_from({:?})", path_str);
                } else {
                    println!("[encoder] WARNING: no onnxruntime.dll available for ort load-dynamic");
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::AtomicBool;

    #[test]
    fn encoder_loads_with_auto_backend_and_runs() {
        // 运行: QWEN_MODEL_DIR=... cargo test --lib encoder_loads_with_auto_backend_and_runs -- --nocapture
        let model_dir = std::env::var("QWEN_MODEL_DIR")
            .unwrap_or_else(|_| {
                let appdata = std::env::var("APPDATA").expect("APPDATA");
                format!(
                    "{}\\com.audio2srt\\audio2srt\\models\\qwen3-asr-0.6b",
                    appdata
                )
            });
        let dir = PathBuf::from(&model_dir);
        assert!(dir.exists(), "model dir not found: {}", model_dir);

        let mut enc = QwenEncoder::load(&model_dir, crate::qwen::backend::EncoderBackend::Auto)
            .expect("encoder load failed");

        let cancel = AtomicBool::new(false);

        // 10 秒静音编码，验证 DirectML GPU 推理
        let samples = vec![0.0f32; 160000];
        let t0 = std::time::Instant::now();
        let out = enc.encode(&samples, &cancel).expect("encoder run failed");
        let dt = t0.elapsed();
        let embd_dim = out.shape.last().copied().unwrap_or(1024).max(1);
        println!(
            "[test] encode 10s audio: {:.2}s (RTF={:.3}, {} tokens), shape {:?}",
            dt.as_secs_f64(),
            dt.as_secs_f64() / 10.0,
            out.embeddings.len() / embd_dim,
            out.shape
        );
        assert!(!out.embeddings.is_empty());
    }
}
