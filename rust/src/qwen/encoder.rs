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
    /// 实际生效的执行提供程序 ("CUDA" / "DirectML" / "CPU")
    pub actual_ep: String,
    #[allow(dead_code)]
    layout: InputLayout,
}

impl QwenEncoder {
    pub fn load(model_dir: &str, backend: EncoderBackend) -> Result<Self, QwenError> {
        ensure_onnxruntime_loaded(backend);

        let dir = Path::new(model_dir);
        // 模型选择策略:
        // - GPU 后端 (CUDA/DirectML 有效): FP16 优先 (张量核加速、体积减半)
        // - CPU 后端: FP32 优先 (CPU EP 无 FP16 原生算子，onnxruntime 会为每个
        //   算子插 Cast 转 FP32 再转回，纯开销; FP32 模型实测显著更快)
        // Auto 按"最终很可能落到哪个 EP"决策: 有 CUDA 特性且 cuDNN 可用 -> GPU;
        // 有 qwen-dml 特性 -> DirectML; 其余 (含 vulkan 特性) -> CPU。
        let gpu_ep_possible = match backend {
            EncoderBackend::Cuda | EncoderBackend::DirectMl => true,
            EncoderBackend::Auto => {
                #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
                {
                    #[cfg(target_os = "windows")]
                    {
                        if cudnn_available() {
                            true
                        } else {
                            cfg!(any(feature = "qwen-dml", feature = "qwen-dml-win"))
                        }
                    }
                    #[cfg(not(target_os = "windows"))]
                    {
                        true
                    }
                }
                #[cfg(not(any(feature = "cuda", feature = "qwen-cuda")))]
                {
                    cfg!(any(feature = "qwen-dml", feature = "qwen-dml-win"))
                }
            }
            EncoderBackend::Cpu => false,
        };
        let candidates = if gpu_ep_possible {
            [
                dir.join("encoder.fp16.onnx"),
                dir.join("asr_encoder_frontend.int4.onnx"),
                dir.join("encoder.int4.onnx"),
                dir.join("encoder.onnx"),
            ]
        } else {
            [
                dir.join("encoder.onnx"),
                dir.join("asr_encoder_frontend.int4.onnx"),
                dir.join("encoder.int4.onnx"),
                dir.join("encoder.fp16.onnx"),
            ]
        };
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
        let mut ep_errors: Vec<String> = Vec::new();
        let mut actual_ep = "CPU".to_string();

        #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
        let cuda_usable = match backend {
            EncoderBackend::Cuda | EncoderBackend::Auto => {
                #[cfg(target_os = "windows")]
                {
                    cudnn_available()
                }
                #[cfg(not(target_os = "windows"))]
                {
                    true
                }
            }
            _ => false,
        };
        #[cfg(not(any(feature = "cuda", feature = "qwen-cuda")))]
        let cuda_usable = false;

        let mut session_res: Option<Session> = None;

        // 1. 尝试 CUDA EP (须具备 cuDNN 9 且驱动匹配; commit 实测验证, 失败自动回退)
        #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
        if cuda_usable {
            use ort::ep::CUDA;
            println!("[encoder] 正在尝试注册并构建 CUDA EP Session...");
            let cuda_builder = builder.clone();
            if let Ok(b) = cuda_builder.with_execution_providers([CUDA::default().build()]) {
                match b.commit_from_file(&chosen_str) {
                    Ok(sess) => {
                        println!("[encoder] ✅ CUDA EP 模式 Session 构建成功 (onnxruntime GPU)");
                        actual_ep = "CUDA".to_string();
                        session_res = Some(sess);
                    }
                    Err(e) => {
                        let msg = format!("{:?}", e);
                        println!("[encoder] ⚠️ CUDA EP Session 构建失败 ({:?}); 准备回退降级", e);
                        ep_errors.push(format!("CUDA EP: {}", msg));
                    }
                }
            }
        }

        // 2. 降级尝试 DirectML EP (Windows 环境下无需 cuDNN，纯 DirectX 12 显卡硬件加速)
        //    仅 qwen-dml / qwen-dml-win 特性启用时编译 (vulkan 特性已不再捆绑 DML:
        //    否则 ort 仅凭 load-dynamic 也会注册成功, 实际却静默跑 CPU, UI 显示失真)
        #[cfg(all(target_os = "windows", any(feature = "qwen-dml", feature = "qwen-dml-win")))]
        if session_res.is_none() && (matches!(backend, EncoderBackend::Auto | EncoderBackend::DirectMl | EncoderBackend::Cuda)) {
            use ort::ep::DirectML;
            println!("[encoder] 正在尝试注册并构建 DirectML EP Session (DX12 GPU 加速, 无需 cuDNN)...");
            let dml_builder = builder.clone();
            if let Ok(b) = dml_builder.with_execution_providers([DirectML::default().build()]) {
                match b.commit_from_file(&chosen_str) {
                    Ok(sess) => {
                        println!("[encoder] ✅ DirectML EP (DX12 GPU) 模式 Session 构建成功");
                        actual_ep = "DirectML".to_string();
                        session_res = Some(sess);
                    }
                    Err(e) => {
                        let msg = format!("{:?}", e);
                        println!("[encoder] ⚠️ DirectML EP Session 构建失败 ({:?}); 准备回退降级", e);
                        ep_errors.push(format!("DirectML EP: {}", msg));
                    }
                }
            }
        }

        // 3. 终极降级：CPU 兜底保障 (确保模型 100% 成功装载)
        let session = match session_res {
            Some(sess) => sess,
            None => {
                println!("[encoder] ================================================================");
                println!("[encoder] ⚠️  提示: 所有 GPU 执行提供程序不可用/注册失败，降级使用 CPU 推理!");
                if !ep_errors.is_empty() {
                    for e in &ep_errors {
                        println!("[encoder]     未成功原因: {}", e);
                    }
                }
                println!("[encoder] ================================================================");
                actual_ep = "CPU".to_string();
                builder
                    .commit_from_file(&chosen_str)
                    .map_err(|e| QwenError::OnnxError(format!("commit_from_file CPU fallback ({}): {}", chosen_str, e)))?
            }
        };

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

        let mut encoder = Self {
            session,
            backend,
            model_path: chosen_str,
            input_name,
            backend_path: String::new(),
            actual_ep,
            layout,
        };
        encoder.warmup();
        Ok(encoder)
    }

    /// 对 ONNX Session 进行零延迟预热：触发底层内存池分配、图优化与 SIMD/AVX 线程池初始化
    pub fn warmup(&mut self) {
        let t0 = std::time::Instant::now();
        let dummy_samples = vec![0.0f32; 16000];
        let cancel = AtomicBool::new(false);
        let _ = self.encode(&dummy_samples, &cancel);
        println!(
            "[encoder] ONNX Session 预热完成: {}ms (EP: {})",
            t0.elapsed().as_millis(),
            self.actual_ep
        );
    }

    /// 编码一段 16k 音频 (ORT Session run 需 &mut, 单实例串行)。
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
        let t_mel_start = std::time::Instant::now();
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
        let mel_ms = t_mel_start.elapsed().as_millis();
        let t_run_start = std::time::Instant::now();

        // 排序：layout 匹配的特征优先；另一特征仅在主路径失败时作为兜底。
        let try_order: [Option<(Vec<f32>, Vec<i64>)>; 2] = [primary, fallback];

        let mut last_err: Option<QwenError> = None;
        let mut embeddings = Vec::new();
        let mut shape_vec = Vec::new();

        // 按值迭代: into_boxed_slice 可零拷贝转移所有权 (避免每次段编码的多余数据拷贝)
        for entry in try_order {
            let Some((data, shape)) = entry else { continue };
            if data.is_empty() {
                continue;
            }
            let shape_for_log = shape.clone(); // 仅 3 个 i64，用于成功日志
            let tensor = match Tensor::<f32>::from_array((
                shape,
                data.into_boxed_slice(),
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
                    println!("[encoder] ONNX inference succeeded with input shape {:?}", shape_for_log);
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

        let run_ms = t_run_start.elapsed().as_millis();
        println!(
            "[encoder] timing: mel={}ms session_run={}ms",
            mel_ms, run_ms
        );

        println!(
            "[ONNX Encoder] ONNX 前向传播计算成功完成: 音频时长 {:.2}s -> 生成声学 Embedding 形状 {:?}",
            samples.len() as f64 / SAMPLE_RATE as f64,
            shape_vec
        );

        Ok(EncoderOutput {
            embeddings,
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
/// 仅 Windows + CUDA 构建启用；非 Windows 的 CUDA 环境假定系统路径可直接加载。
/// 在给定目录及其子目录中搜索包含指定 prefix 前缀的 DLL 文件夹
/// 从字符串提取 (major, minor) 版本号。
/// 例如: "cudnn64_9.dll" -> (9, 0), "v9.24" -> (9, 24),
///       "cudart64_13.dll" -> (13, 0), "v13.3" -> (13, 3)
fn extract_version(s: &str) -> Option<(u32, u32)> {
    let lower = s.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            let mut j = i;
            while j < chars.len() && chars[j].is_ascii_digit() {
                j += 1;
            }
            let major_str: String = chars[i..j].iter().collect();
            if let Ok(major) = major_str.parse::<u32>() {
                if (8..=18).contains(&major) {
                    // 检查是否紧跟 ".数字" 形式的 minor (如 "9.24")
                    let mut minor = 0u32;
                    if j + 1 < chars.len() && chars[j] == '.' && chars[j + 1].is_ascii_digit() {
                        let mut k = j + 1;
                        while k < chars.len() && chars[k].is_ascii_digit() {
                            k += 1;
                        }
                        if let Ok(mi) = chars[j + 1..k].iter().collect::<String>().parse::<u32>() {
                            minor = mi;
                        }
                    }
                    return Some((major, minor));
                }
            }
            i = j;
        } else {
            i += 1;
        }
    }
    None
}

/// 在目录树中搜索包含指定前缀 DLL 的文件夹，并按以下优先级排序：
/// 1. CUDA 变体与系统 CUDA 主版本匹配 (preferred_cuda_major != 0 时启用，
///    用于 cuDNN: bin/<cuda_ver>/x64 目录结构对应不同 CUDA 变体)
/// 2. 与目标主版本 (preferred_major) 距离最近 (如 cuDNN 目标 9, cudart 目标 13)
/// 3. 同主版本时 minor 大者优先 (最新受支持版本，如 cuDNN 9.24 > 9.10)
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn find_best_dll_dir(
    root: &Path,
    dll_prefix: &str,
    preferred_major: u32,
    preferred_cuda_major: u32,
) -> Option<std::path::PathBuf> {
    if !root.exists() {
        return None;
    }
    // (dir, 库主版本, 库 minor, 对应 CUDA 主版本变体, 0=未知)
    let mut candidates: Vec<(std::path::PathBuf, u32, u32, u32)> = Vec::new();
    let mut stack = vec![(root.to_path_buf(), 0)];

    while let Some((dir, depth)) = stack.pop() {
        if depth > 5 {
            continue;
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut subdirs = Vec::new();
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() {
                    if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                        if name.to_lowercase().starts_with(dll_prefix) && name.to_lowercase().ends_with(".dll") {
                            // 版本: DLL 名提供 major (如 cudnn64_9 -> 9)，
                            // minor 从所在目录路径补充 (如 v9.24/bin/13.3/x64 -> minor 24)
                            let ver = match extract_version(name) {
                                Some((m, 0)) => {
                                    dir.to_str()
                                        .and_then(extract_version)
                                        .filter(|(dm, _)| *dm == m)
                                        .map(|(_, dmi)| (m, dmi))
                                        .unwrap_or((m, 0))
                                }
                                v => v.unwrap_or((0, 0)),
                            };
                            // CUDA 变体: cuDNN 目录结构 bin/<cuda_ver>/x64 —— 父目录名即 CUDA 版本
                            let cuda_major = dir
                                .parent()
                                .and_then(|p| p.file_name().and_then(|n| n.to_str()))
                                .and_then(extract_version)
                                .map(|(m, _)| m)
                                .filter(|m| (11..=13).contains(m))
                                .unwrap_or(0);
                            candidates.push((dir.clone(), ver.0, ver.1, cuda_major));
                            break;
                        }
                    }
                } else if p.is_dir() {
                    subdirs.push(p);
                }
            }
            for sub in subdirs {
                stack.push((sub, depth + 1));
            }
        }
    }

    if candidates.is_empty() {
        return None;
    }

    // 排序: CUDA 变体匹配系统 -> minor 最新 -> major 距离 -> major 大
    candidates.sort_by(|a, b| {
        let match_a = if preferred_cuda_major != 0 && a.3 != 0 {
            if a.3 == preferred_cuda_major { 0 } else { 1 }
        } else {
            0 // 未知变体不惩罚 (兼容无版本子目录的安装结构)
        };
        let match_b = if preferred_cuda_major != 0 && b.3 != 0 {
            if b.3 == preferred_cuda_major { 0 } else { 1 }
        } else {
            0
        };
        let dist_a = (a.1 as i32 - preferred_major as i32).abs();
        let dist_b = (b.1 as i32 - preferred_major as i32).abs();
        match_a
            .cmp(&match_b)
            .then_with(|| b.2.cmp(&a.2))
            .then_with(|| dist_a.cmp(&dist_b))
            .then_with(|| b.1.cmp(&a.1))
    });

    Some(candidates.remove(0).0)
}

/// 探测系统可用的 CUDA 主版本，用于匹配 cuDNN 的 CUDA 变体。
/// 简化依据: 显卡驱动与 CUDA 工具包安装时均已做兼容性检查 (驱动向后兼容、
/// 工具包安装器校验驱动版本)，因此这里只需区分 onnxruntime 1.28 支持的
/// CUDA 13 / 12 (CUDA 11 不支持，无需探测)。cuDNN 是解压式安装无检查，
/// 故按此主版本匹配其 bin/<cuda_ver>/x64 变体即可；最终由 CUDA EP
/// commit 实测验证兜底 (失败自动回退 DirectML/CPU)。
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn preferred_cuda_major() -> u32 {
    let base = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    for major in [13u32, 12] {
        let dll = format!("cudart64_{}.dll", major);
        if let Ok(entries) = std::fs::read_dir(base) {
            if entries.flatten().any(|e| e.path().join("bin/x64").join(&dll).exists()) {
                return major;
            }
        }
        if let Ok(path) = std::env::var("PATH") {
            if path
                .split(';')
                .any(|d| !d.is_empty() && Path::new(d).join(&dll).exists())
            {
                return major;
            }
        }
    }
    0
}

/// 自动探测系统的 cuDNN 安装目录。
/// 优先选择与系统 CUDA 主版本匹配的变体 (bin/<cuda_ver>/x64)，
/// 同匹配时选 cuDNN minor 最新 (如 9.24 > 9.10)。
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn probe_cudnn_dir() -> Option<std::path::PathBuf> {
    let cuda_major = preferred_cuda_major();

    // 1. 优先读取显式环境变量 (CUDNN_PATH, CUDNN_ROOT, CUDNN_HOME)
    for env_var in ["CUDNN_PATH", "CUDNN_ROOT", "CUDNN_HOME"] {
        if let Ok(val) = std::env::var(env_var) {
            let p = Path::new(&val);
            if let Some(bin) = find_best_dll_dir(p, "cudnn64_", 9, cuda_major) {
                return Some(bin);
            }
        }
    }

    // 2. 检查常见默认安装路径 (扫描 C:\Program Files 下的所有 CUDNN 版本目录并按匹配度排序)
    let bases = [
        Path::new("C:/Program Files/NVIDIA/CUDNN"),
        Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDNN"),
    ];

    for base in bases {
        if base.exists() {
            if let Some(found) = find_best_dll_dir(base, "cudnn64_", 9, cuda_major) {
                return Some(found);
            }
        }
    }
    None
}

/// 自动探测 CUDA Toolkit 的 bin 目录 (优先系统可用最高 CUDA 主版本)
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn probe_cuda_bin() -> Option<std::path::PathBuf> {
    let cuda_major = preferred_cuda_major();

    // 1. 优先读取 CUDA_PATH 环境变量
    if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
        let p = Path::new(&cuda_path);
        if let Some(bin) = find_best_dll_dir(p, "cudart64_", cuda_major.max(12), 0) {
            return Some(bin);
        }
    }

    // 2. 读取其他以 CUDA_PATH_ 开头的环境变量 (如 CUDA_PATH_V13_0, CUDA_PATH_V12_0 等)
    let mut env_paths = Vec::new();
    for (k, v) in std::env::vars() {
        if k.starts_with("CUDA_PATH_") {
            env_paths.push(v);
        }
    }
    for val in env_paths {
        let p = Path::new(&val);
        if let Some(bin) = find_best_dll_dir(p, "cudart64_", cuda_major.max(12), 0) {
            return Some(bin);
        }
    }

    // 3. 探查常规安装主目录
    let base = Path::new("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA");
    if base.exists() {
        if let Some(found) = find_best_dll_dir(base, "cudart64_", cuda_major.max(12), 0) {
            return Some(found);
        }
    }
    None
}

/// 检查指定路径中是否存在 ONNX Runtime GPU 所必需的 cuDNN 9 核心动态库
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn has_cudnn9_dll(dir: &Path) -> bool {
    if !dir.exists() {
        return false;
    }
    if dir.join("cudnn64_9.dll").exists() || dir.join("bin").join("cudnn64_9.dll").exists() {
        return true;
    }
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() {
                if let Some(name) = p.file_name().and_then(|n| n.to_str()) {
                    if name.to_lowercase().starts_with("cudnn64_9") && name.to_lowercase().ends_with(".dll") {
                        return true;
                    }
                }
            } else if p.is_dir() {
                if p.join("cudnn64_9.dll").exists() {
                    return true;
                }
            }
        }
    }
    false
}

/// 判断 cuDNN 当前是否可加载
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn cudnn_available() -> bool {
    if let Ok(exe_path) = std::env::current_exe() {
        if let Some(dir) = exe_path.parent() {
            if has_cudnn9_dll(dir) {
                return true;
            }
        }
    }
    if let Some(dir) = probe_cudnn_dir() {
        if has_cudnn9_dll(&dir) {
            return true;
        }
    }
    if let Ok(path) = std::env::var("PATH") {
        for d in path.split(';') {
            if !d.is_empty() && has_cudnn9_dll(Path::new(d)) {
                return true;
            }
        }
    }
    false
}

/// 部署 CUDA 版 ONNX Runtime 核心 DLL 到 exe 目录:
/// onnxruntime.dll + onnxruntime_providers_cuda.dll + providers_shared
/// cuDNN/CUDA 运行时 DLL **不拷贝**：直接通过 AddDllDirectory 使用系统安装
#[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
fn deploy_cuda_bundle(exe_dir: &Path) {
    let bundled = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("third_party")
        .join("onnxruntime-cuda");
    let core_files = [
        "onnxruntime.dll",
        "onnxruntime_providers_cuda.dll",
        "onnxruntime_providers_shared.dll",
    ];

    // Windows: 每次启动都注册系统 cuDNN/CUDA 目录到 DLL 搜索路径，
    // 并清理历史版本拷贝到 exe 目录的显卡 DLL (释放磁盘空间)。
    #[cfg(target_os = "windows")]
    {
        if let Some(dir) = probe_cudnn_dir() {
            add_dll_directory(&dir);
            println!("[encoder] Using cuDNN runtime from {} (no copy)", dir.display());
        } else {
            println!("[encoder] INFO: System cuDNN 9 installation not found; CUDA EP will fall back to DirectML/CPU.");
        }
        if let Some(dir) = probe_cuda_bin() {
            add_dll_directory(&dir);
            println!("[encoder] Using CUDA runtime from {} (no copy)", dir.display());
        }
        cleanup_legacy_cuda_copies(exe_dir);
    }

    // 必须所有核心文件存在且 onnxruntime.dll 文件大小与 CUDA 版本完全匹配，才跳过拷贝
    let cuda_dll_src = bundled.join("onnxruntime.dll");
    let cuda_dll_dst = exe_dir.join("onnxruntime.dll");
    let dll_matches = if cuda_dll_src.exists() && cuda_dll_dst.exists() {
        cuda_dll_src.metadata().map(|m| m.len()).unwrap_or(0) == cuda_dll_dst.metadata().map(|m| m.len()).unwrap_or(1)
    } else {
        false
    };
    let already = dll_matches && core_files.iter().all(|f| exe_dir.join(f).exists());
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
}

/// 清理旧版本部署时拷贝到 exe 目录的 cuDNN/CUDA 运行时 DLL (cudnn*.dll 等)。
/// 这些 DLL 现在直接使用系统安装，不再需要本地副本。
#[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
fn cleanup_legacy_cuda_copies(exe_dir: &Path) {
    let legacy_prefixes = [
        "cudnn", "cudart64_13", "cublas64_13", "cublasLt64_13",
        "cudart64_12", "cublas64_12", "cublasLt64_12",
    ];
    if let Ok(entries) = std::fs::read_dir(exe_dir) {
        for e in entries.flatten() {
            let p = e.path();
            let name = p.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.ends_with(".dll")
                && legacy_prefixes.iter().any(|pfx| name.starts_with(pfx))
            {
                match std::fs::remove_file(&p) {
                    Ok(_) => println!("[encoder] Removed legacy copied DLL: {}", name),
                    Err(_) => println!("[encoder] Skipped locked legacy DLL: {}", name),
                }
            }
        }
    }
}

/// 把目录加入进程 DLL 搜索路径 (LoadLibrary 无需拷贝即可找到系统安装的显卡 DLL)。
#[cfg(target_os = "windows")]
fn add_dll_directory(path: &Path) {
    use std::os::windows::ffi::OsStrExt;
    use windows_sys::Win32::System::LibraryLoader::{
        AddDllDirectory, SetDefaultDllDirectories, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS,
    };
    unsafe {
        // 必须先启用默认搜索标志，AddDllDirectory 添加的目录才会被普通 LoadLibrary 搜索
        SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
        let wide: Vec<u16> = path.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
        AddDllDirectory(wide.as_ptr());
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
        // 诊断用: ORT_VERBOSE=1 时输出 onnxruntime 详细日志 (含 EP partition 信息)
        if std::env::var("ORT_VERBOSE").is_ok() {
            if let Ok(env) = ort::environment::get_environment() {
                env.set_log_level(ort::logging::LogLevel::Verbose);
            }
        }
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(exe_dir) = exe_path.parent() {
                println!("[encoder] Checking runtime DLLs in exe_dir: {:?}", exe_dir);

                let want_cuda = match backend {
                    EncoderBackend::Cuda => true,
                    EncoderBackend::DirectMl => false,
                    EncoderBackend::Auto => {
                        #[cfg(target_os = "windows")]
                        {
                            #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
                            {
                                cudnn_available()
                            }
                            #[cfg(not(any(feature = "cuda", feature = "qwen-cuda")))]
                            {
                                false
                            }
                        }
                        #[cfg(not(target_os = "windows"))]
                        {
                            true
                        }
                    }
                    EncoderBackend::Cpu => false,
                };
                println!(
                    "[encoder] backend={:?} want_cuda={}",
                    backend, want_cuda
                );

                #[cfg(any(feature = "cuda", feature = "qwen-cuda"))]
                if want_cuda {
                    deploy_cuda_bundle(&exe_dir);
                }
                #[cfg(not(any(feature = "cuda", feature = "qwen-cuda")))]
                if want_cuda {
                    println!("[encoder] cuda feature not enabled; skip CUDA bundle deploy");
                }
                if !want_cuda {
                    // DirectML / CPU: 部署项目内捆绑的 DirectML 版
                    let bundled_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
                        .join("third_party")
                        .join("onnxruntime-dml");
                    let bundled_dll = bundled_dir.join("onnxruntime.dll");
                    let target_dll = exe_dir.join("onnxruntime.dll");
                    let target_shared = exe_dir.join("onnxruntime_providers_shared.dll");
                    if bundled_dll.exists() {
                        // 覆盖旧拷贝 (可能是 CPU 版)，保证使用 DML 版
                        let target_cuda = exe_dir.join("onnxruntime_providers_cuda.dll");
                        let _ = std::fs::remove_file(&target_cuda);
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
    fn extract_version_parses_major_minor() {
        assert_eq!(extract_version("cudnn64_9.dll"), Some((9, 0)));
        assert_eq!(extract_version("v9.24"), Some((9, 24)));
        assert_eq!(extract_version("cudart64_13.dll"), Some((13, 0)));
        assert_eq!(extract_version("v13.3"), Some((13, 3)));
        assert_eq!(extract_version("cublasLt64_13.dll"), Some((13, 0)));
        assert_eq!(extract_version("cudnn_ops64_9.dll"), Some((9, 0)));
        assert_eq!(extract_version("C:/Program Files/NVIDIA/CUDNN/v9.24/bin/13.3/x64"), Some((9, 24)));
        assert_eq!(extract_version("no-version-here"), None);
    }

    /// 构造临时 cuDNN 安装目录树，验证 CUDA 变体匹配 + minor 最新排序
    #[cfg(all(target_os = "windows", any(feature = "cuda", feature = "qwen-cuda")))]
    #[test]
    fn find_best_dll_dir_matches_cuda_variant() {
        use std::fs;
        let tmp = std::env::temp_dir().join(format!("cudnn_test_{}", std::process::id()));
        let _ = fs::remove_dir_all(&tmp);
        // v9.10 只有 CUDA 12 变体
        fs::create_dir_all(tmp.join("v9.10/bin/12.9/x64")).unwrap();
        fs::write(tmp.join("v9.10/bin/12.9/x64/cudnn64_9.dll"), b"x").unwrap();
        // v9.24 有 CUDA 12 和 13 两个变体
        fs::create_dir_all(tmp.join("v9.24/bin/13.3/x64")).unwrap();
        fs::write(tmp.join("v9.24/bin/13.3/x64/cudnn64_9.dll"), b"x").unwrap();
        fs::create_dir_all(tmp.join("v9.24/bin/12.9/x64")).unwrap();
        fs::write(tmp.join("v9.24/bin/12.9/x64/cudnn64_9.dll"), b"x").unwrap();

        // 系统 CUDA 13: 应选 v9.24 的 CUDA 13 变体 (匹配 + 最新)
        let r = find_best_dll_dir(&tmp, "cudnn64_", 9, 13).unwrap();
        assert!(r.to_string_lossy().contains("v9.24"), "got {}", r.display());
        assert!(r.to_string_lossy().contains("13.3"), "got {}", r.display());

        // 系统 CUDA 12: 应选 v9.24 的 CUDA 12 变体 (匹配 + minor 最新, 而非 v9.10)
        let r2 = find_best_dll_dir(&tmp, "cudnn64_", 9, 12).unwrap();
        assert!(r2.to_string_lossy().contains("v9.24"), "got {}", r2.display());
        assert!(r2.to_string_lossy().contains("12.9"), "got {}", r2.display());

        // 无 CUDA 匹配需求 (0): 按 minor 最新选 v9.24 (13.3 变体, 递归先序不保证, 但必为 v9.24)
        let r3 = find_best_dll_dir(&tmp, "cudnn64_", 9, 0).unwrap();
        assert!(r3.to_string_lossy().contains("v9.24"), "got {}", r3.display());

        let _ = fs::remove_dir_all(&tmp);
    }

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
