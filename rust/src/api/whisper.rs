use crate::frb_generated::StreamSink;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSysContext,
    WhisperSysState, DtwParameters, DtwMode, DtwModelPreset,
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};
use std::sync::Arc;
use std::sync::Mutex;

struct CachedContext {
    model_path: String,
    use_gpu: bool,
    context: Arc<WhisperContext>,
}

static G_CONTEXT: Mutex<Option<CachedContext>> = Mutex::new(None);

pub(crate) fn get_or_create_context(
    model_path: &str,
    use_gpu: bool,
    ctx_params: WhisperContextParameters,
) -> Result<Arc<WhisperContext>, String> {
    let mut cache = G_CONTEXT
        .lock()
        .map_err(|e| format!("Failed to lock global context: {}", e))?;

    if let Some(ref cached) = *cache {
        if cached.model_path == model_path && cached.use_gpu == use_gpu {
            println!("[Rust] Reusing cached Whisper context for model: {}", model_path);
            return Ok(cached.context.clone());
        }
    }

    println!(
        "[Rust] Loading new Whisper context for model: {} (use_gpu={})",
        model_path, use_gpu
    );
    let new_ctx = WhisperContext::new_with_params(model_path, ctx_params).map_err(|e| {
        let err_msg = format!("加载模型失败: {}", e);
        println!("[Rust] {}", err_msg);
        err_msg
    })?;

    let shared_ctx = Arc::new(new_ctx);
    *cache = Some(CachedContext {
        model_path: model_path.to_string(),
        use_gpu,
        context: shared_ctx.clone(),
    });

    Ok(shared_ctx)
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptionSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub enum TranscriptionEvent {
    Progress(i32),
    ProgressDetail { processed_ms: i64, total_ms: i64 },
    Success(Vec<TranscriptionSegment>),
    Failure(String),
    Segment(TranscriptionSegment),
}

#[derive(Clone, Debug, Default)]
pub struct VulkanDeviceInfo {
    pub id: i32,
    pub name: String,
    pub total_vram_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct HardwareAccelerationInfo {
    pub is_vulkan_available: bool,
    pub devices: Vec<VulkanDeviceInfo>,
}

#[cfg(target_os = "windows")]
#[repr(C)]
struct DISPLAY_DEVICEA {
    cb: u32,
    device_name: [u8; 32],
    device_string: [u8; 128],
    state_flags: u32,
    device_id: [u8; 128],
    device_key: [u8; 128],
}

#[cfg(target_os = "windows")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> isize;
    fn FreeLibrary(hLibModule: isize) -> i32;
    fn EnumDisplayDevicesA(
        lpDevice: *const u8,
        iDevNum: u32,
        lpDisplayDevice: *mut DISPLAY_DEVICEA,
        dwFlags: u32,
    ) -> i32;
}

#[cfg(target_os = "windows")]
fn check_vulkan_supported_safely() -> bool {
    unsafe {
        println!("[Rust] Safe Vulkan Check: Loading vulkan-1.dll...");
        let module = LoadLibraryA(b"vulkan-1.dll\0".as_ptr());
        if module == 0 {
            println!("[Rust] Safe Vulkan Check: vulkan-1.dll not found in system paths.");
            return false;
        }
        FreeLibrary(module);

        // 枚举 Windows 显示适配器名称，排除 Microsoft 默认基本适配器以校验是否存在真实图形驱动
        let mut dd = DISPLAY_DEVICEA {
            cb: std::mem::size_of::<DISPLAY_DEVICEA>() as u32,
            device_name: [0; 32],
            device_string: [0; 128],
            state_flags: 0,
            device_id: [0; 128],
            device_key: [0; 128],
        };

        let mut found_real_gpu = false;
        let mut i = 0;
        while EnumDisplayDevicesA(std::ptr::null(), i, &mut dd, 0) != 0 {
            if (dd.state_flags & 0x1) != 0 { // DISPLAY_DEVICE_ACTIVE = 1
                let device_str = std::ffi::CStr::from_ptr(dd.device_string.as_ptr() as *const i8)
                    .to_string_lossy()
                    .to_lowercase();
                println!("[Rust] Safe Vulkan Check: Active display adapter: {}", device_str);
                if !device_str.contains("basic display") && !device_str.contains("basic render") {
                    found_real_gpu = true;
                }
            }
            i += 1;
        }

        if !found_real_gpu {
            println!("[Rust] Safe Vulkan Check: Only Microsoft Basic Display adapter detected. Vulkan disabled.");
            return false;
        }

        println!("[Rust] Safe Vulkan Check: Real hardware GPU adapter is active. Vulkan check passed.");
        true
    }
}

#[cfg(not(target_os = "windows"))]
fn check_vulkan_supported_safely() -> bool {
    true
}

pub fn get_hardware_acceleration_info() -> HardwareAccelerationInfo {
    #[cfg(feature = "vulkan")]
    {
        let is_vulkan_safe = check_vulkan_supported_safely();

        if is_vulkan_safe {
            println!("[Rust] Querying Vulkan devices...");
            let devices = whisper_rs::vulkan::list_devices();
            let is_vulkan_available = !devices.is_empty();
            let mapped_devices = devices
                .into_iter()
                .map(|d| VulkanDeviceInfo {
                    id: d.id,
                    name: d.name,
                    total_vram_bytes: d.vram.total as u64,
                })
                .collect();
            println!("[Rust] Vulkan support: {}, devices: {:?}", is_vulkan_available, mapped_devices);
            HardwareAccelerationInfo {
                is_vulkan_available,
                devices: mapped_devices,
            }
        } else {
            println!("[Rust] Vulkan check failed. Bypassing Vulkan device listing to prevent crash.");
            HardwareAccelerationInfo {
                is_vulkan_available: false,
                devices: vec![],
            }
        }
    }
    #[cfg(not(feature = "vulkan"))]
    {
        println!("[Rust] Vulkan feature disabled/not compiled.");
        HardwareAccelerationInfo {
            is_vulkan_available: false,
            devices: vec![],
        }
    }
}

struct ProgressContext<'a> {
    callback: &'a mut dyn FnMut(TranscriptionEvent),
    current_segment: usize,
    total_segments: usize,
}

// 原始 C++ 风格的进度回调函数（避免闭包的堆内存泄漏以及二次转写端口死锁/卡死问题）
unsafe extern "C" fn progress_callback_trampoline(
    _ctx: *mut WhisperSysContext,
    _state: *mut WhisperSysState,
    progress: std::ffi::c_int,
    user_data: *mut std::ffi::c_void,
) {
    if !user_data.is_null() {
        let context = &mut *(user_data as *mut ProgressContext);
        let scaled_progress = if context.total_segments > 0 {
            let seg_contribution = 100.0 / context.total_segments as f64;
            let current_base = context.current_segment as f64 * seg_contribution;
            let current_progress = (progress as f64 * seg_contribution) / 100.0;
            (current_base + current_progress).round() as i32
        } else {
            progress
        };
        let scaled_progress = scaled_progress.clamp(0, 100);
        println!(
            "[Rust] Progress callback: {}% (raw: {}%, segment {}/{})",
            scaled_progress, progress, context.current_segment + 1, context.total_segments
        );
        (context.callback)(TranscriptionEvent::Progress(scaled_progress));
    }
}



pub fn convert_chinese(text: String, to_simplified: bool) -> String {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    zhconv::zhconv(&text, target)
}

pub fn convert_chinese_list(texts: Vec<String>, to_simplified: bool) -> Vec<String> {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    texts.into_iter().map(|text| zhconv::zhconv(&text, target)).collect()
}

pub fn transcribe(
    sink: StreamSink<TranscriptionEvent>,
    model_path: String,
    vad_model_path: String,
    audio_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    // VAD 参数
    vad_enabled: bool,
    vad_threshold: f32,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
    // Whisper 惩罚与降级参数
    temperature: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
    no_context: bool,
) {
    // 异步执行转写任务，防止界面卡顿
    std::thread::spawn(move || {
        let mut callback = |event| {
            let _ = sink.add(event);
        };
        if let Err(e) = run_transcription_inner(
            &mut callback,
            model_path,
            vad_model_path,
            audio_path,
            language,
            translate,
            threads,
            use_gpu,
            vad_enabled,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            temperature,
            temperature_inc,
            entropy_thold,
            logprob_thold,
            no_speech_thold,
            no_context,
        ) {
            let _ = sink.add(TranscriptionEvent::Failure(e));
        }
    });
}

pub(crate) fn run_transcription_inner(
    callback: &mut dyn FnMut(TranscriptionEvent),
    model_path: String,
    vad_model_path: String,
    audio_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    vad_enabled: bool,
    vad_threshold: f32,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
    temperature: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
    _no_context: bool,
) -> Result<(), String> {
    println!(
        "[Rust] run_transcription: model_path={}, vad_model_path={}, audio_path={}, language={:?}, translate={}, use_gpu={}, vad={}",
        model_path, vad_model_path, audio_path, language, translate, use_gpu, vad_enabled
    );

    // 1. 读取音频数据 (16kHz 单声道 16-bit PCM wav)
    let mut reader = hound::WavReader::open(&audio_path).map_err(|e| {
        let err_msg = format!("无法打开音频文件: {}", e);
        println!("[Rust] {}", err_msg);
        err_msg
    })?;
    
    let spec = reader.spec();
    println!("[Rust] WAV Spec: sample_rate={}, channels={}, bits_per_sample={}", spec.sample_rate, spec.channels, spec.bits_per_sample);
    if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        let err_msg = "音频格式错误：必须是 16kHz, 单声道, 16位 PCM WAV 文件。".to_string();
        println!("[Rust] {}", err_msg);
        return Err(err_msg);
    }

    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        let s = sample.map_err(|e| {
            let err_msg = format!("读取音频采样点失败: {}", e);
            println!("[Rust] {}", err_msg);
            err_msg
        })?;
        samples.push(s as f32 / 32768.0);
    }
    println!("[Rust] Loaded {} audio samples ({:.2} seconds)", samples.len(), samples.len() as f64 / 16000.0);

    // 确保音频总长度不少于 0.5 秒以防崩溃
    if samples.len() < 8000 {
        println!("[Rust] ⚠️ 整体音频过短 ({} samples), 已自动静音填充至 0.5s 以保护 DTW。", samples.len());
        samples.resize(8000, 0.0);
    }

    // 2. 加载模型上下文
    println!("[Rust] Loading Whisper model context (use_gpu={})...", use_gpu);
    let mut ctx_params = WhisperContextParameters::default();
    
    // 启用 DTW 对齐模式
    let dtw_preset = get_dtw_model_preset(&model_path);
    if let Some(preset) = dtw_preset {
        println!("[Rust] DTW alignment enabled with preset for model: {}", model_path);
        let mem_size = calculate_dtw_mem_size(samples.len());
        ctx_params.dtw_parameters(DtwParameters {
            mode: DtwMode::ModelPreset { model_preset: preset },
            dtw_mem_size: mem_size,
        });
    } else {
        println!("[Rust] DTW alignment disabled (no preset for model: {})", model_path);
        ctx_params.dtw_parameters(DtwParameters {
            mode: DtwMode::None,
            dtw_mem_size: 0,
        });
    }
    
    let mut selected_device_name = "CPU".to_string();
    if use_gpu {
        #[cfg(feature = "vulkan")]
        {
            let devices = whisper_rs::vulkan::list_devices();
            if !devices.is_empty() {
                let dgpu = devices.iter().find(|d| {
                    let name_lower = d.name.to_lowercase();
                    let is_igpu = name_lower.contains("integrated")
                        || name_lower.contains("uhd")
                        || name_lower.contains("iris")
                        || name_lower.contains("vega")
                        || (name_lower.contains("intel") && !name_lower.contains("arc"))
                        || name_lower.contains("radeon(tm)");
                    !is_igpu
                });

                let selected = if let Some(d) = dgpu {
                    Some(d)
                } else {
                    devices.first()
                };

                if let Some(device) = selected {
                    println!("[Rust] Selecting Vulkan GPU device {}: {}", device.id, device.name);
                    ctx_params.use_gpu = true;
                    ctx_params.gpu_device = device.id;
                    selected_device_name = format!("GPU: {}", device.name);
                } else {
                    println!("[Rust] No Vulkan devices selected, falling back to CPU.");
                    ctx_params.use_gpu = false;
                }
            } else {
                println!("[Rust] No Vulkan devices found, falling back to CPU.");
                ctx_params.use_gpu = false;
            }
        }
        #[cfg(not(feature = "vulkan"))]
        {
            println!("[Rust] Vulkan feature not compiled, falling back to CPU.");
            ctx_params.use_gpu = false;
        }
    } else {
        ctx_params.use_gpu = false;
    }
    println!("[Rust] Hardware device selected for model execution context: {}", selected_device_name);
    
    let ctx = get_or_create_context(&model_path, use_gpu, ctx_params)?;
    
    // 3. 创建推理状态 (在此处创建一次并复用)
    println!("[Rust] Creating Whisper state...");
    let mut state = ctx.create_state().map_err(|e| {
        let err_msg = format!("创建推理状态失败: {}", e);
        println!("[Rust] {}", err_msg);
        err_msg
    })?;
    
    // 4. 配置转写参数
    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    
    let n_threads = threads.unwrap_or(4);
    params.set_n_threads(n_threads);
    println!("[Rust] Threads set to: {}", n_threads);
    
    if let Some(ref lang) = language {
        if lang != "auto" && !lang.is_empty() {
            println!("[Rust] Explicit language code set to: '{}'", lang);
            params.set_language(Some(lang.as_str()));
        } else {
            println!("[Rust] Language code set to: Auto-detect");
            params.set_language(None);
            params.set_detect_language(false);
        }
    } else {
        println!("[Rust] Language code set to: Auto-detect (None)");
        params.set_language(None);
        params.set_detect_language(false);
    }
    
    params.set_translate(translate);

    // 应用防重复及降级参数
    params.set_temperature(temperature);
    params.set_temperature_inc(temperature_inc);
    params.set_entropy_thold(entropy_thold);
    params.set_logprob_thold(logprob_thold);
    params.set_no_speech_thold(no_speech_thold);
    params.set_no_context(false); // 必须允许使用底层 KV 缓存的 context 串联上下文
    params.set_single_segment(false); // 允许模型自适应处理长句
    println!(
        "[Rust] Whisper Params: temp={}, temp_inc={}, entropy_thold={}, logprob_thold={}, no_speech_thold={}, no_context=false",
        temperature, temperature_inc, entropy_thold, logprob_thold, no_speech_thold
    );

    // 5. 运行转写与 VAD 切片逻辑
    let mut combined_segments = Vec::new();

    if vad_enabled {
        println!("[Rust] Initializing Silero VAD from: {}", vad_model_path);
        let vad_ctx_params = WhisperVadContextParams::new();
        let mut vad = WhisperVadContext::new(&vad_model_path, vad_ctx_params).map_err(|e| e.to_string())?;
        let mut vad_params = WhisperVadParams::new();
        vad_params.set_min_silence_duration(vad_min_silence_ms as i32);
        vad_params.set_min_speech_duration(vad_min_speech_ms as i32);
        let prob_threshold = if vad_threshold > 0.0 && vad_threshold < 1.0 { vad_threshold } else { 0.5f32 };
        vad_params.set_threshold(prob_threshold);

        let segs = vad.segments_from_samples(vad_params, &samples).map_err(|e| e.to_string())?;
        
        let mut speech_segments = Vec::new();
        for s in segs {
            let start_sample = s.start as usize * 160;
            let end_sample = s.end as usize * 160;
            if end_sample > start_sample && end_sample <= samples.len() {
                speech_segments.push((start_sample, end_sample));
            }
        }

        println!(
            "[Rust] VAD enabled. Found {} active speech segments via Silero VAD.",
            speech_segments.len()
        );

        if speech_segments.is_empty() {
            println!("[Rust] No speech segments detected. Returning empty subtitles.");
            callback(TranscriptionEvent::Success(combined_segments));
            return Ok(());
        }

        let mut progress_ctx = ProgressContext {
            callback,
            current_segment: 0,
            total_segments: speech_segments.len(),
        };

        // 设置进度回调
        unsafe {
            params.set_progress_callback(Some(progress_callback_trampoline));
            params.set_progress_callback_user_data(&progress_ctx as *const ProgressContext as *mut std::ffi::c_void);
        }

        for (idx, &(start_sample, end_sample)) in speech_segments.iter().enumerate() {
            // progress_ctx.current_segment = idx;
            
            // 增加 200ms 的平滑余量，且直接提取正确的切片
            let safe_start = start_sample.saturating_sub(3200); 
            let safe_end = (end_sample + 3200).min(samples.len()); 
            let mut segment_samples = samples[safe_start..safe_end].to_vec();

            // 如果噪音切片小于 0.2 秒，可能是杂音，直接跳过防幻觉
            if segment_samples.len() < 3200 {
                continue;
            }
            
            const MIN_SAMPLES_FOR_DTW: usize = 8000; // 0.5s
            if segment_samples.len() < MIN_SAMPLES_FOR_DTW {
                segment_samples.resize(MIN_SAMPLES_FOR_DTW, 0.0);
            }
            
            let global_offset_ms = (safe_start as i64) / 16;
            let current_params = params.clone();

            state.full(current_params, &segment_samples).map_err(|e| {
                let err_msg = format!("VAD 分段转写推理失败 (序号 {}): {}", idx + 1, e);
                println!("[Rust] {}", err_msg);
                err_msg
            })?;

            let final_num_segments = state.full_n_segments();
            for i in 0..final_num_segments {
                if let Some(segment) = state.get_segment(i) {
                    let text = segment.to_str_lossy().unwrap_or_default().into_owned();

                    // 滤除音乐符号
                    let cleaned_text = text.replace("🎵", "")
                                           .replace("[音乐]", "")
                                           .replace("(音乐)", "")
                                           .replace("[Music]", "")
                                           .trim().to_string();

                    if cleaned_text.is_empty() {
                        continue;
                    }

                    combined_segments.push(TranscriptionSegment {
                        start_ms: segment.start_timestamp() * 10 + global_offset_ms,
                        end_ms: segment.end_timestamp() * 10 + global_offset_ms,
                        text: cleaned_text,
                    });
                }
            }
        }
    } else {
        // VAD 未启用：对完整音频进行单次推理
        let progress_ctx = ProgressContext {
            callback,
            current_segment: 0,
            total_segments: 1,
        };

        // 设置进度回调
        unsafe {
            params.set_progress_callback(Some(progress_callback_trampoline));
            params.set_progress_callback_user_data(&progress_ctx as *const ProgressContext as *mut std::ffi::c_void);
        }

        println!("[Rust] VAD disabled. Running transcription on full audio...");
        state.full(params, &samples).map_err(|e| {
            let err_msg = format!("完整音频转写推理失败: {}", e);
            println!("[Rust] {}", err_msg);
            err_msg
        })?;

        let num_segments = state.full_n_segments();
        for i in 0..num_segments {
            if let Some(segment) = state.get_segment(i) {
                let text = segment
                    .to_str_lossy()
                    .unwrap_or_else(|_| std::borrow::Cow::Borrowed(""))
                    .into_owned();
                
                if text.trim().is_empty() {
                    continue;
                }

                let start = segment.start_timestamp();
                let end = segment.end_timestamp();

                combined_segments.push(TranscriptionSegment {
                    start_ms: start * 10,
                    end_ms: end * 10,
                    text,
                });
            }
        }
    }

    // 6. 发送转写成功事件 (进行简体中文转换)
    for seg in &mut combined_segments {
        seg.text = zhconv::zhconv(&seg.text, zhconv::Variant::ZhCN);
    }
    println!("[Rust] Sending Success event with {} segments", combined_segments.len());
    callback(TranscriptionEvent::Success(combined_segments));
    
    Ok(())
}

fn calculate_dtw_mem_size(num_samples: usize) -> usize {
    const FRAME_SAMPLES: usize = 160;
    let num_frames = (num_samples + FRAME_SAMPLES - 1) / FRAME_SAMPLES;

    const BYTES_F32: usize = 4;
    const BYTES_I32: usize = 4;
    const LANES: usize = 4;

    let band_frames = match num_frames {
        0..=15_000 => 96,
        15_001..=45_000 => 128,
        _ => 160,
    };

    let dp_bytes = num_frames
        .saturating_mul(band_frames)
        .saturating_mul(LANES)
        .saturating_mul(BYTES_F32);

    let bt_bytes = num_frames
        .saturating_mul(BYTES_I32);

    const BASELINE_MB: usize = 24;
    let base_bytes = BASELINE_MB * 1024 * 1024;

    let total = base_bytes
        .saturating_add(dp_bytes)
        .saturating_add(bt_bytes);

    let min_bytes = 24 * 1024 * 1024;
    let max_bytes = 768 * 1024 * 1024;
    let clamped = total.clamp(min_bytes, max_bytes);

    const ALIGN: usize = 8 * 1024 * 1024;
    (clamped + (ALIGN - 1)) & !(ALIGN - 1)
}

fn get_dtw_model_preset(model_path: &str) -> Option<DtwModelPreset> {
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("medium.en") {
        Some(DtwModelPreset::MediumEn)
    } else if path_lower.contains("medium") {
        Some(DtwModelPreset::Medium)
    } else if path_lower.contains("large-v3-turbo") {
        Some(DtwModelPreset::LargeV3Turbo)
    } else if path_lower.contains("large") {
        Some(DtwModelPreset::LargeV3)
    } else {
        // Disabling DTW for tiny, base, and small models to prevent median filter width ne[2] assertion crash.
        None
    }
}
