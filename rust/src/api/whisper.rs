use crate::frb_generated::StreamSink;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSysContext,
    WhisperSysState,
};

#[derive(Clone, Debug, Default)]
pub struct TranscriptionSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub enum TranscriptionEvent {
    Progress(i32),
    Success(Vec<TranscriptionSegment>),
    Failure(String),
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

pub fn get_hardware_acceleration_info() -> HardwareAccelerationInfo {
    #[cfg(feature = "vulkan")]
    {
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

struct ProgressContext {
    sink: StreamSink<TranscriptionEvent>,
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
        let context = &*(user_data as *const ProgressContext);
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
        let _ = context.sink.add(TranscriptionEvent::Progress(scaled_progress));
    }
}

/// 基于能量（RMS）的轻量级端点检测（VAD）算法
fn energy_based_vad(
    samples: &[f32],
    sample_rate: usize,
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
) -> Vec<(usize, usize)> {
    let window_ms = 30i32;
    let window_size = (sample_rate * window_ms as usize) / 1000; // 30ms = 480 个采样点
    if window_size == 0 || samples.len() < window_size {
        return vec![(0, samples.len())];
    }

    let min_speech_windows = (min_speech_ms / window_ms) as usize;
    let min_silence_windows = (min_silence_ms / window_ms) as usize;

    let num_windows = samples.len() / window_size;
    let mut window_activities = vec![false; num_windows];

    for i in 0..num_windows {
        let start = i * window_size;
        let end = start + window_size;
        let window_slice = &samples[start..end];
        let mut sum_sq = 0.0;
        for &s in window_slice {
            sum_sq += s * s;
        }
        let rms = (sum_sq / window_size as f32).sqrt();
        window_activities[i] = rms >= threshold;
    }

    let mut segments = Vec::new();
    let mut in_speech = false;
    let mut speech_start_window = 0;
    let mut consecutive_silence_windows = 0;
    let mut consecutive_speech_windows = 0;

    for i in 0..num_windows {
        let is_active = window_activities[i];
        if !in_speech {
            if is_active {
                consecutive_speech_windows += 1;
                if consecutive_speech_windows >= min_speech_windows {
                    in_speech = true;
                    speech_start_window = i + 1 - consecutive_speech_windows;
                    consecutive_silence_windows = 0;
                }
            } else {
                consecutive_speech_windows = 0;
            }
        } else {
            if !is_active {
                consecutive_silence_windows += 1;
                if consecutive_silence_windows >= min_silence_windows {
                    in_speech = false;
                    let speech_end_window = i + 1 - consecutive_silence_windows;
                    segments.push((speech_start_window * window_size, speech_end_window * window_size));
                    consecutive_speech_windows = 0;
                }
            } else {
                consecutive_silence_windows = 0;
            }
        }
    }

    if in_speech {
        segments.push((speech_start_window * window_size, samples.len()));
    }

    segments
}

pub fn transcribe(
    sink: StreamSink<TranscriptionEvent>,
    model_path: String,
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
        if let Err(e) = run_transcription(
            &sink,
            model_path,
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

fn run_transcription(
    sink: &StreamSink<TranscriptionEvent>,
    model_path: String,
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
    no_context: bool,
) -> Result<(), String> {
    println!(
        "[Rust] run_transcription: model_path={}, audio_path={}, language={:?}, translate={}, use_gpu={}, vad={}",
        model_path, audio_path, language, translate, use_gpu, vad_enabled
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

    // 2. 加载模型上下文
    println!("[Rust] Loading Whisper model context (use_gpu={})...", use_gpu);
    let mut ctx_params = WhisperContextParameters::default();
    
    let mut selected_device_name = "CPU".to_string();
    if use_gpu {
        #[cfg(feature = "vulkan")]
        {
            let devices = whisper_rs::vulkan::list_devices();
            if !devices.is_empty() {
                // Find dGPU first
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
    
    let ctx = WhisperContext::new_with_params(&model_path, ctx_params)
        .map_err(|e| {
            let err_msg = format!("加载模型失败: {}", e);
            println!("[Rust] {}", err_msg);
            err_msg
        })?;
    
    // 3. 创建推理状态
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
    params.set_no_context(no_context);
    println!(
        "[Rust] Whisper Params: temp={}, temp_inc={}, entropy_thold={}, logprob_thold={}, no_speech_thold={}, no_context={}",
        temperature, temperature_inc, entropy_thold, logprob_thold, no_speech_thold, no_context
    );

    // 5. 运行转写与 VAD 切片逻辑
    let mut combined_segments = Vec::new();

    if vad_enabled {
        let speech_segments = energy_based_vad(
            &samples,
            16000,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
        );
        println!(
            "[Rust] VAD enabled. Found {} active speech segments.",
            speech_segments.len()
        );

        if speech_segments.is_empty() {
            println!("[Rust] No speech segments detected. Returning empty subtitles.");
            let _ = sink.add(TranscriptionEvent::Success(combined_segments));
            return Ok(());
        }

        let mut progress_ctx = ProgressContext {
            sink: sink.clone(),
            current_segment: 0,
            total_segments: speech_segments.len(),
        };

        // 设置进度回调
        unsafe {
            params.set_progress_callback(Some(progress_callback_trampoline));
            params.set_progress_callback_user_data(&progress_ctx as *const ProgressContext as *mut std::ffi::c_void);
        }

        for (idx, &(start_sample, end_sample)) in speech_segments.iter().enumerate() {
            progress_ctx.current_segment = idx;
            let segment_samples = &samples[start_sample..end_sample];
            
            println!(
                "[Rust] Transcribing VAD segment {}/{}: start={:.2}s, end={:.2}s ({} samples)",
                idx + 1,
                speech_segments.len(),
                start_sample as f64 / 16000.0,
                end_sample as f64 / 16000.0,
                segment_samples.len()
            );

            state.full(params.clone(), segment_samples).map_err(|e| {
                let err_msg = format!("VAD 分段转写推理失败 (序号 {}): {}", idx + 1, e);
                println!("[Rust] {}", err_msg);
                err_msg
            })?;

            let num_segments = state.full_n_segments();
            let offset_cs = (start_sample as i64) / 160; // 16000Hz 下每 10ms (1厘秒) 包含 160 个采样点

            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    let text = segment
                        .to_str_lossy()
                        .unwrap_or_else(|_| std::borrow::Cow::Borrowed(""))
                        .into_owned();
                    
                    // 过滤掉纯空白的无效段
                    if text.trim().is_empty() {
                        continue;
                    }

                    let start = segment.start_timestamp();
                    let end = segment.end_timestamp();

                    combined_segments.push(TranscriptionSegment {
                        start_ms: (start + offset_cs) * 10,
                        end_ms: (end + offset_cs) * 10,
                        text,
                    });
                }
            }
        }
    } else {
        // VAD 未启用：对完整音频进行单次推理
        let progress_ctx = ProgressContext {
            sink: sink.clone(),
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

    // 6. 发送转写成功事件
    println!("[Rust] Sending Success event with {} segments", combined_segments.len());
    let _ = sink.add(TranscriptionEvent::Success(combined_segments));
    
    Ok(())
}
