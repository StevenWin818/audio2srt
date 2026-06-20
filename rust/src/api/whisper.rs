use crate::frb_generated::StreamSink;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSysContext,
    WhisperSysState,
};
use std::sync::Arc;
use std::sync::Mutex;

struct CachedContext {
    model_path: String,
    use_gpu: bool,
    context: Arc<WhisperContext>,
}

static G_CONTEXT: Mutex<Option<CachedContext>> = Mutex::new(None);

fn get_or_create_context(
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

/// 将过长的 VAD 语音分段以自然停顿（RMS 能量最低点）为边界，切割成最长 max_speech_ms 的子分段，防止熔断级联丢弃
fn split_long_segments(
    segments: Vec<(usize, usize)>,
    samples: &[f32],
    sample_rate: usize,
    max_speech_ms: i32,
    search_window_ms: i32,
) -> Vec<(usize, usize)> {
    let max_samples = (sample_rate * max_speech_ms as usize) / 1000;
    let search_samples = (sample_rate * search_window_ms as usize) / 1000;
    let window_ms = 30;
    let window_size = (sample_rate * window_ms) / 1000;

    let mut result = Vec::new();

    for (start, end) in segments {
        let len = end - start;
        if len <= max_samples {
            result.push((start, end));
            continue;
        }

        // 需要分割长片段
        let mut curr_start = start;
        while curr_start < end {
            let remaining = end - curr_start;
            if remaining <= max_samples {
                result.push((curr_start, end));
                break;
            }

            // 寻找分割点：在 [curr_start + max_samples - search_samples, curr_start + max_samples] 范围内
            let search_start = curr_start + max_samples - search_samples;
            let search_end = curr_start + max_samples;
            
            // 确保不超出边界
            let search_start = search_start.clamp(curr_start, end);
            let search_end = search_end.clamp(curr_start, end);

            if search_start >= search_end {
                result.push((curr_start, end));
                break;
            }

            // 在该范围内以 30ms 窗口寻找音能（RMS）最低的点作为分割点
            let mut min_rms = f32::MAX;
            let mut best_split_sample = curr_start + max_samples - search_samples / 2; // 默认中点

            let mut check_start = search_start;
            while check_start + window_size <= search_end {
                let window_slice = &samples[check_start..check_start + window_size];
                let mut sum_sq = 0.0;
                for &s in window_slice {
                    sum_sq += s * s;
                }
                let rms = (sum_sq / window_size as f32).sqrt();
                if rms < min_rms {
                    min_rms = rms;
                    best_split_sample = check_start + window_size; // 分割在窗口结束处
                }
                check_start += window_size;
            }

            result.push((curr_start, best_split_sample));
            curr_start = best_split_sample;
        }
    }

    result
}

/// 清洗文本中的标点符号与空格以实现精确的比对
fn clean_punctuation_and_whitespace(text: &str) -> String {
    text.replace(|c: char| {
        c.is_ascii_punctuation()
            || c.is_whitespace()
            || c == '。'
            || c == '，'
            || c == '！'
            || c == '？'
            || c == '、'
            || c == '“'
            || c == '”'
            || c == '；'
            || c == '：'
    }, "")
}

/// 检测字符串是否包含高频/长句子的连续重复循环（幻觉）
fn has_repetition_loop(text: &str) -> bool {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n < 4 {
        return false;
    }

    // 1. 检测长短语连续重复：长度 L >= 4 的子串，连续重复出现 2 次及以上
    for len in 4..=(n / 2) {
        for i in 0..=(n - 2 * len) {
            let sub1 = &chars[i..i+len];
            let sub2 = &chars[i+len..i+2*len];
            if sub1 == sub2 {
                let first_char = sub1[0];
                if sub1.iter().all(|&c| c == first_char) {
                    continue;
                }
                return true;
            }
        }
    }

    // 2. 检测单字或双字短语的高频连续重复
    // 单字连续重复 5 次及以上
    let mut consecutive_single = 1;
    let mut last_char = ' ';
    for &c in &chars {
        if c.is_whitespace() {
            continue;
        }
        if c == last_char {
            consecutive_single += 1;
            if consecutive_single >= 5 {
                return true;
            }
        } else {
            consecutive_single = 1;
            last_char = c;
        }
    }

    // 双字词组连续重复 4 次及以上
    if n >= 8 {
        for i in 0..=(n - 8) {
            let w1 = &chars[i..i+2];
            let w2 = &chars[i+2..i+4];
            let w3 = &chars[i+4..i+6];
            let w4 = &chars[i+6..i+8];
            if w1 == w2 && w2 == w3 && w3 == w4 {
                return true;
            }
        }
    }

    false
}

/// 对文本中连续重复的部分进行去重，保留最多 1 次（长段）或 2 次（单字/双字）
fn deduplicate_repeats(text: &str) -> String {
    let mut chars: Vec<char> = text.chars().collect();
    let mut changed = true;

    while changed {
        changed = false;
        let n = chars.len();
        if n < 8 {
            break;
        }

        // 优先排重长度较大的子串，保证段落去重完整性
        'outer: for len in (4..=(n / 2)).rev() {
            for i in 0..=(n - 2 * len) {
                let sub1 = &chars[i..i+len];
                let sub2 = &chars[i+len..i+2*len];
                if sub1 == sub2 {
                    let first_char = sub1[0];
                    if sub1.iter().all(|&c| c == first_char) {
                        continue;
                    }
                    chars.drain(i+len..i+2*len);
                    changed = true;
                    break 'outer;
                }
            }
        }
    }

    // 对单字/双字高频连续重复进行去重（最多保留 2 个）
    // 单字
    let mut i = 0;
    while i < chars.len() {
        let mut count = 1;
        while i + count < chars.len() && chars[i + count] == chars[i] && !chars[i].is_whitespace() {
            count += 1;
        }
        if count >= 3 {
            chars.drain(i + 2 .. i + count);
        }
        i += 1;
    }

    // 双字
    let mut j = 0;
    while j + 4 <= chars.len() {
        let w1 = chars[j..j+2].to_vec();
        let w2 = chars[j+2..j+4].to_vec();
        if w1 == w2 {
            let mut count = 2;
            while j + 2 * (count + 1) <= chars.len() && chars[j + 2 * count .. j + 2 * (count + 1)] == w1 {
                count += 1;
            }
            if count >= 3 {
                chars.drain(j + 4 .. j + 2 * count);
            }
        }
        j += 1;
    }

    chars.into_iter().collect()
}

/// 对文本按标点/空格拆分为短语，并进行短语级连续重复去重
fn deduplicate_phrases(text: &str) -> String {
    let mut phrases = Vec::new();
    let mut current_phrase = String::new();
    let mut separators = Vec::new();

    for c in text.chars() {
        let is_sep = c.is_ascii_punctuation()
            || c.is_whitespace()
            || c == '。'
            || c == '，'
            || c == '！'
            || c == '？'
            || c == '、'
            || c == '“'
            || c == '”'
            || c == '；'
            || c == '：';
        
        if is_sep {
            if !current_phrase.trim().is_empty() {
                phrases.push(current_phrase.clone());
                current_phrase.clear();
                separators.push(c.to_string());
            } else if !separators.is_empty() {
                if let Some(last_sep) = separators.last_mut() {
                    last_sep.push(c);
                }
            }
        } else {
            current_phrase.push(c);
        }
    }
    if !current_phrase.trim().is_empty() {
        phrases.push(current_phrase);
        separators.push(String::new());
    }

    if phrases.is_empty() {
        return text.to_string();
    }

    let mut i = 0;
    while i < phrases.len() {
        let mut found_dup = false;
        let max_len = (phrases.len() - i) / 2;
        
        for len in (1..=max_len).rev() {
            let sub1 = &phrases[i..i+len];
            let sub2 = &phrases[i+len..i+2*len];
            
            let match_count = sub1.iter().zip(sub2.iter()).filter(|(s1, s2)| {
                let c1 = clean_punctuation_and_whitespace(s1);
                let c2 = clean_punctuation_and_whitespace(s2);
                c1 == c2 && !c1.is_empty()
            }).count();
            
            if match_count == len {
                phrases.drain(i+len..i+2*len);
                separators.drain(i+len..i+2*len);
                found_dup = true;
                break;
            }
        }
        
        if !found_dup {
            i += 1;
        }
    }

    let mut result = String::new();
    for (p, sep) in phrases.into_iter().zip(separators.into_iter()) {
        result.push_str(&p);
        result.push_str(&sep);
    }
    result
}

/// 后处理已转写出来的字幕段列表，进行跨段去重及最后的质量净化
fn post_process_segments(segments: Vec<TranscriptionSegment>) -> Vec<TranscriptionSegment> {
    let mut result = Vec::new();
    let mut last_cleaned_text = String::new();

    for mut seg in segments {
        // 1. 单句内短语与字符级去重
        let text_dedup_phrases = deduplicate_phrases(&seg.text);
        let cleaned_text = deduplicate_repeats(&text_dedup_phrases);
        
        if cleaned_text.trim().is_empty() {
            continue;
        }
        
        // 2. 检测单句内是否依然存在长字串自我重复（幻觉熔断兜底）
        let cleaned_current = clean_punctuation_and_whitespace(&cleaned_text);
        if has_repetition_loop(&cleaned_current) {
            println!("[Rust] Post-process: 丢弃包含自我循环重复的字幕分段: '{}'", cleaned_text);
            continue;
        }

        // 3. 跨句/跨段级完全重复或极高相似度过滤
        let cleaned_last = clean_punctuation_and_whitespace(&last_cleaned_text);
        if !cleaned_current.is_empty() && cleaned_current == cleaned_last {
            println!("[Rust] Post-process: 丢弃跨段重复字幕分段: '{}'", cleaned_text);
            continue;
        }

        seg.text = cleaned_text;
        last_cleaned_text = seg.text.clone();
        result.push(seg);
    }

    result
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
    
    let ctx = get_or_create_context(&model_path, use_gpu, ctx_params)?;
    
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
        let mut speech_segments = energy_based_vad(
            &samples,
            16000,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
        );
        speech_segments = split_long_segments(
            speech_segments,
            &samples,
            16000,
            15000, // 默认最大分段长度 15 秒
            5000,  // 寻找自然停顿（RMS 能量最低）的 5 秒滑动搜索窗口
        );
        println!(
            "[Rust] VAD enabled. Found {} active speech segments (after splitting long segments).",
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

        let mut last_healthy_text = String::new();
        let mut consecutive_repeats = 0;

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

            // 1. 正常推理尝试（默认允许携带上下文以连贯语义，除非在全局设置中已明确关闭）
            let mut current_params = params.clone();
            if !no_context {
                current_params.set_no_context(false);
            }

            state.full(current_params, segment_samples).map_err(|e| {
                let err_msg = format!("VAD 分段转写推理失败 (序号 {}): {}", idx + 1, e);
                println!("[Rust] {}", err_msg);
                err_msg
            })?;

            // 2. 提取当前正常推理产生的文本
            let mut current_text = String::new();
            let num_segments = state.full_n_segments();
            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    current_text.push_str(&segment.to_str_lossy().unwrap_or_default());
                }
            }

            if current_text.trim().is_empty() {
                println!("[Rust] ⚠️ 模型未输出任何文本！可能是音频太模糊触发了 no_speech_thold (无声阈值)，导致大模型将其误判为静音并跳过。");
            }

            // 清理文本中的标点符号和空格以便于精确比对
            let cleaned_current = clean_punctuation_and_whitespace(&current_text);
            let cleaned_last = clean_punctuation_and_whitespace(&last_healthy_text);

            // 3. 监控重复幻觉（同时检测跨段重复和单段内的自我循环）
            let is_segment_self_repeating = has_repetition_loop(&cleaned_current);
            
            // 增强版跨段重复检测：除了完全相等，如果包含较长的公共子串（>= 5个字符），也视为重复，防止模型附加无意义语气词逃避检测
            let is_cross_segment_repeating = !cleaned_current.is_empty() && !cleaned_last.is_empty() && (
                cleaned_current == cleaned_last || 
                (cleaned_last.chars().count() >= 5 && cleaned_current.contains(&cleaned_last)) ||
                (cleaned_current.chars().count() >= 5 && cleaned_last.contains(&cleaned_current))
            );

            let mut discard_segment = false;

            if is_segment_self_repeating || is_cross_segment_repeating {
                consecutive_repeats += 1;
                println!(
                    "[Rust] ⚠️ 警告: 检测到内容重复 (单段内自我循环: {}, 跨段重复: {} - 连续 {} 次): '{}'",
                    is_segment_self_repeating,
                    is_cross_segment_repeating,
                    consecutive_repeats,
                    current_text.trim()
                );
            } else {
                consecutive_repeats = 0;
                if !current_text.trim().is_empty() {
                    last_healthy_text = current_text.clone();
                }
            }

            // 4. 熔断与回退处理
            if is_segment_self_repeating || consecutive_repeats >= 2 {
                println!("[Rust] 🛑 触发熔断！检测到持续重复或单段内自我循环，正在重新创建推理状态以彻底清除 C++ 侧受污染的 KV 缓存历史...");

                state = ctx.create_state().map_err(|e| {
                    let err_msg = format!("熔断回退重建推理状态失败: {}", e);
                    println!("[Rust] {}", err_msg);
                    err_msg
                })?;

                let mut fallback_params = params.clone();
                // 斩断内部上下文历史
                fallback_params.set_no_context(true);
                // 不传入 initial_prompt，彻底切断幻觉源头
                fallback_params.set_initial_prompt("");

                // 微升温度增加采样随机性
                let fallback_temp = if temperature < 0.2 { 0.3 } else { temperature + 0.2 };
                fallback_params.set_temperature(fallback_temp);

                println!("[Rust] 正在执行回退推理: temp={:.2} (无 prompt)", fallback_temp);

                state.full(fallback_params, segment_samples).map_err(|e| {
                    let err_msg = format!("VAD 分段回退推理失败 (序号 {}): {}", idx + 1, e);
                    println!("[Rust] {}", err_msg);
                    err_msg
                })?;

                // 重新提取重试后的文本
                current_text.clear();
                let num_segments_retry = state.full_n_segments();
                for i in 0..num_segments_retry {
                    if let Some(segment) = state.get_segment(i) {
                        current_text.push_str(&segment.to_str_lossy().unwrap_or_default());
                    }
                }
                
                let cleaned_retry = clean_punctuation_and_whitespace(&current_text);
                let retry_self_repeating = has_repetition_loop(&cleaned_retry);
                
                if retry_self_repeating {
                    println!("[Rust] 🛑 回退重试后依然检测到重复幻觉，保留该分段交由后处理清洗，但会重建状态以防污染: '{}'", current_text.trim());
                    // discard_segment = true; // 移除丢弃逻辑，保留有效内容
                    consecutive_repeats = 0; // 重置重复计数
                    
                    // 重新创建推理状态以彻底清除 C++ 侧受污染 Hendrick/Whisper KV 缓存历史，防止污染后续分段
                    state = ctx.create_state().map_err(|e| {
                        let err_msg = format!("熔断重建推理状态失败: {}", e);
                        println!("[Rust] {}", err_msg);
                        err_msg
                    })?;
                } else {
                    println!("[Rust] ✅ 回退重试成功，新输出: '{}'", current_text.trim());
                    consecutive_repeats = 0;
                    if !current_text.trim().is_empty() {
                        last_healthy_text = current_text.clone();
                    }
                }
            }

            // 5. 最终持久化写入字幕段列表
            if !discard_segment {
                let offset_cs = (start_sample as i64) / 160;
                let final_num_segments = state.full_n_segments();

                for i in 0..final_num_segments {
                    if let Some(segment) = state.get_segment(i) {
                        let text = segment
                            .to_str_lossy()
                            .unwrap_or_else(|_| std::borrow::Cow::Borrowed(""))
                            .into_owned();

                        // 过滤掉纯空白的段
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
    let combined_segments = post_process_segments(combined_segments);
    println!("[Rust] Sending Success event with {} segments", combined_segments.len());
    let _ = sink.add(TranscriptionEvent::Success(combined_segments));
    
    Ok(())
}
