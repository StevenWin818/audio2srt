use crate::frb_generated::StreamSink;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperSysContext,
    WhisperSysState, DtwParameters, DtwMode, DtwModelPreset,
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

    // 计算全局能量用于自适应阈值
    let mut global_sum_sq = 0.0;
    // 每隔一定步长采样，加速计算
    let step = 10.max(samples.len() / 100000);
    let mut count = 0;
    for i in (0..samples.len()).step_by(step) {
        global_sum_sq += samples[i] * samples[i];
        count += 1;
    }
    let global_rms = if count > 0 { (global_sum_sq / count as f32).sqrt() } else { 0.0 };
    // 动态阈值：取用户设定阈值和 (全局能量的 0.6 倍 + 0.005) 的较小值
    // 这能有效防止对于整体音量偏小的音频使用固定 0.05 导致大面积误判
    let adaptive_threshold = threshold.min(global_rms * 0.6 + 0.005);
    println!("[Rust] VAD global RMS: {:.4}, original threshold: {:.4}, adaptive threshold: {:.4}", global_rms, threshold, adaptive_threshold);

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
        window_activities[i] = rms >= adaptive_threshold;
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

pub fn convert_chinese(text: String, to_simplified: bool) -> String {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    zhconv::zhconv(&text, target)
}

pub fn convert_chinese_list(texts: Vec<String>, to_simplified: bool) -> Vec<String> {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    texts.into_iter().map(|text| zhconv::zhconv(&text, target)).collect()
}

/// 清洗文本中的标点符号与空格以实现精确的比对
pub(crate) fn clean_punctuation_and_whitespace(text: &str) -> String {
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
pub(crate) fn has_repetition_loop(text: &str) -> bool {
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
pub(crate) fn deduplicate_repeats(text: &str) -> String {
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
        // 0. 统一转换为简体中文，消除 Whisper 输出中简繁混用的现象
        seg.text = zhconv::zhconv(&seg.text, zhconv::Variant::ZhCN);

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
        let mut callback = |event| {
            let _ = sink.add(event);
        };
        if let Err(e) = run_transcription_inner(
            &mut callback,
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

pub(crate) fn run_transcription_inner(
    callback: &mut dyn FnMut(TranscriptionEvent),
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

    // 确保音频总长度不少于 0.5 秒以防崩溃
    if samples.len() < 8000 {
        println!("[Rust] ⚠️ 整体音频过短 ({} samples), 已自动静音填充至 0.5s 以保护 DTW。", samples.len());
        samples.resize(8000, 0.0);
    }

    // 2. 加载模型上下文
    println!("[Rust] Loading Whisper model context (use_gpu={})...", use_gpu);
    let mut ctx_params = WhisperContextParameters::default();
    
    // Enable DTW mode using the model preset if available
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
        // 准备状态追踪变量
        let mut recent_history: std::collections::VecDeque<String> = std::collections::VecDeque::new();
        
        // 新增：用于纯文本语义传递的滑动提示词
        let mut rolling_prompt = String::new();

        #[allow(unused_assignments)]
        for (idx, &(start_sample, end_sample)) in speech_segments.iter().enumerate() {
            progress_ctx.current_segment = idx;
            
            // 提取为可变的 Vector 并强行对超短音频使用静音填充以保护 DTW 机制
            let mut segment_samples = samples[start_sample..end_sample].to_vec();
            const MIN_SAMPLES_FOR_DTW: usize = 8000; // 16kHz 下 0.5 秒 = 8000 个采样点
            if segment_samples.len() < MIN_SAMPLES_FOR_DTW {
                println!("[Rust] ⚠️ 拦截到超短音频 ({} samples), 已自动静音填充至 0.5s 以保护 DTW。", segment_samples.len());
                segment_samples.resize(MIN_SAMPLES_FOR_DTW, 0.0);
            }
            
            let global_offset_ms = (start_sample as i64) / 16;
            
            println!(
                "[Rust] Transcribing VAD segment {}/{}: start={:.2}s, end={:.2}s ({} samples)",
                idx + 1,
                speech_segments.len(),
                start_sample as f64 / 16000.0,
                end_sample as f64 / 16000.0,
                segment_samples.len()
            );

            // 1. 正常推理尝试：永远斩断底层声学上下文，通过 prompt 传递纯文本语义
            let mut current_params = params.clone();
            
            // 强制清空上一段的音频 KV 缓存，防止死循环幻觉
            current_params.set_no_context(true);
            
            // 如果全局设置允许携带上下文，且已有历史文本，则将其作为提示词注入
            if !no_context && !rolling_prompt.is_empty() {
                current_params.set_initial_prompt(&rolling_prompt);
            }

            state.full(current_params, &segment_samples).map_err(|e| {
                let err_msg = format!("VAD 分段转写推理失败 (序号 {}): {}", idx + 1, e);
                println!("[Rust] {}", err_msg);
                err_msg
            })?;

            // 2. 提取当前正常推理产生的文本
            let num_segments = state.full_n_segments();
            let mut current_text = String::new();
            
            // 用于检测单 VAD 块内部的交替幻觉
            let mut exact_match_count = 0;
            let mut partial_match_count = 0;
            let mut is_cross_segment_repeating = false;
            let mut is_segment_self_repeating = false;

            for i in 0..num_segments {
                if let Some(segment) = state.get_segment(i) {
                    let seg_text = segment.to_str_lossy().unwrap_or_default().into_owned();
                    current_text.push_str(&seg_text);
                    
                    let cleaned_seg = clean_punctuation_and_whitespace(&seg_text);
                    if cleaned_seg.trim().is_empty() {
                        continue;
                    }

                    // 检测单句自我循环
                    if has_repetition_loop(&cleaned_seg) {
                        is_segment_self_repeating = true;
                    }

                    // 在历史中查找匹配
                    let char_count = cleaned_seg.chars().count();
                    for past_text in &recent_history {
                        if *past_text == cleaned_seg {
                            exact_match_count += 1;
                        } else if (char_count >= 5 && past_text.contains(&cleaned_seg)) ||
                            (past_text.chars().count() >= 5 && cleaned_seg.contains(past_text)) {
                            partial_match_count += 1;
                        }
                    }

                    // 加入历史
                    recent_history.push_back(cleaned_seg.clone());
                    if recent_history.len() > 25 {
                        recent_history.pop_front();
                    }
                    
                    if exact_match_count >= 3 {
                        if char_count >= 4 {
                            is_cross_segment_repeating = true;
                        } else if exact_match_count >= 5 {
                            is_cross_segment_repeating = true;
                        }
                    }
                    if partial_match_count >= 3 {
                        is_cross_segment_repeating = true;
                    }
                }
            }

            if current_text.trim().is_empty() {
                println!("[Rust] ⚠️ 模型未输出任何文本！可能是音频太模糊触发了 no_speech_thold (无声阈值)，导致大模型将其误判为静音并跳过。");
            }

            let discard_segment = false;

            if is_segment_self_repeating || is_cross_segment_repeating {
                println!(
                    "[Rust] ⚠️ 警告: 检测到内容重复 (单段自我循环: {}, 精确匹配: {}, 部分匹配: {})",
                    is_segment_self_repeating,
                    exact_match_count,
                    partial_match_count
                );
            }

            // 4. 熔断与回退处理
            if is_segment_self_repeating || is_cross_segment_repeating {
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
                let fallback_temp = if temperature < 0.2_f32 { 0.3_f32 } else { temperature + 0.2_f32 };
                fallback_params.set_temperature(fallback_temp);

                println!("[Rust] 正在执行回退推理: temp={:.2} (无 prompt)", fallback_temp);

                state.full(fallback_params, &segment_samples).map_err(|e| {
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
                    
                    // 重新创建推理状态以彻底清除 C++ 侧受污染 Hendrick/Whisper KV 缓存历史，防止污染后续分段
                    state = ctx.create_state().map_err(|e| {
                        let err_msg = format!("熔断重建推理状态失败: {}", e);
                        println!("[Rust] {}", err_msg);
                        err_msg
                    })?;
                } else {
                    println!("[Rust] ✅ 回退重试成功，新输出: '{}'", current_text.trim());
                    if !cleaned_retry.is_empty() {
                        recent_history.push_back(cleaned_retry.clone());
                        if recent_history.len() > 10 {
                            recent_history.pop_front();
                        }
                    }
                }
            }

            // 5. 最终持久化写入字幕段列表
            if !discard_segment {
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

                        // 新增：更新滑动提示词
                        if !no_context {
                            rolling_prompt.push_str(&text);
                            // 截断滑动窗口，保留最后 100 个字符
                            let char_vec: Vec<char> = rolling_prompt.chars().collect();
                            if char_vec.len() > 100 {
                                rolling_prompt = char_vec[char_vec.len() - 100 ..].iter().collect();
                            }
                        }

                        combined_segments.push(TranscriptionSegment {
                            start_ms: segment.start_timestamp() * 10 + global_offset_ms,
                            end_ms: segment.end_timestamp() * 10 + global_offset_ms,
                            text,
                        });
                    }
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

    // 6. 发送转写成功事件
    let combined_segments = post_process_segments(combined_segments);
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
