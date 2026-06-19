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

// 原始 C++ 风格的进度回调函数（避免闭包的堆内存泄漏以及二次转写端口死锁/卡死问题）
unsafe extern "C" fn progress_callback_trampoline(
    _ctx: *mut WhisperSysContext,
    _state: *mut WhisperSysState,
    progress: std::ffi::c_int,
    user_data: *mut std::ffi::c_void,
) {
    if !user_data.is_null() {
        let sink = &*(user_data as *const StreamSink<TranscriptionEvent>);
        println!("[Rust] Progress callback: {}%", progress);
        let _ = sink.add(TranscriptionEvent::Progress(progress));
    }
}

pub fn transcribe(
    sink: StreamSink<TranscriptionEvent>,
    model_path: String,
    audio_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
) {
    // 异步执行转写任务，防止界面卡顿
    std::thread::spawn(move || {
        if let Err(e) = run_transcription(&sink, model_path, audio_path, language, translate, threads, use_gpu) {
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
) -> Result<(), String> {
    println!("[Rust] run_transcription: model_path={}, audio_path={}, language={:?}, translate={}, use_gpu={}", model_path, audio_path, language, translate, use_gpu);

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
    ctx_params.use_gpu = use_gpu;
    
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
    
    // 设置线程数
    let n_threads = threads.unwrap_or(4);
    params.set_n_threads(n_threads);
    println!("[Rust] Threads set to: {}", n_threads);
    
    // 设置语言 (如果为 None 或 "auto"，显式清理以启用 Whisper 自动检测并转写，若设为 detect_language(true) 会使 whisper.cpp 检测完语言后立即退出而不进行转写)
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

    // 设置进度回调 (使用原始的 C 回调，借用栈上 StreamSink 指针，避免堆上闭包泄漏导致端口泄漏)
    unsafe {
        params.set_progress_callback(Some(progress_callback_trampoline));
        params.set_progress_callback_user_data(sink as *const StreamSink<TranscriptionEvent> as *mut std::ffi::c_void);
    }

    // 5. 开始推理
    println!("[Rust] Starting Whisper model inference...");
    state.full(params, &samples).map_err(|e| {
        let err_msg = format!("转写推理失败: {}", e);
        println!("[Rust] {}", err_msg);
        err_msg
    })?;
    println!("[Rust] Whisper model inference completed successfully!");

    // 6. 获取结果
    let num_segments = state.full_n_segments();
    println!("[Rust] Number of transcribed segments: {}", num_segments);
    let mut segments = Vec::new();

    for i in 0..num_segments {
        if let Some(segment) = state.get_segment(i) {
            let text = segment.to_str_lossy().unwrap_or_else(|_| std::borrow::Cow::Borrowed("")).into_owned();
            let start = segment.start_timestamp();
            let end = segment.end_timestamp();
            
            println!("[Rust] Segment {}: [{}-{} ms] {}", i + 1, start * 10, end * 10, text);
            
            segments.push(TranscriptionSegment {
                // t0 和 t1 的单位是厘秒 (10ms)，所以乘以 10 得到毫秒 (ms)
                start_ms: start * 10,
                end_ms: end * 10,
                text,
            });
        }
    }

    // 发送转写成功事件
    println!("[Rust] Sending Success event with {} segments", segments.len());
    let _ = sink.add(TranscriptionEvent::Success(segments));
    
    Ok(())
}
