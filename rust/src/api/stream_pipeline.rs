use std::process::{Command, Stdio};
use std::io::{Read, BufRead, BufReader};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::path::Path;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use anyhow::{Result, Context, anyhow};
use flate2::read::GzDecoder;
use tar::Archive;
use deepfilter_rt::DeepFilterStream;
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use whisper_rs::{WhisperVadContext, WhisperVadContextParams, WhisperVadParams, WhisperVadSegment};
use crate::frb_generated::StreamSink;
use crate::api::silero_vad::{
    TranscriptionSegment, TranscriptionEvent, WordItem,
    convert_chinese, register_thread_as_pro_audio,
};

static PERF_LOG_ENABLED: AtomicBool = AtomicBool::new(false);

#[flutter_rust_bridge::frb(sync)]
pub fn set_rust_perf_logging(enable: bool) {
    PERF_LOG_ENABLED.store(enable, Ordering::Relaxed);
}

static SHOULD_CANCEL: AtomicBool = AtomicBool::new(false);
/// 转写串行化锁: 一次只允许一个转写线程运行 (见 transcribe_stream)
static TRANS_LOCK: parking_lot::Mutex<()> = parking_lot::Mutex::new(());
static ACTIVE_FFMPEG_CHILD: Mutex<Option<std::process::Child>> = Mutex::new(None);

/// 当前缓存的 Qwen 运行时状态 (供 UI 显示模型实际加载在哪)。
#[derive(Debug, Clone)]
pub struct QwenRuntimeStatus {
    /// 模型目录
    pub model_dir: String,
    /// decoder GGUF 文件名 (量化信息)
    pub decoder_file: String,
    /// encoder 实际执行提供程序 ("CUDA" / "DirectML" / "CPU")
    pub encoder_ep: String,
    /// decoder 实际后端 ("CUDA" / "Vulkan" / "CPU")
    pub decoder_backend: String,
    /// decoder offload 状态 ("GPU (N/N layers)" / "CPU")
    pub decoder_offload: String,
}

/// 查询当前缓存的 Qwen 运行时状态 (模型/encoder/decoder 实际加载位置)。
pub fn get_qwen_runtime_status() -> Option<QwenRuntimeStatus> {
    let cache = crate::qwen::context::GLOBAL_QWEN_CACHE.lock();
    let (model_dir, decoder_file, encoder_ep, decoder_backend, decoder_offload) =
        cache.runtime_status()?;
    Some(QwenRuntimeStatus {
        model_dir,
        decoder_file,
        encoder_ep,
        decoder_backend,
        decoder_offload,
    })
}

#[flutter_rust_bridge::frb(ignore)]
pub fn get_or_create_qwen_runtime(
    qwen_dir: &str,
    aligner_dir: Option<&str>,
    encoder_backend: crate::qwen::backend::EncoderBackend,
    decoder_backend: crate::qwen::backend::DecoderBackend,
    decoder_file: Option<&str>,
) -> Result<Arc<crate::qwen::runtime::QwenRuntime>> {
    let enc_backend_resolved = match encoder_backend {
        crate::qwen::backend::EncoderBackend::Auto => {
            if cfg!(any(feature = "cuda", feature = "qwen-cuda")) {
                crate::qwen::backend::EncoderBackend::Cuda
            } else if cfg!(all(target_os = "windows", any(feature = "vulkan", feature = "qwen-dml", feature = "qwen-dml-win"))) {
                crate::qwen::backend::EncoderBackend::DirectMl
            } else {
                crate::qwen::backend::EncoderBackend::Cpu
            }
        }
        b => b,
    };
    let dec_backend_resolved = match decoder_backend {
        crate::qwen::backend::DecoderBackend::Auto => {
            if cfg!(any(feature = "cuda", feature = "qwen-cuda")) {
                crate::qwen::backend::DecoderBackend::Cuda
            } else if cfg!(any(feature = "vulkan", feature = "qwen-vulkan")) {
                crate::qwen::backend::DecoderBackend::Vulkan
            } else {
                crate::qwen::backend::DecoderBackend::Cpu
            }
        }
        b => b,
    };

    let norm_asr = qwen_dir.replace('\\', "/").trim_end_matches('/').to_lowercase();
    let norm_aligner = aligner_dir.map(|s| s.replace('\\', "/").trim_end_matches('/').to_lowercase());
    let norm_decoder_file = decoder_file.map(|s| s.replace('\\', "/").to_lowercase());

    let key = crate::qwen::context::RuntimeCacheKey {
        asr_model_id: norm_asr,
        aligner_model_id: norm_aligner,
        encoder_backend: enc_backend_resolved,
        decoder_backend: dec_backend_resolved,
        decoder_file: norm_decoder_file,
    };

    let mut cache = crate::qwen::context::GLOBAL_QWEN_CACHE.lock();
    if let Some(runtime) = cache.get(&key) {
        println!("[Rust Cache HIT] 成功秒级复用显存中的预加载 QwenRuntime (Model: {})", qwen_dir);
        runtime.reset_cancel();
        return Ok(runtime);
    }

    // 目录或 Backend 变更：先清理旧的 Cache，触发原 QwenDecoder 的 Drop，彻底释放 GPU 显存！
    println!("[Rust Cache MISS] 未命中预加载缓存! 请求键: {:?}. 正在释放旧模型并载入新模型...", key);
    cache.clear();

    println!("[Rust] Loading new QwenRuntime into GPU VRAM from model dir: {}", qwen_dir);
    let runtime = Arc::new(
        crate::qwen::runtime::QwenRuntime::load(
            qwen_dir,
            aligner_dir,
            encoder_backend,
            decoder_backend,
            decoder_file,
        )
        .map_err(|e| anyhow!("Failed to load Qwen runtime: {:?}", e))?,
    );

    cache.set(key, runtime.clone());
    Ok(runtime)
}

#[flutter_rust_bridge::frb(sync)]
pub fn unload_qwen_runtime() {
    let mut cache = crate::qwen::context::GLOBAL_QWEN_CACHE.lock();
    println!("[Rust] unload_qwen_runtime: Dropping cached QwenRuntime and freeing GPU VRAM...");
    cache.clear();
}

#[flutter_rust_bridge::frb(sync)]
pub fn preload_qwen_model(
    asr_model_dir: String,
    aligner_model_dir: Option<String>,
    decoder_file: Option<String>,
) -> Result<(), String> {
    println!("[Rust] Spawning background thread for preloading Qwen model: {} (decoder={:?})", asr_model_dir, decoder_file);
    std::thread::spawn(move || {
        let app_data = std::env::var("APPDATA").unwrap_or_default();
        let default_qwen = format!("{}/com.audio2srt/audio2srt/models/qwen3-asr-0.6b", app_data.replace('\\', "/"));

        let qwen_dir = if is_qwen_model_dir(&asr_model_dir) {
            asr_model_dir
        } else if Path::new(&default_qwen).exists() {
            default_qwen
        } else {
            println!("[Rust Preload] Model dir not valid: {}", asr_model_dir);
            return;
        };

        match get_or_create_qwen_runtime(
            &qwen_dir,
            aligner_model_dir.as_deref(),
            crate::qwen::backend::EncoderBackend::Auto,
            crate::qwen::backend::DecoderBackend::Auto,
            decoder_file.as_deref(),
        ) {
            Ok(_) => println!("[Rust Preload] Successfully preloaded model into VRAM: {}", qwen_dir),
            Err(e) => println!("[Rust Preload] Preload model error: {:?}", e),
        }
    });
    Ok(())
}

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_transcription_backend() {
    SHOULD_CANCEL.store(true, Ordering::SeqCst);
    // 同时标记当前运行时取消: encoder/decoder 的 cancel 检查立即生效，
    // 加速编码/解码线程退出，避免与下一次转写并发干扰
    {
        let cache = crate::qwen::context::GLOBAL_QWEN_CACHE.lock();
        cache.cancel_runtime();
    }
    if let Ok(mut lock) = ACTIVE_FFMPEG_CHILD.lock() {
        if let Some(mut child) = lock.take() {
            println!("[Rust] cancel_transcription_backend: Killing active FFmpeg child process.");
            let _ = child.kill();
        }
    }
}

fn add_perf_record(stage: &str, elapsed_ms: u64, audio_duration_sec: f64) {
    if !PERF_LOG_ENABLED.load(Ordering::Relaxed) {
        return;
    }

    use std::time::{SystemTime, UNIX_EPOCH};
    use std::fs::OpenOptions;
    use std::io::Write;
    use std::sync::Mutex;

    static FILE_MUTEX: Mutex<()> = Mutex::new(());
    let _lock = FILE_MUTEX.lock().unwrap();

    let timestamp_ms = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64;
    let rtf = if audio_duration_sec > 0.0 {
        (elapsed_ms as f64 / 1000.0) / audio_duration_sec
    } else {
        0.0
    };
    println!(
        "[PERF] Stage: {}, Time: {}ms, Audio: {:.2}s, RTF: {:.4}",
        stage, elapsed_ms, audio_duration_sec, rtf
    );

    let filename = "rust_pipeline_perf.csv";
    let file_exists = std::path::Path::new(filename).exists();
    if let Ok(mut file) = OpenOptions::new()
        .create(true)
        .write(true)
        .append(true)
        .open(filename)
    {
        if !file_exists {
            let _ = writeln!(file, "timestamp_ms,stage,elapsed_ms,audio_duration_sec,rtf");
        }
        let _ = writeln!(
            file,
            "{},{},{},{:.4},{:.4}",
            timestamp_ms, stage, elapsed_ms, audio_duration_sec, rtf
        );
    }
}

macro_rules! log_perf {
    ($stage:expr, $elapsed_ms:expr, $audio_duration_sec:expr) => {
        add_perf_record($stage, $elapsed_ms, $audio_duration_sec);
    };
}

fn parse_duration_str(time_str: &str) -> Option<f64> {
    let parts: Vec<&str> = time_str.split(':').collect();
    if parts.len() == 3 {
        let h: f64 = parts[0].parse().unwrap_or(0.0);
        let m: f64 = parts[1].parse().unwrap_or(0.0);
        let s: f64 = parts[2].parse().unwrap_or(0.0);
        Some(h * 3600.0 + m * 60.0 + s)
    } else {
        None
    }
}

fn get_media_duration_secs(ffmpeg_path: &str, input_path: &str) -> Result<f64> {
    let mut cmd = Command::new(ffmpeg_path);
    cmd.arg("-i").arg(input_path);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }
    let output = cmd.output().context("Failed to run FFmpeg for duration detection")?;
    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let mut duration_secs = 0.0;
    for line in stderr_str.lines() {
        if line.contains("Duration:") {
            if let Some(duration_str) = line.split("Duration:").nth(1) {
                if let Some(time_str) = duration_str.split(',').next() {
                    if let Some(parts) = parse_duration_str(time_str.trim()) {
                        duration_secs = parts;
                    }
                }
            }
        }
    }
    if duration_secs > 0.0 {
        Ok(duration_secs)
    } else {
        Err(anyhow!("Could not detect media duration"))
    }
}

pub trait TranscriptionSink: Send + Sync {
    fn add(&self, event: TranscriptionEvent) -> Result<(), String>;
}

impl TranscriptionSink for StreamSink<TranscriptionEvent> {
    fn add(&self, event: TranscriptionEvent) -> Result<(), String> {
        StreamSink::<TranscriptionEvent, flutter_rust_bridge::SseCodec>::add(self, event)
            .map_err(|e| e.to_string())
    }
}

#[derive(Clone, Debug, Copy, PartialEq, Eq)]
pub enum TimestampMode {
    Fast,
    Precise,
}

// ✅ 配置结构体以消除“过长参数列表”
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub ffmpeg_path: String,
    pub input_path: String,
    pub model_path: String,
    pub vad_model_path: String,
    pub df_model_path: String,
    pub asr_model_dir: String,
    pub aligner_model_dir: Option<String>,
    /// 解码器 GGUF 文件名 (量化选择)，如 "decoder.q4_k_m.gguf"；None 时按优先级自动扫描
    pub decoder_file: Option<String>,
    pub context_prompt: Option<String>,
    pub encoder_backend: crate::qwen::backend::EncoderBackend,
    pub decoder_backend: crate::qwen::backend::DecoderBackend,
    pub timestamp_mode: TimestampMode,
    // Whisper 设置
    pub language: Option<String>,
    pub translate: bool,
    pub threads: Option<i32>,
    pub use_gpu: bool,
    pub to_simplified: bool,
    pub no_context: bool,
    pub no_state_history: bool,
    // VAD & DFN 设置
    pub enable_denoise: bool,
    pub vad_enabled: bool,
    pub vad_threshold: f64,
    pub vad_min_speech_ms: i32,
    pub vad_min_silence_ms: i32,
    // 音轨选择（从0开始的相对音轨索引）
    pub selected_audio_track: Option<usize>,
}

pub fn transcribe_stream(
    sink: StreamSink<TranscriptionEvent>,
    config: PipelineConfig,
) {
    let sink_arc: Arc<dyn TranscriptionSink> = Arc::new(sink);
    let sink_clone = sink_arc.clone();
    thread::spawn(move || {
        // 串行化: 等上一次转写线程完全退出后再启动。
        // 取消后旧线程可能仍在向已关闭的 StreamSink 发消息
        // (frb 报 "Fail to post message to Dart")，若此时并发启动第二次
        // 转写，frb 事件流可能异常导致 Dart 侧收不到任何事件。
        // 带超时兜底，避免旧线程异常卡死时第二次永远无法启动。
        use std::time::{Duration, Instant};
        let deadline = Instant::now() + Duration::from_secs(30);
        let _guard = loop {
            match TRANS_LOCK.try_lock() {
                Some(g) => break g,
                None if Instant::now() > deadline => {
                    println!("[Rust] previous transcription thread still active after 30s, proceeding anyway");
                    break TRANS_LOCK.lock();
                }
                None => std::thread::sleep(Duration::from_millis(50)),
            }
        };
        if let Err(e) = run_stream_pipeline_inner(
            sink_clone.clone(),
            config,
        ) {
            let _ = sink_clone.add(TranscriptionEvent::Failure(e.to_string()));
        }
    });
}

fn run_stream_pipeline_inner(
    sink: Arc<dyn TranscriptionSink>,
    config: PipelineConfig,
) -> Result<()> {
    SHOULD_CANCEL.store(false, Ordering::SeqCst);
    ASR_TASK_SEQ.store(0, Ordering::SeqCst);

    #[cfg(target_os = "windows")]
    {
        lock_high_priority();
        disable_power_throttling();
    }

    // 1. 获取视频总时长
    let total_duration = match get_media_duration_secs(&config.ffmpeg_path, &config.input_path) {
        Ok(d) => d,
        Err(e) => {
            println!("[Rust] Warning: Failed to get duration: {}. Defaulting to 1.0", e);
            1.0
        }
    };
    println!("[Rust] Media total duration: {} seconds", total_duration);

    // 2. 初始化 Qwen ASR 管道
    let app_data = std::env::var("APPDATA").unwrap_or_default();
    let default_qwen = format!("{}/com.audio2srt/audio2srt/models/qwen3-asr-0.6b", app_data.replace('\\', "/"));

    let qwen_dir = if is_qwen_model_dir(&config.asr_model_dir) {
        config.asr_model_dir.clone()
    } else if is_qwen_model_dir(&config.model_path) {
        config.model_path.clone()
    } else if Path::new(&default_qwen).exists() {
        default_qwen
    } else {
        return Err(anyhow!("未找到有效的 Qwen ASR 模型目录: {}", config.asr_model_dir));
    };

    let runtime_arc = get_or_create_qwen_runtime(
        &qwen_dir,
        config.aligner_model_dir.as_deref(),
        config.encoder_backend,
        config.decoder_backend,
        config.decoder_file.as_deref(),
    )?;
    runtime_arc.reset_cancel();

    let (tx_raw_48k, rx_raw_48k) = sync_channel::<Vec<f32>>(16);
    let (tx_clean_48k, rx_clean_48k) = sync_channel::<Vec<f32>>(16);
    let (tx_asr_task, rx_asr_task) = sync_channel::<AsrTask>(16);

    let ffmpeg_handle = spawn_ffmpeg_pump(&config, tx_raw_48k)?;
    let dfn_handle = spawn_dfn_worker(&config, rx_raw_48k, tx_clean_48k);
    let vad_handle = spawn_vad_worker(&config, rx_clean_48k, tx_asr_task);
    let qwen_handle = spawn_qwen_worker(runtime_arc, &config, total_duration, rx_asr_task, sink);

    ffmpeg_handle.join()?;
    dfn_handle.join().map_err(|_| anyhow!("DFN3 thread panicked"))??;
    vad_handle.join().map_err(|_| anyhow!("VAD thread panicked"))??;
    qwen_handle.join().map_err(|_| anyhow!("Failed to join Qwen GPU thread"))??;

    Ok(())
}

// ==== 工作线程结构体与辅助函数 ====

struct AsrTask {
    seq: u64,
    samples: Vec<f32>,
    start_ms: i64,
    end_ms: i64,
    }

/// VAD 切段全局序号 (编码线程池并行处理后按此恢复顺序)
static ASR_TASK_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct FfmpegPumpHandle {
    pump_thread: thread::JoinHandle<Result<()>>,
    stderr_thread: thread::JoinHandle<String>,
}

impl FfmpegPumpHandle {
    fn join(self) -> Result<()> {
        let _ = self.pump_thread.join();
        let mut child_opt = None;
        if let Ok(mut lock) = ACTIVE_FFMPEG_CHILD.lock() {
            child_opt = lock.take();
        }
        let stderr_logs = self.stderr_thread.join().unwrap();
        if let Some(mut child) = child_opt {
            let ffmpeg_status = child.wait().unwrap();
            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                return Ok(());
            }
            if !ffmpeg_status.success() {
                return Err(anyhow!("FFmpeg exited with error: {:?}\nStderr logs:\n{}", ffmpeg_status.code(), stderr_logs));
            }
        }
        Ok(())
    }
}

fn spawn_ffmpeg_pump(
    config: &PipelineConfig,
    tx_raw_48k: std::sync::mpsc::SyncSender<Vec<f32>>,
) -> Result<FfmpegPumpHandle> {
    println!("[Rust] Starting FFmpeg audio pump process...");
    
    // 动态探测音频流数量
    let mut probe_cmd = Command::new(&config.ffmpeg_path);
    probe_cmd.arg("-i").arg(&config.input_path);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        probe_cmd.creation_flags(0x08000000);
    }

    let audio_stream_count = if let Ok(output) = probe_cmd.output() {
        let stderr_str = String::from_utf8_lossy(&output.stderr);
        let mut count = 0;
        for line in stderr_str.lines() {
            if line.contains("Stream #") && line.contains("Audio:") {
                count += 1;
            }
        }
        count
    } else {
        1
    };
    println!("[Rust] transcribe_stream 检测到音频流数量: {}，文件: {}", audio_stream_count, config.input_path);

    let mut cmd = Command::new(&config.ffmpeg_path);
    cmd.arg("-y")
       .arg("-i")
       .arg(&config.input_path)
       .arg("-vn")
       .arg("-sn")
       .arg("-dn");

    if let Some(track_idx) = config.selected_audio_track {
        cmd.arg("-map")
           .arg(format!("0:a:{}", track_idx));
    } else if audio_stream_count > 1 {
        println!("[Rust] 未指定音轨，检测到多音轨，默认提取第一条音轨 (0:a:0)");
        cmd.arg("-map")
           .arg("0:a:0"); 
    } else {
        cmd.arg("-map")
           .arg("0:a?");
    }

    cmd.arg("-f")
       .arg("f32le")
       .arg("-ar")
       .arg("48000")
       .arg("-ac")
       .arg("1")
       .arg("-");

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x08000000);
    }

    cmd.stdout(Stdio::piped())
       .stderr(Stdio::piped());

    let child = cmd.spawn().map_err(|e| anyhow!("Failed to start FFmpeg: {}", e))?;
    if let Ok(mut lock) = ACTIVE_FFMPEG_CHILD.lock() {
        *lock = Some(child);
    }

    let mut stdout = {
        // We need to access child's stdout. We lock to get it.
        let mut stdout_taken = None;
        if let Ok(mut lock) = ACTIVE_FFMPEG_CHILD.lock() {
            if let Some(ref mut c) = *lock {
                stdout_taken = c.stdout.take();
            }
        }
        stdout_taken.context("Failed to take FFmpeg stdout")?
    };

    let stderr = {
        let mut stderr_taken = None;
        if let Ok(mut lock) = ACTIVE_FFMPEG_CHILD.lock() {
            if let Some(ref mut c) = *lock {
                stderr_taken = c.stderr.take();
            }
        }
        stderr_taken.context("Failed to take FFmpeg stderr")?
    };

    let stderr_thread = thread::spawn(move || {
        let reader = BufReader::new(stderr);
        let mut last_lines = std::collections::VecDeque::new();
        for line in reader.lines() {
            if let Ok(l) = line {
                last_lines.push_back(l);
                if last_lines.len() > 20 {
                    last_lines.pop_front();
                }
            }
        }
        last_lines.into_iter().collect::<Vec<String>>().join("\n")
    });

    let pump_thread = thread::spawn(move || -> Result<()> {
        let mut temp_buf = [0u8; 4096];
        loop {
            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                break;
            }
            let start_time = std::time::Instant::now();
            let mut chunk_bytes = Vec::with_capacity(192000);
            while chunk_bytes.len() < 192000 {
                if SHOULD_CANCEL.load(Ordering::SeqCst) {
                    break;
                }
                let to_read = std::cmp::min(temp_buf.len(), 192000 - chunk_bytes.len());
                match stdout.read(&mut temp_buf[..to_read]) {
                    Ok(0) => break,
                    Ok(n) => {
                        chunk_bytes.extend_from_slice(&temp_buf[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e.into()),
                }
            }

            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                break;
            }

            if chunk_bytes.is_empty() {
                break;
            }

            let n_samples = chunk_bytes.len() / 4;
            let mut samples = vec![0.0f32; n_samples];
            for (i, bytes) in chunk_bytes.chunks_exact(4).enumerate() {
                samples[i] = f32::from_le_bytes(bytes.try_into().unwrap());
            }

            let _elapsed_ms = start_time.elapsed().as_millis() as u64;
            let _audio_dur_sec = samples.len() as f64 / 48000.0;
            log_perf!("FFmpeg", _elapsed_ms, _audio_dur_sec);

            if tx_raw_48k.send(samples).is_err() {
                break;
            }
        }
        Ok(())
    });

    Ok(FfmpegPumpHandle {
        pump_thread,
        stderr_thread,
    })
}

fn spawn_dfn_worker(
    config: &PipelineConfig,
    rx_raw_48k: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_clean_48k: std::sync::mpsc::SyncSender<Vec<f32>>,
) -> thread::JoinHandle<Result<()>> {
    let enable_denoise = config.enable_denoise;
    let df_model_path = config.df_model_path.clone();

    thread::spawn(move || -> Result<()> {
        if !enable_denoise {
            println!("[Rust] DeepFilterNet3 降噪已禁用，音频流直通处理。");
            let tx_clean_48k_clone = tx_clean_48k.clone();
            let mut buffer = Vec::with_capacity(48000);
            while let Ok(raw_chunk_48k) = rx_raw_48k.recv() {
                buffer.extend_from_slice(&raw_chunk_48k);
                while buffer.len() >= 48000 {
                    let chunk: Vec<f32> = buffer.drain(..48000).collect();
                    if tx_clean_48k_clone.send(chunk).is_err() {
                        return Ok(());
                    }
                }
            }
            if !buffer.is_empty() {
                buffer.resize(48000, 0.0);
                let _ = tx_clean_48k_clone.send(buffer);
            }
            drop(tx_clean_48k_clone);
            return Ok(());
        }

        // 在工作线程内部延迟解压 DeepFilterNet 模型
        println!("[Rust] Preparing DeepFilterNet3 models from tar.gz...");
        let extracted_dir = extract_tar_gz_if_needed(Path::new(&df_model_path))?;
        println!("[Rust] DeepFilterStream prepared at: {:?}", extracted_dir);

        struct DfnTask {
            seq_id: usize,
            warmup_samples: Vec<f32>,
            real_samples: Vec<f32>,
        }
        
        struct DfnResult {
            seq_id: usize,
            cleaned_samples: Vec<f32>,
        }

        let (task_tx, task_rx) = crossbeam_channel::bounded::<DfnTask>(16);
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<DfnResult>();

        let num_workers = 4;
        let mut worker_handles = Vec::with_capacity(num_workers);
        let model_dir_arc = Arc::new(extracted_dir);
        let p_core_mask = get_physical_pcore_mask();

        for worker_id in 0..num_workers {
            let rx = task_rx.clone();
            let tx = result_tx.clone();
            let model_dir = Arc::clone(&model_dir_arc);
            let mask = p_core_mask;

            let handle = thread::spawn(move || -> Result<()> {
                println!("[Rust] Worker {} starting thread...", worker_id);
                #[cfg(target_os = "windows")]
                set_thread_affinity_mask(mask);
                register_thread_as_pro_audio();

                println!("[Rust] Worker {} loading DeepFilterStream from {:?}", worker_id, model_dir);
                let mut stream = DeepFilterStream::with_threads(&model_dir, 1)
                    .map_err(|e| anyhow!("Worker {} failed to create DeepFilterStream: {:?}", worker_id, e))?;
                println!("[Rust] Worker {} warming up DeepFilterStream...", worker_id);
                stream.warmup()
                    .map_err(|e| anyhow!("Worker {} failed to warmup: {:?}", worker_id, e))?;
                println!("[Rust] Worker {} warmup complete and ready!", worker_id);

                while let Ok(task) = rx.recv() {
                    let DfnTask { seq_id, warmup_samples, real_samples } = task;
                    let mut cleaned_samples = Vec::with_capacity(real_samples.len());
                    let start_time = std::time::Instant::now();

                    // 1. 重置 GRU 隐藏状态
                    stream.reset();

                    // 2. 状态预热 (Warm-up) - 丢弃此区间的输出
                    if !warmup_samples.is_empty() {
                        let _ = stream.process(&warmup_samples)
                            .map_err(|e| anyhow!("Worker {} warmup error: {:?}", worker_id, e))?;
                    }

                    // 3. 真实数据降噪
                    let cleaned = stream.process(&real_samples)
                        .map_err(|e| anyhow!("Worker {} processing error: {:?}", worker_id, e))?;
                    cleaned_samples.extend_from_slice(&cleaned);

                    // 4. 冲刷缓存
                    let flushed = stream.flush()
                        .map_err(|e| anyhow!("Worker {} flush error: {:?}", worker_id, e))?;
                    cleaned_samples.extend_from_slice(&flushed);

                    let _elapsed_ms = start_time.elapsed().as_millis() as u64;
                    let _audio_dur_sec = real_samples.len() as f64 / 48000.0;
                    log_perf!("DFN3", _elapsed_ms, _audio_dur_sec);

                    if tx.send(DfnResult { seq_id, cleaned_samples }).is_err() {
                        break;
                    }
                }
                Ok(())
            });
            worker_handles.push(handle);
        }

        drop(result_tx);

        let tx_clean_48k_clone = tx_clean_48k.clone();
        let collector_thread = thread::spawn(move || {
            let mut reorder_map = BTreeMap::new();
            let mut next_expected_id = 0;

            while let Ok(result) = result_rx.recv() {
                reorder_map.insert(result.seq_id, result.cleaned_samples);

                while let Some(entry) = reorder_map.first_entry() {
                    if *entry.key() == next_expected_id {
                        let samples = entry.remove();
                        let mut failed = false;
                        for chunk in samples.chunks(48000) {
                            let mut chunk_vec = chunk.to_vec();
                            if chunk_vec.len() < 48000 {
                                chunk_vec.resize(48000, 0.0);
                            }
                            if tx_clean_48k_clone.send(chunk_vec).is_err() {
                                failed = true;
                                break;
                            }
                        }
                        if failed {
                            break;
                        }
                        next_expected_id += 1;
                    } else {
                        break;
                    }
                }
            }
        });

        let block_size = 480000;
        let warmup_size = 48000;

        let mut current_block = Vec::with_capacity(block_size);
        let mut history_1s = Vec::with_capacity(warmup_size);
        let mut seq_id = 0;

        while let Ok(raw_chunk_48k) = rx_raw_48k.recv() {
            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                break;
            }
            current_block.extend_from_slice(&raw_chunk_48k);

            while current_block.len() >= block_size {
                let real_samples: Vec<f32> = current_block.drain(..block_size).collect();

                let warmup_samples = if seq_id == 0 {
                    Vec::new()
                } else {
                    history_1s.clone()
                };

                let start_idx = real_samples.len().saturating_sub(warmup_size);
                history_1s = real_samples[start_idx..].to_vec();

                let task = DfnTask {
                    seq_id,
                    warmup_samples,
                    real_samples,
                };
                
                if task_tx.send(task).is_err() {
                    break;
                }
                seq_id += 1;
            }
        }

        if !current_block.is_empty() {
            let warmup_samples = if seq_id == 0 {
                Vec::new()
            } else {
                history_1s.clone()
            };

            let task = DfnTask {
                seq_id,
                warmup_samples,
                real_samples: current_block,
            };
            let _ = task_tx.send(task);
        }

        drop(task_tx);

        for handle in worker_handles {
            match handle.join() {
                Ok(Err(e)) => println!("[Rust] DFN3 Worker thread returned error: {:?}", e),
                Err(e) => println!("[Rust] DFN3 Worker thread panicked: {:?}", e),
                _ => {}
            }
        }

        let _ = collector_thread.join();
        drop(tx_clean_48k);
        Ok(())
    })
}

/// VAD 增量扫描的重叠长度: 每次把上次扫描位置回退 0.5s 一起喂入,
/// 捕捉跨扫描边界结束的语音尾部, 防止短句丢失。
const VAD_SCAN_OVERLAP: usize = 8000; // 0.5s @ 16kHz

/// 对给定音频切片运行一次 Silero VAD 扫描。
/// whisper.cpp 的 VAD 每次只处理传入的采样 (无内部累积), 可安全重复调用。
fn vad_scan(
    vad_ctx: &mut WhisperVadContext,
    vad_params: &WhisperVadParams,
    audio: &[f32],
) -> Vec<WhisperVadSegment> {
    vad_ctx
        .segments_from_samples(vad_params.clone(), audio)
        .map(|segs| segs.into_iter().collect())
        .unwrap_or_default()
}

fn spawn_vad_worker(
    config: &PipelineConfig,
    rx_clean_48k: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_asr_task: std::sync::mpsc::SyncSender<AsrTask>,
) -> thread::JoinHandle<Result<()>> {
    let vad_enabled = config.vad_enabled;
    let vad_model_path = config.vad_model_path.clone();
    let vad_min_silence_ms = config.vad_min_silence_ms;
    let vad_min_speech_ms = config.vad_min_speech_ms;
    let vad_threshold = config.vad_threshold;

    thread::spawn(move || -> Result<()> {
        register_thread_as_pro_audio();
        // 混合架构下把 VAD 绑到 E 核: P 核全部让给 encoder (非混合架构自动跳过)
        bind_vad_to_e_cores();

        let resampler_params = SincInterpolationParameters {
            sinc_len: 64,
            f_cutoff: 0.95,
            interpolation: SincInterpolationType::Nearest,
            oversampling_factor: 128,
            window: WindowFunction::BlackmanHarris2,
        };
        let mut resampler = SincFixedIn::<f32>::new(
            16000_f64 / 48000_f64,
            2.0,
            resampler_params,
            48000,
            1,
        ).map_err(|e| anyhow!("Failed to initialize Rubato resampler: {:?}", e))?;

        let mut audio_buffer: Vec<f32> = Vec::with_capacity(30 * 16000);
        let mut current_offset_ms: i64 = 0;

        let mut vad = if vad_enabled {
            println!("[Rust] 正在为流式处理初始化 Silero VAD 模型: {}", vad_model_path);
            let mut vad_ctx_params = WhisperVadContextParams::new();
            // 增量扫描后每次 VAD 计算量很小 (毫秒级), 单线程即可;
            // 且线程已绑 E 核, GGML 池线程不受线程亲和性约束 (会跑到 P 核抢资源)
            vad_ctx_params.set_n_threads(1);
            match WhisperVadContext::new(&vad_model_path, vad_ctx_params) {
                Ok(v) => Some(v),
                Err(e) => {
                    println!("[Rust] 初始化 WhisperVadContext 失败: {:?}", e);
                    None
                }
            }
        } else {
            None
        };

        let mut vad_params = WhisperVadParams::new();
        vad_params.set_min_silence_duration(vad_min_silence_ms as i32);
        vad_params.set_min_speech_duration(vad_min_speech_ms as i32);
        let prob_threshold = if vad_threshold > 0.0 && vad_threshold < 1.0 { vad_threshold as f32 } else { 0.5f32 };
        vad_params.set_threshold(prob_threshold);

        // 增量扫描游标: audio_buffer 中已喂给 VAD 的起始位置。whisper.cpp 的 VAD
        // 每次只处理传入的采样 (无内部累积), 之前每次全量喂整个缓冲导致扫描成本
        let mut vad_scan_cursor = 0usize;
        // 缓冲内已知最早的语音起点 (供 30s 保底强切时尽量保留语音不丢)
        let mut speech_anchor = 0usize;

        while let Ok(mut clean_48k_batch) = rx_clean_48k.recv() {
            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                break;
            }
            let start_time = std::time::Instant::now();
            let mut audio_to_process = Vec::new();
            
            loop {
                let resampled = match resampler.process(&[&clean_48k_batch], None) {
                    Ok(r) => r,
                    Err(e) => {
                        println!("[Rust] 重采样出错: {:?}", e);
                        break;
                    }
                };
                let clean_block_16k = &resampled[0];
                audio_to_process.extend_from_slice(clean_block_16k);
                
                match rx_clean_48k.try_recv() {
                    Ok(b) => {
                        clean_48k_batch = b;
                    }
                    Err(_) => {
                        break;
                    }
                }
            }

            if audio_to_process.is_empty() {
                continue;
            }

            let _audio_dur_sec = audio_to_process.len() as f64 / 16000.0;
            audio_buffer.extend_from_slice(&audio_to_process);

            // 步长过滤：仅当新增采样点 >= 8000 (500ms 音频) 时才触发神经网络 VAD 计算，
            if audio_buffer.len().saturating_sub(vad_scan_cursor) < 8000 {
                continue;
            }

            let mut check_and_cut = true;
            while check_and_cut {
                check_and_cut = false;

                if let Some(ref mut vad_ctx) = vad {
                    // 增量扫描: 只喂"上次扫描之后新增的音频 + 0.5s 重叠"。
                    // 重叠用于捕捉跨扫描边界结束的语音尾部 (防止短句丢失)。
                    let mut scan_base = vad_scan_cursor.saturating_sub(VAD_SCAN_OVERLAP);
                    let mut segs_vec = vad_scan(vad_ctx, &vad_params, &audio_buffer[scan_base..]);

                    // 增量窗口内无语音但缓冲已积累较多: 语音可能刚在扫描窗口前结束
                    // (窗口截断会漏掉跨边界的段尾), 对全缓冲重扫一次验证,
                    // 避免"语音结束未切段 -> 静音清理把整段语音丢掉"。
                    if segs_vec.is_empty() && audio_buffer.len() > 80000 && scan_base > 0 {
                        let full = vad_scan(vad_ctx, &vad_params, &audio_buffer);
                        if !full.is_empty() {
                            segs_vec = full;
                            // 全缓冲扫描的段索引相对缓冲起点 (0), 基准随之切换
                            scan_base = 0;
                        }
                    }

                    // 记录缓冲内已知最早的"真实语音起点": 仅当段起点在窗口内部
                    // (first.start > 0) 才更新 —— 起点落在窗口开头说明该段可能从更早
                    // 延续而来 (起点被截断到窗口开头), 用 min 保住最早真实起点;
                    // 切段时从锚点起算, 避免跨窗口的语音开头被丢弃
                    if let Some(first) = segs_vec.first() {
                        if (first.start as usize * 160) > 0 {
                            let abs = scan_base + (first.start as usize * 160);
                            if speech_anchor == 0 || abs < speech_anchor {
                                speech_anchor = abs;
                            }
                        }
                    }
                    let mut cut_performed = false;
                    for (i, s) in segs_vec.iter().enumerate() {
                        // 段时间戳 (0.01s 单位) 是相对喂入窗口的, 换算回缓冲绝对位置
                        let start_idx =
                            (scan_base + (s.start as usize * 160)).min(audio_buffer.len());
                        let end_idx =
                            (scan_base + (s.end as usize * 160)).min(audio_buffer.len());

                        // 智能切段判定:
                        // 1. 后面已出现下一个语音段 (i + 1 < segs_vec.len()): 证明当前段已百分之百结束，触发极速切段！
                        // 2. 当前是最后一个语音段: 需等待缓冲区末尾有 >= 2400 采样点 (150ms 静音余量)
                        let is_followed_by_next = i + 1 < segs_vec.len();
                        let has_silence_tail = audio_buffer.len() >= end_idx + 2400;

                        if is_followed_by_next || has_silence_tail {
                            // 用语音锚点找回真正起点的两种情形:
                            // 1. 段起点被截断到窗口开头 (start_idx == scan_base):
                            //    语音从更早延续而来, 起点落在窗口外;
                            // 2. 段起点明显晚于锚点 (旧语音结尾已滑出窗口):
                            //    Silero 可能把旧语音+新语音合并成一个段导致未切,
                            //    窗口推进后旧语音只剩在缓冲里, 此时新段出现必须
                            //    从锚点起算, 否则 drain 会把旧语音整段丢掉。
                            let cut_start = if speech_anchor != 0
                                && (start_idx <= scan_base + 2
                                    || speech_anchor < start_idx.saturating_sub(16000))
                            {
                                speech_anchor.min(audio_buffer.len())
                            } else {
                                start_idx
                            };
                            let safe_start = cut_start.saturating_sub(3200); 
                            let safe_end = (end_idx + 3200).min(audio_buffer.len()); 
                            
                            let segment_samples = audio_buffer[safe_start..safe_end].to_vec();
                            let start_ms = current_offset_ms + (safe_start as i64 * 1000 / 16000);
                            let end_ms = current_offset_ms + (safe_end as i64 * 1000 / 16000);

                            // 诊断: 记录每次切段的真实时间戳与锚点, 便于定位首条字幕延迟
                            println!(
                                "[VAD] cut: start_ms={} end_ms={} len={}ms (anchor={} start_idx={} end_idx={})",
                                start_ms,
                                end_ms,
                                segment_samples.len() * 1000 / 16000,
                                speech_anchor,
                                start_idx,
                                end_idx
                            );

                            if segment_samples.len() > 3200 {
                                if tx_asr_task.send(AsrTask { seq: ASR_TASK_SEQ.fetch_add(1, Ordering::Relaxed), samples: segment_samples, start_ms, end_ms }).is_err() {
                                    break;
                                }
                            }
                            
                            audio_buffer.drain(..safe_end);
                            current_offset_ms += (safe_end as i64 * 1000) / 16000;
                            cut_performed = true;
                            // 缓冲整体前移: 全部视为未扫描, 语音锚点重置
                            vad_scan_cursor = 0;
                            speech_anchor = 0;
                            break;
                        }
                    }

                    if cut_performed {
                        check_and_cut = true;
                        continue;
                    }

                            // 本次未切段: 游标推进到缓冲末尾 (下次只扫增量, 不再全量重扫)
                            vad_scan_cursor = audio_buffer.len();

                            // 及时清理静音 Buffer: 当积累超过 5s 且 VAD 未能检出任何有效语音段时，
                            // 丢弃前面 4s 的纯静音，只留 1s 边沿，防止 Buffer 膨胀到 30s 导致 VAD 扫描变慢
                            if segs_vec.is_empty() && audio_buffer.len() > 80000 {
                                let drop_samples = audio_buffer.len() - 16000;
                                audio_buffer.drain(..drop_samples);
                                current_offset_ms += (drop_samples as i64 * 1000) / 16000;
                                vad_scan_cursor = 0;
                                speech_anchor = 0;
                            }

                            // 极端无停顿长文本保底保护：Buffer 满了 30s 强切，避免越界
                            if audio_buffer.len() >= 480000 {
                                if !segs_vec.is_empty() {
                                    // 从已知最早的语音起点切起 (连续语音时约等于缓冲起点),
                                    // 避免只保留最后 0.5s 增量导致长段语音丢失
                                    let anchor = speech_anchor.min(audio_buffer.len());
                                    let safe_start = anchor.saturating_sub(3200).min(audio_buffer.len());
                                    let segment_samples = audio_buffer[safe_start..].to_vec();
                                    let start_ms = current_offset_ms + (safe_start as i64 * 1000 / 16000);
                                    let end_ms = current_offset_ms + (audio_buffer.len() as i64 * 1000 / 16000);

                                    println!(
                                        "[VAD] force-cut(30s): start_ms={} end_ms={} len={}ms (anchor={})",
                                        start_ms,
                                        end_ms,
                                        segment_samples.len() * 1000 / 16000,
                                        anchor
                                    );
                                    
                                    let _ = tx_asr_task.send(AsrTask { seq: ASR_TASK_SEQ.fetch_add(1, Ordering::Relaxed), samples: segment_samples, start_ms, end_ms });
                                } else {
                                    println!("[Rust] VAD 过滤: 成功丢弃 30 秒的非语音/纯音乐数据");
                                }
                                
                                current_offset_ms += (audio_buffer.len() as i64 * 1000) / 16000;
                                audio_buffer.clear();
                                vad_scan_cursor = 0;
                                speech_anchor = 0;
                            }
                } else {
                    // VAD 关闭时的回退逻辑 (按 10s 死切)
                    if audio_buffer.len() >= 160000 {
                        let segment_samples: Vec<f32> = audio_buffer.drain(..160000).collect();
                        let start_ms = current_offset_ms;
                        let end_ms = current_offset_ms + 10000;
                        current_offset_ms = end_ms;
                        if tx_asr_task.send(AsrTask { seq: ASR_TASK_SEQ.fetch_add(1, Ordering::Relaxed), samples: segment_samples, start_ms, end_ms }).is_err() {
                            break;
                        }
                        check_and_cut = true;
                    }
                }
            }

            let _elapsed_ms = start_time.elapsed().as_millis() as u64;
            log_perf!("SileroVAD", _elapsed_ms, _audio_dur_sec);
        }

        if !audio_buffer.is_empty() {
            let seg_duration_ms = (audio_buffer.len() as f64 / 16000.0 * 1000.0) as i64;
            let start_ms = current_offset_ms;
            let end_ms = current_offset_ms + seg_duration_ms;
            let _ = tx_asr_task.send(AsrTask { seq: ASR_TASK_SEQ.fetch_add(1, Ordering::Relaxed), samples: audio_buffer, start_ms, end_ms });
        }

        drop(tx_asr_task);
        Ok(())
    })
}

struct EncodedTask {
    seq: u64,
    samples: Vec<f32>,
    /// None 表示编码失败/跳过
    enc_out: Option<crate::qwen::encoder::EncoderOutput>,
    start_ms: i64,
    end_ms: i64,
}

/// 连续解码为空(模型跳过)的音频最大合并时长，超过则丢弃避免污染后续识别。
const MAX_MERGE_SECONDS: usize = 10;

fn spawn_qwen_worker(
    runtime: Arc<crate::qwen::runtime::QwenRuntime>,
    config: &PipelineConfig,
    total_duration: f64,
    rx_task: std::sync::mpsc::Receiver<AsrTask>,
    sink: Arc<dyn TranscriptionSink>,
) -> thread::JoinHandle<Result<()>> {
    let to_simplified = config.to_simplified;
    let language = config.language.clone();
    let context_prompt = config.context_prompt.clone();

    // 编码/解码双线程流水线：段 N 在 GPU 上自回归解码时，段 N+1 已在编码。
    // 编码线程池: encoder 无状态 (每段独立 mel+推理)，CPU 推理时多线程并行编码
    // 可显著提升吞吐 (decoder 侧按 seq 恢复顺序)。
    let (tx_encoded, rx_encoded) = std::sync::mpsc::sync_channel::<EncodedTask>(16);
    let actual_ep = runtime.encoder_actual_ep();
    // 编码线程池策略：
    // - GPU 模式 (CUDA/DirectML): 预加载的主 Encoder 结合 GPU 极速计算足够支撑全量吞吐，
    //   仅使用 1 个 worker 线程复用主 Encoder (预加载已完成)，不加载多余 Session.
    // - CPU 模式: 仅开启 2 个线程 (主 Worker 复用预加载 Session + 1 个辅助 Worker)
    let num_encoders = if actual_ep == "CPU" { 2 } else { 1 };

    let rx_task_shared = Arc::new(std::sync::Mutex::new(rx_task));
    let mut encoder_handles = Vec::new();
    for worker_idx in 0..num_encoders {
        let runtime = runtime.clone();
        let tx_encoded = tx_encoded.clone();
        let rx_task_shared = rx_task_shared.clone();
        encoder_handles.push(thread::spawn(move || -> Result<()> {
            register_thread_as_pro_audio();
            set_worker_pcore_affinity(worker_idx);
            // 仅对 worker_idx > 0 (即 CPU 模式下的辅助 worker) 懒加载辅助 Session；
            // worker_idx == 0 始终 100% 直接复用预加载已完备的主 runtime encoder (零启动开销)。
            let rx_loaded = if worker_idx > 0 {
                let (tx, rx) = std::sync::mpsc::channel::<Result<crate::qwen::encoder::QwenEncoder, String>>();
                let model_dir = runtime.encoder_model_path();
                let backend = runtime.encoder_backend();
                thread::spawn(move || {
                    let res = crate::qwen::encoder::QwenEncoder::load(&model_dir, backend)
                        .map_err(|e| format!("{:?}", e));
                    let _ = tx.send(res);
                });
                Some(rx)
            } else {
                None
            };

            let mut worker_enc: Option<crate::qwen::encoder::QwenEncoder> = None;
            let mut load_failed = false;
            loop {
                // 轮询辅助 worker encoder 加载结果 (完成后切换为独立 Session 并行)
                if let Some(ref rx) = rx_loaded {
                    if !load_failed && worker_enc.is_none() {
                        match rx.try_recv() {
                            Ok(Ok(e)) => {
                                println!("[encoder] worker encoder session ready (parallel)");
                                worker_enc = Some(e);
                            }
                            Ok(Err(e)) => {
                                println!("[encoder] worker encoder load failed, using shared encoder: {}", e);
                                load_failed = true;
                            }
                            Err(_) => {}
                        }
                    }
                }
                let task = {
                    let guard = rx_task_shared.lock().unwrap();
                    match guard.recv() {
                        Ok(t) => t,
                        Err(_) => break, // 通道关闭
                    }
                };
                if SHOULD_CANCEL.load(Ordering::SeqCst) {
                    break;
                }
                if task.samples.is_empty() {
                    // 空任务也发送 None 标记，保证 seq 连续 (decoder 直接跳过)
                    if tx_encoded
                        .send(EncodedTask {
                            seq: task.seq,
                            samples: Vec::new(),
                            enc_out: None,
                            start_ms: task.start_ms,
                            end_ms: task.end_ms,
                        })
                        .is_err()
                    {
                        break;
                    }
                    continue;
                }
                let start_time = std::time::Instant::now();
                // worker 独立 Session 就绪前使用主 runtime encoder (预加载)
                let enc_out = match &mut worker_enc {
                    Some(e) => e.encode(&task.samples, &runtime.cancel),
                    None => runtime.encode_segment(&task.samples),
                };
                let enc_out = match enc_out {
                    Ok(e) => Some(e),
                    Err(e) => {
                        // 编码失败也发送 None 标记保持 seq 连续，避免 decoder 卡住
                        println!("[Rust] Qwen encode segment error: {:?}", e);
                        if tx_encoded
                            .send(EncodedTask {
                                seq: task.seq,
                                samples: Vec::new(),
                                enc_out: None,
                                start_ms: task.start_ms,
                                end_ms: task.end_ms,
                            })
                            .is_err()
                        {
                            break;
                        }
                        continue;
                    }
                };
                let _elapsed_ms = start_time.elapsed().as_millis() as u64;
                let _audio_dur_sec = task.samples.len() as f64 / 16000.0;
                let embd_dim = enc_out
                    .as_ref()
                    .map(|o| o.shape.last().copied().unwrap_or(1024).max(1))
                    .unwrap_or(1024);
                println!(
                    "[encoder] Encoded {:.2}s audio in {}ms ({} audio tokens)",
                    _audio_dur_sec,
                    _elapsed_ms,
                    enc_out.as_ref().map(|o| o.embeddings.len() / embd_dim).unwrap_or(0)
                );
                log_perf!("QwenEncode", _elapsed_ms, _audio_dur_sec);
                if tx_encoded
                    .send(EncodedTask {
                        seq: task.seq,
                        samples: task.samples,
                        enc_out,
                        start_ms: task.start_ms,
                        end_ms: task.end_ms,
                    })
                    .is_err()
                {
                    break;
                }
            }
            Ok(())
        }));
    }
    // 主线程不再持有 sender (各 worker 持有 clone)
    drop(tx_encoded);

    let decoder_handle = {
        thread::spawn(move || -> Result<()> {
            register_thread_as_pro_audio();
            let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
            let max_merge_samples = MAX_MERGE_SECONDS * 16000;
            // 合并缓冲：模型对嘈杂/语言切换初期的短片段倾向输出空结果(跳过)。
            // 空结果不直接丢弃，而是与下一个片段拼接后整体重编码重试，
            // 直到模型能识别为止。
            let mut pending: Option<(Vec<f32>, i64)> = None;

            // 编码线程池并行完成，按 seq 恢复原始顺序后再处理
            let mut next_seq = 0u64;
            let mut ordered: std::collections::BTreeMap<u64, EncodedTask> =
                std::collections::BTreeMap::new();

            loop {
                match rx_encoded.recv() {
                    Ok(task) => {
                        ordered.insert(task.seq, task);
                    }
                    Err(_) => break,
                }
                if SHOULD_CANCEL.load(Ordering::SeqCst) {
                    break;
                }

                // 按 seq 顺序取出连续可处理的段
                while let Some(mut task) = ordered.remove(&next_seq) {
                    next_seq += 1;
                    process_encoded_task(
                        &mut task,
                        &mut pending,
                        &runtime,
                        max_merge_samples,
                        &language,
                        &context_prompt,
                        &sink,
                        &mut all_segments,
                        to_simplified,
                        total_duration,
                    )?;
                }
            }
            // 通道关闭后清空剩余的乱序缓冲 (按序处理)；
            // 若仍有 seq 空洞 (理论上 worker 已保证连续，这里兜底) 跳过缺失 seq
            loop {
                match ordered.keys().next() {
                    Some(&k) if k == next_seq => {
                        let mut task = ordered.remove(&k).unwrap();
                        next_seq += 1;
                        process_encoded_task(
                            &mut task,
                            &mut pending,
                            &runtime,
                            max_merge_samples,
                            &language,
                            &context_prompt,
                            &sink,
                            &mut all_segments,
                            to_simplified,
                            total_duration,
                        )?;
                    }
                    Some(&k) => {
                        // 空洞: 缺失的 seq 永远不会到达，跳过
                        println!("[Rust] Qwen: skipping missing seq {} (hole)", next_seq);
                        while next_seq < k {
                            next_seq += 1;
                        }
                    }
                    None => break,
                }
            }

            let _ = sink.add(TranscriptionEvent::Success(all_segments));
            Ok(())
        })
    };

    thread::spawn(move || {
        let mut enc_ok = true;
        for handle in encoder_handles {
            if handle.join().map_err(|_| anyhow!("Qwen encoder thread panicked"))?.is_err() {
                enc_ok = false;
            }
        }
        let dec_res = decoder_handle
            .join()
            .map_err(|_| anyhow!("Qwen decoder thread panicked"))?;
        if enc_ok {
            enc_ok = dec_res.is_ok();
        }
        if !enc_ok {
            // 编码线程返回 Err 不致命 (单个段编码失败已跳过)，仅当解码失败才视为错误
            return dec_res;
        }
        Ok(())
    })
}

/// 解码一个已编码的段 (含合并重试逻辑)，由 decoder 线程按 seq 顺序调用。
/// 返回 Ok(()) 表示继续处理；sink 关闭或通道结束由调用方处理。
#[allow(clippy::too_many_arguments)]
fn process_encoded_task(
    task: &mut EncodedTask,
    pending: &mut Option<(Vec<f32>, i64)>,
    runtime: &Arc<crate::qwen::runtime::QwenRuntime>,
    max_merge_samples: usize,
    language: &Option<String>,
    context_prompt: &Option<String>,
    sink: &Arc<dyn TranscriptionSink>,
    all_segments: &mut Vec<TranscriptionSegment>,
    to_simplified: bool,
    total_duration: f64,
) -> Result<()> {
    // 编码失败/跳过的标记段 (enc_out=None): 直接跳过，不参与合并
    let Some(mut enc_out) = task.enc_out.take() else {
        return Ok(());
    };

    // 与上一个"空结果"片段拼接
    if let Some((p_samples, p_start_ms)) = pending.take() {
        if p_samples.len() + task.samples.len() > max_merge_samples {
            println!(
                "[Rust] Qwen: dropping {}ms of unrecognized (empty) audio (from {}ms)",
                p_samples.len() * 1000 / 16000,
                p_start_ms
            );
        } else {
            let mut merged = Vec::with_capacity(p_samples.len() + task.samples.len());
            merged.extend_from_slice(&p_samples);
            merged.extend_from_slice(&task.samples);
            match runtime.encode_segment(&merged) {
                Ok(enc_out2) => {
                    task.samples = merged;
                    task.start_ms = p_start_ms;
                    enc_out = enc_out2;
                }
                Err(e) => {
                    // 合并重编码失败: 恢复 pending 等下一个片段再试,
                    // 避免"已取走未合并"导致这段语音静默丢失
                    println!("[Rust] Qwen merge re-encode error: {:?}", e);
                    *pending = Some((p_samples, p_start_ms));
                }
            }
        }
    }

    let start_time = std::time::Instant::now();
    let decode_res = match runtime.decode_segment(
        &enc_out,
        language.as_deref(),
        context_prompt.as_deref(),
    ) {
        Ok(res) => res,
        Err(e) => {
            println!("[Rust] Qwen transcribe segment error: {:?}", e);
            return Ok(());
        }
    };

    let _elapsed_ms = start_time.elapsed().as_millis() as u64;
    let _audio_dur_sec = task.samples.len() as f64 / 16000.0;
    log_perf!("Qwen", _elapsed_ms, _audio_dur_sec);

    let mut final_text = decode_res.text.trim().to_string();
    if final_text.is_empty() {
        // 模型跳过：保留音频待与下一个片段合并重试
        *pending = Some((task.samples.clone(), task.start_ms));
        return Ok(());
    }

    if to_simplified {
        final_text = convert_chinese(final_text, true);
    }

    // 强制对齐
    let word_items = match runtime.align_segment(
        &task.samples,
        &final_text,
        task.start_ms as u64,
        task.end_ms as u64,
    ) {
        Ok(align_res) => align_res
            .units
            .into_iter()
            .map(|u| WordItem {
                text: u.text,
                start_ms: u.start_ms as i64,
                end_ms: u.end_ms as i64,
                confidence: u.confidence.unwrap_or(1.0),
            })
            .collect(),
        Err(_) => Vec::new(),
    };

    let new_seg = TranscriptionSegment {
        start_ms: task.start_ms,
        end_ms: task.end_ms,
        text: final_text,
        timestamp_quality: "Qwen3-ForcedAligned".to_string(),
        words: word_items,
    };

    all_segments.push(new_seg.clone());
    if sink.add(TranscriptionEvent::Segment(new_seg)).is_err() {
        println!("[Rust] Sink closed. Aborting Qwen loop.");
        return Ok(());
    }

    // Progress
    let progress = ((task.end_ms as f64 / 1000.0) / total_duration * 100.0) as i32;
    if sink.add(TranscriptionEvent::Progress(progress.clamp(0, 100))).is_err() {
        println!("[Rust] Sink closed. Aborting Qwen loop.");
        return Ok(());
    }
    if sink.add(TranscriptionEvent::ProgressDetail {
        processed_ms: task.end_ms,
        total_ms: (total_duration * 1000.0) as i64,
    }).is_err() {
        println!("[Rust] Sink closed. Aborting Qwen loop.");
        return Ok(());
    }
    Ok(())
}

// ==== 管道辅助函数 ====

fn extract_tar_gz_if_needed(tar_gz_path: &Path) -> Result<std::path::PathBuf> {
    let parent = tar_gz_path.parent().ok_or_else(|| anyhow!("No parent dir"))?;
    let file_stem = tar_gz_path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.replace(".tar.gz", "_extracted"))
        .ok_or_else(|| anyhow!("Invalid model filename"))?;
    let dest_dir = parent.join(file_stem);
    
    let expected_files = [
        "config.ini",
        "enc_conv_streaming.onnx",
        "enc_gru_streaming.onnx",
        "erb_dec_streaming.onnx",
        "df_dec_streaming.onnx",
    ];
    
    let nested_dir = dest_dir.join("tmp").join("export");
    let check_dir = if nested_dir.join("config.ini").exists() {
        &nested_dir
    } else {
        &dest_dir
    };
    
    let all_exist = expected_files.iter().all(|name| check_dir.join(name).exists());
    if all_exist {
        println!("[Rust] Model already extracted at: {:?}", check_dir);
        return Ok(check_dir.to_path_buf());
    }
    
    if dest_dir.exists() {
        let _ = std::fs::remove_dir_all(&dest_dir);
    }
    
    println!("[Rust] Extracting model {:?} to {:?}", tar_gz_path, dest_dir);
    let _ = std::fs::create_dir_all(&dest_dir);
    
    let file = std::fs::File::open(tar_gz_path)
        .map_err(|e| anyhow!("Failed to open tar.gz: {}", e))?;
    let tar_file = GzDecoder::new(file);
    let mut archive = Archive::new(tar_file);
    archive.unpack(&dest_dir)
        .map_err(|e| anyhow!("Failed to unpack tarfile: {}", e))?;
        
    println!("[Rust] Extraction complete.");
    
    if nested_dir.join("config.ini").exists() {
        Ok(nested_dir)
    } else {
        Ok(dest_dir)
    }
}

#[cfg(target_os = "windows")]
fn lock_high_priority() {
    use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass, HIGH_PRIORITY_CLASS};
    unsafe {
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
}

#[cfg(target_os = "windows")]
fn disable_power_throttling() {
    use std::mem::size_of;
    use windows_sys::Win32::System::Threading::{
        GetCurrentProcess, SetProcessInformation, ProcessPowerThrottling,
        PROCESS_POWER_THROTTLING_STATE, PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
    };

    unsafe {
        let mut state = PROCESS_POWER_THROTTLING_STATE {
            Version: 1,
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0,
        };

        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &mut state as *mut _ as *mut _,
            size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
    }
}

fn get_physical_pcore_mask() -> usize {
    // 优先使用运行时探测的混合架构拓扑 (精确 P 核集合)；
    // 探测失败或非混合架构时回退偶数逻辑 id 启发式 (假设超线程成对)。
    if let Some((p_cores, _)) = cached_cpu_topology() {
        let mask: u64 = p_cores
            .iter()
            .filter(|c| c.group == 0)
            .map(|c| c.mask)
            .fold(0, |a, m| a | m);
        if mask != 0 {
            return mask as usize;
        }
    }
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let mut mask = 0_usize;
    for core in core_ids {
        if core.id % 2 == 0 {
            mask |= 1_usize.checked_shl(core.id as u32).unwrap_or(0);
        }
    }
    if mask == 0 {
        !0
    } else {
        mask
    }
}

fn set_worker_pcore_affinity(worker_idx: usize) {
    if core_pinning_disabled() {
        return;
    }
    if let Some((p_cores, _)) = cached_cpu_topology() {
        if !p_cores.is_empty() {
            let target = &p_cores[worker_idx % p_cores.len()];
            let ok = set_thread_group_affinity(&[*target]);
            println!(
                "[encoder] ONNX Worker {} bound to group {} mask {:#x} (affinity success: {})",
                worker_idx, target.group, target.mask, ok
            );
            return;
        }
    }
    if let Some(core_ids) = core_affinity::get_core_ids() {
        let pcores: Vec<_> = core_ids.into_iter().filter(|c| c.id % 2 == 0).collect();
        if !pcores.is_empty() {
            let target_core = pcores[worker_idx % pcores.len()];
            let res = core_affinity::set_for_current(target_core);
            println!("[encoder] ONNX Worker {} bound to P-Core ID {} (affinity success: {})", worker_idx, target_core.id, res);
        }
    }
}

#[cfg(target_os = "windows")]
fn set_thread_affinity_mask(mask: usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}

// ==== CPU 混合架构 (P/E 核) 运行时探测 ====
//
// 兼容性设计:
// 1. 全部通过 GetLogicalProcessorInformationEx 运行时探测，零硬编码拓扑假设；
// 2. EfficiencyClass 区分 P/E 核 (Win10 2004+ / Win11 混合架构)。API 失败或
//    所有核 EfficiencyClass 相同 (非混合架构) 时，E 核集合为空 -> VAD 不绑核
//    (交给系统调度器)，P 核集合回退旧的"偶数 id"启发式 (超线程成对机型仍正确)；
// 3. 所有绑定调用 best-effort，失败只打日志绝不报错；
// 4. 环境变量 AUDIO2SRT_NO_CORE_PIN=1 可一键关闭全部绑核。

#[cfg(target_os = "windows")]
#[derive(Clone, Copy, Debug)]
struct CoreSet {
    group: u16,
    mask: u64,
}

#[cfg(target_os = "windows")]
fn core_pinning_disabled() -> bool {
    std::env::var_os("AUDIO2SRT_NO_CORE_PIN").is_some()
}

#[cfg(not(target_os = "windows"))]
fn core_pinning_disabled() -> bool {
    true
}

#[cfg(target_os = "windows")]
fn cached_cpu_topology() -> Option<&'static (Vec<CoreSet>, Vec<CoreSet>)> {
    static TOPOLOGY: std::sync::OnceLock<Option<(Vec<CoreSet>, Vec<CoreSet>)>> =
        std::sync::OnceLock::new();
    TOPOLOGY.get_or_init(detect_cpu_topology).as_ref()
}

#[cfg(not(target_os = "windows"))]
fn cached_cpu_topology() -> Option<&'static (Vec<CoreSet>, Vec<CoreSet>)> {
    None
}

/// 运行时探测 CPU 拓扑，返回 (P 核集合, E 核集合)。E 核为空 = 非混合架构。
/// 失败返回 None (调用方回退旧启发式)。
#[cfg(target_os = "windows")]
fn detect_cpu_topology() -> Option<(Vec<CoreSet>, Vec<CoreSet>)> {
    use windows_sys::Win32::System::SystemInformation::{
        GetLogicalProcessorInformationEx, RelationProcessorCore, PROCESSOR_RELATIONSHIP,
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
    };

    unsafe {
        let mut len: u32 = 0;
        GetLogicalProcessorInformationEx(RelationProcessorCore, std::ptr::null_mut(), &mut len);
        if len == 0 {
            return None;
        }
        let mut buf = vec![0u8; len as usize];
        if GetLogicalProcessorInformationEx(
            RelationProcessorCore,
            buf.as_mut_ptr() as *mut SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX,
            &mut len,
        ) == 0
        {
            return None;
        }

        // (efficiency_class, 该核的 group/mask 列表)
        let mut cores: Vec<(u8, Vec<CoreSet>)> = Vec::new();
        let mut offset = 0usize;
        while offset + std::mem::size_of::<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>() <= buf.len() {
            let entry = (buf.as_ptr() as *const u8).add(offset)
                as *const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX;
            let rel = &*entry;
            let size = rel.Size as usize;
            if size == 0 {
                break;
            }
            if rel.Relationship == RelationProcessorCore {
                let proc_rel: &PROCESSOR_RELATIONSHIP = &rel.Anonymous.Processor;
                let group_count = proc_rel.GroupCount as usize;
                if group_count > 0 {
                    let group_masks =
                        std::slice::from_raw_parts(proc_rel.GroupMask.as_ptr(), group_count);
                    let sets: Vec<CoreSet> = group_masks
                        .iter()
                        .map(|gm| CoreSet {
                            group: gm.Group,
                            mask: gm.Mask as u64,
                        })
                        .collect();
                    cores.push((proc_rel.EfficiencyClass, sets));
                }
            }
            offset += size;
        }

        if cores.is_empty() {
            return None;
        }
        let min_class = cores.iter().map(|c| c.0).min().unwrap_or(0);
        let max_class = cores.iter().map(|c| c.0).max().unwrap_or(0);
        let collect = |f: &dyn Fn(u8) -> bool| -> Vec<CoreSet> {
            cores
                .iter()
                .filter(|(c, _)| f(*c))
                .flat_map(|(_, m)| m.iter().cloned())
                .collect()
        };
        let p = collect(&|c| c == min_class);
        if min_class == max_class {
            // 非混合架构: 全部是 P 核, 无 E 核
            Some((p, Vec::new()))
        } else {
            Some((p, collect(&|c| c > min_class)))
        }
    }
}

/// 把当前线程绑定到给定核集合。任何一组合格即返回 true。
#[cfg(target_os = "windows")]
fn set_thread_group_affinity(cores: &[CoreSet]) -> bool {
    use windows_sys::Win32::System::SystemInformation::GROUP_AFFINITY;
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadGroupAffinity};
    for cs in cores {
        let ga = GROUP_AFFINITY {
            Mask: cs.mask as usize,
            Group: cs.group,
            Reserved: [0; 3],
        };
        if unsafe { SetThreadGroupAffinity(GetCurrentThread(), &ga, std::ptr::null_mut()) } != 0 {
            return true;
        }
    }
    false
}

/// VAD 线程绑定到 E 核 (混合架构): 把 P 核全部让给 encoder。
/// 非混合架构/探测失败/被环境变量禁用时静默跳过。
fn bind_vad_to_e_cores() {
    if core_pinning_disabled() {
        return;
    }
    #[cfg(target_os = "windows")]
    {
        if let Some((_, e_cores)) = cached_cpu_topology() {
            if e_cores.is_empty() {
                println!("[VAD] 非混合架构 (无 E 核), 不绑核, 由系统调度器分配");
                return;
            }
            let ok = set_thread_group_affinity(e_cores);
            println!(
                "[VAD] 已绑定到 E 核集合 ({} 组, success: {})",
                e_cores.len(),
                ok
            );
        }
    }
}

fn is_qwen_model_dir(dir_or_file: &str) -> bool {
    let path = Path::new(dir_or_file);
    let lower = dir_or_file.to_lowercase();
    if lower.contains("qwen") {
        return true;
    }
    if path.is_dir() {
        if path.join("encoder.fp16.onnx").exists()
            || path.join("encoder.onnx").exists()
            || path.join("encoder.int4.onnx").exists()
            || path.join("asr_encoder_frontend.int4.onnx").exists()
        {
            // decoder 任一量化命名均可
            let decoder_names = [
                "decoder.bf16.gguf",
                "decoder.f16.gguf",
                "decoder.q8_0.gguf",
                "decoder.q6_k.gguf",
                "decoder.q5_k_m.gguf",
                "decoder.q4_k_m.gguf",
                "decoder.q4_k.gguf",
                "decoder.gguf",
                "asr_decoder.q4_k.gguf",
            ];
            if decoder_names.iter().any(|f| path.join(f).exists()) {
                return true;
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::silero_vad::TranscriptionEvent;

    struct MockSink;

    impl TranscriptionSink for MockSink {
        fn add(&self, event: TranscriptionEvent) -> Result<(), String> {
            println!("[MockSink] Received event: {:?}", event);
            Ok(())
        }
    }

    #[test]
    fn test_pipeline_performance() {
        set_rust_perf_logging(true);
        let movie_file = "C:\\FFOutput\\testmovie.mkv";
        let fallback_audio = "C:\\Projects\\audio2srt\\testaudio.wav";
        let input_path = if std::path::Path::new(movie_file).exists() {
            movie_file.to_string()
        } else {
            fallback_audio.to_string()
        };

        let large_model = "C:\\Users\\Steven\\AppData\\Roaming\\com.audio2srt\\audio2srt\\models\\ggml-large-v3-q8_0.bin";
        let base_model = "C:\\Users\\Steven\\AppData\\Roaming\\com.audio2srt\\audio2srt\\models\\ggml-base.bin";
        let model_path = if std::path::Path::new(large_model).exists() {
            large_model.to_string()
        } else {
            base_model.to_string()
        };

        let vad_model_path = "C:\\Users\\Steven\\AppData\\Roaming\\com.audio2srt\\audio2srt\\models\\ggml-silero-v5.1.2.bin".to_string();
        let df_model_path = "C:\\Users\\Steven\\AppData\\Roaming\\com.audio2srt\\audio2srt\\models\\DeepFilterNet3_onnx.tar.gz".to_string();

        println!("Running performance test using input: {}", input_path);
        println!("Model: {}", model_path);

        let ffmpeg_path = "ffmpeg".to_string();
        let sink = Arc::new(MockSink) as Arc<dyn TranscriptionSink>;

        let sink_clone = sink.clone();
        
        let config = PipelineConfig {
            ffmpeg_path,
            input_path,
            model_path: model_path.clone(),
            vad_model_path,
            df_model_path,
            asr_model_dir: model_path,
            aligner_model_dir: None,
            decoder_file: None,
            context_prompt: None,
            encoder_backend: crate::qwen::backend::EncoderBackend::Cpu,
            decoder_backend: crate::qwen::backend::DecoderBackend::Cpu,
            timestamp_mode: TimestampMode::Fast,
            language: Some("zh".to_string()),
            translate: false,
            threads: Some(4),
            use_gpu: true,
            to_simplified: true,
            enable_denoise: true,
            vad_enabled: true,
            vad_threshold: 0.5,
            vad_min_speech_ms: 300,
            vad_min_silence_ms: 400,
            no_context: true,
            no_state_history: true,
            selected_audio_track: None,
        };

        let _handle = thread::spawn(move || {
            let res = run_stream_pipeline_inner(sink_clone, config);
            println!("Pipeline run result: {:?}", res);
        });

        // 运行 60 秒后退出测试。因为性能记录是实时追加写入的，所以退出前已经缓存好了性能数据。
        thread::sleep(std::time::Duration::from_secs(60));
        println!("Test timed out after 60s, exiting to terminate background threads.");
    }

    #[test]
    fn test_auto_language_srt_export() {
        let ffmpeg_path = "ffmpeg".to_string();
        let app_data = std::env::var("APPDATA").unwrap_or_default();
        let large_path = format!("{}/com.audio2srt/audio2srt/models/ggml-large-v3-q8_0.bin", app_data);
        let base_path = format!("{}/com.audio2srt/audio2srt/models/ggml-base.bin", app_data);
        let model_path = if Path::new(&large_path).exists() {
            large_path
        } else {
            base_path
        };

        let files = vec![
            (
                "C:/Projects/测试用例/简单-新闻/《新闻联播》26-06-21.mp4",
                "C:/Projects/测试用例/简单-新闻/《新闻联播》26-06-21.srt",
            ),
            (
                "C:/Projects/测试用例/简单-新闻/BBC_Great-progress-at-US-Iran-talks-says-US-_Media_dsTNWkWYB50_001_720p.mp4",
                "C:/Projects/测试用例/简单-新闻/BBC_Great-progress-at-US-Iran-talks-says-US-_Media_dsTNWkWYB50_001_720p.srt",
            ),
        ];

        fn fmt_srt(ms_i64: i64) -> String {
            let ms = ms_i64.max(0) as u64;
            let sec = ms / 1000;
            let millis = ms % 1000;
            let min = sec / 60;
            let hr = min / 60;
            format!("{:02}:{:02}:{:02},{:03}", hr, min % 60, sec % 60, millis)
        }

        for (input_path, srt_path) in files {
            if !Path::new(input_path).exists() {
                println!("Input file not found, skipping: {}", input_path);
                continue;
            }

            println!("=== Testing Auto-Detect Language on File: {} ===", input_path);

            let (done_tx, done_rx) = std::sync::mpsc::channel();

            struct ExportSink {
                srt_path: String,
                segments: Mutex<Vec<TranscriptionSegment>>,
                done_tx: std::sync::mpsc::Sender<()>,
            }

            impl TranscriptionSink for ExportSink {
                fn add(&self, event: TranscriptionEvent) -> Result<(), String> {
                    match event {
                        TranscriptionEvent::Segment(seg) => {
                            println!("[SRT Segment] {} -> {} | {}", fmt_srt(seg.start_ms), fmt_srt(seg.end_ms), seg.text);
                            let mut lock = self.segments.lock().unwrap();
                            lock.push(seg);
                        }
                        TranscriptionEvent::Success(_) | TranscriptionEvent::Failure(_) => {
                            let _ = self.done_tx.send(());
                        }
                        _ => {}
                    }
                    Ok(())
                }
            }

            let sink = Arc::new(ExportSink {
                srt_path: srt_path.to_string(),
                segments: Mutex::new(Vec::new()),
                done_tx,
            });

            let config = PipelineConfig {
                ffmpeg_path: ffmpeg_path.clone(),
                input_path: input_path.to_string(),
                model_path: model_path.clone(),
                vad_model_path: "".to_string(),
                df_model_path: "".to_string(),
                asr_model_dir: model_path.clone(),
                aligner_model_dir: None,
                decoder_file: None,
                context_prompt: None,
                encoder_backend: crate::qwen::backend::EncoderBackend::Cpu,
                decoder_backend: crate::qwen::backend::DecoderBackend::Cpu,
                timestamp_mode: TimestampMode::Fast,
                language: Some("auto".to_string()),
                translate: false,
                threads: Some(4),
                use_gpu: true,
                to_simplified: true,
                enable_denoise: false,
                vad_enabled: true,
                vad_threshold: 0.5,
                vad_min_speech_ms: 300,
                vad_min_silence_ms: 400,
                no_context: true,
                no_state_history: true,
                selected_audio_track: None,
            };

            let res = run_stream_pipeline_inner(sink.clone(), config);
            println!("Pipeline started for {} with result: {:?}", input_path, res);

            let _ = done_rx.recv_timeout(std::time::Duration::from_secs(60));
            thread::sleep(std::time::Duration::from_millis(500));

            let lock = sink.segments.lock().unwrap();
            let mut srt_content = String::new();
            for (i, s) in lock.iter().enumerate() {
                srt_content.push_str(&format!("{}\n{} --> {}\n{}\n\n", i + 1, fmt_srt(s.start_ms), fmt_srt(s.end_ms), s.text));
            }

            use std::os::windows::ffi::OsStrExt;
            use std::os::windows::ffi::OsStringExt;
            let wide: Vec<u16> = std::ffi::OsString::from(srt_path)
                .encode_wide()
                .chain(std::iter::once(0))
                .collect();
            let pb = std::path::PathBuf::from(std::ffi::OsString::from_wide(&wide[..wide.len() - 1]));
            use std::io::Write;
            match std::fs::File::create(&pb) {
                Ok(mut f) => {
                    let _ = f.write_all(srt_content.as_bytes());
                    let _ = f.flush();
                    println!("[Test Success] Wrote {} segments ({} bytes) to SRT at: {:?}", lock.len(), srt_content.len(), pb);
                }
                Err(e) => println!("[Test Error] Failed to write SRT file {:?}: {}", pb, e),
            }
        }
    }
}
