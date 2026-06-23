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
use whisper_rs::{FullParams, SamplingStrategy, WhisperContextParameters, WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
use crate::frb_generated::StreamSink;
use crate::api::whisper::{
    TranscriptionSegment, TranscriptionEvent, get_or_create_context,
    convert_chinese, calculate_dtw_mem_size, get_dtw_model_preset,
    register_thread_as_pro_audio,
};

static PERF_LOG_ENABLED: AtomicBool = AtomicBool::new(false);

#[flutter_rust_bridge::frb(sync)]
pub fn set_rust_perf_logging(enable: bool) {
    PERF_LOG_ENABLED.store(enable, Ordering::Relaxed);
}

static SHOULD_CANCEL: AtomicBool = AtomicBool::new(false);
static ACTIVE_FFMPEG_CHILD: Mutex<Option<std::process::Child>> = Mutex::new(None);

#[flutter_rust_bridge::frb(sync)]
pub fn cancel_transcription_backend() {
    SHOULD_CANCEL.store(true, Ordering::SeqCst);
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
        self.add(event).map_err(|e| e.to_string())
    }
}

// ✅ 配置结构体以消除“过长参数列表”
#[derive(Clone, Debug)]
pub struct PipelineConfig {
    pub ffmpeg_path: String,
    pub input_path: String,
    pub model_path: String,
    pub vad_model_path: String,
    pub df_model_path: String,
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

    // 2. 初始化 Whisper 上下文
    println!("[Rust] Loading Whisper model context...");
    let mut ctx_params = WhisperContextParameters::default();
    
    // 如果可用，使用模型预设启用 DTW 模式
    let dtw_preset = get_dtw_model_preset(&config.model_path);
    if let Some(preset) = dtw_preset {
        println!("[Rust] DTW alignment enabled with preset for model: {}", config.model_path);
        let num_samples = (total_duration * 16000.0) as usize;
        let mem_size = calculate_dtw_mem_size(num_samples);
        ctx_params.dtw_parameters(whisper_rs::DtwParameters {
            mode: whisper_rs::DtwMode::ModelPreset { model_preset: preset },
            dtw_mem_size: mem_size,
        });
    } else {
        println!("[Rust] DTW alignment disabled (no preset for model: {})", config.model_path);
        ctx_params.dtw_parameters(whisper_rs::DtwParameters {
            mode: whisper_rs::DtwMode::None,
            dtw_mem_size: 0,
        });
    }
    
    let mut selected_device_name = "CPU".to_string();
    if config.use_gpu {
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
                let selected = dgpu.or(devices.first());
                if let Some(device) = selected {
                    println!("[Rust] Selecting Vulkan GPU device {}: {}", device.id, device.name);
                    ctx_params.use_gpu = true;
                    ctx_params.gpu_device = device.id;
                    selected_device_name = format!("GPU: {}", device.name);
                }
            }
        }
    } else {
        ctx_params.use_gpu = false;
    }
    println!("[Rust] Hardware device for Whisper: {}", selected_device_name);

    let ctx = get_or_create_context(&config.model_path, config.use_gpu, ctx_params)
        .map_err(|e| anyhow!("Failed to load Whisper model: {}", e))?;

    // 3. 创建流式管道 (容量均为 10)
    let (tx_raw_48k, rx_raw_48k) = sync_channel::<Vec<f32>>(10);
    let (tx_clean_48k, rx_clean_48k) = sync_channel::<Vec<f32>>(10);
    let (tx_whisper_task, rx_whisper_task) = sync_channel::<WhisperTask>(10);

    // 4. 启动各个独立的工作节点
    let ffmpeg_handle = spawn_ffmpeg_pump(&config, tx_raw_48k)?;
    let dfn_handle = spawn_dfn_worker(&config, rx_raw_48k, tx_clean_48k);
    let vad_handle = spawn_vad_worker(&config, rx_clean_48k, tx_whisper_task);
    let whisper_handle = spawn_whisper_worker(ctx, &config, total_duration, rx_whisper_task, sink);

    // 5. 等待收尾
    ffmpeg_handle.join()?;
    dfn_handle.join().map_err(|_| anyhow!("DFN3 thread panicked"))??;
    vad_handle.join().map_err(|_| anyhow!("VAD thread panicked"))??;
    whisper_handle.join().map_err(|_| anyhow!("Failed to join Whisper GPU thread"))??;

    Ok(())
}

// ==== 工作线程结构体与辅助函数 ====

struct WhisperTask {
    samples: Vec<f32>,
    start_ms: i64,
    end_ms: i64,
}

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

        // ✅ 在工作线程内部延迟解压 DeepFilterNet 模型
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

fn spawn_vad_worker(
    config: &PipelineConfig,
    rx_clean_48k: std::sync::mpsc::Receiver<Vec<f32>>,
    tx_whisper_task: std::sync::mpsc::SyncSender<WhisperTask>,
) -> thread::JoinHandle<Result<()>> {
    let vad_enabled = config.vad_enabled;
    let vad_model_path = config.vad_model_path.clone();
    let vad_min_silence_ms = config.vad_min_silence_ms;
    let vad_min_speech_ms = config.vad_min_speech_ms;
    let vad_threshold = config.vad_threshold;

    thread::spawn(move || -> Result<()> {
        register_thread_as_pro_audio();

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
            let vad_ctx_params = WhisperVadContextParams::new();
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

            let mut check_and_cut = true;
            while check_and_cut {
                check_and_cut = false;

                if let Some(ref mut vad_ctx) = vad {
                    match vad_ctx.segments_from_samples(vad_params.clone(), &audio_buffer) {
                        Ok(segs) => {
                            let segs_vec: Vec<_> = segs.into_iter().collect();
                            let mut cut_performed = false;
                            for s in &segs_vec {
                                let start_idx = (s.start as usize * 160).min(audio_buffer.len());
                                let end_idx = (s.end as usize * 160).min(audio_buffer.len());

                                // 寻找一个"已经结束"的语音段 (后面有 >= 4000 采样点 即 250ms 的静音)
                                if audio_buffer.len() >= end_idx + 4000 {
                                    let safe_start = start_idx.saturating_sub(3200); 
                                    let safe_end = (end_idx + 3200).min(audio_buffer.len()); 
                                    
                                    let segment_samples = audio_buffer[safe_start..safe_end].to_vec();
                                    let start_ms = current_offset_ms + (safe_start as i64 * 1000 / 16000);
                                    let end_ms = current_offset_ms + (safe_end as i64 * 1000 / 16000);

                                    if segment_samples.len() > 3200 {
                                        if tx_whisper_task.send(WhisperTask { samples: segment_samples, start_ms, end_ms }).is_err() {
                                            break;
                                        }
                                    }
                                    
                                    audio_buffer.drain(..safe_end);
                                    current_offset_ms += (safe_end as i64 * 1000) / 16000;
                                    cut_performed = true;
                                    break;
                                }
                            }

                            if cut_performed {
                                check_and_cut = true;
                                continue;
                            }

                            // 如果 Buffer 满了 30s，触发红线保护
                            if audio_buffer.len() >= 480000 {
                                if let Some(s) = segs_vec.last() {
                                    let start_idx = (s.start as usize * 160).min(audio_buffer.len());
                                    let safe_start = start_idx.saturating_sub(3200);
                                    let segment_samples = audio_buffer[safe_start..].to_vec();
                                    let start_ms = current_offset_ms + (safe_start as i64 * 1000 / 16000);
                                    let end_ms = current_offset_ms + (audio_buffer.len() as i64 * 1000 / 16000);
                                    
                                    let _ = tx_whisper_task.send(WhisperTask { samples: segment_samples, start_ms, end_ms });
                                } else {
                                    println!("[Rust] VAD 过滤: 成功丢弃 30 秒的非语音/纯音乐数据");
                                }
                                
                                current_offset_ms += (audio_buffer.len() as i64 * 1000) / 16000;
                                audio_buffer.clear();
                            }
                        }
                        Err(_) => {}
                    }
                } else {
                    // VAD 关闭时的回退逻辑 (按 10s 死切)
                    if audio_buffer.len() >= 160000 {
                        let segment_samples: Vec<f32> = audio_buffer.drain(..160000).collect();
                        let start_ms = current_offset_ms;
                        let end_ms = current_offset_ms + 10000;
                        current_offset_ms = end_ms;
                        if tx_whisper_task.send(WhisperTask { samples: segment_samples, start_ms, end_ms }).is_err() {
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
            let _ = tx_whisper_task.send(WhisperTask { samples: audio_buffer, start_ms, end_ms });
        }

        drop(tx_whisper_task);
        Ok(())
    })
}

fn spawn_whisper_worker(
    ctx: Arc<whisper_rs::WhisperContext>,
    config: &PipelineConfig,
    total_duration: f64,
    rx_whisper_task: std::sync::mpsc::Receiver<WhisperTask>,
    sink: Arc<dyn TranscriptionSink>,
) -> thread::JoinHandle<Result<()>> {
    let threads = config.threads;
    let translate = config.translate;
    let language = config.language.clone();
    let no_context = config.no_context;
    let to_simplified = config.to_simplified;

    thread::spawn(move || -> Result<()> {
        register_thread_as_pro_audio();

        let mut whisper_state = ctx.create_state()
            .map_err(|e| anyhow!("Failed to create Whisper state: {}", e))?;

        let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
        
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let n_threads = threads.unwrap_or(4);
        params.set_n_threads(n_threads);
        params.set_translate(translate);
        if let Some(ref lang) = language {
            if lang != "auto" && !lang.is_empty() { 
                params.set_language(Some(lang.as_str())); 
            } else { 
                params.set_language(None); 
                params.set_detect_language(false); 
            }
        } else {
            params.set_language(None); 
            params.set_detect_language(false);
        }
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);
        params.set_no_speech_thold(0.6);
        params.set_single_segment(false);

        while let Ok(task) = rx_whisper_task.recv() {
            if SHOULD_CANCEL.load(Ordering::SeqCst) {
                break;
            }
            let WhisperTask { samples: mut samples_16k, start_ms, end_ms } = task;
            
            if !samples_16k.is_empty() {
                // 强行对超短音频使用静音填充以保护 DTW 机制
                const MIN_SAMPLES_FOR_DTW: usize = 8000; // 0.5s
                if samples_16k.len() < MIN_SAMPLES_FOR_DTW {
                    samples_16k.resize(MIN_SAMPLES_FOR_DTW, 0.0);
                }

                let mut current_params = params.clone();
                current_params.set_no_context(no_context);
                current_params.set_single_segment(false); 
                current_params.set_suppress_blank(true);

                let start_time = std::time::Instant::now();
                let full_res = whisper_state.full(current_params, &samples_16k);
                let _elapsed_ms = start_time.elapsed().as_millis() as u64;
                let _audio_dur_sec = samples_16k.len() as f64 / 16000.0;
                log_perf!("Whisper", _elapsed_ms, _audio_dur_sec);

                full_res.map_err(|e| anyhow!("Whisper inference error: {:?}", e))?;

                let n_segments = whisper_state.full_n_segments();
                for i in 0..n_segments {
                    if let Some(segment) = whisper_state.get_segment(i) {
                        let text = segment.to_str_lossy().unwrap_or_default().into_owned();
                        
                        // 滤除音乐和叹气符号
                        let mut final_text = text.replace("♪", "")
                                                 .replace("[音乐]", "")
                                                 .replace("(音乐)", "")
                                                 .replace("[Music]", "")
                                                 .replace("(Music)", "");
                        final_text = final_text.trim().to_string();

                        if final_text.is_empty() {
                            continue;
                        }

                        let sub_start_ms = start_ms + segment.start_timestamp() * 10;
                        let sub_end_ms = start_ms + segment.end_timestamp() * 10;

                        if to_simplified {
                            final_text = convert_chinese(final_text, true);
                        }

                        let new_seg = TranscriptionSegment {
                            start_ms: sub_start_ms,
                            end_ms: sub_end_ms,
                            text: final_text,
                        };
                        all_segments.push(new_seg.clone());
                        if sink.add(TranscriptionEvent::Segment(new_seg)).is_err() {
                            println!("[Rust] Sink is closed. Aborting whisper loop.");
                            return Ok(());
                        }
                    }
                }
            }

            // 推进进度条
            let progress = ((end_ms as f64 / 1000.0) / total_duration * 100.0) as i32;
            if sink.add(TranscriptionEvent::Progress(progress.clamp(0, 100))).is_err() {
                println!("[Rust] Sink is closed. Aborting whisper loop.");
                return Ok(());
            }
            if sink.add(TranscriptionEvent::ProgressDetail {
                processed_ms: end_ms,
                total_ms: (total_duration * 1000.0) as i64,
            }).is_err() {
                println!("[Rust] Sink is closed. Aborting whisper loop.");
                return Ok(());
            }
        }
        
        let _ = sink.add(TranscriptionEvent::Success(all_segments));
        Ok(())
    })
}

// ==== 现有的辅助函数 ====

fn extract_tar_gz_if_needed(tar_gz_path: &Path) -> Result<std::path::PathBuf> {
    let parent = tar_gz_path.parent().ok_or_else(|| anyhow!("No parent dir"))?;
    let file_stem = tar_gz_path.file_name()
        .and_then(|n| n.to_str())
        .map(|s| s.replace(".tar.gz", "_extracted"))
        .ok_or_else(|| anyhow!("Invalid model filename"))?;
    let dest_dir = parent.join(file_stem);
    
    // 检查所有预期的模型文件是否都已存在
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
    
    // 如果目标目录已存在，先进行清理，避免残留不完整的解压文件
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

#[cfg(target_os = "windows")]
fn set_thread_affinity_mask(mask: usize) {
    use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadAffinityMask};
    unsafe {
        SetThreadAffinityMask(GetCurrentThread(), mask);
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::whisper::TranscriptionEvent;

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
            model_path,
            vad_model_path,
            df_model_path,
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

        let handle = thread::spawn(move || {
            let res = run_stream_pipeline_inner(sink_clone, config);
            println!("Pipeline run result: {:?}", res);
        });

        // 运行 60 秒后退出测试。因为性能记录是实时追加写入的，所以退出前已经缓存好了性能数据。
        thread::sleep(std::time::Duration::from_secs(60));
        println!("Test timed out after 60s, exiting to terminate background threads.");
    }
}
