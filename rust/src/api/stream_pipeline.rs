use std::process::{Command, Stdio};
use std::io::{Read, BufRead, BufReader};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::path::Path;
use std::collections::BTreeMap;
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};
use flate2::read::GzDecoder;
use tar::Archive;
use deepfilter_rt::DeepFilterStream;
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContextParameters, WhisperVadContext, WhisperVadContextParams, WhisperVadParams};
use crate::frb_generated::StreamSink;
use crate::api::whisper::{
    TranscriptionSegment, TranscriptionEvent, get_or_create_context,
    convert_chinese,
};



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
                        break;
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

pub fn transcribe_stream(
    sink: StreamSink<TranscriptionEvent>,
    ffmpeg_path: String,
    input_path: String,
    model_path: String,
    vad_model_path: String,
    df_model_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    to_simplified: bool,
    enable_denoise: bool,
    vad_enabled: bool,
    vad_threshold: f64,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
) {
    let sink_clone = sink.clone();
    thread::spawn(move || {
        if let Err(e) = run_stream_pipeline_inner(
            &sink_clone,
            ffmpeg_path,
            input_path,
            model_path,
            vad_model_path,
            df_model_path,
            language,
            translate,
            threads,
            use_gpu,
            to_simplified,
            enable_denoise,
            vad_enabled,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
        ) {
            let _ = sink_clone.add(TranscriptionEvent::Failure(e.to_string()));
        }
    });
}

fn run_stream_pipeline_inner(
    sink: &StreamSink<TranscriptionEvent>,
    ffmpeg_path: String,
    input_path: String,
    model_path: String,
    vad_model_path: String,
    df_model_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    to_simplified: bool,
    enable_denoise: bool,
    vad_enabled: bool,
    vad_threshold: f64,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
) -> Result<()> {
    #[cfg(target_os = "windows")]
    {
        lock_high_priority();
        disable_power_throttling();
    }

    // 1. 获取视频总时长
    let total_duration = match get_media_duration_secs(&ffmpeg_path, &input_path) {
        Ok(d) => d,
        Err(e) => {
            println!("[Rust] Warning: Failed to get duration: {}. Defaulting to 1.0", e);
            1.0
        }
    };
    println!("[Rust] Media total duration: {} seconds", total_duration);

    let chunk_size_48k = 48000;

    // 提前在主流程中解压并准备好模型目录，避免在专属计算线程中执行重度 I/O 操作
    let extracted_dir = if enable_denoise {
        println!("[Rust] Preparing DeepFilterNet3 models from tar.gz...");
        let extracted = extract_tar_gz_if_needed(Path::new(&df_model_path))?;
        println!("[Rust] DeepFilterStream prepared at: {:?}", extracted);
        extracted
    } else {
        std::path::PathBuf::new()
    };

    // 4. 初始化 Whisper 上下文
    println!("[Rust] Loading Whisper model context...");
    let mut ctx_params = WhisperContextParameters::default();
    
    // Enable DTW mode using the model preset if available
    let dtw_preset = get_dtw_model_preset(&model_path);
    if let Some(preset) = dtw_preset {
        println!("[Rust] DTW alignment enabled with preset for model: {}", model_path);
        let num_samples = (total_duration * 16000.0) as usize;
        let mem_size = calculate_dtw_mem_size(num_samples);
        ctx_params.dtw_parameters(whisper_rs::DtwParameters {
            mode: whisper_rs::DtwMode::ModelPreset { model_preset: preset },
            dtw_mem_size: mem_size,
        });
    } else {
        println!("[Rust] DTW alignment disabled (no preset for model: {})", model_path);
        ctx_params.dtw_parameters(whisper_rs::DtwParameters {
            mode: whisper_rs::DtwMode::None,
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

    let ctx = get_or_create_context(&model_path, use_gpu, ctx_params)
        .map_err(|e| anyhow!("Failed to load Whisper model: {}", e))?;
    
    let mut whisper_state = ctx.create_state()
        .map_err(|e| anyhow!("Failed to create Whisper state: {}", e))?;

    // 5. 启动 FFmpeg 流提取进程
    println!("[Rust] Starting FFmpeg audio pump process...");
    
    // 动态探测音频流数量
    let mut probe_cmd = Command::new(&ffmpeg_path);
    probe_cmd.arg("-i").arg(&input_path);
    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        probe_cmd.creation_flags(0x08000000); // 隐藏窗口
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
    println!("[Rust] transcribe_stream 检测到音频流数量: {}，文件: {}", audio_stream_count, input_path);

    let mut cmd = Command::new(&ffmpeg_path);
    cmd.arg("-y")
       .arg("-i")
       .arg(&input_path)
       .arg("-vn")
       .arg("-sn")
       .arg("-dn");

    // 多音频流时使用 amix 进行声道混合，并配合 volume 还原音量
    if audio_stream_count > 1 {
        cmd.arg("-filter_complex")
           .arg(format!("amix=inputs={}:duration=longest:dropout_transition=0,volume={}[a]", audio_stream_count, audio_stream_count))
           .arg("-map")
           .arg("[a]");
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
        cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }

    cmd.stdout(Stdio::piped())
       .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| anyhow!("Failed to start FFmpeg: {}", e))?;
    let mut stdout = child.stdout.take().context("Failed to take FFmpeg stdout")?;
    let stderr = child.stderr.take().context("Failed to take FFmpeg stderr")?;

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

    // 6. 建立带背压的通信通道 (最多缓冲 10 个数据块，即 10 秒音频)
    let (tx, rx) = sync_channel::<Vec<f32>>(10);

    let pump_thread = thread::spawn(move || -> Result<()> {
        let mut temp_buf = [0u8; 4096];
        loop {
            let mut chunk_bytes = Vec::with_capacity(192000);
            while chunk_bytes.len() < 192000 {
                let to_read = std::cmp::min(temp_buf.len(), 192000 - chunk_bytes.len());
                match stdout.read(&mut temp_buf[..to_read]) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        chunk_bytes.extend_from_slice(&temp_buf[..n]);
                    }
                    Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                    Err(e) => return Err(e.into()),
                }
            }

            if chunk_bytes.is_empty() {
                break;
            }

            let n_samples = chunk_bytes.len() / 4;
            let mut samples = vec![0.0f32; n_samples];
            for (i, bytes) in chunk_bytes.chunks_exact(4).enumerate() {
                samples[i] = f32::from_le_bytes(bytes.try_into().unwrap());
            }

            if tx.send(samples).is_err() {
                break;
            }
        }
        Ok(())
    });

    // 7. 解耦 CPU(降噪)、CPU(重采样与 VAD) 与 GPU(推理) 为三级并行流水线
    
    // 定义在 GPU 线程中执行的推理任务
    struct WhisperTask {
        samples: Vec<f32>,
        start_ms: i64,
        end_ms: i64,
    }

    // 管道 2: DFN3 降噪完毕 (48kHz) -> 重采样器 (容量10)
    let (tx_clean_48k, rx_clean_48k) = sync_channel::<Vec<f32>>(10);
    
    // 管道 3: VAD 切割完毕 (16kHz完整句子) -> Whisper GPU (容量10)
    let (whisper_tx, whisper_rx) = sync_channel::<WhisperTask>(10);
    let sink_for_whisper = sink.clone();

    // ==== 线程 C: Whisper 纯 GPU 推理线程 ====
    let whisper_thread = thread::spawn(move || -> Result<()> {
        #[cfg(target_os = "windows")]
        register_thread_as_pro_audio();

        let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
        
        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        let n_threads = threads.unwrap_or(4);
        params.set_n_threads(n_threads);
        params.set_translate(translate);
        if let Some(ref lang) = language {
            if lang != "auto" && !lang.is_empty() { params.set_language(Some(lang.as_str())); } 
            else { params.set_language(None); params.set_detect_language(false); }
        } else {
            params.set_language(None); params.set_detect_language(false);
        }
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);
        params.set_no_speech_thold(0.6);
        params.set_single_segment(false);

        while let Ok(task) = whisper_rx.recv() {
            let WhisperTask { samples: mut samples_16k, start_ms, end_ms } = task;
            
            if !samples_16k.is_empty() {
                // 强行对超短音频使用静音填充以保护 DTW 机制
                const MIN_SAMPLES_FOR_DTW: usize = 8000; // 16kHz 下 0.5 秒 = 8000 个采样点
                if samples_16k.len() < MIN_SAMPLES_FOR_DTW {
                    println!("[Rust] ⚠️ 拦截到超短音频 ({} samples), 已自动静音填充至 0.5s 以保护 DTW。", samples_16k.len());
                    samples_16k.resize(MIN_SAMPLES_FOR_DTW, 0.0);
                }

                let mut current_params = params.clone();
                current_params.set_no_context(false);

                println!("[Rust] Running Whisper inference on segment: [{:.2}s - {:.2}s], samples: {}", 
                    start_ms as f64 / 1000.0, end_ms as f64 / 1000.0, samples_16k.len());

                whisper_state.full(current_params, &samples_16k)
                    .map_err(|e| anyhow!("Whisper inference error: {:?}", e))?;

                let n_segments = whisper_state.full_n_segments();
                for i in 0..n_segments {
                    if let Some(segment) = whisper_state.get_segment(i) {
                        let text = segment.to_str_lossy().unwrap_or_default().trim().to_string();
                        if text.is_empty() {
                            continue;
                        }

                        // 依据 Whisper 内部判定时间戳与当前 VAD 分段起始偏移，计算出准确的绝对时间
                        let sub_start_ms = start_ms + segment.start_timestamp() * 10;
                        let sub_end_ms = start_ms + segment.end_timestamp() * 10;

                        let mut final_text = text;
                        if to_simplified {
                            final_text = convert_chinese(final_text, true);
                        }

                        if !final_text.is_empty() {
                            let new_seg = TranscriptionSegment {
                                start_ms: sub_start_ms,
                                end_ms: sub_end_ms,
                                text: final_text,
                            };
                            all_segments.push(new_seg.clone());
                            let _ = sink_for_whisper.add(TranscriptionEvent::Segment(new_seg));
                        }
                    }
                }
            }

            // 无论是否跳过推理，都必须推进进度条，让 UI 保持流畅响应！
            let progress = ((end_ms as f64 / 1000.0) / total_duration * 100.0) as i32;
            let _ = sink_for_whisper.add(TranscriptionEvent::Progress(progress.clamp(0, 100)));
            let _ = sink_for_whisper.add(TranscriptionEvent::ProgressDetail {
                processed_ms: end_ms,
                total_ms: (total_duration * 1000.0) as i64,
            });
        }
        
        let _ = sink_for_whisper.add(TranscriptionEvent::Success(all_segments));
        Ok(())
    });

    // ==== 核心线程 1: DFN3 专属降噪线程 ====
    let extracted_dir_clone = extracted_dir.clone();
    let dfn_thread = thread::spawn(move || -> Result<()> {
        if !enable_denoise {
            println!("[Rust] DeepFilterNet3 降噪已禁用，音频流直通处理。");
            let tx_clean_48k_clone = tx_clean_48k.clone();
            let mut buffer = Vec::with_capacity(chunk_size_48k);
            while let Ok(raw_chunk_48k) = rx.recv() {
                buffer.extend_from_slice(&raw_chunk_48k);
                while buffer.len() >= chunk_size_48k {
                    let chunk: Vec<f32> = buffer.drain(..chunk_size_48k).collect();
                    if tx_clean_48k_clone.send(chunk).is_err() {
                        return Ok(());
                    }
                }
            }
            if !buffer.is_empty() {
                buffer.resize(chunk_size_48k, 0.0);
                let _ = tx_clean_48k_clone.send(buffer);
            }
            drop(tx_clean_48k_clone);
            return Ok(());
        }

        struct DfnTask {
            seq_id: usize,
            warmup_samples: Vec<f32>,
            real_samples: Vec<f32>,
        }
        
        struct DfnResult {
            seq_id: usize,
            cleaned_samples: Vec<f32>,
        }

        // 用 crossbeam_channel 来实现多生产者多消费者的高效通道
        let (task_tx, task_rx) = crossbeam_channel::bounded::<DfnTask>(16);
        let (result_tx, result_rx) = crossbeam_channel::unbounded::<DfnResult>();

        let num_workers = 6;
        let mut worker_handles = Vec::with_capacity(num_workers);

        // 使用 Arc 来让所有工作线程免拷贝共享模型目录
        let model_dir_arc = Arc::new(extracted_dir_clone);

        let p_core_mask = get_physical_pcore_mask();

        // 启动并发工作池线程
        for worker_id in 0..num_workers {
            let rx = task_rx.clone();
            let tx = result_tx.clone();
            let model_dir = Arc::clone(&model_dir_arc);
            let mask = p_core_mask;

            let handle = thread::spawn(move || -> Result<()> {
                #[cfg(target_os = "windows")]
                {
                    set_thread_affinity_mask(mask);
                    register_thread_as_pro_audio();
                }

                // 独占一个模型推理流，配置 1 线程以达到极佳流式推理速度
                let mut stream = DeepFilterStream::with_threads(&model_dir, 1)
                    .map_err(|e| anyhow!("Worker {} failed to create DeepFilterStream: {:?}", worker_id, e))?;
                stream.warmup()
                    .map_err(|e| anyhow!("Worker {} failed to warmup: {:?}", worker_id, e))?;

                while let Ok(task) = rx.recv() {
                    let DfnTask { seq_id, warmup_samples, real_samples } = task;
                    let mut cleaned_samples = Vec::with_capacity(real_samples.len());

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

                    if tx.send(DfnResult { seq_id, cleaned_samples }).is_err() {
                        break;
                    }
                }
                Ok(())
            });
            worker_handles.push(handle);
        }

        // 把 result_tx 丢掉，这样在所有 worker 退出后，result_rx.recv() 能自动返回 EOF
        drop(result_tx);

        // 启动汇总重排线程 (Reorder Buffer)
        let tx_clean_48k_clone = tx_clean_48k.clone();
        let collector_thread = thread::spawn(move || {
            let mut reorder_map = BTreeMap::new();
            let mut next_expected_id = 0;

            while let Ok(result) = result_rx.recv() {
                reorder_map.insert(result.seq_id, result.cleaned_samples);

                // 有序释放连续分片数据
                while let Some(entry) = reorder_map.first_entry() {
                    if *entry.key() == next_expected_id {
                        let samples = entry.remove();
                        let mut failed = false;
                        for chunk in samples.chunks(chunk_size_48k) {
                            let mut chunk_vec = chunk.to_vec();
                            if chunk_vec.len() < chunk_size_48k {
                                chunk_vec.resize(chunk_size_48k, 0.0);
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

        // ======= 主调度器逻辑 (Scheduler) =======
        // 积攒音频：分块设定为 10 秒（480,000 个采样点）
        let block_size = 480000;
        let warmup_size = 48000; // 预热长度为 1 秒

        let mut current_block = Vec::with_capacity(block_size);
        let mut history_1s = Vec::with_capacity(warmup_size);
        let mut seq_id = 0;

        while let Ok(raw_chunk_48k) = rx.recv() {
            current_block.extend_from_slice(&raw_chunk_48k);

            while current_block.len() >= block_size {
                let real_samples: Vec<f32> = current_block.drain(..block_size).collect();

                let warmup_samples = if seq_id == 0 {
                    Vec::new()
                } else {
                    history_1s.clone()
                };

                // 提取本次分片结尾的 1 秒数据，作为下一次分片的预热源
                history_1s = real_samples[real_samples.len() - warmup_size..].to_vec();

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

        // 处理 EOF 残留音频数据
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

        // 分发完所有任务后，主动关闭任务通道以让工作线程读取退出
        drop(task_tx);

        // 等待所有工作线程完成
        for handle in worker_handles {
            if let Err(e) = handle.join() {
                println!("[Rust] DFN3 Worker thread panicked: {:?}", e);
            }
        }

        // 等待 Collector 重排收集完毕
        let _ = collector_thread.join();

        drop(tx_clean_48k);
        Ok(())
    });

    // ==== 核心线程 2: 重采样与 VAD 寻峰线程 ====
    let vad_model_path_clone = vad_model_path.clone();
    let vad_thread = thread::spawn(move || -> Result<()> {
        #[cfg(target_os = "windows")]
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
            chunk_size_48k,
            1,
        ).map_err(|e| anyhow!("Failed to initialize Rubato resampler: {:?}", e))?;

        let mut audio_buffer: Vec<f32> = Vec::with_capacity(30 * 16000);
        let mut current_offset_ms: i64 = 0;

        let mut vad = if vad_enabled {
            println!("[Rust] 正在为流式处理初始化 Silero VAD 模型: {}", vad_model_path_clone);
            let vad_ctx_params = WhisperVadContextParams::new();
            match WhisperVadContext::new(&vad_model_path_clone, vad_ctx_params) {
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

        while let Ok(clean_48k_batch) = rx_clean_48k.recv() {
            let resampled = match resampler.process(&[&clean_48k_batch], None) {
                Ok(r) => r,
                Err(e) => {
                    println!("[Rust] 重采样出错: {:?}", e);
                    break;
                }
            };
            let clean_block_16k = &resampled[0];
            
            if clean_block_16k.is_empty() {
                continue;
            }

            audio_buffer.extend_from_slice(clean_block_16k);

            let mut check_and_cut = true;
            while check_and_cut {
                check_and_cut = false;

                // 30 秒红线强制截断，防止 Whisper 溢出或幻觉
                if audio_buffer.len() >= 480000 {
                    println!("[Rust] VAD: audio_buffer 达到 30s 红线，强制截断。");
                    let segment_samples: Vec<f32> = audio_buffer.drain(..480000).collect();
                    let seg_duration_ms = 30000;
                    let start_ms = current_offset_ms;
                    let end_ms = current_offset_ms + seg_duration_ms;
                    current_offset_ms = end_ms;

                    if whisper_tx.send(WhisperTask { samples: segment_samples, start_ms, end_ms }).is_err() {
                        break;
                    }
                    check_and_cut = true;
                    continue;
                }

                if let Some(ref mut vad_ctx) = vad {
                    match vad_ctx.segments_from_samples(vad_params.clone(), &audio_buffer) {
                        Ok(segs) => {
                            // 寻找完整的语音段：如果该段的 end_sample 后面有至少 500ms（即 8000 采样点）的静音，
                            // 则切出并发送给推理线程
                            let mut cut_point = None;
                            for s in segs {
                                let end_sample = s.end as usize * 160;
                                if audio_buffer.len() >= end_sample + 8000 {
                                    // 切出点设为 end_sample 加上 4000 个采样点 (250ms 尾部静音)，以保证段落自然过渡
                                    cut_point = Some(end_sample + 4000);
                                    break;
                                }
                            }

                            if let Some(cut_idx) = cut_point {
                                let cut_idx = cut_idx.min(audio_buffer.len());
                                let segment_samples: Vec<f32> = audio_buffer.drain(..cut_idx).collect();
                                let seg_duration_ms = (segment_samples.len() as f64 / 16000.0 * 1000.0) as i64;
                                let start_ms = current_offset_ms;
                                let end_ms = current_offset_ms + seg_duration_ms;
                                current_offset_ms = end_ms;

                                if whisper_tx.send(WhisperTask { samples: segment_samples, start_ms, end_ms }).is_err() {
                                    break;
                                }
                                check_and_cut = true;
                            }
                        }
                        Err(e) => {
                            println!("[Rust] Silero VAD segments_from_samples 出错: {:?}", e);
                        }
                    }
                }
            }

            // 如果 VAD 未启用，则按 10 秒固定长度截断
            if !vad_enabled {
                while audio_buffer.len() >= 160000 {
                    let segment_samples: Vec<f32> = audio_buffer.drain(..160000).collect();
                    let seg_duration_ms = 10000;
                    let start_ms = current_offset_ms;
                    let end_ms = current_offset_ms + seg_duration_ms;
                    current_offset_ms = end_ms;

                    if whisper_tx.send(WhisperTask { samples: segment_samples, start_ms, end_ms }).is_err() {
                        break;
                    }
                }
            }
        }

        // 处理 EOF 残余音频
        if !audio_buffer.is_empty() {
            let seg_duration_ms = (audio_buffer.len() as f64 / 16000.0 * 1000.0) as i64;
            let start_ms = current_offset_ms;
            let end_ms = current_offset_ms + seg_duration_ms;
            
            // 直接发送
            let _ = whisper_tx.send(WhisperTask { samples: audio_buffer, start_ms, end_ms });
        }

        drop(whisper_tx);
        Ok(())
    });

    // 9. Shutdown & Wait
    let _ = pump_thread.join();
    let ffmpeg_status = child.wait().unwrap();
    let stderr_logs = stderr_thread.join().unwrap();
    let _ = dfn_thread.join().map_err(|_| anyhow!("DFN3 thread panicked"))??;
    let _ = vad_thread.join().map_err(|_| anyhow!("VAD thread panicked"))??;
    let _ = whisper_thread.join().map_err(|_| anyhow!("Failed to join Whisper GPU thread"))??;

    if !ffmpeg_status.success() {
        return Err(anyhow!("FFmpeg exited with error: {:?}\nStderr logs:\n{}", ffmpeg_status.code(), stderr_logs));
    }
    
    Ok(())
}


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
        // 强制锁定进程为“高优先级”
        // 哪怕软件最小化到系统托盘，Windows 也绝不敢把这些线程扔进 E-Core
        SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    }
    // println!("[Rust] Process priority locked to HIGH_PRIORITY_CLASS.");
}

#[cfg(target_os = "windows")]
fn register_thread_as_pro_audio() {
    use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
    use windows_sys::core::PCWSTR;

    unsafe {
        // 告诉 Windows 应当用最高调度质量
        let task_name: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
        let mut task_index = 0;
        
        let handle = AvSetMmThreadCharacteristicsW(
            task_name.as_ptr() as PCWSTR, 
            &mut task_index
        );

        if handle.is_null() {
            // println!("[Rust] MMCSS 注册失败，退回普通调度");
        } else {
            // println!("[Rust] 线程成功接入 MMCSS 绿色通道！");
        }
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
            Version: 1, // PROCESS_POWER_THROTTLING_CURRENT_VERSION
            ControlMask: PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            StateMask: 0, // 0 代表关闭节流 (如果是执行速度控制的话)
        };

        SetProcessInformation(
            GetCurrentProcess(),
            ProcessPowerThrottling,
            &mut state as *mut _ as *mut _,
            size_of::<PROCESS_POWER_THROTTLING_STATE>() as u32,
        );
        // println!("[Rust] Windows 11 电源节流 (EcoQoS) 已被禁用，后台满血运行。");
    }
}

fn get_physical_pcore_mask() -> usize {
    let core_ids = core_affinity::get_core_ids().unwrap_or_default();
    let mut mask = 0_usize;
    for core in core_ids {
        if core.id % 2 == 0 {
            // 使用 checked_shl 防止在 64+ 核 CPU 上位移溢出崩溃
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

fn get_dtw_model_preset(model_path: &str) -> Option<whisper_rs::DtwModelPreset> {
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("medium.en") {
        Some(whisper_rs::DtwModelPreset::MediumEn)
    } else if path_lower.contains("medium") {
        Some(whisper_rs::DtwModelPreset::Medium)
    } else if path_lower.contains("large-v3-turbo") {
        Some(whisper_rs::DtwModelPreset::LargeV3Turbo)
    } else if path_lower.contains("large") {
        Some(whisper_rs::DtwModelPreset::LargeV3)
    } else {
        // Disabling DTW for tiny, base, and small models to prevent median filter width ne[2] assertion crash.
        None
    }
}
