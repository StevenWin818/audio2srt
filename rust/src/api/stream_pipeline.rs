use std::process::{Command, Stdio};
use std::io::{Read, BufRead, BufReader};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::path::Path;
use anyhow::{Result, Context, anyhow};
use flate2::read::GzDecoder;
use tar::Archive;
use deepfilter_rt::DeepFilterStream;
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContextParameters};
use crate::frb_generated::StreamSink;
use crate::api::whisper::{
    TranscriptionSegment, TranscriptionEvent, get_or_create_context,
    has_repetition_loop, deduplicate_repeats, convert_chinese,
};

#[derive(Clone, Debug)]
pub struct AudioFrameState {
    pub rms: f32,
    pub is_silent: bool,
    pub samples_count: usize,
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
    df_model_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    to_simplified: bool,
) {
    let sink_clone = sink.clone();
    thread::spawn(move || {
        if let Err(e) = run_stream_pipeline_inner(
            &sink_clone,
            ffmpeg_path,
            input_path,
            model_path,
            df_model_path,
            language,
            translate,
            threads,
            use_gpu,
            to_simplified,
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
    df_model_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    to_simplified: bool,
) -> Result<()> {
    // 1. 获取视频总时长
    let total_duration = match get_media_duration_secs(&ffmpeg_path, &input_path) {
        Ok(d) => d,
        Err(e) => {
            println!("[Rust] Warning: Failed to get duration: {}. Defaulting to 1.0", e);
            1.0
        }
    };
    println!("[Rust] Media total duration: {} seconds", total_duration);

    let chunk_size_48k = 24000;

    // 提前在主流程中解压并准备好模型目录，避免在专属计算线程中执行重度 I/O 操作
    println!("[Rust] Preparing DeepFilterNet3 models from tar.gz...");
    let extracted_dir = extract_tar_gz_if_needed(Path::new(&df_model_path))?;
    println!("[Rust] DeepFilterStream prepared at: {:?}", extracted_dir);

    // 4. 初始化 Whisper 上下文
    println!("[Rust] Loading Whisper model context...");
    let mut ctx_params = WhisperContextParameters::default();
    
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
    }
    println!("[Rust] Hardware device for Whisper: {}", selected_device_name);

    let ctx = get_or_create_context(&model_path, use_gpu, ctx_params)
        .map_err(|e| anyhow!("Failed to load Whisper model: {}", e))?;
    
    let mut whisper_state = ctx.create_state()
        .map_err(|e| anyhow!("Failed to create Whisper state: {}", e))?;

    // 5. 启动 FFmpeg 流提取进程
    println!("[Rust] Starting FFmpeg audio pump process...");
    let mut cmd = Command::new(&ffmpeg_path);
    cmd.arg("-y")
       .arg("-i")
       .arg(&input_path)
       .arg("-vn")
       .arg("-sn")
       .arg("-dn")
       .arg("-f")
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

    // 6. 建立带背压的通信通道 (最多缓冲 10 个数据块，即 5 秒音频)
    let (tx, rx) = sync_channel::<Vec<f32>>(10);

    let pump_thread = thread::spawn(move || -> Result<()> {
        let mut temp_buf = [0u8; 4096];
        loop {
            let mut chunk_bytes = Vec::with_capacity(96000);
            while chunk_bytes.len() < 96000 {
                let to_read = std::cmp::min(temp_buf.len(), 96000 - chunk_bytes.len());
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
        let mut rolling_prompt = String::new();
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

        while let Ok(task) = whisper_rx.recv() {
            let WhisperTask { samples: samples_16k, start_ms, end_ms } = task;
            
            let mut current_params = params.clone();
            current_params.set_no_context(true);
            
            if !rolling_prompt.is_empty() {
                current_params.set_initial_prompt(&rolling_prompt);
            }

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

                    let mut final_text = deduplicate_repeats(&text);
                    if to_simplified {
                        final_text = convert_chinese(final_text, true);
                    }

                    if !final_text.is_empty() {
                        if has_repetition_loop(&final_text) {
                            println!("[Rust] 检测到子句幻觉循环: '{}'。清理滑动提示词。", final_text);
                            rolling_prompt.clear();
                            final_text = deduplicate_repeats(&final_text);
                        } else {
                            rolling_prompt.push_str(&final_text);
                            let char_vec: Vec<char> = rolling_prompt.chars().collect();
                            if char_vec.len() > 100 {
                                rolling_prompt = char_vec[char_vec.len() - 100 ..].iter().collect();
                            }
                        }

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

            let progress = ((end_ms as f64 / 1000.0) / total_duration * 100.0) as i32;
            let _ = sink_for_whisper.add(TranscriptionEvent::Progress(progress.clamp(0, 100)));
        }
        
        let _ = sink_for_whisper.add(TranscriptionEvent::Success(all_segments));
        Ok(())
    });

    // ==== 核心线程 1: DFN3 专属降噪线程 ====
    let extracted_dir_clone = extracted_dir.clone();
    let dfn_thread = thread::spawn(move || -> Result<()> {
        println!("[Rust] Initializing DeepFilterStream in thread from: {:?}", extracted_dir_clone);
        
        // 对于极小矩阵流式推理，关闭多线程，消除线程锁与调度开销
        let mut stream = DeepFilterStream::with_threads(&extracted_dir_clone, 1)
            .map_err(|e| anyhow!("Failed to create DeepFilterStream: {:?}", e))?;
        stream.warmup()
            .map_err(|e| anyhow!("Failed to warmup DeepFilterStream: {:?}", e))?;
        println!("[Rust] DFN3 ONNX Runtime (ort) initialized with 1 thread.");

        let mut raw_48k_buf: Vec<f32> = Vec::with_capacity(chunk_size_48k * 8);
        let mut read_pos = 0;

        while let Ok(raw_chunk_48k) = rx.recv() {
            raw_48k_buf.extend_from_slice(&raw_chunk_48k);

            while raw_48k_buf.len() - read_pos >= chunk_size_48k {
                // 性能优化：直接使用切片引用传入流式推理，零 collect()，零 heap 分配
                let process_block = &raw_48k_buf[read_pos..read_pos + chunk_size_48k];
                let clean_48k_batch = stream.process(process_block)
                    .map_err(|e| anyhow!("DFN3 processing error: {:?}", e))?;
                
                if tx_clean_48k.send(clean_48k_batch).is_err() {
                    break;
                }
                read_pos += chunk_size_48k;
            }

            // 定期清理已消耗的数据，避免内存无限累加，同时极低频次的前移整理开销可忽略
            if read_pos >= chunk_size_48k * 4 {
                raw_48k_buf.drain(..read_pos);
                read_pos = 0;
            }
        }

        // 处理 EOF 残余数据
        let remaining = &raw_48k_buf[read_pos..];
        if !remaining.is_empty() {
            let mut remainder_buf = remaining.to_vec();
            let remainder = remainder_buf.len() % 480;
            if remainder > 0 {
                let padding = 480 - remainder;
                remainder_buf.extend(std::iter::repeat(0.0f32).take(padding));
            }
            let clean_48k_batch = stream.process(&remainder_buf)
                .map_err(|e| anyhow!("DFN3 EOF processing error: {:?}", e))?;
            if !clean_48k_batch.is_empty() {
                let _ = tx_clean_48k.send(clean_48k_batch);
            }
        }

        // 冲刷流缓存
        let flushed = stream.flush()
            .map_err(|e| anyhow!("DFN3 flush error: {:?}", e))?;
        if !flushed.is_empty() {
            let _ = tx_clean_48k.send(flushed);
        }

        drop(tx_clean_48k);
        Ok(())
    });

    // ==== 核心线程 2: 重采样与 VAD 寻峰线程 ====
    let vad_thread = thread::spawn(move || -> Result<()> {
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
        let mut frame_states: Vec<AudioFrameState> = Vec::with_capacity(1500);
        let mut current_offset_ms: i64 = 0;

        while let Ok(clean_48k_batch) = rx_clean_48k.recv() {
            let resampled = match resampler.process(&[&clean_48k_batch], None) {
                Ok(r) => r,
                Err(e) => {
                    println!("[Rust] Resampler error: {:?}", e);
                    break;
                }
            };
            let clean_block_16k = &resampled[0];
            
            if clean_block_16k.is_empty() {
                continue;
            }

            audio_buffer.extend_from_slice(clean_block_16k);
            for frame_16k in clean_block_16k.chunks(320) {
                let sum_sq: f32 = frame_16k.iter().map(|&s| s * s).sum();
                let rms = (sum_sq / frame_16k.len() as f32).sqrt();
                frame_states.push(AudioFrameState {
                    rms,
                    is_silent: rms < 0.01,
                    samples_count: frame_16k.len(),
                });
            }

            let mut check_and_cut = true;
            while check_and_cut {
                check_and_cut = false;
                let current_frames = frame_states.len();
                let current_secs = current_frames as f32 * 0.02;

                if current_frames < 500 {
                    break;
                }

                let mut cut_frame_idx = None;

                if current_secs <= 20.0 {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 500, 40) {
                        cut_frame_idx = Some(start + (end - start) / 2);
                    }
                } else if current_secs <= 28.0 {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 500, 15) {
                        cut_frame_idx = Some(start + (end - start) / 2);
                    }
                } else {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 500, 15) {
                        cut_frame_idx = Some(start + (end - start) / 2);
                    } else if current_frames >= 1400 {
                        let search_range_start = current_frames - 250;
                        let mut min_rms = f32::MAX;
                        let mut min_idx = search_range_start;
                        for idx in search_range_start..current_frames {
                            if frame_states[idx].rms < min_rms {
                                min_rms = frame_states[idx].rms;
                                min_idx = idx;
                            }
                        }
                        cut_frame_idx = Some(min_idx);
                        println!("[Rust] Redline force cut at frame {} (rms: {:.4})", min_idx, min_rms);
                    }
                }

                if let Some(cut_idx) = cut_frame_idx {
                    let cut_sample_idx: usize = frame_states[..cut_idx].iter().map(|f| f.samples_count).sum();
                    let segment_samples: Vec<f32> = audio_buffer.drain(..cut_sample_idx).collect();
                    frame_states.drain(..cut_idx);

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
        }

        if !audio_buffer.is_empty() {
            let seg_duration_ms = (audio_buffer.len() as f64 / 16000.0 * 1000.0) as i64;
            let start_ms = current_offset_ms;
            let end_ms = current_offset_ms + seg_duration_ms;
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

fn find_silence_sequence(states: &[AudioFrameState], search_start: usize, min_len: usize) -> Option<(usize, usize)> {
    let mut consecutive = 0;
    let mut start_idx = 0;
    for i in search_start..states.len() {
        if states[i].is_silent {
            if consecutive == 0 {
                start_idx = i;
            }
            consecutive += 1;
            if consecutive >= min_len {
                let mut end_idx = i;
                for j in (i + 1)..states.len() {
                    if !states[j].is_silent {
                        break;
                    }
                    end_idx = j;
                }
                return Some((start_idx, end_idx));
            }
        } else {
            consecutive = 0;
        }
    }
    None
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
