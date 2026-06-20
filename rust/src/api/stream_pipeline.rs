use std::process::{Command, Stdio};
use std::io::{Read, BufRead, BufReader};
use std::sync::mpsc::sync_channel;
use std::thread;
use std::path::Path;
use std::sync::Arc;
use anyhow::{Result, Context, anyhow};
use ndarray::ArrayView2;
use deep_filter::tract::{DfParams, DfTract, RuntimeParams};
use rubato::{Resampler, SincFixedIn, SincInterpolationType, SincInterpolationParameters, WindowFunction};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};
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

    // 2. 初始化 DFN3 模型
    println!("[Rust] Initializing DeepFilterNet3 from: {}", df_model_path);
    let df_params = DfParams::new(Path::new(&df_model_path).to_path_buf())
        .map_err(|e| anyhow!("Failed to load DeepFilterNet3 model parameters: {}", e))?;
    let mut df_tract = DfTract::new(df_params, &RuntimeParams::default())
        .map_err(|e| anyhow!("Failed to create DfTract instance: {}", e))?;
    let hop_size = df_tract.hop_size; // typically 960 (20ms at 48kHz)
    println!("[Rust] DFN3 initialized. Hop size: {}", hop_size);

    // 3. 初始化 rubato 重采样器 (48kHz -> 16kHz)
    let resampler_params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };
    let mut resampler = SincFixedIn::<f32>::new(
        16000_f64 / 48000_f64,
        2.0,
        resampler_params,
        hop_size,
        1,
    ).map_err(|e| anyhow!("Failed to initialize Rubato resampler: {:?}", e))?;

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

    let pump_tx = tx.clone();
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

            if pump_tx.send(samples).is_err() {
                break;
            }
        }
        Ok(())
    });

    // 7. 主管道循环
    let mut audio_buffer: Vec<f32> = Vec::new();
    let mut frame_states: Vec<AudioFrameState> = Vec::new();
    let mut raw_48k_buf: Vec<f32> = Vec::new();
    
    let mut rolling_prompt = String::new();
    let mut all_segments: Vec<TranscriptionSegment> = Vec::new();
    let mut current_offset_ms: i64 = 0;

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

    let mut run_whisper_on_segment = |samples_16k: &[f32], start_ms: i64, end_ms: i64| -> Result<()> {
        if samples_16k.is_empty() {
            return Ok(());
        }

        let mut current_params = params.clone();
        if !rolling_prompt.is_empty() {
            current_params.set_initial_prompt(&rolling_prompt);
            current_params.set_no_context(false);
        } else {
            current_params.set_no_context(true);
        }

        println!("[Rust] Running Whisper inference on segment: [{:.2}s - {:.2}s], samples: {}", 
            start_ms as f64 / 1000.0, end_ms as f64 / 1000.0, samples_16k.len());

        whisper_state.full(current_params, samples_16k)
            .map_err(|e| anyhow!("Whisper inference error: {:?}", e))?;

        let n_segments = whisper_state.full_n_segments();

        let mut segment_text = String::new();
        for i in 0..n_segments {
            if let Some(segment) = whisper_state.get_segment(i) {
                let text = segment
                    .to_str_lossy()
                    .unwrap_or_else(|_| std::borrow::Cow::Borrowed(""))
                    .into_owned();
                segment_text.push_str(&text);
            }
        }

        let cleaned_text = segment_text.trim().to_string();
        if !cleaned_text.is_empty() {
            let mut final_text = deduplicate_repeats(&cleaned_text);
            if to_simplified {
                final_text = convert_chinese(final_text, true);
            }

            if !final_text.is_empty() {
                if has_repetition_loop(&final_text) {
                    println!("[Rust] Hallucination loop detected in: '{}'. Clearing rolling prompt.", final_text);
                    rolling_prompt.clear();
                    final_text = deduplicate_repeats(&final_text);
                } else {
                    rolling_prompt = final_text.clone();
                    if rolling_prompt.chars().count() > 100 {
                        let char_vec: Vec<char> = rolling_prompt.chars().collect();
                        rolling_prompt = char_vec[char_vec.len() - 100 ..].iter().collect();
                    }
                }

                let new_seg = TranscriptionSegment {
                    start_ms,
                    end_ms,
                    text: final_text,
                };
                all_segments.push(new_seg.clone());
                let _ = sink.add(TranscriptionEvent::Segment(new_seg));
            }
        }

        let progress = ((end_ms as f64 / 1000.0) / total_duration * 100.0) as i32;
        let progress = progress.clamp(0, 100);
        let _ = sink.add(TranscriptionEvent::Progress(progress));

        Ok(())
    };

    while let Ok(raw_chunk_48k) = rx.recv() {
        raw_48k_buf.extend_from_slice(&raw_chunk_48k);

        while raw_48k_buf.len() >= hop_size {
            let frame_48k: Vec<f32> = raw_48k_buf.drain(..hop_size).collect();
            
            // A. DFN3 Denoise
            let noisy_arr = ArrayView2::from_shape((1, hop_size), &frame_48k)
                .map_err(|e| anyhow!("Failed to create ndarray view: {}", e))?;
            let mut clean_frame_48k = vec![0.0f32; hop_size];
            {
                let mut clean_arr = ndarray::ArrayViewMut2::from_shape((1, hop_size), &mut clean_frame_48k)
                    .map_err(|e| anyhow!("Failed to create ndarray mut view: {}", e))?;
                df_tract.process(noisy_arr, clean_arr)
                    .map_err(|e| anyhow!("DFN3 process error: {:?}", e))?;
            }

            // B. Rubato Resample (48kHz -> 16kHz)
            let resampled = resampler.process(&[clean_frame_48k], None)
                .map_err(|e| anyhow!("Resampling error: {:?}", e))?;
            let clean_frame_16k = &resampled[0];

            if clean_frame_16k.is_empty() {
                continue;
            }

            // C. VAD Update
            audio_buffer.extend_from_slice(clean_frame_16k);

            let sum_sq: f32 = clean_frame_16k.iter().map(|&s| s * s).sum();
            let rms = (sum_sq / clean_frame_16k.len() as f32).sqrt();
            let is_silent = rms < 0.01;
            frame_states.push(AudioFrameState { rms, is_silent, samples_count: clean_frame_16k.len() });

            let mut check_and_cut = true;
            while check_and_cut {
                check_and_cut = false;
                let current_frames = frame_states.len();
                let current_secs = current_frames as f32 * 0.02;

                if current_frames < 150 {
                    break;
                }

                let mut cut_frame_idx = None;

                if current_secs <= 15.0 {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 150, 40) {
                        cut_frame_idx = Some(start + (end - start) / 2);
                    }
                } else if current_secs <= 25.0 {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 150, 15) {
                        cut_frame_idx = Some(start + (end - start) / 2);
                    }
                } else {
                    if let Some((start, end)) = find_silence_sequence(&frame_states, 150, 15) {
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

                    run_whisper_on_segment(&segment_samples, start_ms, end_ms)?;
                    check_and_cut = true;
                }
            }
        }
    }

    // 8. EOF Residuals
    if !raw_48k_buf.is_empty() {
        let padding_needed = hop_size - raw_48k_buf.len();
        raw_48k_buf.extend(std::iter::repeat(0.0f32).take(padding_needed));
        
        let noisy_arr = ArrayView2::from_shape((1, hop_size), &raw_48k_buf)
            .map_err(|e| anyhow!("Failed to create ndarray view: {}", e))?;
        let mut clean_frame_48k = vec![0.0f32; hop_size];
        {
            let mut clean_arr = ndarray::ArrayViewMut2::from_shape((1, hop_size), &mut clean_frame_48k)
                .map_err(|e| anyhow!("Failed to create ndarray mut view: {}", e))?;
            df_tract.process(noisy_arr, clean_arr)
                .map_err(|e| anyhow!("DFN3 process error: {:?}", e))?;
        }

        let resampled = resampler.process(&[clean_frame_48k], None)
            .map_err(|e| anyhow!("Resampling error: {:?}", e))?;
        let clean_frame_16k = &resampled[0];
        
        audio_buffer.extend_from_slice(clean_frame_16k);
    }

    if !audio_buffer.is_empty() {
        let seg_duration_ms = (audio_buffer.len() as f64 / 16000.0 * 1000.0) as i64;
        let start_ms = current_offset_ms;
        let end_ms = current_offset_ms + seg_duration_ms;
        run_whisper_on_segment(&audio_buffer, start_ms, end_ms)?;
    }

    // 9. Shutdown & wait
    let _ = pump_thread.join().map_err(|_| anyhow!("Failed to join pump thread"))?;
    let ffmpeg_status = child.wait().map_err(|e| anyhow!("Failed to wait for FFmpeg: {}", e))?;
    let stderr_logs = stderr_thread.join().map_err(|_| anyhow!("Failed to join stderr thread"))?;

    if !ffmpeg_status.success() {
        return Err(anyhow!("FFmpeg exited with error: {:?}\nStderr logs:\n{}", ffmpeg_status.code(), stderr_logs));
    }

    let _ = sink.add(TranscriptionEvent::Success(all_segments));
    
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
