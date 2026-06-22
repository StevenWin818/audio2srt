use std::process::{Command, Stdio};
use std::io::{BufRead, BufReader};
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;
use crate::frb_generated::StreamSink;

#[derive(Clone)]
pub struct FfmpegEvent {
    pub progress: Option<i32>,
    pub success: Option<String>,
    pub error: Option<String>,
}

fn parse_time_to_secs(time_str: &str) -> Option<f64> {
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

/// 格式化 Windows 路径以适应 FFmpeg 的 subtitles 滤镜
fn format_path_for_filter(path: &str) -> String {
    // 将反斜杠替换为正斜杠，并转义冒号和单引号
    let mut p = path.replace("\\", "/");
    p = p.replace(":", "\\:");
    p = p.replace("'", "\\'");
    p
}

fn run_ffmpeg_with_progress(
    mut cmd: Command,
    sink: StreamSink<FfmpegEvent>,
    output_path: String,
) -> Result<(), String> {
    cmd.stdout(Stdio::null())
       .stderr(Stdio::piped());

    let mut child = cmd.spawn().map_err(|e| format!("无法启动 FFmpeg: {}", e))?;
    let stderr = child.stderr.take().unwrap();
    let reader = BufReader::new(stderr);

    let mut total_duration_secs = 0.0;
    let mut last_progress = -1;

    for line_result in reader.lines() {
        if let Ok(line) = line_result {
            // Duration: 00:01:23.45,
            if line.contains("Duration:") && total_duration_secs == 0.0 {
                if let Some(duration_str) = line.split("Duration:").nth(1) {
                    if let Some(time_str) = duration_str.split(',').next() {
                        if let Some(secs) = parse_time_to_secs(time_str.trim()) {
                            total_duration_secs = secs;
                        }
                    }
                }
            }
            // time=00:00:12.34
            if line.contains("time=") && total_duration_secs > 0.0 {
                if let Some(time_part) = line.split("time=").nth(1) {
                    if let Some(time_str) = time_part.split(' ').next() {
                        if let Some(current_secs) = parse_time_to_secs(time_str.trim()) {
                            let mut progress = ((current_secs / total_duration_secs) * 100.0) as i32;
                            if progress > 100 { progress = 100; }
                            if progress < 0 { progress = 0; }
                            
                            if progress != last_progress {
                                let _ = sink.add(FfmpegEvent { progress: Some(progress), success: None, error: None });
                                last_progress = progress;
                            }
                        }
                    }
                }
            }
        }
    }

    let status = child.wait().map_err(|e| e.to_string())?;
    if status.success() {
        let _ = sink.add(FfmpegEvent { progress: None, success: Some(output_path), error: None });
    } else {
        let _ = sink.add(FfmpegEvent { progress: None, success: None, error: Some("FFmpeg 执行失败".to_string()) });
    }
    
    Ok(())
}

/// 提取音视频文件中的音频并重采样为 16kHz 单声道 PCM WAV 格式
pub fn extract_audio_from_media(
    sink: StreamSink<FfmpegEvent>,
    ffmpeg_path: String,
    input_path: String,
    output_path: String,
) -> Result<(), String> {
    // 动态探测音频流数量
    let mut probe_cmd = Command::new(&ffmpeg_path);
    probe_cmd.arg("-i").arg(&input_path);
    #[cfg(target_os = "windows")]
    probe_cmd.creation_flags(0x08000000); // 隐藏窗口

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
    println!("[Rust] 检测到音频流数量: {}，文件: {}", audio_stream_count, input_path);

    let mut cmd = Command::new(&ffmpeg_path);
    
    cmd.arg("-y")
       .arg("-i")
       .arg(&input_path)
       .arg("-vn") // 忽略视频
       .arg("-sn") // 忽略字幕
       .arg("-dn"); // 忽略数据

    // 如果检测到多音轨，不要混音！明确只提取第一条音频流（Index 为 0）
    // 避免中英双语同时播放导致 Whisper 识别崩溃
    if audio_stream_count > 1 {
        println!("[Rust] 检测到多音轨，放弃混音，默认提取第一条音轨 (0:a:0)");
        cmd.arg("-map")
           .arg("0:a:0"); 
    } else {
        // 如果只有一个音轨，或者没检测出音轨，让 FFmpeg 自动决定默认流
        cmd.arg("-map")
           .arg("0:a?"); // 0:a? 表示尝试映射音频，如果没有也不会报错退出
    }

    cmd.arg("-ar")
       .arg("16000")
       .arg("-ac")
       .arg("1")
       .arg("-c:a")
       .arg("pcm_s16le")
       .arg(&output_path);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    run_ffmpeg_with_progress(cmd, sink, output_path)
}

/// 将 SRT 字幕文件集成到视频中（软封装或硬压制）
pub fn mux_srt_to_video(
    sink: StreamSink<FfmpegEvent>,
    ffmpeg_path: String,
    video_path: String,
    srt_path: String,
    output_path: String,
    hard_burn: bool,
) -> Result<(), String> {
    let mut cmd = Command::new(&ffmpeg_path);
    cmd.arg("-y")
       .arg("-i")
       .arg(&video_path);

    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    if hard_burn {
        let escaped_srt = format_path_for_filter(&srt_path);
        let filter_arg = format!("subtitles='{}'", escaped_srt);
        
        cmd.arg("-vf")
           .arg(filter_arg)
           .arg("-c:a")
           .arg("copy")
           .arg(&output_path);
    } else {
        cmd.arg("-i")
           .arg(&srt_path)
           .arg("-c")
           .arg("copy");

        if output_path.to_lowercase().ends_with(".mp4") {
            cmd.arg("-c:s").arg("mov_text");
        } else {
            cmd.arg("-c:s").arg("srt");
        }
        cmd.arg(&output_path);
    }

    run_ffmpeg_with_progress(cmd, sink, output_path)
}
