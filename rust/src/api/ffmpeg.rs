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

// 辅助函数：根据 ffmpeg 路径推导 ffprobe 路径
fn get_ffprobe_path(ffmpeg_path: &str) -> String {
    let path = std::path::Path::new(ffmpeg_path);
    if let Some(parent) = path.parent() {
        if let Some(file_name) = path.file_name().and_then(|n| n.to_str()) {
            let ffprobe_name = file_name.replace("ffmpeg", "ffprobe");
            let ffprobe_path = parent.join(ffprobe_name);
            if ffprobe_path.exists() {
                return ffprobe_path.to_string_lossy().to_string();
            }
        }
    }
    ffmpeg_path.replace("ffmpeg", "ffprobe")
}

// 音轨详细元数据结构体
#[derive(serde::Serialize, serde::Deserialize, Debug, Clone)]
pub struct AudioTrackInfo {
    pub index: usize,          // FFmpeg 里的音轨相对索引（从0开始）
    pub codec_name: String,    // 编码名称，例如 "aac", "ac3"
    pub language: Option<String>, // 语言，例如 "chi", "eng"
    pub title: Option<String>, // 轨道标题，例如 "Director's Commentary"
}

// ffprobe JSON 输出对应的内部结构体
#[derive(serde::Deserialize, Debug, Clone)]
struct ProbeOutput {
    streams: Option<Vec<ProbeStream>>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ProbeStream {
    // index: usize,
    codec_name: Option<String>,
    tags: Option<ProbeTags>,
}

#[derive(serde::Deserialize, Debug, Clone)]
struct ProbeTags {
    language: Option<String>,
    title: Option<String>,
}

// 暴露给 Dart 的解析函数，通过 ffprobe 探测媒体文件中的音轨列表（若失败则使用 ffmpeg 备选解析）
pub fn probe_audio_tracks(ffmpeg_path: String, file_path: String) -> Vec<AudioTrackInfo> {
    let ffprobe_path = get_ffprobe_path(&ffmpeg_path);
    let mut cmd = Command::new(ffprobe_path);
    cmd.arg("-v").arg("error")
       .arg("-select_streams").arg("a")
       .arg("-show_entries").arg("stream=index,codec_name:stream_tags=language,title")
       .arg("-of").arg("json")
       .arg(&file_path);

    #[cfg(target_os = "windows")]
    {
        cmd.creation_flags(0x08000000); // 隐藏命令行窗口
    }

    let output = match cmd.output() {
        Ok(out) => out,
        Err(e) => {
            println!("[Rust] 启动 ffprobe 失败: {:?}. 正在尝试使用 ffmpeg 作为备选解析器...", e);
            return probe_audio_tracks_fallback(&ffmpeg_path, &file_path);
        }
    };

    let probe_out: ProbeOutput = match serde_json::from_slice(&output.stdout) {
        Ok(parsed) => parsed,
        Err(e) => {
            println!("[Rust] 解析 ffprobe json 输出失败: {:?}. 正在尝试使用 ffmpeg 作为备选解析器...", e);
            return probe_audio_tracks_fallback(&ffmpeg_path, &file_path);
        }
    };

    let mut tracks = vec![];
    if let Some(streams) = probe_out.streams {
        for (i, s) in streams.into_iter().enumerate() {
            let codec_name = s.codec_name.unwrap_or_else(|| "unknown".to_string());
            let (language, title) = if let Some(tags) = s.tags {
                let mapped_lang = tags.language.as_ref().map(|l| convert_iso639_2(l).1);
                (mapped_lang.or(tags.language), tags.title)
            } else {
                (None, None)
            };
            tracks.push(AudioTrackInfo {
                index: i, // 相对音轨索引
                codec_name,
                language,
                title,
            });
        }
    }

    tracks
}

// 备选解析函数：使用 ffmpeg -i 读取 stderr 信息并手动解析音轨
fn probe_audio_tracks_fallback(ffmpeg_path: &str, file_path: &str) -> Vec<AudioTrackInfo> {
    use std::process::Command;
    #[cfg(target_os = "windows")]
    use std::os::windows::process::CommandExt;

    let mut cmd = Command::new(ffmpeg_path);
    // 使用 -hide_banner 让输出更纯粹，去除版权等无用信息
    cmd.arg("-hide_banner").arg("-i").arg(file_path);
    
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // 隐藏黑框

    let output = match cmd.output() {
        Ok(out) => out,
        Err(e) => {
            println!("[Rust-Fallback] 启动 ffmpeg 失败: {:?}", e);
            return vec![];
        }
    };

    let stderr_str = String::from_utf8_lossy(&output.stderr);
    let mut tracks = vec![];
    let mut current_track: Option<AudioTrackInfo> = None;
    let mut relative_audio_index = 0;

    for line in stderr_str.lines() {
        let trimmed = line.trim();
        
        // 发现新的一条音频流
        if trimmed.starts_with("Stream #") && trimmed.contains(": Audio:") {
            // 保存上一条记录
            if let Some(t) = current_track.take() {
                tracks.push(t);
            }

            let mut language = None;
            let mut codec_name = "unknown".to_string();

            // 粗略切分 Stream 前缀部分 和 Audio 后缀部分
            if let Some((stream_prefix, audio_info)) = trimmed.split_once(": Audio:") {
                
                // 解析语言，提取括号里的内容
                if let Some(start_idx) = stream_prefix.find('(') {
                    if let Some(end_idx) = stream_prefix.find(')') {
                        if start_idx < end_idx {
                            language = Some(stream_prefix[start_idx + 1..end_idx].to_string());
                        }
                    }
                }

                // 解析编码，取冒号后的第一个逗号或空格前的词
                let first_word = audio_info.trim().split(|c| c == ',' || c == ' ').next();
                if let Some(codec) = first_word {
                    codec_name = codec.to_string();
                }
            }

            let mapped_lang = language.as_ref().map(|l| convert_iso639_2(l).1);
            current_track = Some(AudioTrackInfo {
                index: relative_audio_index,
                codec_name,
                language: mapped_lang.or(language),
                title: None,
            });
            relative_audio_index += 1;
            
        } 
        // 正在解析属于该音频流的 Metadata
        else if trimmed.starts_with("title") && current_track.is_some() {
            if let Some(ref mut track) = current_track {
                // 使用 split_once 防止截断带冒号的 title
                if let Some((_, title_val)) = trimmed.split_once(':') {
                    track.title = Some(title_val.trim().to_string());
                }
            }
        }
    }

    if let Some(t) = current_track {
        tracks.push(t);
    }

    println!("[Rust-Fallback] 备选解析器探测到 {} 条音轨", tracks.len());
    tracks
}

/// 将 ISO 639-2 (三字母) 转换为 (ISO 639-1 二字母, 语言中文名称)
pub fn convert_iso639_2(code: &str) -> (String, String) {
    match code.to_lowercase().as_str() {
        // 中文 (chi / zho) -> zh
        "chi" | "zho" => ("zh".to_string(), "中文".to_string()),
        // 英文 (eng) -> en
        "eng" => ("en".to_string(), "English".to_string()),
        // 日文 (jpn) -> ja
        "jpn" => ("ja".to_string(), "日本語".to_string()),
        // 韩文 (kor) -> ko
        "kor" => ("ko".to_string(), "한국어".to_string()),
        // 法文 (fre / fra) -> fr
        "fre" | "fra" => ("fr".to_string(), "Français".to_string()),
        // 西班牙文 (spa) -> es
        "spa" => ("es".to_string(), "Español".to_string()),
        // 德文 (ger / deu) -> de
        "ger" | "deu" => ("de".to_string(), "Deutsch".to_string()),
        // 俄文 (rus) -> ru
        "rus" => ("ru".to_string(), "Русский".to_string()),
        // 意大利文 (ita) -> it
        "ita" => ("it".to_string(), "Italiano".to_string()),
        // 葡萄牙文 (por) -> pt
        "por" => ("pt".to_string(), "Português".to_string()),
        // 未定义或无标签
        "und" | "" => ("auto".to_string(), "未知语言".to_string()),
        // 兜底策略：如果不认识，原样返回
        other => ("auto".to_string(), other.to_string()),
    }
}

