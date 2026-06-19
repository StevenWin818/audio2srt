use std::process::Command;
#[cfg(target_os = "windows")]
use std::os::windows::process::CommandExt;

/// 格式化 Windows 路径以适应 FFmpeg 的 subtitles 滤镜
fn format_path_for_filter(path: &str) -> String {
    // 将反斜杠替换为正斜杠，并转义冒号
    let p = path.replace("\\", "/");
    p.replace(":", "\\:")
}

/// 提取音视频文件中的音频并重采样为 16kHz 单声道 PCM WAV 格式
pub fn extract_audio_from_media(
    ffmpeg_path: String,
    input_path: String,
    output_path: String,
) -> Result<String, String> {
    let mut cmd = Command::new(&ffmpeg_path);
    
    // 设置 FFmpeg 参数：
    // -y: 覆盖输出文件
    // -i: 输入文件路径
    // -ar 16000: 采样率 16kHz
    // -ac 1: 单声道
    // -c:a pcm_s16le: 16位 signed PCM
    cmd.arg("-y")
       .arg("-i")
       .arg(&input_path)
       .arg("-ar")
       .arg("16000")
       .arg("-ac")
       .arg("1")
       .arg("-c:a")
       .arg("pcm_s16le")
       .arg(&output_path);

    // 在 Windows 上隐藏控制台窗口
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    match cmd.output() {
        Ok(output) => {
            if output.status.success() {
                Ok(output_path)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("FFmpeg 提取音频失败: {}", stderr))
            }
        }
        Err(e) => Err(format!("无法运行 FFmpeg (路径: {}): {}", ffmpeg_path, e)),
    }
}

/// 将 SRT 字幕文件集成到视频中（软封装或硬压制）
pub fn mux_srt_to_video(
    ffmpeg_path: String,
    video_path: String,
    srt_path: String,
    output_path: String,
    hard_burn: bool,
) -> Result<String, String> {
    let mut cmd = Command::new(&ffmpeg_path);
    cmd.arg("-y")
       .arg("-i")
       .arg(&video_path);

    // 在 Windows 上隐藏控制台窗口
    #[cfg(target_os = "windows")]
    cmd.creation_flags(0x08000000); // CREATE_NO_WINDOW

    if hard_burn {
        // 硬压制字幕：使用 subtitles 滤镜
        let escaped_srt = format_path_for_filter(&srt_path);
        let filter_arg = format!("subtitles='{}'", escaped_srt);
        
        cmd.arg("-vf")
           .arg(filter_arg)
           .arg("-c:a")
           .arg("copy") // 复制音频流以提升速度
           .arg(&output_path);
    } else {
        // 软封装字幕：作为独立字幕轨写入
        cmd.arg("-i")
           .arg(&srt_path)
           .arg("-c")
           .arg("copy"); // 复制所有视频、音频和字幕编码

        // 针对不同视频格式使用不同的字幕编码格式
        if output_path.to_lowercase().ends_with(".mp4") {
            cmd.arg("-c:s").arg("mov_text");
        } else {
            cmd.arg("-c:s").arg("srt");
        }
        cmd.arg(&output_path);
    }

    match cmd.output() {
        Ok(output) => {
            if output.status.success() {
                Ok(output_path)
            } else {
                let stderr = String::from_utf8_lossy(&output.stderr);
                Err(format!("FFmpeg 字幕合成失败: {}", stderr))
            }
        }
        Err(e) => Err(format!("无法运行 FFmpeg (路径: {}): {}", ffmpeg_path, e)),
    }
}
