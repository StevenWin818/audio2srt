use whisper_rs::{
    WhisperVadContext, WhisperVadContextParams, WhisperVadParams,
};
use serde::{Deserialize, Serialize};
use crate::frb_generated::StreamSink;

/// 兼容性存根（底层已升级为 Qwen3-ASR 流式管道）
#[allow(unused_variables)]
pub fn transcribe(
    sink: StreamSink<TranscriptionEvent>,
    model_path: String,
    vad_model_path: String,
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
    no_state_history: bool,
) {}

/// 兼容性存根
#[allow(unused_variables)]
pub fn warmup_whisper_context(model_path: String, use_gpu: bool, total_duration: f64) -> Result<(), String> {
    Ok(())
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct WordItem {
    pub text: String,
    pub start_ms: i64,
    pub end_ms: i64,
    pub confidence: f32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TranscriptionSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
    pub words: Vec<WordItem>,
    pub timestamp_quality: String,
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
#[allow(dead_code)]
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
#[allow(dead_code)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> isize;
    fn FreeLibrary(hLibModule: isize) -> i32;
}

#[cfg(target_os = "windows")]
#[allow(dead_code)]
#[link(name = "user32")]
extern "system" {
    fn EnumDisplayDevicesA(
        lpDevice: *const u8,
        iDevNum: u32,
        lpDisplayDevice: *mut DISPLAY_DEVICEA,
        dwFlags: u32,
    ) -> i32;
}

#[cfg(target_os = "windows")]
#[allow(dead_code)]
fn check_vulkan_supported_safely() -> bool {
    unsafe {
        let module = LoadLibraryA(b"vulkan-1.dll\0".as_ptr());
        if module == 0 {
            return false;
        }
        FreeLibrary(module);

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
            if (dd.state_flags & 0x1) != 0 {
                let device_str = std::ffi::CStr::from_ptr(dd.device_string.as_ptr() as *const i8)
                    .to_string_lossy()
                    .to_lowercase();
                if !device_str.contains("basic display") && !device_str.contains("basic render") {
                    found_real_gpu = true;
                }
            }
            i += 1;
        }

        found_real_gpu
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
        HardwareAccelerationInfo {
            is_vulkan_available: is_vulkan_safe,
            devices: if is_vulkan_safe {
                vec![VulkanDeviceInfo {
                    id: 0,
                    name: "Vulkan GPU Acceleration".to_string(),
                    total_vram_bytes: 0,
                }]
            } else {
                vec![]
            },
        }
    }
    #[cfg(feature = "cuda")]
    {
        HardwareAccelerationInfo {
            is_vulkan_available: true,
            devices: vec![VulkanDeviceInfo {
                id: 0,
                name: "NVIDIA CUDA GPU".to_string(),
                total_vram_bytes: 0,
            }],
        }
    }
    #[cfg(not(any(feature = "vulkan", feature = "cuda")))]
    {
        HardwareAccelerationInfo {
            is_vulkan_available: false,
            devices: vec![],
        }
    }
}

pub fn convert_chinese(text: String, to_simplified: bool) -> String {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    zhconv::zhconv(&text, target)
}

pub fn convert_chinese_list(texts: Vec<String>, to_simplified: bool) -> Vec<String> {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    texts.into_iter().map(|text| zhconv::zhconv(&text, target)).collect()
}

#[cfg(target_os = "windows")]
pub fn register_thread_as_pro_audio() {
    use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsA;

    thread_local! {
        static MMCSS_HANDLE: std::cell::RefCell<Option<usize>> = const { std::cell::RefCell::new(None) };
    }

    MMCSS_HANDLE.with(|cell| {
        let mut guard = cell.borrow_mut();
        if guard.is_none() {
            let task_name = std::ffi::CString::new("Pro Audio").unwrap();
            let mut task_index: u32 = 0;
            let handle = unsafe {
                AvSetMmThreadCharacteristicsA(task_name.as_ptr() as *const u8, &mut task_index)
            };
            if handle != std::ptr::null_mut() {
                *guard = Some(handle as usize);
            }
        }
    });
}

#[cfg(not(target_os = "windows"))]
pub fn register_thread_as_pro_audio() {}

/// Silero VAD 语音活动检测引擎包装结构
pub struct SileroVadEngine {
    context: WhisperVadContext,
}

impl SileroVadEngine {
    pub fn load(vad_model_path: &str) -> Result<Self, String> {
        println!("[Rust] 初始化 Silero VAD 引擎: {}", vad_model_path);
        let ctx_params = WhisperVadContextParams::new();
        let context = WhisperVadContext::new(vad_model_path, ctx_params).map_err(|e| e.to_string())?;
        Ok(Self { context })
    }

    pub fn detect_speech_segments(
        &mut self,
        samples: &[f32],
        min_speech_ms: i32,
        min_silence_ms: i32,
        threshold: f32,
    ) -> Result<Vec<(usize, usize)>, String> {
        let mut vad_params = WhisperVadParams::new();
        vad_params.set_min_speech_duration(min_speech_ms);
        vad_params.set_min_silence_duration(min_silence_ms);
        let prob_threshold = if threshold > 0.0 && threshold < 1.0 { threshold } else { 0.5f32 };
        vad_params.set_threshold(prob_threshold);

        let segs = self.context.segments_from_samples(vad_params, samples).map_err(|e| e.to_string())?;
        let mut speech_segments = Vec::new();
        for s in segs {
            let start_sample = s.start as usize * 160;
            let end_sample = s.end as usize * 160;
            if end_sample > start_sample && end_sample <= samples.len() {
                speech_segments.push((start_sample, end_sample));
            }
        }
        Ok(speech_segments)
    }
}
