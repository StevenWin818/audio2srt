use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncoderBackend {
    Auto,
    DirectMl,
    Cpu,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecoderBackend {
    Auto,
    Vulkan,
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputeDeviceInfo {
    pub name: String,
    pub vendor: String,
    pub vram_mb: u64,
    pub supports_vulkan: bool,
    pub supports_directml: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenHardwareInfo {
    pub vulkan_available: bool,
    pub directml_available: bool,
    pub cuda_available: bool,
    pub cpu_available: bool,
    pub devices: Vec<ComputeDeviceInfo>,
    pub recommended_encoder_backend: EncoderBackend,
    pub recommended_decoder_backend: DecoderBackend,
}

// ----- Vulkan 运行时探针（编译功能门控） -----
//
// llama.cpp::sys 公开了在运行时查询后端可用性的方法
// `llama_supports_gpu_offload()` （大约在 0.1.x 之后添加）
// `qwen-vulkan` 功能关闭时，退回到安全存根

#[cfg(all(feature = "qwen-vulkan", target_os = "windows"))]
fn vulkan_runtime_supported() -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn LoadLibraryA(lpLibFileName: *const u8) -> isize;
    }
    // 校验是否存在 Vulkan ICD 加载器 (vulkan-1.dll)
    let h = unsafe { LoadLibraryA(b"vulkan-1.dll\0".as_ptr()) };
    h != 0
}

#[cfg(all(feature = "qwen-vulkan", not(target_os = "windows")))]
fn vulkan_runtime_supported() -> bool {
    // 尽力尝试：在 Linux/macOS 上默认假设存在；llama.cpp 将在运行时自动降级至 CPU。
    true
}

#[cfg(not(feature = "qwen-vulkan"))]
fn vulkan_runtime_supported() -> bool {
    false
}

pub fn detect_hardware_acceleration() -> QwenHardwareInfo {
    let vulkan_supported = vulkan_runtime_supported();
    #[cfg(target_os = "windows")]
    let directml_supported = cfg!(feature = "qwen-dml");
    #[cfg(not(target_os = "windows"))]
    let directml_supported = false;

    // 在没有 Vulkan 加载器绑定的纯 Rust 环境下无 GPU 详细枚举；
    // 当运行时探针检测成功时，构建通用设备记录，以便 UI 能正常展示显卡状态。
    let devices: Vec<ComputeDeviceInfo> = if vulkan_supported {
        vec![ComputeDeviceInfo {
            name: "GPU".to_string(),
            vendor: "Unknown".to_string(),
            vram_mb: 0,
            supports_vulkan: true,
            supports_directml: directml_supported,
        }]
    } else {
        vec![]
    };

    let rec_enc = if directml_supported {
        EncoderBackend::DirectMl
    } else {
        EncoderBackend::Cpu
    };
    let rec_dec = if vulkan_supported {
        DecoderBackend::Vulkan
    } else {
        DecoderBackend::Cpu
    };

    QwenHardwareInfo {
        vulkan_available: vulkan_supported,
        directml_available: directml_supported,
        cuda_available: false,
        cpu_available: true,
        devices,
        recommended_encoder_backend: rec_enc,
        recommended_decoder_backend: rec_dec,
    }
}
