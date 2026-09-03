use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EncoderBackend {
    Auto,
    DirectMl,
    Cuda,
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

#[cfg(target_os = "windows")]
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
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(lpLibFileName: *const u8) -> isize;
    fn FreeLibrary(hLibModule: isize) -> i32;
}

#[cfg(target_os = "windows")]
fn enumerate_windows_gpus() -> Vec<String> {
    let mut names = Vec::new();
    let mut dd = DISPLAY_DEVICEA {
        cb: std::mem::size_of::<DISPLAY_DEVICEA>() as u32,
        device_name: [0; 32],
        device_string: [0; 128],
        state_flags: 0,
        device_id: [0; 128],
        device_key: [0; 128],
    };
    let mut i = 0;
    unsafe {
        while EnumDisplayDevicesA(std::ptr::null(), i, &mut dd, 0) != 0 {
            // 过滤虚拟镜像驱动 (DISPLAY_DEVICE_MIRRORING_DRIVER = 0x8)
            // 注意: 绝不可过滤 (state_flags & 0x1 == 0)，双显卡/Optimus 笔记本上独显通常不直接绑定桌面
            if (dd.state_flags & 0x8) == 0 {
                let name = std::ffi::CStr::from_ptr(dd.device_string.as_ptr() as *const i8)
                    .to_string_lossy()
                    .trim()
                    .to_string();
                let lower = name.to_lowercase();
                if !lower.is_empty()
                    && !lower.contains("basic display")
                    && !lower.contains("basic render")
                    && !names.contains(&name)
                {
                    names.push(name);
                }
            }
            i += 1;
        }
    }
    // 排序: 独立显卡 (NVIDIA / AMD Radeon RX / Intel Arc) 优先排在前面，集成核显排在后面
    names.sort_by_key(|n| {
        let l = n.to_lowercase();
        let is_igpu = l.contains("integrated")
            || l.contains("uhd")
            || l.contains("iris")
            || l.contains("vega")
            || l.contains("xe ")
            || l.contains("xe graphics")
            || (l.contains("intel") && !l.contains("arc"))
            || l.contains("radeon(tm)");
        if is_igpu { 1 } else { 0 }
    });
    names
}

#[cfg(target_os = "windows")]
fn vulkan_runtime_supported() -> bool {
    unsafe {
        let module = LoadLibraryA(b"vulkan-1.dll\0".as_ptr());
        if module == 0 {
            return false;
        }
        FreeLibrary(module);
        true
    }
}

#[cfg(target_os = "windows")]
fn cuda_runtime_supported() -> bool {
    unsafe {
        let module = LoadLibraryA(b"nvcuda.dll\0".as_ptr());
        if module == 0 {
            return false;
        }
        FreeLibrary(module);
        true
    }
}

#[cfg(not(target_os = "windows"))]
fn vulkan_runtime_supported() -> bool {
    true
}

#[cfg(not(target_os = "windows"))]
fn cuda_runtime_supported() -> bool {
    true
}

pub fn detect_hardware_acceleration() -> QwenHardwareInfo {
    let has_vulkan_feat = cfg!(any(feature = "vulkan", feature = "qwen-vulkan"));
    let has_cuda_feat = cfg!(any(feature = "cuda", feature = "qwen-cuda"));
    let has_dml_feat = cfg!(any(feature = "qwen-dml", feature = "qwen-dml-win"));

    #[cfg(target_os = "windows")]
    let (vulkan_runtime, cuda_runtime, gpu_names) = {
        let v = if has_vulkan_feat { vulkan_runtime_supported() } else { false };
        let c = if has_cuda_feat { cuda_runtime_supported() } else { false };
        let gpus = enumerate_windows_gpus();
        (v, c, gpus)
    };

    #[cfg(not(target_os = "windows"))]
    let (vulkan_runtime, cuda_runtime, gpu_names) = {
        (has_vulkan_feat, has_cuda_feat, vec!["GPU Acceleration".to_string()])
    };

    let vulkan_available = has_vulkan_feat && vulkan_runtime;
    let cuda_available = has_cuda_feat && cuda_runtime;
    let directml_available = has_dml_feat;

    let is_gpu_ready = vulkan_available || cuda_available || directml_available;

    let mut devices = Vec::new();
    if is_gpu_ready {
        if gpu_names.is_empty() {
            devices.push(ComputeDeviceInfo {
                name: if cuda_available {
                    "NVIDIA CUDA GPU".to_string()
                } else {
                    "Vulkan GPU Acceleration".to_string()
                },
                vendor: if cuda_available {
                    "NVIDIA".to_string()
                } else {
                    "Unknown".to_string()
                },
                vram_mb: 0,
                supports_vulkan: vulkan_available,
                supports_directml: directml_available,
            });
        } else {
            for name in gpu_names {
                let lower = name.to_lowercase();
                let vendor = if lower.contains("nvidia")
                    || lower.contains("geforce")
                    || lower.contains("quadro")
                    || lower.contains("rtx")
                    || lower.contains("gtx")
                {
                    "NVIDIA"
                } else if lower.contains("amd") || lower.contains("radeon") {
                    "AMD"
                } else if lower.contains("intel")
                    || lower.contains("arc")
                    || lower.contains("iris")
                    || lower.contains("uhd")
                {
                    "Intel"
                } else {
                    "Unknown"
                };
                devices.push(ComputeDeviceInfo {
                    name,
                    vendor: vendor.to_string(),
                    vram_mb: 0,
                    supports_vulkan: vulkan_available,
                    supports_directml: directml_available,
                });
            }
        }
    }

    let rec_enc = if directml_available {
        EncoderBackend::DirectMl
    } else if cuda_available {
        EncoderBackend::Cuda
    } else {
        EncoderBackend::Cpu
    };

    let rec_dec = if cuda_available {
        DecoderBackend::Cuda
    } else if vulkan_available {
        DecoderBackend::Vulkan
    } else {
        DecoderBackend::Cpu
    };

    QwenHardwareInfo {
        vulkan_available,
        directml_available,
        cuda_available,
        cpu_available: true,
        devices,
        recommended_encoder_backend: rec_enc,
        recommended_decoder_backend: rec_dec,
    }
}
