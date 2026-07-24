pub use crate::qwen::backend::{
    detect_hardware_acceleration, ComputeDeviceInfo, DecoderBackend, EncoderBackend,
    QwenHardwareInfo,
};

pub fn get_qwen_hardware_info() -> QwenHardwareInfo {
    detect_hardware_acceleration()
}
