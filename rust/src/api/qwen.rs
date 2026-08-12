use crate::api::hardware::{get_qwen_hardware_info, QwenHardwareInfo};
use crate::api::models::{validate_qwen_model, ModelInfo};
use crate::api::stream_pipeline::get_or_create_qwen_runtime;
use crate::qwen::backend::{DecoderBackend, EncoderBackend};
use crate::qwen::context::GLOBAL_QWEN_CACHE;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenWarmupConfig {
    pub asr_model_dir: String,
    pub aligner_model_dir: Option<String>,
    pub encoder_backend: EncoderBackend,
    pub decoder_backend: DecoderBackend,
}

pub fn get_qwen_runtime_info() -> QwenHardwareInfo {
    get_qwen_hardware_info()
}

pub fn validate_qwen_model_dir(model_dir: String) -> Result<ModelInfo, String> {
    validate_qwen_model(model_dir)
}

/// 预热运行时: 与转写管道共用同一缓存键构造与缓存逻辑,
/// 保证 warmup 之后的首轮转写能直接 cache HIT (后端解析 + 路径归一化一致)。
pub fn warmup_qwen_runtime(config: QwenWarmupConfig) -> Result<(), String> {
    get_or_create_qwen_runtime(
        &config.asr_model_dir,
        config.aligner_model_dir.as_deref(),
        config.encoder_backend,
        config.decoder_backend,
        None,
        true,
    )
    .map(|_| ())
    .map_err(|e| e.to_string())
}

pub fn unload_qwen_runtime() -> Result<(), String> {
    let old = {
        let mut cache = GLOBAL_QWEN_CACHE.lock();
        cache.clear()
    };
    if old.is_some() {
        println!("[Rust] unload_qwen_runtime: Dropping cached QwenRuntime and freeing GPU VRAM...");
    }
    drop(old);
    Ok(())
}
