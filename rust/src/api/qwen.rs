use crate::api::hardware::{get_qwen_hardware_info, QwenHardwareInfo};
use crate::api::models::{validate_qwen_model, ModelInfo};
use crate::qwen::backend::{DecoderBackend, EncoderBackend};
use crate::qwen::context::{RuntimeCacheKey, GLOBAL_QWEN_CACHE};
use crate::qwen::runtime::QwenRuntime;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

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

pub fn warmup_qwen_runtime(config: QwenWarmupConfig) -> Result<(), String> {
    let key = RuntimeCacheKey {
        asr_model_id: config.asr_model_dir.clone(),
        aligner_model_id: config.aligner_model_dir.clone(),
        encoder_backend: config.encoder_backend,
        decoder_backend: config.decoder_backend,
    };

    let runtime = QwenRuntime::load(
        &config.asr_model_dir,
        config.aligner_model_dir.as_deref(),
        config.encoder_backend,
        config.decoder_backend,
    )
    .map_err(|e| e.to_string())?;

    let mut cache = GLOBAL_QWEN_CACHE.lock();
    cache.set(key, Arc::new(runtime));
    Ok(())
}

pub fn unload_qwen_runtime() -> Result<(), String> {
    let mut cache = GLOBAL_QWEN_CACHE.lock();
    cache.clear();
    Ok(())
}
