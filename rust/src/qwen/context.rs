use crate::qwen::backend::{DecoderBackend, EncoderBackend};
use crate::qwen::runtime::QwenRuntime;
use parking_lot::Mutex;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RuntimeCacheKey {
    pub asr_model_id: String,
    pub aligner_model_id: Option<String>,
    pub encoder_backend: EncoderBackend,
    pub decoder_backend: DecoderBackend,
    /// 解码器 GGUF 文件名 (量化选择)
    pub decoder_file: Option<String>,
}

pub struct QwenContextCache {
    current_key: Option<RuntimeCacheKey>,
    runtime: Option<Arc<QwenRuntime>>,
}

impl QwenContextCache {
    pub fn new() -> Self {
        Self {
            current_key: None,
            runtime: None,
        }
    }

    pub fn get(&self, key: &RuntimeCacheKey) -> Option<Arc<QwenRuntime>> {
        if self.current_key.as_ref() == Some(key) {
            self.runtime.clone()
        } else {
            None
        }
    }

    pub fn set(&mut self, key: RuntimeCacheKey, runtime: Arc<QwenRuntime>) {
        if let Some(old_runtime) = self.runtime.take() {
            println!("[Rust GLOBAL_QWEN_CACHE] Replacing old runtime & releasing VRAM...");
            drop(old_runtime);
        }
        self.current_key = Some(key);
        self.runtime = Some(runtime);
    }

    pub fn clear(&mut self) {
        self.current_key = None;
        if let Some(old_runtime) = self.runtime.take() {
            println!("[Rust GLOBAL_QWEN_CACHE] Clearing previous runtime & releasing VRAM...");
            drop(old_runtime);
        }
    }

    /// 当前缓存的运行时状态 (模型目录 / decoder 文件 / encoder EP / decoder 后端 / offload)
    pub fn runtime_status(&self) -> Option<(String, String, String, String, String)> {
        self.runtime.as_ref().map(|r| {
            (
                r.model_dir.clone(),
                r.decoder_file.clone(),
                r.encoder_ep.clone(),
                r.decoder_backend.clone(),
                r.decoder_offload.clone(),
            )
        })
    }

    /// 标记当前缓存的运行时取消 (encoder/decode 的 cancel 检查立即生效，
    /// 加速取消后的线程退出；下次转写 cache HIT 时 reset_cancel 清除)
    pub fn cancel_runtime(&self) {
        if let Some(r) = &self.runtime {
            r.cancel();
        }
    }
}

pub static GLOBAL_QWEN_CACHE: once_cell::sync::Lazy<Mutex<QwenContextCache>> =
    once_cell::sync::Lazy::new(|| Mutex::new(QwenContextCache::new()));
