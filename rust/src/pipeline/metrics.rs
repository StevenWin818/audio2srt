use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PipelineMetrics {
    pub job_id: String,
    pub processed_audio_ms: u64,
    pub total_audio_ms: u64,
    pub ffmpeg_wait_ms: u64,
    pub denoise_ms: u64,
    pub vad_ms: u64,
    pub asr_encoder_ms: u64,
    pub asr_decode_ms: u64,
    pub aligner_ms: u64,
    pub total_pipeline_ms: u64,
    pub peak_ram_mb: u64,
    pub peak_vram_mb: u64,
}
