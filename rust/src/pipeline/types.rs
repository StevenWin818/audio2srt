use crate::qwen::aligner::AlignedToken;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimestampQuality {
    Vad,
    ForcedAligned,
}

#[derive(Debug, Clone)]
pub struct AudioChunk {
    pub seq_id: u64,
    pub samples: Vec<f32>,
    pub sample_rate: u32,
}

#[derive(Debug, Clone)]
pub struct SpeechSegment {
    pub seq_id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples_16k: Arc<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub struct RecognizedSegment {
    pub seq_id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub samples_16k: Arc<Vec<f32>>,
    pub text: String,
    pub language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinalSegment {
    pub seq_id: u64,
    pub start_ms: u64,
    pub end_ms: u64,
    pub text: String,
    pub words: Vec<AlignedToken>,
    pub timestamp_quality: TimestampQuality,
}
