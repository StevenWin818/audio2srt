use thiserror::Error;

#[derive(Error, Debug)]
pub enum QwenError {
    #[error("Model directory or manifest not found: {0}")]
    ModelNotFound(String),

    #[error("Invalid model manifest or hash mismatch: {0}")]
    InvalidManifest(String),

    #[error("ONNX Runtime error: {0}")]
    OnnxError(String),

    #[error("Decoder / llama.cpp error: {0}")]
    DecoderError(String),

    #[error("Alignment error: {0}")]
    AlignmentError(String),

    #[error("Audio processing error: {0}")]
    AudioError(String),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Task cancelled")]
    Cancelled,
}
