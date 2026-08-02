use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelFileManifest {
    pub name: String,
    pub sha256: String,
    pub size_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPackageManifest {
    pub model_id: String,
    pub name: String,
    pub version: String,
    pub model_type: String, // "asr" or "aligner"
    pub files: Vec<ModelFileManifest>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub model_id: String,
    pub path: String,
    pub valid: bool,
    pub total_size_bytes: u64,
    pub missing_files: Vec<String>,
}

pub fn validate_qwen_model(model_dir: String) -> Result<ModelInfo, String> {
    let dir = Path::new(&model_dir);
    if !dir.exists() {
        return Err(format!("Model directory does not exist: {}", model_dir));
    }

    let manifest_path = dir.join("manifest.json");
    if manifest_path.exists() {
        let content = fs::read_to_string(&manifest_path)
            .map_err(|e| format!("Failed to read manifest.json: {}", e))?;
        let manifest: ModelPackageManifest = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid manifest.json format: {}", e))?;

        let mut missing_files = Vec::new();
        let mut total_size = 0u64;

        for file_info in &manifest.files {
            let file_path = dir.join(&file_info.name);
            if !file_path.exists() {
                missing_files.push(file_info.name.clone());
            } else if let Ok(meta) = fs::metadata(&file_path) {
                total_size += meta.len();
            }
        }

        let valid = missing_files.is_empty();
        Ok(ModelInfo {
            model_id: manifest.model_id,
            path: model_dir,
            valid,
            total_size_bytes: total_size,
            missing_files,
        })
    } else {
        // Simple fallback check without manifest.json
        let mut missing_files = Vec::new();
        let enc_exists = dir.join("asr_encoder_frontend.int4.onnx").exists()
            || dir.join("encoder.onnx").exists();
        let decoder_names = [
            "decoder.bf16.gguf",
            "decoder.f16.gguf",
            "decoder.q8_0.gguf",
            "decoder.q6_k.gguf",
            "decoder.q5_k_m.gguf",
            "decoder.q4_k_m.gguf",
            "decoder.q4_k.gguf",
            "decoder.gguf",
            "asr_decoder.q4_k.gguf",
        ];
        let dec_exists = decoder_names.iter().any(|f| dir.join(f).exists());

        if !enc_exists {
            missing_files.push("Encoder ONNX model".into());
        }
        if !dec_exists {
            missing_files.push("Decoder GGUF model".into());
        }

        Ok(ModelInfo {
            model_id: "custom-qwen".into(),
            path: model_dir,
            valid: missing_files.is_empty(),
            total_size_bytes: 0,
            missing_files,
        })
    }
}
