use std::path::Path;

pub struct DeepFilterStream;

impl DeepFilterStream {
    pub fn with_threads(_model_dir: &Path, _threads: usize) -> Result<Self, String> {
        Ok(Self)
    }

    pub fn warmup(&mut self) -> Result<(), String> {
        Ok(())
    }

    pub fn reset(&mut self) {}

    pub fn process(&mut self, chunk: &[f32]) -> Result<Vec<f32>, String> {
        Ok(chunk.to_vec())
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, String> {
        Ok(Vec::new())
    }
}
