#[cxx::bridge]
pub mod ffi {
    unsafe extern "C++" {
        include!("ct2_wrapper.h");

        type WhisperWrapper;

        fn create_whisper_model(
            model_path: &str,
            device: &str,
            device_index: i32,
            compute_type: &str,
            intra_threads: i32,
        ) -> UniquePtr<WhisperWrapper>;

        unsafe fn transcribe(
            self: &WhisperWrapper,
            mel_data: *const f32,
            n_mels: usize,
            n_frames: usize,
            beam_size: usize,
            patience: f32,
            temperature: f32,
            prompt_tokens: &[usize],
            repetition_penalty: f32,
            no_repeat_ngram_size: usize,
            no_speech_prob: &mut f32,
            avg_logprob: &mut f32,
        ) -> Vec<usize>;
    }
}
