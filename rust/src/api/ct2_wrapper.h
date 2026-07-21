#pragma once

#include "rust/cxx.h"
#include <memory>
#include <string>

namespace ctranslate2 {
  namespace models {
    class Whisper;
  }
}

struct BatchResult;

class WhisperWrapper {
public:
  WhisperWrapper(const std::string& model_path,
                 const std::string& device,
                 int device_index,
                 const std::string& compute_type,
                 int intra_threads);
  ~WhisperWrapper();

  rust::Vec<size_t> transcribe(
      const float* mel_data,
      size_t n_mels,
      size_t n_frames,
      size_t beam_size,
      float patience,
      float temperature,
      rust::Slice<const size_t> prompt_tokens,
      float repetition_penalty,
      size_t no_repeat_ngram_size,
      float& no_speech_prob,
      float& avg_logprob) const;

  rust::Vec<BatchResult> transcribe_batch(
      const float* mel_data,
      size_t batch_size,
      size_t n_mels,
      size_t n_frames,
      size_t beam_size,
      float patience,
      float temperature,
      rust::Slice<const size_t> prompt_tokens,
      float repetition_penalty,
      size_t no_repeat_ngram_size) const;

  rust::String detect_language(
      const float* mel_data,
      size_t n_mels,
      size_t n_frames) const;

private:
  std::unique_ptr<ctranslate2::models::Whisper> model_;
};

std::unique_ptr<WhisperWrapper> create_whisper_model(
    rust::Str model_path,
    rust::Str device,
    int device_index,
    rust::Str compute_type,
    int intra_threads);

size_t get_gpu_free_vram_mb();
