#include "ct2_wrapper.h"
#include "ctranslate2/models/whisper.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"
#include <iostream>
#include <vector>

namespace {
  ctranslate2::Device str_to_device(const std::string& device) {
    if (device == "cuda" || device == "CUDA") {
      return ctranslate2::Device::CUDA;
    }
    return ctranslate2::Device::CPU;
  }

  ctranslate2::ComputeType str_to_compute_type(const std::string& compute_type) {
    if (compute_type == "float16") return ctranslate2::ComputeType::FLOAT16;
    if (compute_type == "int16") return ctranslate2::ComputeType::INT16;
    if (compute_type == "int8") return ctranslate2::ComputeType::INT8;
    if (compute_type == "int8_float16") return ctranslate2::ComputeType::INT8_FLOAT16;
    return ctranslate2::ComputeType::DEFAULT;
  }
}

WhisperWrapper::WhisperWrapper(const std::string& model_path,
                               const std::string& device,
                               int device_index,
                               const std::string& compute_type,
                               int intra_threads) {
  ctranslate2::Device dev = str_to_device(device);
  ctranslate2::ComputeType comp_type = str_to_compute_type(compute_type);

  ctranslate2::ReplicaPoolConfig config;
  config.num_threads_per_replica = intra_threads;

  // Load the model
  model_ = std::make_unique<ctranslate2::models::Whisper>(
      model_path,
      dev,
      comp_type,
      std::vector<int>{device_index},
      false,
      config
  );
}

WhisperWrapper::~WhisperWrapper() = default;

rust::Vec<size_t> WhisperWrapper::transcribe(
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
    float& avg_logprob) const {

  // Create the features StorageView
  // Shape: [batch_size, n_mels, n_frames] -> [1, n_mels, n_frames]
  std::vector<int64_t> shape = {1, static_cast<int64_t>(n_mels), static_cast<int64_t>(n_frames)};
  std::vector<float> mel_vector(mel_data, mel_data + (n_mels * n_frames));
  ctranslate2::StorageView features(shape, mel_vector);

  // Options
  ctranslate2::models::WhisperOptions options;
  options.beam_size = beam_size;
  options.patience = patience;
  options.sampling_temperature = temperature;
  if (temperature > 0.0f) {
    options.sampling_topk = 0; // Enable full sampling
  } else {
    options.sampling_topk = 1; // Greedy search
  }
  options.return_scores = true;
  options.return_no_speech_prob = true;
  options.repetition_penalty = repetition_penalty;
  options.no_repeat_ngram_size = no_repeat_ngram_size;

  // Prompts (passed from Rust)
  std::vector<std::vector<size_t>> prompts = {{}};
  for (size_t token : prompt_tokens) {
    prompts[0].push_back(token);
  }

  // Run generation
  auto futures = model_->generate(features, prompts, options);
  
  rust::Vec<size_t> result_token_ids;
  avg_logprob = 0.0f;
  if (!futures.empty()) {
    try {
      auto result = futures[0].get();
      no_speech_prob = result.no_speech_prob;
      if (!result.scores.empty()) {
        avg_logprob = result.scores[0];
      }
      if (!result.sequences_ids.empty()) {
        for (size_t id : result.sequences_ids[0]) {
          result_token_ids.push_back(id);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[C++] CTranslate2 Transcription Error: " << e.what() << std::endl;
    }
  }

  return result_token_ids;
}

std::unique_ptr<WhisperWrapper> create_whisper_model(
    rust::Str model_path,
    rust::Str device,
    int device_index,
    rust::Str compute_type,
    int intra_threads) {
  try {
    return std::make_unique<WhisperWrapper>(
        std::string(model_path),
        std::string(device),
        device_index,
        std::string(compute_type),
        intra_threads
    );
  } catch (const std::exception& e) {
    std::cerr << "[C++] Error creating Whisper model: " << e.what() << std::endl;
    return nullptr;
  }
}
