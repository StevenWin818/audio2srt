#include "ct2_wrapper.h"
#include "rust/cxx.h"
#include "rust_lib_audio2srt/src/ctranslate2_bridge.rs.h"
#include "ctranslate2/models/whisper.h"
#include "ctranslate2/devices.h"
#include "ctranslate2/types.h"
#include "ctranslate2/storage_view.h"
#include <iostream>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

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

size_t get_gpu_free_vram_mb() {
#ifdef _WIN32
  typedef int (*cudaMemGetInfoFunc)(size_t*, size_t*);
  HMODULE hModule = LoadLibraryA("nvcuda.dll");
  if (!hModule) {
    hModule = LoadLibraryA("cudart64_12.dll");
  }
  if (!hModule) {
    hModule = LoadLibraryA("cudart64_110.dll");
  }
  if (hModule) {
    cudaMemGetInfoFunc cudaMemGetInfo = (cudaMemGetInfoFunc)GetProcAddress(hModule, "cudaMemGetInfo");
    if (cudaMemGetInfo) {
      size_t free_bytes = 0, total_bytes = 0;
      if (cudaMemGetInfo(&free_bytes, &total_bytes) == 0 && free_bytes > 0) {
        FreeLibrary(hModule);
        return free_bytes / (1024 * 1024);
      }
    }
    FreeLibrary(hModule);
  }
#endif
  return 0;
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

  std::vector<int64_t> shape = {1, static_cast<int64_t>(n_mels), static_cast<int64_t>(n_frames)};
  std::vector<float> mel_vector(mel_data, mel_data + (n_mels * n_frames));
  ctranslate2::StorageView features(shape, mel_vector);

  ctranslate2::models::WhisperOptions options;
  options.beam_size = beam_size;
  options.patience = patience;
  options.sampling_temperature = temperature;
  if (temperature > 0.0f) {
    options.sampling_topk = 0;
  } else {
    options.sampling_topk = 1;
  }
  options.return_scores = true;
  options.return_no_speech_prob = true;
  options.repetition_penalty = repetition_penalty;
  options.no_repeat_ngram_size = no_repeat_ngram_size;

  std::vector<std::vector<size_t>> prompts = {{}};
  for (size_t token : prompt_tokens) {
    prompts[0].push_back(token);
  }

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
        for (size_t token_id : result.sequences_ids[0]) {
          result_token_ids.push_back(token_id);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[WhisperWrapper] C++ exception in generate: " << e.what() << std::endl;
    }
  }

  return result_token_ids;
}

rust::Vec<BatchResult> WhisperWrapper::transcribe_batch(
    const float* mel_data,
    size_t batch_size,
    size_t n_mels,
    size_t n_frames,
    size_t beam_size,
    float patience,
    float temperature,
    rust::Slice<const size_t> prompt_tokens,
    float repetition_penalty,
    size_t no_repeat_ngram_size) const {

  std::vector<int64_t> shape = {
      static_cast<int64_t>(batch_size),
      static_cast<int64_t>(n_mels),
      static_cast<int64_t>(n_frames)
  };
  std::vector<float> mel_vector(mel_data, mel_data + (batch_size * n_mels * n_frames));
  ctranslate2::StorageView features(shape, mel_vector);

  ctranslate2::models::WhisperOptions options;
  options.beam_size = beam_size;
  options.patience = patience;
  options.sampling_temperature = temperature;
  if (temperature > 0.0f) {
    options.sampling_topk = 0;
  } else {
    options.sampling_topk = 1;
  }
  options.return_scores = true;
  options.return_no_speech_prob = true;
  options.repetition_penalty = repetition_penalty;
  options.no_repeat_ngram_size = no_repeat_ngram_size;
  options.max_length = 256;

  std::vector<size_t> single_prompt;
  for (size_t token : prompt_tokens) {
    single_prompt.push_back(token);
  }
  std::vector<std::vector<size_t>> prompts(batch_size, single_prompt);

  auto futures = model_->generate(features, prompts, options);

  rust::Vec<BatchResult> results;
  for (size_t i = 0; i < futures.size(); ++i) {
    BatchResult item;
    item.no_speech_prob = 0.0f;
    item.avg_logprob = 0.0f;
    try {
      auto result = futures[i].get();
      item.no_speech_prob = result.no_speech_prob;
      if (!result.scores.empty()) {
        item.avg_logprob = result.scores[0];
      }
      if (!result.sequences_ids.empty()) {
        for (size_t token_id : result.sequences_ids[0]) {
          item.token_ids.push_back(token_id);
        }
      }
    } catch (const std::exception& e) {
      std::cerr << "[WhisperWrapper] C++ exception in generate_batch: " << e.what() << std::endl;
    }
    results.push_back(item);
  }

  return results;
}

rust::String WhisperWrapper::detect_language(
    const float* mel_data,
    size_t n_mels,
    size_t n_frames) const {
  std::vector<int64_t> shape = {1, static_cast<int64_t>(n_mels), static_cast<int64_t>(n_frames)};
  std::vector<float> mel_vector(mel_data, mel_data + (n_mels * n_frames));
  ctranslate2::StorageView features(shape, mel_vector);

  auto futures = model_->detect_language(features);
  if (!futures.empty()) {
    auto lang_res = futures[0].get();
    if (!lang_res.empty()) {
      std::string lang = lang_res[0].first;
      if (lang.size() > 4 && lang.substr(0, 2) == "<|" && lang.substr(lang.size() - 2) == "|>") {
        lang = lang.substr(2, lang.size() - 4);
      }
      return rust::String(lang);
    }
  }
  return rust::String("");
}

std::unique_ptr<WhisperWrapper> create_whisper_model(
    rust::Str model_path,
    rust::Str device,
    int device_index,
    rust::Str compute_type,
    int intra_threads) {
  return std::make_unique<WhisperWrapper>(
      std::string(model_path),
      std::string(device),
      device_index,
      std::string(compute_type),
      intra_threads
  );
}
