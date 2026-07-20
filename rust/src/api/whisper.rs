use crate::frb_generated::StreamSink;
use crate::api::ctranslate2_bridge::ffi::{WhisperWrapper, create_whisper_model};
use std::sync::{Arc, Mutex};
use std::collections::HashMap;

struct CachedContext {
    model_path: String,
    use_gpu: bool,
    model: Arc<WhisperModel>,
}

pub struct WhisperModel {
    pub inner: cxx::UniquePtr<WhisperWrapper>,
}
unsafe impl Send for WhisperModel {}
unsafe impl Sync for WhisperModel {}

static G_CONTEXT: Mutex<Option<CachedContext>> = Mutex::new(None);

pub(crate) fn get_or_create_context(
    model_path: &str,
    requested_use_gpu: bool,
) -> Result<Arc<WhisperModel>, String> {
    let path = std::path::Path::new(model_path);
    if !path.is_dir() {
        return Err(format!("Model path is not a directory: {}", model_path));
    }

    let mut cache = G_CONTEXT
        .lock()
        .map_err(|e| format!("Failed to lock global context: {}", e))?;

    if let Some(ref cached) = *cache {
        if cached.model_path == model_path && cached.use_gpu == requested_use_gpu {
            println!("[Rust] Reusing cached CTranslate2 Whisper context for model: {}", model_path);
            return Ok(cached.model.clone());
        }
    }

    let mut use_gpu = requested_use_gpu;

    println!(
        "[Rust] Loading new CTranslate2 Whisper context for model: {} (requested use_gpu={})",
        model_path, use_gpu
    );
    
    if use_gpu {
        // Check if we are using the dummy cudnn64_8.dll or if it is missing
        if let Ok(exe_path) = std::env::current_exe() {
            if let Some(dir) = exe_path.parent() {
                let cudnn8_path = dir.join("cudnn64_8.dll");
                let cudnn9_path = dir.join("cudnn64_9.dll");
                
                let mut found_valid_cudnn = false;
                
                if let Ok(meta) = std::fs::metadata(&cudnn9_path) {
                    if meta.len() >= 200 * 1024 {
                        found_valid_cudnn = true;
                    }
                } else if let Ok(meta) = std::fs::metadata(&cudnn8_path) {
                    if meta.len() >= 200 * 1024 {
                        found_valid_cudnn = true;
                    } else {
                        println!("[Rust] Tiny dummy cudnn64_8.dll detected. Forcing CPU fallback to avoid CUDA crash.");
                    }
                }
                
                if !found_valid_cudnn {
                    println!("[Rust] Valid cuDNN DLL not found (checked 8 and 9). Forcing CPU fallback to avoid CUDA crash.");
                    use_gpu = false;
                }
            }
        }
    }
    
    let mut device = if use_gpu { "cuda" } else { "cpu" };
    let mut compute_type = if use_gpu { "float16" } else { "int8" };
    
    let mut threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4) as i32;
    // Don't use all logical cores, as hyperthreading doesn't help matrix multiplication much.
    // Use half the logical cores, max 12, min 4.
    threads = (threads / 2).max(4).min(12);

    println!("[Rust] Using {} threads for CTranslate2 context.", threads);

    let mut wrapper = create_whisper_model(model_path, device, 0, compute_type, threads);
    
    // Fallback to CPU if GPU initialization failed
    if wrapper.is_null() && use_gpu {
        println!("[Rust] GPU initialization failed (unsupported device or out of memory). Falling back to CPU.");
        device = "cpu";
        compute_type = "int8";
        wrapper = create_whisper_model(model_path, device, 0, compute_type, threads);
    }
    if wrapper.is_null() {
        return Err("Failed to load CTranslate2 model".to_string());
    }

    let shared_model = Arc::new(WhisperModel { inner: wrapper });
    *cache = Some(CachedContext {
        model_path: model_path.to_string(),
        use_gpu: requested_use_gpu,
        model: shared_model.clone(),
    });

    Ok(shared_model)
}

#[derive(Clone, Debug, Default)]
pub struct TranscriptionSegment {
    pub start_ms: i64,
    pub end_ms: i64,
    pub text: String,
}

#[derive(Clone, Debug)]
pub enum TranscriptionEvent {
    Progress(i32),
    ProgressDetail { processed_ms: i64, total_ms: i64 },
    Success(Vec<TranscriptionSegment>),
    Failure(String),
    Segment(TranscriptionSegment),
}

#[derive(Clone, Debug, Default)]
pub struct VulkanDeviceInfo {
    pub id: i32,
    pub name: String,
    pub total_vram_bytes: u64,
}

#[derive(Clone, Debug, Default)]
pub struct HardwareAccelerationInfo {
    pub is_vulkan_available: bool,
    pub devices: Vec<VulkanDeviceInfo>,
}

pub fn get_hardware_acceleration_info() -> HardwareAccelerationInfo {
    // CTranslate2 doesn't use Vulkan, it uses CUDA.
    // For simplicity, we just return empty Vulkan device list.
    HardwareAccelerationInfo {
        is_vulkan_available: false,
        devices: vec![],
    }
}

pub fn convert_chinese(text: String, to_simplified: bool) -> String {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    zhconv::zhconv(&text, target)
}

pub fn convert_chinese_list(texts: Vec<String>, to_simplified: bool) -> Vec<String> {
    let target = if to_simplified { zhconv::Variant::ZhCN } else { zhconv::Variant::ZhTW };
    texts.into_iter().map(|text| zhconv::zhconv(&text, target)).collect()
}

pub fn transcribe(
    sink: StreamSink<TranscriptionEvent>,
    model_path: String,
    vad_model_path: String,
    audio_path: String,
    language: Option<String>,
    translate: bool,
    threads: Option<i32>,
    use_gpu: bool,
    vad_enabled: bool,
    vad_threshold: f32,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
    temperature: f32,
    temperature_inc: f32,
    entropy_thold: f32,
    logprob_thold: f32,
    no_speech_thold: f32,
    no_context: bool,
    no_state_history: bool,
) {
    std::thread::spawn(move || {
        #[cfg(target_os = "windows")]
        register_thread_as_pro_audio();

        let mut callback = |event| {
            let _ = sink.add(event);
        };
        if let Err(e) = run_transcription_inner(
            &mut callback,
            model_path,
            vad_model_path,
            audio_path,
            language,
            translate,
            threads,
            use_gpu,
            vad_enabled,
            vad_threshold,
            vad_min_speech_ms,
            vad_min_silence_ms,
            temperature,
            temperature_inc,
            entropy_thold,
            logprob_thold,
            no_speech_thold,
            no_context,
            no_state_history,
        ) {
            let _ = sink.add(TranscriptionEvent::Failure(e));
        }
    });
}

pub(crate) fn run_transcription_inner(
    callback: &mut dyn FnMut(TranscriptionEvent),
    model_path: String,
    vad_model_path: String,
    audio_path: String,
    language: Option<String>,
    translate: bool,
    _threads: Option<i32>,
    use_gpu: bool,
    vad_enabled: bool,
    vad_threshold: f32,
    vad_min_speech_ms: i32,
    vad_min_silence_ms: i32,
    temperature: f32,
    _temperature_inc: f32,
    _entropy_thold: f32,
    _logprob_thold: f32,
    _no_speech_thold: f32,
    _no_context: bool,
    _no_state_history: bool,
) -> Result<(), String> {
    println!(
        "[Rust] run_transcription (CTranslate2): model_path={}, vad_model_path={}, audio_path={}, language={:?}, translate={}, use_gpu={}, vad={}",
        model_path, vad_model_path, audio_path, language, translate, use_gpu, vad_enabled
    );

    // 1. 读取音频数据 (16kHz 单声道 16-bit PCM wav)
    let mut reader = hound::WavReader::open(&audio_path).map_err(|e| {
        let err_msg = format!("无法打开音频文件: {}", e);
        println!("[Rust] {}", err_msg);
        err_msg
    })?;
    
    let spec = reader.spec();
    if spec.sample_rate != 16000 || spec.channels != 1 || spec.bits_per_sample != 16 {
        return Err("音频格式错误：必须是 16kHz, 单声道, 16位 PCM WAV 文件。".to_string());
    }

    let mut samples = Vec::new();
    for sample in reader.samples::<i16>() {
        let s = sample.map_err(|e| format!("读取音频采样点失败: {}", e))?;
        samples.push(s as f32 / 32768.0);
    }

    if samples.len() < 8000 {
        samples.resize(8000, 0.0);
    }

    // 2. VAD 切片
    let mut speech_segments = Vec::new();
    if vad_enabled {
        println!("[Rust] Initializing Silero VAD from: {}", vad_model_path);
        let threshold = if vad_threshold > 0.0 && vad_threshold < 1.0 { vad_threshold } else { 0.5f32 };
        let segments = run_ort_vad(&vad_model_path, &samples, threshold, vad_min_speech_ms, vad_min_silence_ms)?;
        println!("[Rust] VAD enabled. Found {} active speech segments.", segments.len());
        for (start, end) in segments {
            speech_segments.push((start, end));
        }
        if speech_segments.is_empty() {
            callback(TranscriptionEvent::Success(Vec::new()));
            return Ok(());
        }
    } else {
        speech_segments.push((0, samples.len()));
    }

    // 3. 加载 CTranslate2 模型与词表
    let model = get_or_create_context(&model_path, use_gpu)?;
    let vocab = load_vocabulary(&model_path)?;
    let unicode_to_bytes = get_unicode_to_bytes();
    let n_mels = get_num_mel_bins(&model_path).unwrap_or(80);

    let mut combined_segments = Vec::new();

    // 4. 逐段推理
    for (idx, &(start_sample, end_sample)) in speech_segments.iter().enumerate() {
        let safe_start = start_sample.saturating_sub(3200); 
        let safe_end = (end_sample + 3200).min(samples.len());
        let segment_samples = samples[safe_start..safe_end].to_vec();

        if segment_samples.len() < 3200 {
            continue;
        }

        let rms = calculate_rms(&segment_samples);
        if rms < 0.002 {
            continue;
        }
        
        let global_offset_ms = (safe_start as i64) / 16;
        let total_samples = segment_samples.len();
        let chunk_size = 480000; // 30s chunks required by CTranslate2 Whisper model
        let mut offset = 0;

        while offset < total_samples {
            let chunk_start_ms = global_offset_ms + (offset as i64 * 1000) / 16000;
            let chunk_actual_len = (total_samples - offset).min(chunk_size);
            let chunk_actual_duration_ms = (chunk_actual_len as i64 * 1000) / 16000;

            let chunk_samples = segment_samples[offset..offset + chunk_actual_len].to_vec();

            // 1. 计算所有 FFT 帧
            let fft_frames = mel_spec::stft::Spectrogram::compute_all_cpu(
                &chunk_samples,
                400,
                160,
            );

            // 2. 生成 Mel 滤波器组权重
            let mel_filters = mel_spec::mel::mel(
                16000.0,
                400,
                n_mels,
                None,
                None,
                false,
                true,
            );

            // 3. 将各帧 STFT 映射至未归一化的 log10 Mel 能量 (Array2)
            let n_frames = fft_frames.len(); // Will always be 3000 due to padding
            let mut combined_mel = ndarray_016::Array2::zeros((n_mels, n_frames));
            for (f, frame) in fft_frames.into_iter().enumerate() {
                let fft_array = ndarray_016::Array1::from_vec(frame);
                let log_mel_frame = mel_spec::mel::log_mel_spectrogram(&fft_array, &mel_filters);
                for m in 0..n_mels {
                    combined_mel[[m, f]] = log_mel_frame[[m, 0]];
                }
            }

            // 4. 对整段进行全局 Whisper 归一化
            let normalized = mel_spec::mel::norm_mel(&combined_mel);

            let mut flat_mel = vec![0.0f32; n_mels * n_frames];
            for f in 0..n_frames {
                for m in 0..n_mels {
                    flat_mel[m * n_frames + f] = normalized[[m, f]] as f32;
                }
            }

            let mut actual_language = language.clone();
            let is_auto = actual_language.is_none() || actual_language.as_deref() == Some("auto");
            if is_auto {
                let detected_lang = unsafe {
                    model.inner.detect_language(
                        flat_mel.as_ptr(),
                        n_mels,
                        n_frames
                    )
                };
                if !detected_lang.is_empty() {
                    println!("[Rust] Auto language detected: {}", detected_lang);
                    actual_language = Some(detected_lang);
                } else {
                    println!("[Rust] Auto language detection returned empty string! Falling back to en");
                    actual_language = Some("en".to_string());
                }
            }

            let prompt_tokens = get_prompt_tokens(&vocab, &actual_language, translate);
            println!("[Rust] Final prompt tokens: {:?}", prompt_tokens);
            let repetition_penalty = 1.0f32;
            let no_repeat_ngram_size = 0;

            let get_compression_ratio = |text: &str| -> f32 {
                if text.is_empty() {
                    return 0.0;
                }
                use flate2::write::GzEncoder;
                use flate2::Compression;
                use std::io::Write;
                let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
                if encoder.write_all(text.as_bytes()).is_err() {
                    return 1.0;
                }
                if let Ok(compressed) = encoder.finish() {
                    text.len() as f32 / compressed.len() as f32
                } else {
                    1.0
                }
            };

            let mut final_segs = Vec::new();
            let mut final_no_speech_prob = 0.0f32;
            let patience = 1.0;
            
            let temps = vec![temperature, 0.2f32, 0.4f32, 0.6f32, 0.8f32, 1.0f32];
            for (idx, &temp) in temps.iter().enumerate() {
                let mut no_speech_prob = 0.0f32;
                let mut avg_logprob = 0.0f32;
                let token_ids = unsafe {
                    model.inner.transcribe(
                        flat_mel.as_ptr(),
                        n_mels,
                        n_frames,
                        1, // beam size: use 1 (greedy)
                        payout_or_default(patience),
                        temp,
                        &prompt_tokens,
                        repetition_penalty,
                        no_repeat_ngram_size,
                        &mut no_speech_prob,
                        &mut avg_logprob,
                    )
                };
                
                let segs = parse_tokens_to_segments(
                    &token_ids,
                    &vocab,
                    &unicode_to_bytes,
                    chunk_start_ms,
                    chunk_actual_duration_ms,
                );
                
                let full_text: String = segs.iter().map(|s| s.text.as_str()).collect();
                let ratio = get_compression_ratio(&full_text);
                
                final_segs = segs;
                final_no_speech_prob = no_speech_prob;
                
                let is_unconfident = avg_logprob < -1.0;
                let is_repetitive = ratio > 2.4;
                
                if (is_unconfident || is_repetitive) && idx < temps.len() - 1 {
                    println!(
                        "[Rust] Fallback triggered at temp {}: avg_logprob = {:.3} (threshold = -1.0), compression_ratio = {:.3} (threshold = 2.4). Retrying...",
                        temp, avg_logprob, ratio
                    );
                    continue;
                }
                
                break;
            }

            if final_no_speech_prob > 0.6 {
                println!("[Rust] Whisper detected no_speech_prob: {:.2} > 0.6, skipping music/silence segment", final_no_speech_prob);
                offset += chunk_size;
                continue;
            }

            for mut s in final_segs {
                // Ignore any segments starting after the actual audio content ends (to drop silence hallucinations)
                if s.start_ms >= chunk_start_ms + chunk_actual_duration_ms - 200 {
                    continue;
                }
                if s.end_ms > chunk_start_ms + chunk_actual_duration_ms {
                    s.end_ms = chunk_start_ms + chunk_actual_duration_ms;
                }
                let cleaned_text = s.text.replace("🎵", "")
                                         .replace("[音乐]", "")
                                         .replace("(音乐)", "")
                                         .replace("[Music]", "")
                                         .trim().to_string();
                if !cleaned_text.is_empty() {
                    s.text = zhconv::zhconv(&cleaned_text, zhconv::Variant::ZhCN);
                    combined_segments.push(s);
                }
            }

            offset += chunk_size;
        }

        let progress = ((idx + 1) as f32 / speech_segments.len() as f32 * 100.0) as i32;
        callback(TranscriptionEvent::Progress(progress.clamp(0, 100)));
    }

    callback(TranscriptionEvent::Success(combined_segments));
    Ok(())
}

fn payout_or_default(patience: f32) -> f32 {
    if patience <= 0.0 { 1.0 } else { patience }
}

pub fn get_unicode_to_bytes() -> HashMap<char, u8> {
    let mut bs: Vec<u8> = Vec::new();
    bs.extend(b'!'..=b'~');
    bs.extend(0xA1..=0xAC);
    bs.extend(0xAE..=0xFF);
    
    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n: u32 = 0;
    for b in 0..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    
    bs.iter().zip(cs.iter()).map(|(&b, &c)| (std::char::from_u32(c).unwrap(), b)).collect()
}

pub fn decode_tokens(tokens: &[String], unicode_to_bytes: &HashMap<char, u8>) -> String {
    let mut bytes = Vec::new();
    for token in tokens {
        for c in token.chars() {
            if let Some(&b) = unicode_to_bytes.get(&c) {
                bytes.push(b);
            } else {
                bytes.extend_from_slice(c.to_string().as_bytes());
            }
        }
    }
    String::from_utf8_lossy(&bytes).into_owned()
}

pub fn parse_tokens_to_segments(
    token_ids: &[usize],
    vocab: &[String],
    unicode_to_bytes: &HashMap<char, u8>,
    global_offset_ms: i64,
    segment_duration_ms: i64,
) -> Vec<TranscriptionSegment> {
    let mut segments = Vec::new();
    let mut current_text_tokens = Vec::new();
    let mut current_start_ms = global_offset_ms;

    for &id in token_ids {
        if id >= vocab.len() {
            continue;
        }
        let s = &vocab[id];

        if s.starts_with("<|") && s.ends_with("|>") {
            let inner = &s[2..s.len() - 2];
            if let Ok(val) = inner.parse::<f32>() {
                let timestamp_ms = (val * 1000.0) as i64 + global_offset_ms;
                if !current_text_tokens.is_empty() {
                    let text = decode_tokens(&current_text_tokens, unicode_to_bytes);
                    segments.push(TranscriptionSegment {
                        start_ms: current_start_ms,
                        end_ms: timestamp_ms,
                        text,
                    });
                    current_text_tokens.clear();
                }
                current_start_ms = timestamp_ms;
            }
        } else {
            if s.starts_with("<|") && s.ends_with("|>") {
                continue;
            }
            current_text_tokens.push(s.clone());
        }
    }

    if !current_text_tokens.is_empty() {
        let text = decode_tokens(&current_text_tokens, unicode_to_bytes);
        segments.push(TranscriptionSegment {
            start_ms: current_start_ms,
            end_ms: global_offset_ms + segment_duration_ms,
            text,
        });
    }

    segments
}

pub fn get_prompt_tokens(
    vocab: &[String],
    language: &Option<String>,
    translate: bool,
) -> Vec<usize> {
    let mut prompts = Vec::new();
    if let Some(start_idx) = vocab.iter().position(|s| s == "<|startoftranscript|>") {
        prompts.push(start_idx);
        
        let mut pushed_lang = false;
        if let Some(ref lang) = language {
            if lang != "auto" && !lang.is_empty() {
                let lang_token = format!("<|{}|>", lang);
                if let Some(lang_idx) = vocab.iter().position(|s| s == &lang_token) {
                    prompts.push(lang_idx);
                    pushed_lang = true;
                }
            }
        }
        
        // Only push task token if a language token was pushed, OR if language is not "auto"
        // (to prevent malformed prompt "<|startoftranscript|> <|transcribe|>" which crashes large-v3)
        let is_auto = language.as_deref() == Some("auto");
        if pushed_lang || !is_auto {
            let task_token = if translate { "<|translate|>" } else { "<|transcribe|>" };
            if let Some(task_idx) = vocab.iter().position(|s| s == task_token) {
                prompts.push(task_idx);
            }
        }
    }
    prompts
}

fn parse_json_vocabulary(content: &str) -> Option<Vec<String>> {
    // Try parsing as a list of strings first
    if let Ok(vocab) = serde_json::from_str::<Vec<String>>(content) {
        return Some(vocab);
    }
    // Try parsing as a map (token -> ID)
    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, usize>>(content) {
        let mut vocab = vec![String::new(); map.len()];
        for (token, id) in map {
            if id < vocab.len() {
                vocab[id] = token;
            }
        }
        return Some(vocab);
    }
    // Try parsing as a map (ID as string -> token)
    if let Ok(map) = serde_json::from_str::<std::collections::HashMap<String, String>>(content) {
        let mut vocab = vec![String::new(); map.len()];
        for (id_str, token) in map {
            if let Ok(id) = id_str.parse::<usize>() {
                if id < vocab.len() {
                    vocab[id] = token;
                }
            }
        }
        return Some(vocab);
    }
    None
}

pub fn load_vocabulary(model_path: &str) -> Result<Vec<String>, String> {
    let path_txt = std::path::Path::new(model_path).join("vocabulary.txt");
    let path_json = std::path::Path::new(model_path).join("vocabulary.json");
    let path_vocab_json = std::path::Path::new(model_path).join("vocab.json");

    let paths_to_try = [path_txt, path_json, path_vocab_json];

    for path in &paths_to_try {
        if path.exists() {
            let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
            let trimmed = content.trim();
            if trimmed.starts_with('[') || trimmed.starts_with('{') {
                if let Some(vocab) = parse_json_vocabulary(trimmed) {
                    return Ok(vocab);
                }
            }
            // Fall back to newline separated txt
            let vocab: Vec<String> = content.lines().map(|s| s.to_string()).collect();
            return Ok(vocab);
        }
    }

    Err(format!("Vocabulary file not found in {:?}", model_path))
}

pub fn get_num_mel_bins(model_path: &str) -> Option<usize> {
    // 1. Try preprocessor_config.json -> feature_size
    let prep_path = std::path::Path::new(model_path).join("preprocessor_config.json");
    if prep_path.exists() {
        if let Ok(content) = std::fs::read_to_string(prep_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(n) = json["feature_size"].as_u64() {
                    return Some(n as usize);
                }
            }
        }
    }

    // 2. Try config.json -> num_mel_bins
    let config_path = std::path::Path::new(model_path).join("config.json");
    if config_path.exists() {
        if let Ok(content) = std::fs::read_to_string(config_path) {
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(n) = json["num_mel_bins"].as_u64() {
                    return Some(n as usize);
                }
                if let Some(n) = json["model_spec"]["num_mel_bins"].as_u64() {
                    return Some(n as usize);
                }
            }
        }
    }

    // 3. Try fallback based on model path name
    let path_lower = model_path.to_lowercase();
    if path_lower.contains("large-v3") || path_lower.contains("large_v3") {
        return Some(128);
    }

    None
}

pub fn run_ort_vad(
    model_path: &str,
    samples: &[f32],
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
) -> Result<Vec<(usize, usize)>, String> {
    let mut session = ort::session::Session::builder()
        .map_err(|e| e.to_string())?
        .commit_from_file(model_path)
        .map_err(|e| e.to_string())?;
    run_ort_vad_with_session(&mut session, samples, threshold, min_speech_ms, min_silence_ms)
}

pub struct VadSessionState {
    pub state_v5: ndarray::Array3<f32>,
    pub h_v4: ndarray::Array3<f32>,
    pub c_v4: ndarray::Array3<f32>,
    pub context: Vec<f32>,
    pub detected_segments: Vec<(usize, usize)>,
    pub triggered: bool,
    pub speech_start: usize,
    pub temp_end: usize,
    pub processed_samples: usize,
}

impl VadSessionState {
    pub fn new() -> Self {
        Self {
            state_v5: ndarray::Array3::<f32>::zeros((2, 1, 128)),
            h_v4: ndarray::Array3::<f32>::zeros((2, 1, 64)),
            c_v4: ndarray::Array3::<f32>::zeros((2, 1, 64)),
            context: vec![0.0f32; 64],
            detected_segments: Vec::new(),
            triggered: false,
            speech_start: 0,
            temp_end: 0,
            processed_samples: 0,
        }
    }

    pub fn shift(&mut self, offset: usize) {
        self.processed_samples = self.processed_samples.saturating_sub(offset);
        self.speech_start = self.speech_start.saturating_sub(offset);
        self.temp_end = self.temp_end.saturating_sub(offset);
        
        // Shift all detected segments and filter out those that are fully drained
        self.detected_segments = self.detected_segments.iter()
            .map(|&(start, end)| (start.saturating_sub(offset), end.saturating_sub(offset)))
            .filter(|&(_, end)| end > 0)
            .collect();
    }

    pub fn reset(&mut self) {
        self.state_v5 = ndarray::Array3::<f32>::zeros((2, 1, 128));
        self.h_v4 = ndarray::Array3::<f32>::zeros((2, 1, 64));
        self.c_v4 = ndarray::Array3::<f32>::zeros((2, 1, 64));
        self.context = vec![0.0f32; 64];
        self.detected_segments.clear();
        self.triggered = false;
        self.speech_start = 0;
        self.temp_end = 0;
        self.processed_samples = 0;
    }
}

pub fn run_ort_vad_with_state(
    session: &mut ort::session::Session,
    samples: &[f32],
    state: &mut VadSessionState,
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
) -> Result<Vec<(usize, usize)>, String> {
    let chunk_size = 512;
    let min_speech_samples = (min_speech_ms as f32 * 16.0) as usize;
    let min_silence_samples = (min_silence_ms as f32 * 16.0) as usize;

    let is_v5 = session.inputs().iter().any(|i| i.name() == "state");

    let total_samples = samples.len();
    let mut offset = state.processed_samples;

    if is_v5 {
        let sr = ndarray::arr0(16000i64);

        while offset + chunk_size <= total_samples {
            let chunk = &samples[offset..offset + chunk_size];
            
            // Prepend 64-sample context
            let mut input_vec = Vec::with_capacity(64 + chunk_size);
            input_vec.extend_from_slice(&state.context);
            input_vec.extend_from_slice(chunk);

            let input = ndarray::Array2::from_shape_vec((1, 64 + chunk_size), input_vec)
                .map_err(|e| e.to_string())?;

            let input_val = ort::value::Value::from_array(input).map_err(|e| e.to_string())?;
            let sr_val = ort::value::Value::from_array(sr.clone()).map_err(|e| e.to_string())?;
            let state_val = ort::value::Value::from_array(state.state_v5.clone()).map_err(|e| e.to_string())?;

            let outputs = session.run(ort::inputs![
                "input" => input_val,
                "sr" => sr_val,
                "state" => state_val,
            ]).map_err(|e| e.to_string())?;

            let output_tensor = outputs["output"].try_extract_array::<f32>().map_err(|e| e.to_string())?;
            let speech_prob = output_tensor[[0, 0]];
            
            if offset % 16000 == 0 {
                let chunk_rms = {
                    if chunk.is_empty() { 0.0f32 } else {
                        let sum: f32 = chunk.iter().map(|&x| x * x).sum();
                        (sum / chunk.len() as f32).sqrt()
                    }
                };
                println!("[Rust] VAD v5 speech_prob: {:.4} at offset {}, RMS: {:.6}", speech_prob, offset, chunk_rms);
            }
            
            let next_state = outputs["stateN"].try_extract_array::<f32>().map_err(|e| e.to_string())?
                .into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
            state.state_v5 = next_state.to_owned();
            state.context.copy_from_slice(&chunk[chunk_size - 64..]);

            if speech_prob >= threshold {
                if state.temp_end > 0 {
                    state.temp_end = 0;
                }
                if !state.triggered {
                    state.triggered = true;
                    state.speech_start = offset;
                }
            } else if speech_prob < threshold && state.triggered {
                if state.temp_end == 0 {
                    state.temp_end = offset;
                }
                if offset - state.temp_end >= min_silence_samples {
                    if state.temp_end - state.speech_start >= min_speech_samples {
                        state.detected_segments.push((state.speech_start, state.temp_end));
                    }
                    state.triggered = false;
                    state.temp_end = 0;
                }
            }

            offset += chunk_size;
        }
    } else {
        let sr = ndarray::arr0(16000i64);

        while offset + chunk_size <= total_samples {
            let chunk = &samples[offset..offset + chunk_size];
            
            // Prepend 64-sample context
            let mut input_vec = Vec::with_capacity(64 + chunk_size);
            input_vec.extend_from_slice(&state.context);
            input_vec.extend_from_slice(chunk);

            let input = ndarray::Array2::from_shape_vec((1, 64 + chunk_size), input_vec)
                .map_err(|e| e.to_string())?;

            let input_val = ort::value::Value::from_array(input).map_err(|e| e.to_string())?;
            let sr_val = ort::value::Value::from_array(sr.clone()).map_err(|e| e.to_string())?;
            let h_val = ort::value::Value::from_array(state.h_v4.clone()).map_err(|e| e.to_string())?;
            let c_val = ort::value::Value::from_array(state.c_v4.clone()).map_err(|e| e.to_string())?;

            let outputs = session.run(ort::inputs![
                "input" => input_val,
                "sr" => sr_val,
                "h" => h_val,
                "c" => c_val,
            ]).map_err(|e| e.to_string())?;

            let output_tensor = outputs["output"].try_extract_array::<f32>().map_err(|e| e.to_string())?;
            let speech_prob = output_tensor[[0, 0]];

            if offset % 16000 == 0 {
                let chunk_rms = {
                    if chunk.is_empty() { 0.0f32 } else {
                        let sum: f32 = chunk.iter().map(|&x| x * x).sum();
                        (sum / chunk.len() as f32).sqrt()
                    }
                };
                println!("[Rust] VAD v4 speech_prob: {:.4} at offset {}, RMS: {:.6}", speech_prob, offset, chunk_rms);
            }

            let hn = outputs["hn"].try_extract_array::<f32>().map_err(|e| e.to_string())?
                .into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
            let cn = outputs["cn"].try_extract_array::<f32>().map_err(|e| e.to_string())?
                .into_dimensionality::<ndarray::Ix3>().map_err(|e| e.to_string())?;
            state.h_v4 = hn.to_owned();
            state.c_v4 = cn.to_owned();
            state.context.copy_from_slice(&chunk[chunk_size - 64..]);

            if speech_prob >= threshold {
                if state.temp_end > 0 {
                    state.temp_end = 0;
                }
                if !state.triggered {
                    state.triggered = true;
                    state.speech_start = offset;
                }
            } else if speech_prob < threshold && state.triggered {
                if state.temp_end == 0 {
                    state.temp_end = offset;
                }
                if offset - state.temp_end >= min_silence_samples {
                    if state.temp_end - state.speech_start >= min_speech_samples {
                        state.detected_segments.push((state.speech_start, state.temp_end));
                    }
                    state.triggered = false;
                    state.temp_end = 0;
                }
            }

            offset += chunk_size;
        }
    }

    state.processed_samples = offset;
    Ok(state.detected_segments.clone())
}

pub fn run_ort_vad_with_session(
    session: &mut ort::session::Session,
    samples: &[f32],
    threshold: f32,
    min_speech_ms: i32,
    min_silence_ms: i32,
) -> Result<Vec<(usize, usize)>, String> {
    let mut state = VadSessionState::new();
    let mut segments = run_ort_vad_with_state(session, samples, &mut state, threshold, min_speech_ms, min_silence_ms)?;
    let total_samples = samples.len();
    if state.triggered && total_samples - state.speech_start >= (min_speech_ms as f32 * 16.0) as usize {
        let end = if state.temp_end > 0 { state.temp_end } else { total_samples };
        segments.push((state.speech_start, end));
    }
    Ok(segments)
}

pub(crate) fn calculate_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum: f32 = samples.iter().map(|&x| x * x).sum();
    (sum / samples.len() as f32).sqrt()
}

#[cfg(target_os = "windows")]
pub(crate) fn register_thread_as_pro_audio() {
    use windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW;
    use windows_sys::core::PCWSTR;

    unsafe {
        let task_name: Vec<u16> = "Pro Audio\0".encode_utf16().collect();
        let mut task_index = 0;
        let _handle = AvSetMmThreadCharacteristicsW(
            task_name.as_ptr() as PCWSTR, 
            &mut task_index
        );
    }
}

#[cfg(not(target_os = "windows"))]
pub(crate) fn register_thread_as_pro_audio() {}

pub fn warmup_whisper_context(model_path: String, use_gpu: bool, _total_duration: f64) {
    std::thread::spawn(move || {
        #[cfg(target_os = "windows")]
        {
            use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
            unsafe {
                let thread = GetCurrentThread();
                let _ = SetThreadPriority(thread, THREAD_PRIORITY_LOWEST);
            }
        }
        
        println!("[Rust] Preloading CTranslate2 Whisper context in background for model: {}", model_path);
        let _ = get_or_create_context(&model_path, use_gpu);
    });
}
