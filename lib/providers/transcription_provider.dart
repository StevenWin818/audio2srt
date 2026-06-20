import 'dart:async';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import '../src/rust/api/ffmpeg.dart' as rust_ffmpeg;
import '../src/rust/api/whisper.dart' as rust_whisper;
import '../services/ffmpeg_service.dart';
import '../services/model_service.dart';

enum TranscriptionStatus {
  idle,
  extractingAudio,
  transcribing,
  completed,
  failed,
}

class SubtitleItem {
  int startMs;
  int endMs;
  String text;

  SubtitleItem({
    required this.startMs,
    required this.endMs,
    required this.text,
  });

  SubtitleItem clone() {
    return SubtitleItem(startMs: startMs, endMs: endMs, text: text);
  }
}

class TranscriptionProvider with ChangeNotifier {
  final FFmpegService _ffmpegService = FFmpegService();
  final ModelService _modelService = ModelService();

  FFmpegService get ffmpegService => _ffmpegService;
  ModelService get modelService => _modelService;

  // 状态属性
  int _currentTab = 0;
  int get currentTab => _currentTab;

  File? _inputMediaFile;
  File? get inputMediaFile => _inputMediaFile;

  String? _selectedModel;
  String? get selectedModel => _selectedModel;

  String _selectedLanguage = 'auto';
  String get selectedLanguage => _selectedLanguage;

  bool _translateToEnglish = false;
  bool get translateToEnglish => _translateToEnglish;

  bool _useGpu = false;
  bool get useGpu => _useGpu;

  bool _isGpuAvailable = false;
  bool get isGpuAvailable => _isGpuAvailable;

  List<rust_whisper.VulkanDeviceInfo> _vulkanDevices = [];
  List<rust_whisper.VulkanDeviceInfo> get vulkanDevices => _vulkanDevices;

  // VAD 配置
  bool _vadEnabled = false;
  bool get vadEnabled => _vadEnabled;

  double _vadThreshold = 0.010; // RMS 音能阈值
  double get vadThreshold => _vadThreshold;

  int _vadMinSpeechMs = 250; // 最小语音长度 (ms)
  int get vadMinSpeechMs => _vadMinSpeechMs;

  int _vadMinSilenceMs = 800; // 最小静音判定时间 (ms)
  int get vadMinSilenceMs => _vadMinSilenceMs;

  // Whisper 惩罚与降级参数配置
  double _temperature = 0.0;
  double get temperature => _temperature;

  double _temperatureInc = 0.2;
  double get temperatureInc => _temperatureInc;

  double _entropyThold = 2.4;
  double get entropyThold => _entropyThold;

  double _logprobThold = -1.0;
  double get logprobThold => _logprobThold;

  double _noSpeechThold = 0.6;
  double get noSpeechThold => _noSpeechThold;

  bool _noContext = false;
  bool get noContext => _noContext;

  TranscriptionStatus _status = TranscriptionStatus.idle;
  TranscriptionStatus get status => _status;

  String _statusMessage = '';
  String get statusMessage => _statusMessage;

  int _progress = 0;
  int get progress => _progress;

  List<SubtitleItem> _subtitles = [];
  List<SubtitleItem> get subtitles => _subtitles;

  // 全局模型下载状态
  String? _downloadingModelFile;
  String? get downloadingModelFile => _downloadingModelFile;

  double _downloadProgress = 0.0;
  double get downloadProgress => _downloadProgress;

  String _downloadError = '';
  String get downloadError => _downloadError;

  List<String> _downloadedModels = [];
  List<String> get downloadedModels => _downloadedModels;

  StreamSubscription? _transcriptionSub;

  Future<void> init() async {
    // 搜索系统 FFmpeg
    await _ffmpegService.findSystemFFmpeg();

    // 查询 Vulkan 设备列表
    try {
      final info = await rust_whisper.getHardwareAccelerationInfo();
      _isGpuAvailable = info.isVulkanAvailable;
      _vulkanDevices = info.devices;
      debugPrint('[TranscriptionProvider] Loaded Vulkan hardware info: isAvailable=$_isGpuAvailable, devices=${_vulkanDevices.map((d) => d.name).toList()}');
    } catch (e) {
      debugPrint('[TranscriptionProvider] Failed to load Vulkan hardware info: $e');
    }
    
    // 加载已下载模型
    _downloadedModels = await _modelService.getDownloadedModels();
    
    // 如果有已下载的模型，默认选中第一个
    if (_downloadedModels.isNotEmpty) {
      _selectedModel = _downloadedModels.first;
    } else {
      _selectedModel = ModelService.availableModels.first.filename;
    }
    notifyListeners();
  }

  /// 获取当前活跃的计算设备描述
  String get activeDeviceName {
    if (!_useGpu) {
      return 'CPU';
    }
    if (_vulkanDevices.isEmpty) {
      return 'CPU (安全回退 - 未检测到加速显卡)';
    }

    // 独立显卡 (dGPU) -> 集成显卡 (iGPU) 优先寻址
    rust_whisper.VulkanDeviceInfo? selectedDevice;
    for (final dev in _vulkanDevices) {
      final nameLower = dev.name.toLowerCase();
      final isIgpu = nameLower.contains('integrated') ||
          nameLower.contains('uhd') ||
          nameLower.contains('iris') ||
          nameLower.contains('vega') ||
          (nameLower.contains('intel') && !nameLower.contains('arc')) ||
          nameLower.contains('radeon(tm)');
      if (!isIgpu) {
        selectedDevice = dev;
        break;
      }
    }
    selectedDevice ??= _vulkanDevices.first;
    return 'GPU: ${selectedDevice.name}';
  }

  /// 智能低算力预警：未开启加速或开启但没有硬件加速显卡，且模型大小大于 400MB
  bool get showLowPowerWarning {
    if (_selectedModel == null) return false;
    final modelInfo = ModelService.availableModels.firstWhere(
      (m) => m.filename == _selectedModel,
      orElse: () => ModelService.availableModels.first,
    );

    final isGpuActive = _useGpu && _vulkanDevices.isNotEmpty;
    return !isGpuActive && modelInfo.sizeMB > 400.0;
  }

  void setCurrentTab(int index) {
    _currentTab = index;
    notifyListeners();
  }

  Future<void> downloadModel(WhisperModelInfo model) async {
    _downloadingModelFile = model.filename;
    _downloadProgress = 0.0;
    _downloadError = '';
    notifyListeners();

    await _modelService.downloadModel(
      model: model,
      onProgress: (p) {
        _downloadProgress = p;
        notifyListeners();
      },
      onSuccess: () async {
        _downloadingModelFile = null;
        _downloadProgress = 0.0;
        _selectedModel = model.filename; // 自动选中刚下载好的模型
        _downloadedModels = await _modelService.getDownloadedModels(); // 重新加载已下载列表
        notifyListeners();
      },
      onFailure: (err) {
        _downloadingModelFile = null;
        _downloadProgress = 0.0;
        _downloadError = err;
        notifyListeners();
      },
    );
  }

  void cancelDownload() {
    _modelService.cancelDownload();
    _downloadingModelFile = null;
    _downloadProgress = 0.0;
    _downloadError = '';
    notifyListeners();
  }

  Future<void> deleteModel(String filename) async {
    await _modelService.deleteModel(filename);
    _downloadedModels = await _modelService.getDownloadedModels();
    if (_selectedModel == filename) {
      _selectedModel = _downloadedModels.isNotEmpty ? _downloadedModels.first : null;
    }
    notifyListeners();
  }

  void setInputFile(File file) {
    _inputMediaFile = file;
    _status = TranscriptionStatus.idle;
    _progress = 0;
    _subtitles = [];
    _statusMessage = '已导入文件: ${p.basename(file.path)}';
    notifyListeners();
  }

  void setSelectedModel(String filename) {
    _selectedModel = filename;
    notifyListeners();
  }

  void setSelectedLanguage(String langCode) {
    _selectedLanguage = langCode;
    notifyListeners();
  }

  void setTranslate(bool translate) {
    _translateToEnglish = translate;
    notifyListeners();
  }

  void setUseGpu(bool value) {
    _useGpu = value;
    notifyListeners();
  }

  void setVadEnabled(bool value) {
    _vadEnabled = value;
    notifyListeners();
  }

  void setVadThreshold(double value) {
    _vadThreshold = value;
    notifyListeners();
  }

  void setVadMinSpeechMs(int value) {
    _vadMinSpeechMs = value;
    notifyListeners();
  }

  void setVadMinSilenceMs(int value) {
    _vadMinSilenceMs = value;
    notifyListeners();
  }

  void setTemperature(double value) {
    _temperature = value;
    notifyListeners();
  }

  void setTemperatureInc(double value) {
    _temperatureInc = value;
    notifyListeners();
  }

  void setEntropyThold(double value) {
    _entropyThold = value;
    notifyListeners();
  }

  void setLogprobThold(double value) {
    _logprobThold = value;
    notifyListeners();
  }

  void setNoSpeechThold(double value) {
    _noSpeechThold = value;
    notifyListeners();
  }

  void setNoContext(bool value) {
    _noContext = value;
    notifyListeners();
  }

  /// 核心流程：一键开始提取并转写
  Future<void> startTranscription() async {
    if (_inputMediaFile == null) {
      _setError('请先导入音频或视频文件');
      return;
    }

    if (_selectedModel == null) {
      _setError('请先选择推理模型');
      return;
    }

    // 检查模型文件是否存在
    final modelExists = await _modelService.isModelDownloaded(_selectedModel!);
    if (!modelExists) {
      _setError('所选模型未下载，请先前往模型管理面板进行下载');
      return;
    }

    // 检查 FFmpeg 可用性
    final ffmpegAvailable = await _ffmpegService.checkFFmpegAvailable();
    if (!ffmpegAvailable) {
      _setError('未找到 FFmpeg。请在设置中指定正确的 ffmpeg.exe 路径，或确保其已加入系统环境变量 (PATH)');
      return;
    }

    try {
      _progress = 0;
      _subtitles = [];
      
      // 1. 提取音频 (16kHz WAV)
      _status = TranscriptionStatus.extractingAudio;
      _statusMessage = '正在提取和重采样音频流...';
      notifyListeners();

      final tempDir = await getTemporaryDirectory();
      final timestamp = DateTime.now().millisecondsSinceEpoch;
      final wavOutputPath = p.join(
        tempDir.path,
        '${p.basenameWithoutExtension(_inputMediaFile!.path)}_temp_${timestamp}_16k.wav',
      );

      final extractedWavPath = await rust_ffmpeg.extractAudioFromMedia(
        ffmpegPath: _ffmpegService.ffmpegPath,
        inputPath: _inputMediaFile!.path,
        outputPath: wavOutputPath,
      );

      // 2. 开始 Whisper 推理转写
      _status = TranscriptionStatus.transcribing;
      _statusMessage = '正在加载 Whisper 模型进行语音转文字...';
      _progress = 0;
      notifyListeners();

      final modelPath = await _modelService.getModelPath(_selectedModel!);

      final eventStream = rust_whisper.transcribe(
        modelPath: modelPath,
        audioPath: extractedWavPath,
        language: _selectedLanguage == 'auto' ? null : _selectedLanguage,
        translate: _translateToEnglish,
        threads: 4,
        useGpu: _useGpu,
        vadEnabled: _vadEnabled,
        vadThreshold: _vadThreshold,
        vadMinSpeechMs: _vadMinSpeechMs,
        vadMinSilenceMs: _vadMinSilenceMs,
        temperature: _temperature,
        temperatureInc: _temperatureInc,
        entropyThold: _entropyThold,
        logprobThold: _logprobThold,
        noSpeechThold: _noSpeechThold,
        noContext: _noContext,
      );

      await _transcriptionSub?.cancel();
      _transcriptionSub = eventStream.listen(
        (event) {
          event.when(
            progress: (val) {
              _progress = val;
              _statusMessage = '转写推理中...';
              notifyListeners();
            },
            success: (segments) {
              _subtitles = segments
                  .map((seg) => SubtitleItem(
                        startMs: seg.startMs.toInt(),
                        endMs: seg.endMs.toInt(),
                        text: seg.text,
                      ))
                  .toList();
              _status = TranscriptionStatus.completed;
              _statusMessage = '语音转字幕完成！共生成 ${_subtitles.length} 条字幕';
              
              // 尝试删除临时 wav 文件
              try {
                final wavFile = File(extractedWavPath);
                if (wavFile.existsSync()) {
                  wavFile.deleteSync();
                }
              } catch (_) {}
              
              notifyListeners();
            },
            failure: (err) {
              _setError('转写失败: $err');
            },
          );
        },
        onError: (err) {
          _setError('桥接通信异常: $err');
        },
      );
    } catch (e) {
      _setError('处理过程中发生错误: $e');
    }
  }

  /// 更新某一条字幕的文本
  void updateSubtitleText(int index, String newText) {
    if (index >= 0 && index < _subtitles.length) {
      _subtitles[index].text = newText;
      notifyListeners();
    }
  }

  /// 更新某一条字幕的时间戳
  void updateSubtitleTimes(int index, int startMs, int endMs) {
    if (index >= 0 && index < _subtitles.length) {
      _subtitles[index].startMs = startMs;
      _subtitles[index].endMs = endMs;
      notifyListeners();
    }
  }

  /// 导出字幕文件到指定路径
  Future<void> exportSubtitles(String filePath, {bool isVtt = false}) async {
    final file = File(filePath);
    final content = isVtt ? generateVtt(_subtitles) : generateSrt(_subtitles);
    await file.writeAsString(content);
  }

  /// 压制/封装字幕到视频
  Future<String> muxSubtitlesToVideo({
    required String srtPath,
    required String outputPath,
    required bool hardBurn,
  }) async {
    if (_inputMediaFile == null) {
      throw '未导入视频源文件';
    }

    final result = await rust_ffmpeg.muxSrtToVideo(
      ffmpegPath: _ffmpegService.ffmpegPath,
      videoPath: _inputMediaFile!.path,
      srtPath: srtPath,
      outputPath: outputPath,
      hardBurn: hardBurn,
    );

    return result;
  }

  void _setError(String msg) {
    _status = TranscriptionStatus.failed;
    _statusMessage = msg;
    _progress = 0;
    notifyListeners();
  }

  // 格式化时间戳显示 00:00:00,000
  static String formatSrtTimestamp(int ms) {
    int hours = ms ~/ 3600000;
    int minutes = (ms % 3600000) ~/ 60000;
    int seconds = (ms % 60000) ~/ 1000;
    int milliseconds = ms % 1000;

    return '${hours.toString().padLeft(2, '0')}:${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')},${milliseconds.toString().padLeft(3, '0')}';
  }

  static String generateSrt(List<SubtitleItem> items) {
    final buffer = StringBuffer();
    for (int i = 0; i < items.length; i++) {
      final item = items[i];
      buffer.writeln(i + 1);
      buffer.writeln('${formatSrtTimestamp(item.startMs)} --> ${formatSrtTimestamp(item.endMs)}');
      buffer.writeln(item.text.trim());
      buffer.writeln();
    }
    return buffer.toString();
  }

  static String generateVtt(List<SubtitleItem> items) {
    final buffer = StringBuffer();
    buffer.writeln('WEBVTT');
    buffer.writeln();
    for (int i = 0; i < items.length; i++) {
      final item = items[i];
      buffer.writeln(i + 1);
      buffer.writeln('${formatSrtTimestamp(item.startMs).replaceFirst(',', '.')} --> ${formatSrtTimestamp(item.endMs).replaceFirst(',', '.')}');
      buffer.writeln(item.text.trim());
      buffer.writeln();
    }
    return buffer.toString();
  }

  @override
  void dispose() {
    _transcriptionSub?.cancel();
    super.dispose();
  }
}
