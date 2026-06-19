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
