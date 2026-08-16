import 'dart:async';
import 'dart:convert';
import 'dart:ffi';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import 'package:local_notifier/local_notifier.dart';
import 'package:ffi/ffi.dart';
import 'package:window_manager/window_manager.dart';
import '../src/rust/api/ffmpeg.dart' as rust_ffmpeg;
import '../src/rust/api/silero_vad.dart' as rust_whisper;
import '../src/rust/api/stream_pipeline.dart' as rust_stream;
import '../src/rust/qwen/backend.dart' as rust_qwen_backend;
import '../services/ffmpeg_service.dart';
import '../services/model_service.dart';

// Windows FFI functions to flash taskbar icon
typedef _FindWindowWFunc = Int32 Function(Pointer<Utf16> lpClassName, Pointer<Utf16> lpWindowName);
typedef _FindWindowW = int Function(Pointer<Utf16> lpClassName, Pointer<Utf16> lpWindowName);

typedef _FlashWindowFunc = Int32 Function(Int32 hWnd, Int32 bInvert);
typedef _FlashWindow = int Function(int hWnd, int bInvert);

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

  // 高频状态：定义为私有 ValueNotifier 并对外暴露只读 ValueListenable
  final ValueNotifier<int> _progressNotifier = ValueNotifier<int>(0);
  ValueListenable<int> get progressNotifier => _progressNotifier;

  final ValueNotifier<String> _statusMessageNotifier = ValueNotifier<String>('');
  ValueListenable<String> get statusMessageNotifier => _statusMessageNotifier;

  final ValueNotifier<TranscriptionStatus> _statusNotifier = ValueNotifier<TranscriptionStatus>(TranscriptionStatus.idle);
  ValueListenable<TranscriptionStatus> get statusNotifier => _statusNotifier;

  final ValueNotifier<String> _progressDetailNotifier = ValueNotifier<String>('');
  ValueListenable<String> get progressDetailNotifier => _progressDetailNotifier;

  final ValueNotifier<String> _etaNotifier = ValueNotifier<String>('');
  ValueListenable<String> get etaNotifier => _etaNotifier;

  // Microtask 批处理通知，合并同一帧内的多次调用
  bool _isNotifyScheduled = false;
  bool _disposed = false;

  void _safeNotifyListeners() {
    if (!_isNotifyScheduled) {
      _isNotifyScheduled = true;
      scheduleMicrotask(() {
        if (!_disposed) {
          notifyListeners();
        }
        _isNotifyScheduled = false;
      });
    }
  }

  void _syncHighFreqNotifiers() {
    _progressNotifier.value = _progress;
    _statusMessageNotifier.value = _statusMessage;
    _statusNotifier.value = _status;
    _progressDetailNotifier.value = '$processedStr / $remainingStr';
    _etaNotifier.value = etaStr;
  }

  FFmpegService get ffmpegService => _ffmpegService;
  ModelService get modelService => _modelService;

  // 状态属性
  int _currentTab = 0;
  int get currentTab => _currentTab;

  bool _isSidebarExpanded = true;
  bool get isSidebarExpanded => _isSidebarExpanded;

  void setSidebarExpanded(bool value) {
    _isSidebarExpanded = value;
    _safeNotifyListeners();
  }

  bool _showInterruptConfirm = false;
  bool get showInterruptConfirm => _showInterruptConfirm;

  void setShowInterruptConfirm(bool value) {
    _showInterruptConfirm = value;
    _safeNotifyListeners();
  }

  String? _thumbnailPath;
  String? get thumbnailPath => _thumbnailPath;

  File? _inputMediaFile;
  File? get inputMediaFile => _inputMediaFile;

  // 音轨选择状态
  List<rust_ffmpeg.AudioTrackInfo> _availableTracks = [];
  List<rust_ffmpeg.AudioTrackInfo> get availableTracks => _availableTracks;

  rust_ffmpeg.AudioTrackInfo? _selectedTrack;
  rust_ffmpeg.AudioTrackInfo? get selectedTrack => _selectedTrack;

  void setSelectedTrack(rust_ffmpeg.AudioTrackInfo track) {
    _selectedTrack = track;
    _safeNotifyListeners();
  }

  Future<void> _probeAudioTracks(File file, int gen) async {
    try {
      final tracks = await rust_ffmpeg.probeAudioTracks(
        ffmpegPath: _ffmpegService.ffmpegPath,
        filePath: file.path,
      );
      if (gen != _fileGeneration) return;
      _availableTracks = tracks;
      if (_availableTracks.isNotEmpty) {
        _selectedTrack = _availableTracks.first;
      } else {
        _selectedTrack = null;
      }
      _safeNotifyListeners();
    } catch (e) {
      if (gen != _fileGeneration) return;
      debugPrint('[TranscriptionProvider] 探测音轨失败: $e');
    }
  }

  /// 模型版本 (基础模型 id，如 'qwen3-asr-0.6b')
  String? _selectedModelBase;
  String? get selectedModelBase => _selectedModelBase;

  /// 量化等级 id (如 'f16' / 'q8_0' / 'q4_k_m')
  String _selectedQuant = 'f16';
  String get selectedQuant => _selectedQuant;

  /// 兼容旧 getter: 返回模型目录名 (基础模型 id)
  String? get selectedModel => _selectedModelBase;

  /// 解码器 GGUF 文件名 (传给 Rust)
  String? get decoderFile {
    if (_selectedModelBase == null) return null;
    return 'decoder.$_selectedQuant.gguf';
  }

  /// 当前 Qwen 运行时实际加载状态 (模型/encoder/decoder 后端)
  rust_stream.QwenRuntimeStatus? _qwenRuntimeStatus;
  rust_stream.QwenRuntimeStatus? get qwenRuntimeStatus => _qwenRuntimeStatus;

  /// 查询 Rust 侧当前缓存的运行时状态
  Future<void> refreshQwenRuntimeStatus() async {
    try {
      _qwenRuntimeStatus = await rust_stream.getQwenRuntimeStatus();
      _safeNotifyListeners();
    } catch (e) {
      debugPrint('[TranscriptionProvider] getQwenRuntimeStatus error: $e');
    }
  }

  String _selectedLanguage = 'auto';
  String get selectedLanguage => _selectedLanguage;

  bool _enableDenoise = false;
  bool get enableDenoise => _enableDenoise;

  void setEnableDenoise(bool value) {
    _enableDenoise = value;
    _safeNotifyListeners();
  }

  bool _useGpu = true;
  bool get useGpu => _useGpu;

  bool _isGpuAvailable = false;
  bool get isGpuAvailable => _isGpuAvailable;

  List<rust_whisper.VulkanDeviceInfo> _vulkanDevices = [];
  List<rust_whisper.VulkanDeviceInfo> get vulkanDevices => _vulkanDevices;

  /// 获取当前活跃的 GPU 加速后端技术名称 ("CUDA" / "Vulkan")
  String get gpuTechnologyName {
    if (_vulkanDevices.any((d) => d.name.toUpperCase().contains('CUDA'))) {
      return 'CUDA';
    }
    if (_qwenRuntimeStatus != null) {
      if (_qwenRuntimeStatus!.decoderBackend.toUpperCase().contains('CUDA') ||
          _qwenRuntimeStatus!.encoderEp.toUpperCase().contains('CUDA')) {
        return 'CUDA';
      }
    }
    return 'Vulkan';
  }

  // VAD 配置
  bool _vadEnabled = true;
  bool get vadEnabled => _vadEnabled;

  double _vadThreshold = 0.5; // Silero VAD 语音概率阈值
  double get vadThreshold => _vadThreshold;

  int _vadMinSpeechMs = 300; // 最小语音长度 (ms)
  int get vadMinSpeechMs => _vadMinSpeechMs;

  int _vadMinSilenceMs = 400; // 最小静音判定时间 (ms)
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

  bool _noContext = true;
  bool get noContext => _noContext;

  bool _noStateHistory = true; // 禁用 KV 缓存记忆
  bool get noStateHistory => _noStateHistory;

  TranscriptionStatus _status = TranscriptionStatus.idle;
  TranscriptionStatus get status => _status;

  String _statusMessage = '';
  String get statusMessage => _statusMessage;

  int _progress = 0;
  int get progress => _progress;

  int _processedMs = 0;
  int _totalMs = 0;
  double _etaSeconds = 0.0;
  DateTime? _transcribeStartTime;

  String _formatDuration(double seconds) {
    if (seconds.isNaN || seconds.isInfinite || seconds < 0) {
      return '00:00';
    }
    int s = seconds.round();
    int h = s ~/ 3600;
    int m = (s % 3600) ~/ 60;
    int sec = s % 60;
    if (h > 0) {
      return '${h.toString().padLeft(2, '0')}:${m.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
    } else {
      return '${m.toString().padLeft(2, '0')}:${sec.toString().padLeft(2, '0')}';
    }
  }

  int get totalMs => _totalMs;
  String get processedStr => _formatDuration(_processedMs.toDouble() / 1000.0);
  String get remainingStr => _formatDuration((_totalMs - _processedMs).toDouble() / 1000.0);
  String get etaStr => _formatDuration(_etaSeconds);

  String get progressText {
    if (_totalMs == 0) return '$_progress%';
    return '$processedStr / $remainingStr   --$etaStr';
  }

  List<SubtitleItem> _subtitles = [];
  List<SubtitleItem> get subtitles => _subtitles;

  bool _isExported = false;
  bool get isExported => _isExported;

  bool get needsCloseConfirmation {
    if (_status == TranscriptionStatus.extractingAudio || 
        _status == TranscriptionStatus.transcribing) {
      return true;
    }
    if (_subtitles.isNotEmpty && !_isExported) {
      return true;
    }
    return false;
  }

  // 全局模型下载状态
  /// 下载标识: 基础模型为 `<baseId>|<quantId>`，aligner 为 `forced-aligner-0.6b`
  String? _downloadingModelFile;
  String? get downloadingModelFile => _downloadingModelFile;

  double _downloadProgress = 0.0;
  double get downloadProgress => _downloadProgress;

  String _downloadError = '';
  String get downloadError => _downloadError;

  /// 已下载的基础模型 id 列表
  List<String> _downloadedModels = [];
  List<String> get downloadedModels => _downloadedModels;

  /// 已下载的量化组合: `<baseId>|<quantId>`
  Set<String> _readyQuants = {};
  Set<String> get readyQuants => _readyQuants;

  bool isQuantReady(String baseId, String quantId) =>
      _readyQuants.contains('$baseId|$quantId');

  Future<void> _refreshReadyQuants() async {
    final set = <String>{};
    for (final m in ModelService.availableBaseModels) {
      for (final q in m.quants) {
        if (await _modelService.isQuantDownloaded(m.id, q.id)) {
          set.add('${m.id}|${q.id}');
        }
      }
    }
    _readyQuants = set;
  }

  StreamSubscription? _transcriptionSub;

  String? _selectedAlignerModel;
  String? get selectedAlignerModel => _selectedAlignerModel;

  void setAlignerModel(String dirName) {
    _selectedAlignerModel = dirName;
    _safeNotifyListeners();
  }

  /// 下载某个基础模型的一个量化版本
  Future<void> downloadQuant(QwenBaseModel model, QwenQuantVersion quant) async {
    _downloadingModelFile = '${model.id}|${quant.id}';
    _downloadProgress = 0.0;
    _downloadError = '';
    _safeNotifyListeners();

    final mirror = await _resolveMirror();
    await _modelService.downloadQuant(
      model: model,
      quant: quant,
      mirror: mirror,
      onProgress: (progress) {
        _downloadProgress = progress;
        _safeNotifyListeners();
      },
      onSuccess: () async {
        _downloadingModelFile = null;
        _downloadedModels = await _modelService.getDownloadedBases();
        await _refreshReadyQuants();
        _selectedModelBase = model.id;
        _selectedQuant = quant.id;
        await _saveSelectedModelPref('${model.id}|${quant.id}');
        _safeNotifyListeners();
        _preloadWhisperContext();
      },
      onFailure: (error) {
        _downloadingModelFile = null;
        _downloadError = error;
        _safeNotifyListeners();
      },
    );
  }

  /// 下载 ForcedAligner
  Future<void> downloadAligner() async {
    _downloadingModelFile = ModelService.alignerModel.dirName;
    _downloadProgress = 0.0;
    _downloadError = '';
    _safeNotifyListeners();

    final mirror = await _resolveMirror();
    await _modelService.downloadAligner(
      mirror: mirror,
      onProgress: (progress) {
        _downloadProgress = progress;
        _safeNotifyListeners();
      },
      onSuccess: () async {
        _downloadingModelFile = null;
        _selectedAlignerModel = ModelService.alignerModel.dirName;
        _safeNotifyListeners();
      },
      onFailure: (error) {
        _downloadingModelFile = null;
        _downloadError = error;
        _safeNotifyListeners();
      },
    );
  }

  Future<bool> checkAndRepairModel(String dirName) async {
    // ForcedAligner 目录的文件布局与基础模型完全不同,
    // 不能用 encoder/config 逻辑判断 (否则恒误报损坏)
    if (dirName == ModelService.alignerModel.dirName) {
      final corrupted = await _modelService.isAlignerCorrupted(dirName);
      if (!corrupted) return false;
      debugPrint('[TranscriptionProvider] Aligner $dirName corrupted. Auto repairing...');
      await _modelService.repairAligner();
      if (_selectedAlignerModel == dirName) {
        _selectedAlignerModel = null;
      }
      _statusMessage = '检测到 ForcedAligner 组件损坏，已自动清除。请前往模型管理器重新下载。';
      _safeNotifyListeners();
      return true;
    }

    final isCorrupted = await _modelService.isBaseCorrupted(dirName);
    if (isCorrupted) {
      debugPrint('[TranscriptionProvider] Model $dirName base files corrupted. Auto repairing...');
      // 只清理损坏的基础文件 (encoder/config)，保留已下载的量化 decoder
      final cleaned = await _modelService.repairBase(dirName);
      _downloadedModels = await _modelService.getDownloadedBases();
      final stillBroken = !(await _modelService.isBaseDownloaded(dirName));
      if (_selectedModelBase == dirName && stillBroken) {
        _selectedModelBase = _downloadedModels.isNotEmpty ? _downloadedModels.first : null;
      }
      if (_selectedAlignerModel == dirName) {
        _selectedAlignerModel = null;
      }
      _statusMessage = cleaned
          ? '检测到模型基础文件损坏，已清理损坏文件（已下载的量化版本已保留）。请重新下载缺失部分。'
          : '模型基础文件缺失，请重新下载。';
      _safeNotifyListeners();
      return true;
    }
    return false;
  }

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
    
    // 加载已下载模型与持久化模型偏好
    await _modelService.migrateLegacyLayout();
    _downloadedModels = await _modelService.getDownloadedBases();
    await _refreshReadyQuants();

    final savedModel = await _loadSelectedModelPref();
    if (savedModel != null) {
      // 新格式: "baseId|quantId"
      final parts = savedModel.split('|');
      final baseId = parts[0];
      final quantId = parts.length > 1 ? parts[1] : 'f16';
      final base = ModelService.baseById(baseId);
      if (base != null && await _modelService.isBaseDownloaded(baseId)) {
        _selectedModelBase = baseId;
        if (base.quantById(quantId) != null) {
          _selectedQuant = quantId;
        }
      }
    }
    if (_selectedModelBase == null) {
      if (_downloadedModels.isNotEmpty) {
        _selectedModelBase = _downloadedModels.first;
        // 该模型已下载的量化中优先 f16，否则取第一个
        final base = ModelService.baseById(_selectedModelBase!);
        if (base != null) {
          if (await _modelService.isQuantDownloaded(_selectedModelBase!, 'f16')) {
            _selectedQuant = 'f16';
          } else {
            for (final q in base.quants) {
              if (await _modelService.isQuantDownloaded(_selectedModelBase!, q.id)) {
                _selectedQuant = q.id;
                break;
              }
            }
          }
        }
      } else {
        _selectedModelBase = ModelService.availableBaseModels.first.id;
        _selectedQuant = ModelService.availableBaseModels.first.quants.first.id;
      }
    }
    notifyListeners();
    _warmupMirrorSpeed();
    // 不做启动预加载: 实测预加载对推理总时长几乎无提升
    Future.delayed(const Duration(seconds: 4), () => refreshQwenRuntimeStatus());
  }

  Future<void> _saveSelectedModelPref(String modelKey) async {
    try {
      final dir = await getApplicationSupportDirectory();
      final file = File(p.join(dir.path, 'app_settings.json'));
      Map<String, dynamic> map = {};
      if (await file.exists()) {
        try {
          map = jsonDecode(await file.readAsString()) as Map<String, dynamic>;
        } catch (_) {}
      }
      map['selected_model'] = modelKey;
      await file.writeAsString(jsonEncode(map));
    } catch (e) {
      debugPrint('[TranscriptionProvider] Save model pref error: $e');
    }
  }

  Future<String?> _loadSelectedModelPref() async {
    try {
      final dir = await getApplicationSupportDirectory();
      final file = File(p.join(dir.path, 'app_settings.json'));
      if (await file.exists()) {
        final map = jsonDecode(await file.readAsString()) as Map<String, dynamic>;
        final raw = map['selected_model'] as String?;
        if (raw == null) return null;
        // 旧格式迁移: "qwen3-asr-0.6b-f16" -> "qwen3-asr-0.6b|f16"
        if (!raw.contains('|')) {
          const suffixes = [
            ('-f16', 'f16'),
            ('-q8_0', 'q8_0'),
            ('-q6_k', 'q6_k'),
            ('-q4_k_m', 'q4_k_m'),
            ('-q4_k', 'q4_k_m'),
          ];
          for (final (suffix, quant) in suffixes) {
            if (raw.endsWith(suffix)) {
              return '${raw.substring(0, raw.length - suffix.length)}|$quant';
            }
          }
          return '$raw|f16';
        }
        return raw;
      }
    } catch (e) {
      debugPrint('[TranscriptionProvider] Load model pref error: $e');
    }
    return null;
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

  /// 智能低算力预警：未开启加速或开启但没有硬件加速显卡，且模型为 1.7B (大模型)
  bool get showLowPowerWarning {
    if (_selectedModelBase == null) return false;
    final isBigModel = _selectedModelBase == 'qwen3-asr-1.7b';
    final isGpuActive = _useGpu && _vulkanDevices.isNotEmpty;
    return !isGpuActive && isBigModel;
  }

  void setCurrentTab(int index) {
    _currentTab = index;
    notifyListeners();
  }

  void cancelDownload() {
    _modelService.cancelDownload();
    _downloadingModelFile = null;
    _downloadProgress = 0.0;
    _downloadError = '';
    notifyListeners();
  }

  // ===== 下载源选择 =====
  /// 'auto' 表示下载前自动测速选择最快源
  String _selectedMirrorId = 'auto';
  String get selectedMirrorId => _selectedMirrorId;

  /// 当前实际生效的镜像 (auto 模式下为最近一次测速结果)
  ModelMirror? _activeMirror;
  ModelMirror? get activeMirror => _activeMirror;

  /// 显示文本: "自动（当前：HF-Mirror 国内镜像站）" 或源名
  String get selectedMirrorLabel {
    if (_selectedMirrorId == 'auto') {
      final name = _activeMirror?.name;
      return name == null ? '自动（待测速）' : '自动（当前：$name）';
    }
    return ModelService.mirrorById(_selectedMirrorId)?.name ?? _selectedMirrorId;
  }

  void setSelectedMirrorId(String id) {
    if (ModelService.mirrorById(id) == null) return;
    _selectedMirrorId = id;
    notifyListeners();
  }

  /// 解析下载源: auto 时测速并记住当前生效源
  Future<ModelMirror> _resolveMirror() async {
    if (_selectedMirrorId == 'auto') {
      final picked = await _modelService.pickFastestMirror();
      _activeMirror = picked;
      _safeNotifyListeners();
      return picked;
    }
    return ModelService.mirrorById(_selectedMirrorId)!;
  }

  /// 后台预热一次自动测速 (仅 auto 模式, 不阻塞 UI)
  Future<void> _warmupMirrorSpeed() async {
    if (_selectedMirrorId != 'auto') return;
    try {
      await _resolveMirror();
    } catch (e) {
      debugPrint('[TranscriptionProvider] Mirror speed warmup error: $e');
    }
  }

  /// 查找指定基础模型下已下载的量化 (f16 优先, 其次按官方档位顺序)
  Future<String?> _findDownloadedQuant(String baseId) async {
    if (await _modelService.isQuantDownloaded(baseId, 'f16')) return 'f16';
    final base = ModelService.baseById(baseId);
    if (base != null) {
      for (final q in base.quants) {
        if (await _modelService.isQuantDownloaded(baseId, q.id)) return q.id;
      }
    }
    return null;
  }

  Future<void> deleteQuant(String baseId, String quantId) async {
    await _modelService.deleteQuant(baseId, quantId);
    _downloadedModels = await _modelService.getDownloadedBases();
    await _refreshReadyQuants();

    // 删除的是当前正在使用的量化时, 自动回退到该模型其他已下载档位;
    // 若该模型已无任何量化, 则切换到其他已下载模型, 避免转写启动时报"未下载"
    if (_selectedModelBase == baseId && _selectedQuant == quantId) {
      final fallback = await _findDownloadedQuant(baseId);
      if (fallback != null) {
        _selectedQuant = fallback;
        _saveSelectedModelPref('$baseId|$fallback');
        _preloadWhisperContext();
      } else {
        final otherBase = _downloadedModels.firstWhere(
          (id) => id != baseId,
          orElse: () => baseId,
        );
        if (otherBase != baseId) {
          _selectedModelBase = otherBase;
          _selectedQuant = await _findDownloadedQuant(otherBase) ?? 'f16';
          _saveSelectedModelPref('$otherBase|$_selectedQuant');
          _preloadWhisperContext();
        } else {
          // 全部量化均已删除: 保留模型选择, 下次下载后自动复用
          _selectedQuant = 'f16';
          _saveSelectedModelPref('$baseId|f16');
        }
      }
    }
    notifyListeners();
  }

  Future<void> deleteAlignerModel() async {
    await _modelService.deleteAligner();
    _selectedAlignerModel = null;
    notifyListeners();
  }

  int _fileGeneration = 0;

  void setInputFile(File file) {
    final gen = ++_fileGeneration;
    _inputMediaFile = file;
    _status = TranscriptionStatus.idle;
    _progress = 0;
    _subtitles = const [];
    _isExported = false;
    _statusMessage = '已导入文件: ${p.basename(file.path)}';
    _thumbnailPath = null;
    _totalMs = 0;
    _processedMs = 0;
    _showInterruptConfirm = false;
    
    // 清空旧音轨状态并触发异步探测
    _availableTracks = [];
    _selectedTrack = null;
    _probeAudioTracks(file, gen);
    _probeMediaDuration(file, gen);
    _extractThumbnail(file, gen);

    _syncHighFreqNotifiers();
    _safeNotifyListeners();
  }

  Future<void> _extractThumbnail(File file, int gen) async {
    final ext = p.extension(file.path).toLowerCase();
    final isVideoExt = ['.mp4', '.mkv', '.avi', '.mov', '.wmv', '.flv', '.webm'].contains(ext);
    if (!isVideoExt) {
      if (gen != _fileGeneration) return;
      _thumbnailPath = null;
      _safeNotifyListeners();
      return;
    }

    try {
      final tempDir = await getTemporaryDirectory();
      final thumbName = 'thumb_${DateTime.now().millisecondsSinceEpoch}.jpg';
      final thumbPath = p.join(tempDir.path, thumbName);
      
      final result = await Process.run(
        _ffmpegService.ffmpegPath,
        [
          '-y',
          '-ss', '00:00:01',
          '-i', file.path,
          '-vframes', '1',
          '-f', 'image2',
          thumbPath,
        ],
      );
      if (gen != _fileGeneration) return;
      if (result.exitCode == 0 && await File(thumbPath).exists()) {
        _thumbnailPath = thumbPath;
        _safeNotifyListeners();
      } else {
        _thumbnailPath = null;
        _safeNotifyListeners();
      }
    } catch (e) {
      if (gen != _fileGeneration) return;
      debugPrint('[TranscriptionProvider] Extract thumbnail failed: $e');
      _thumbnailPath = null;
      _safeNotifyListeners();
    }
  }

  Future<void> _probeMediaDuration(File file, int gen) async {
    try {
      final result = await Process.run(
        _ffmpegService.ffmpegPath,
        ['-i', file.path],
      );
      if (gen != _fileGeneration) return;
      final output = result.stderr.toString();
      final durationRegex = RegExp(r'Duration:\s*(\d+):(\d+):(\d+\.\d+)');
      final match = durationRegex.firstMatch(output);
      if (match != null) {
        final hours = int.parse(match.group(1)!);
        final minutes = int.parse(match.group(2)!);
        final seconds = double.parse(match.group(3)!);
        final totalSeconds = hours * 3600 + minutes * 60 + seconds;
        _totalMs = (totalSeconds * 1000).toInt();
        _processedMs = 0;
        _syncHighFreqNotifiers();
        _safeNotifyListeners();
        _preloadWhisperContext();
      }
    } catch (e) {
      if (gen != _fileGeneration) return;
      debugPrint('[TranscriptionProvider] Probe duration failed: $e');
    }
  }

  Future<void> setSelectedModelBase(String baseId) async {
    if (_selectedModelBase == baseId) return;
    _selectedModelBase = baseId;

    // 自动更正量化精度：若当前量化在目标 baseId 未下载，自动校准为该模型已下载的有效量化
    final base = ModelService.baseById(baseId);
    if (base != null) {
      final isQuantValid = await _modelService.isQuantDownloaded(baseId, _selectedQuant);
      if (!isQuantValid) {
        if (await _modelService.isQuantDownloaded(baseId, 'f16')) {
          _selectedQuant = 'f16';
        } else {
          for (final q in base.quants) {
            if (await _modelService.isQuantDownloaded(baseId, q.id)) {
              _selectedQuant = q.id;
              break;
            }
          }
        }
      }
    }

    _saveSelectedModelPref('$baseId|$_selectedQuant');
    _safeNotifyListeners();
    _preloadWhisperContext();
  }

  void setSelectedQuant(String quantId) {
    if (_selectedQuant == quantId) return;
    _selectedQuant = quantId;
    if (_selectedModelBase != null) {
      _saveSelectedModelPref('$_selectedModelBase|$quantId');
    }
    _safeNotifyListeners();
    _preloadWhisperContext();
  }

  void setSelectedLanguage(String langCode) {
    _selectedLanguage = langCode;
    _safeNotifyListeners();
  }

  void setUseGpu(bool value) {
    _useGpu = value;
    _safeNotifyListeners();
    _preloadWhisperContext();
  }

  Timer? _preloadTimer;

  Future<void> _preloadWhisperContext() async {
    _preloadTimer?.cancel();
    _preloadTimer = Timer(const Duration(milliseconds: 200), () async {
      final baseId = _selectedModelBase;
      var quant = _selectedQuant;
      if (baseId == null) return;
      try {
        final baseExists = await _modelService.isBaseDownloaded(baseId);
        var quantExists = await _modelService.isQuantDownloaded(baseId, quant);

        // 如果当前 quant 在新 baseId 上未下载，自动寻找并修复为有效量化
        if (!quantExists) {
          final base = ModelService.baseById(baseId);
          if (base != null) {
            if (await _modelService.isQuantDownloaded(baseId, 'f16')) {
              quant = 'f16';
              _selectedQuant = 'f16';
              quantExists = true;
            } else {
              for (final q in base.quants) {
                if (await _modelService.isQuantDownloaded(baseId, q.id)) {
                  quant = q.id;
                  _selectedQuant = q.id;
                  quantExists = true;
                  break;
                }
              }
            }
          }
        }

        if (!baseExists || !quantExists) {
          _downloadedModels = await _modelService.getDownloadedBases();
          _safeNotifyListeners();
          return;
        }

        final modelDir = await _modelService.getModelPath(baseId);
        final alignerDir = _selectedAlignerModel != null ? await _modelService.getModelPath(_selectedAlignerModel!) : null;
        final decoderFile = 'decoder.$quant.gguf';
        debugPrint('[TranscriptionProvider] Preloading Qwen decoder (VRAM) in background: $modelDir (decoder=$decoderFile, useGpu=$_useGpu)');
        // 预加载只加载 decoder GGUF (+ aligner) 到显存/内存缓存;
        // encoder ONNX (~1GB RAM) 由 Rust 端在推理开始时懒加载, 推理结束后释放。
        rust_stream.preloadQwenModel(
          asrModelDir: modelDir,
          alignerModelDir: alignerDir,
          decoderFile: decoderFile,
          encoderBackend: _useGpu
              ? rust_qwen_backend.EncoderBackend.auto
              : rust_qwen_backend.EncoderBackend.cpu,
          decoderBackend: _useGpu
              ? rust_qwen_backend.DecoderBackend.auto
              : rust_qwen_backend.DecoderBackend.cpu,
        );
        // 预加载完成后刷新"模型加载状态"面板 (模型在后台线程加载, 延迟轮询)。
        // 开关 GPU 加速 / 切换模型量化后, 面板显示的实际加载位置随之更新。
        // 后台加载 (尤其 CPU 冷启动) 可能超过数秒, 轮询直到运行时状态刷新,
        // 最多约 24s 后停止, 避免长时间占用定时器。
        _preloadTimer?.cancel();
        var pollAttempts = 0;
        String? lastSignature;
        _preloadTimer = Timer.periodic(const Duration(seconds: 4), (timer) {
          pollAttempts++;
          refreshQwenRuntimeStatus();
          final st = _qwenRuntimeStatus;
          final signature = st == null
              ? null
              : '${st.encoderEp}|${st.decoderBackend}|${st.decoderFile}';
          final reloaded = st != null && signature != lastSignature && pollAttempts > 1;
          lastSignature = signature ?? lastSignature;
          if (reloaded || pollAttempts >= 6) {
            timer.cancel();
          }
        });
      } catch (e) {
        debugPrint('[TranscriptionProvider] Qwen ASR model preload error: $e');
      }
    });
  }

  void setVadEnabled(bool value) {
    _vadEnabled = value;
    _safeNotifyListeners();
  }

  void setVadThreshold(double value) {
    _vadThreshold = value;
    _safeNotifyListeners();
  }

  void setVadMinSpeechMs(int value) {
    _vadMinSpeechMs = value;
    _safeNotifyListeners();
  }

  void setVadMinSilenceMs(int value) {
    _vadMinSilenceMs = value;
    _safeNotifyListeners();
  }

  void setTemperature(double value) {
    _temperature = value;
    _safeNotifyListeners();
  }

  void setTemperatureInc(double value) {
    _temperatureInc = value;
    _safeNotifyListeners();
  }

  void setEntropyThold(double value) {
    _entropyThold = value;
    _safeNotifyListeners();
  }

  void setLogprobThold(double value) {
    _logprobThold = value;
    _safeNotifyListeners();
  }

  void setNoContext(bool value) {
    _noContext = value;
    _safeNotifyListeners();
  }

  void setNoStateHistory(bool value) {
    _noStateHistory = value;
    _safeNotifyListeners();
  }

  /// 核心流程：一键开始提取并转写 (全新三级流式降噪与转写管道)
  Future<void> startTranscription() async {
    if (_inputMediaFile == null) {
      _setError('请先导入音频或视频文件');
      return;
    }

    if (_selectedModelBase == null) {
      _setError('请先选择推理模型');
      return;
    }

    final baseId = _selectedModelBase!;
    final quant = _selectedQuant;

    // 检查模型文件是否存在与完整性
    final baseExists = await _modelService.isBaseDownloaded(baseId);
    if (!baseExists) {
      _setError('所选模型未下载，请先前往模型管理面板进行下载');
      return;
    }
    final quantExists = await _modelService.isQuantDownloaded(baseId, quant);
    if (!quantExists) {
      _setError('所选量化版本 ($quant) 未下载，请先前往模型管理面板下载');
      return;
    }

    final isCorrupted = await _modelService.isBaseCorrupted(baseId);
    if (isCorrupted) {
      await checkAndRepairModel(baseId);
      _setError('检测到所选模型已损坏，已自动为您清除损坏缓存！请前往模型管理器重新下载。');
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
      _processedMs = 0;
      _totalMs = 0;
      _etaSeconds = 0.0;
      _transcribeStartTime = DateTime.now();
      _subtitles = const [];
      _status = TranscriptionStatus.transcribing;
      _isExported = false;
      _syncHighFreqNotifiers();

      // 在推理（转写）开始时，初始化任务栏进度条为 0%
      try {
        windowManager.setProgressBar(0.0);
      } catch (e) {
        debugPrint('Failed to set taskbar progress: $e');
      }
      
      String dfModelPath = "";
      if (_enableDenoise) {
        _statusMessage = '正在初始化 DeepFilterNet 降噪引擎...';
        _syncHighFreqNotifiers();
        _safeNotifyListeners();
        // 准备 DeepFilterNet 降噪模型
        dfModelPath = await _modelService.prepareDFModel();
      }

      String vadModelPath = "";
      if (_vadEnabled) {
        _statusMessage = '正在准备 Silero VAD 引擎...';
        _syncHighFreqNotifiers();
        _safeNotifyListeners();
        vadModelPath = await _modelService.prepareVADModel();
      }

      _statusMessage = '正在准备语音识别模型 (大型模型首次加载可能需要数秒)...';
      _syncHighFreqNotifiers();
      _safeNotifyListeners();

      final modelPath = await _modelService.getModelPath(baseId);
      final decoderFile = 'decoder.$quant.gguf';

      final eventStream = rust_stream.transcribeStream(
        config: rust_stream.PipelineConfig(
          ffmpegPath: _ffmpegService.ffmpegPath,
          inputPath: _inputMediaFile!.path,
          modelPath: modelPath,
          asrModelDir: modelPath,
          alignerModelDir: _selectedAlignerModel != null ? await _modelService.getModelPath(_selectedAlignerModel!) : null,
          decoderFile: decoderFile,
          contextPrompt: null,
          encoderBackend: _useGpu
              ? rust_qwen_backend.EncoderBackend.auto
              : rust_qwen_backend.EncoderBackend.cpu,
          decoderBackend: _useGpu
              ? rust_qwen_backend.DecoderBackend.auto
              : rust_qwen_backend.DecoderBackend.cpu,
          timestampMode: rust_stream.TimestampMode.precise,
          vadModelPath: vadModelPath,
          dfModelPath: dfModelPath,
          language: _selectedLanguage == 'auto' ? null : _selectedLanguage,
          threads: 4,
          useGpu: _useGpu,
          toSimplified: _selectedLanguage == 'zh' || _selectedLanguage == 'auto',
          enableDenoise: _enableDenoise,
          vadEnabled: _vadEnabled,
          vadThreshold: _vadThreshold,
          vadMinSpeechMs: _vadMinSpeechMs,
          vadMinSilenceMs: _vadMinSilenceMs,
          noContext: _noContext,
          noStateHistory: _noStateHistory,
          temperature: _temperature,
          temperatureInc: _temperatureInc,
          entropyThold: _entropyThold,
          logprobThold: _logprobThold,
          selectedAudioTrack: _selectedTrack?.index,
        ),
      );

      await _transcriptionSub?.cancel();
      _transcriptionSub = eventStream.listen(
        (event) async {
          switch (event) {
            case rust_whisper.TranscriptionEvent_Progress(:final field0):
              _progress = field0;
              _statusMessage = '正在生成字幕...';
              _syncHighFreqNotifiers();

              try {
                windowManager.setProgressBar(field0 / 100.0);
              } catch (e) {
                debugPrint('Failed to set taskbar progress: $e');
              }

            case rust_whisper.TranscriptionEvent_ProgressDetail(:final processedMs, :final totalMs):
              _processedMs = processedMs.toInt();
              _totalMs = totalMs.toInt();
              if (_transcribeStartTime != null) {
                final elapsedRealTimeSecs = DateTime.now().difference(_transcribeStartTime!).inMilliseconds / 1000.0;
                final processedMediaSecs = _processedMs / 1000.0;
                final remainingMediaSecs = (_totalMs - _processedMs) / 1000.0;
                if (processedMediaSecs > 0) {
                  final speed = processedMediaSecs / elapsedRealTimeSecs;
                  _etaSeconds = remainingMediaSecs / speed;
                } else {
                  _etaSeconds = 0.0;
                }
              }
              _statusMessage = '正在生成字幕...';
              _syncHighFreqNotifiers();

            case rust_whisper.TranscriptionEvent_Segment(:final field0):
              _subtitles = [
                ..._subtitles,
                SubtitleItem(
                  startMs: field0.startMs.toInt(),
                  endMs: field0.endMs.toInt(),
                  text: field0.text,
                )
              ];
              _safeNotifyListeners();

            case rust_whisper.TranscriptionEvent_Success(:final field0):
              _subtitles = field0
                  .map((seg) => SubtitleItem(
                        startMs: seg.startMs.toInt(),
                        endMs: seg.endMs.toInt(),
                        text: seg.text,
                      ))
                  .toList();
              _status = TranscriptionStatus.completed;
              _statusMessage = '语音转字幕完成！共生成 ${_subtitles.length} 条字幕';
              _syncHighFreqNotifiers();
              _safeNotifyListeners();

              try {
                windowManager.setProgressBar(-1.0);
              } catch (e) {
                debugPrint('Failed to clear taskbar progress: $e');
              }

              _flashTaskbarIcon();

              try {
                final filename = _inputMediaFile != null ? p.basename(_inputMediaFile!.path) : '音视频文件';
                final notification = LocalNotification(
                  title: '语音识别已完成',
                  body: '文件: $filename\n成功生成 ${_subtitles.length} 条字幕。',
                );
                notification.onClick = () async {
                  try {
                    await windowManager.show();
                    await windowManager.focus();
                  } catch (e) {
                    debugPrint('Failed to show window on notification click: $e');
                  }
                };
                notification.show();
              } catch (e) {
                debugPrint('[TranscriptionProvider] 发送成功通知异常: $e');
              }

            case rust_whisper.TranscriptionEvent_Failure(:final field0):
              if (field0.contains('Failed to load Qwen runtime') ||
                  field0.contains('model not found') ||
                  field0.contains('Corrupt') ||
                  field0.contains('ONNX') ||
                  field0.contains('manifest')) {
                var repaired = false;
                if (_selectedModelBase != null) {
                  repaired = await checkAndRepairModel(_selectedModelBase!) || repaired;
                }
                if (_selectedAlignerModel != null) {
                  repaired = await checkAndRepairModel(_selectedAlignerModel!) || repaired;
                }
                if (repaired) {
                  _setError('模型加载失败（检测到文件已损毁），已自动清除损坏缓存！请在模型管理器中重新下载。');
                } else {
                  _setError('模型加载失败: $field0');
                }
              } else {
                _setError('转写失败: $field0');
              }

              // 清除状态栏进度条
              try {
                windowManager.setProgressBar(-1.0);
              } catch (e) {
                debugPrint('Failed to clear taskbar progress: $e');
              }

              // 推理出错也闪烁提醒用户
              _flashTaskbarIcon();

              // 推理失败发送本地通知
              try {
                final filename = _inputMediaFile != null ? p.basename(_inputMediaFile!.path) : '音视频文件';
                final notification = LocalNotification(
                  title: '语音识别失败',
                  body: '文件: $filename\n转换出错: $field0',
                );
                notification.show();
              } catch (e) {
                debugPrint('[TranscriptionProvider] 发送失败通知异常: $e');
              }
          }
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
      final updatedList = List<SubtitleItem>.from(_subtitles);
      updatedList[index] = SubtitleItem(
        startMs: _subtitles[index].startMs,
        endMs: _subtitles[index].endMs,
        text: newText,
      );
      _subtitles = updatedList;
      _safeNotifyListeners();
    }
  }

  /// 更新某一条字幕的时间戳
  void updateSubtitleTimes(int index, int startMs, int endMs) {
    if (index >= 0 && index < _subtitles.length) {
      final updatedList = List<SubtitleItem>.from(_subtitles);
      updatedList[index] = SubtitleItem(
        startMs: startMs,
        endMs: endMs,
        text: _subtitles[index].text,
      );
      _subtitles = updatedList;
      _safeNotifyListeners();
    }
  }
  Future<void> convertSubtitlesToChinese(bool toSimplified) async {
    try {
      final texts = _subtitles.map((e) => e.text).toList();
      final converted = await rust_whisper.convertChineseList(texts: texts, toSimplified: toSimplified);
      if (converted.length != _subtitles.length) {
        debugPrint('[TranscriptionProvider] Converted list length mismatch: expected ${_subtitles.length}, got ${converted.length}');
        return;
      }
      final newList = <SubtitleItem>[];
      for (var i = 0; i < _subtitles.length; i++) {
        newList.add(SubtitleItem(
          startMs: _subtitles[i].startMs,
          endMs: _subtitles[i].endMs,
          text: converted[i],
        ));
      }
      _subtitles = newList;
    } catch (e) {
      debugPrint('Conversion error: $e');
    }
    _safeNotifyListeners();
  }

  /// 导出字幕文件到指定路径
  Future<void> exportSubtitles(String filePath, {bool isVtt = false}) async {
    final file = File(filePath);
    final content = isVtt ? generateVtt(_subtitles) : generateSrt(_subtitles);
    await file.writeAsString(content);
    _isExported = true;
    _safeNotifyListeners();
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

    String? resultPath;
    final muxStream = rust_ffmpeg.muxSrtToVideo(
      ffmpegPath: _ffmpegService.ffmpegPath,
      videoPath: _inputMediaFile!.path,
      srtPath: srtPath,
      outputPath: outputPath,
      hardBurn: hardBurn,
    );

    await for (final event in muxStream) {
      if (event.progress != null) {
        _progress = event.progress!;
        _syncHighFreqNotifiers();
      } else if (event.success != null) {
        resultPath = event.success;
      } else if (event.error != null) {
        throw Exception(event.error);
      }
    }

    if (resultPath == null) {
      throw Exception('Failed to mux subtitles');
    }
    return resultPath;
  }

  void _setError(String msg) {
    _status = TranscriptionStatus.failed;
    _statusMessage = msg;
    _progress = 0;
    _syncHighFreqNotifiers();
    _safeNotifyListeners();
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

  Future<void> cancelTranscription() async {
    rust_stream.cancelTranscriptionBackend();
    await _transcriptionSub?.cancel();
    _transcriptionSub = null;
    _status = TranscriptionStatus.idle;
    _statusMessage = '转写任务已手动停止';
    _syncHighFreqNotifiers();
    _safeNotifyListeners();

    // 手动取消也清除状态栏进度条
    try {
      windowManager.setProgressBar(-1.0);
    } catch (e) {
      debugPrint('Failed to clear taskbar progress: $e');
    }
  }

  static DynamicLibrary? _user32;

  // Windows FFI 动态查找并闪烁状态栏/任务栏图标
  void _flashTaskbarIcon() {
    if (!Platform.isWindows) return;
    try {
      _user32 ??= DynamicLibrary.open('user32.dll');
      final user32 = _user32!;
      final findWindow = user32.lookupFunction<_FindWindowWFunc, _FindWindowW>('FindWindowW');
      final flashWindow = user32.lookupFunction<_FlashWindowFunc, _FlashWindow>('FlashWindow');

      final className = 'FLUTTER_RUNNER_WIN32_WINDOW'.toNativeUtf16();
      final hwnd = findWindow(className, nullptr);
      calloc.free(className);

      if (hwnd != 0) {
        int count = 0;
        Timer.periodic(const Duration(milliseconds: 500), (timer) {
          if (count >= 6) {
            timer.cancel();
          } else {
            flashWindow(hwnd, 1);
            count++;
          }
        });
      }
    } catch (e) {
      debugPrint('[TranscriptionProvider] _flashTaskbarIcon error: $e');
    }
  }

  @override
  void dispose() {
    _disposed = true;
    _preloadTimer?.cancel();
    _transcriptionSub?.cancel();
    _progressNotifier.dispose();
    _statusMessageNotifier.dispose();
    _statusNotifier.dispose();
    _progressDetailNotifier.dispose();
    _etaNotifier.dispose();
    super.dispose();
  }
}
