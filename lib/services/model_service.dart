import 'dart:io';
import 'dart:ffi';
import 'package:dio/dio.dart';
import 'package:ffi/ffi.dart' show malloc;
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/services.dart' show rootBundle;

// dart:io 的 File 在此 SDK 中没有 hard link API，直接调用 kernel32.CreateHardLinkW
// 创建 NTFS 硬链接 (无需管理员权限，同卷不占额外磁盘空间)。
typedef _CreateHardLinkWNative = Int32 Function(
    Pointer<Uint16> linkPath, Pointer<Uint16> existingPath, Pointer<Void> securityAttributes);
typedef _CreateHardLinkWDart = int Function(
    Pointer<Uint16> linkPath, Pointer<Uint16> existingPath, Pointer<Void> securityAttributes);

final _createHardLinkW = DynamicLibrary.open('kernel32.dll')
    .lookupFunction<_CreateHardLinkWNative, _CreateHardLinkWDart>('CreateHardLinkW');

Pointer<Uint16> _toNativeUtf16(String s) {
  final units = s.codeUnits;
  final ptr = malloc<Uint16>(units.length + 1);
  for (var i = 0; i < units.length; i++) {
    ptr[i] = units[i];
  }
  ptr[units.length] = 0;
  return ptr;
}

bool _hardLink(String existingPath, String linkPath) {
  final existing = _toNativeUtf16(existingPath);
  final link = _toNativeUtf16(linkPath);
  try {
    return _createHardLinkW(link, existing, nullptr) != 0;
  } finally {
    malloc.free(existing);
    malloc.free(link);
  }
}

enum ModelType { asr, aligner }

class ModelMirror {
  final String id;
  final String name;
  final String baseUrl;
  final String description;
  int? latencyMs;

  ModelMirror({
    required this.id,
    required this.name,
    required this.baseUrl,
    required this.description,
    this.latencyMs,
  });
}

class QwenModelFile {
  final String filename;
  final String urlPath;
  final double sizeMB;

  QwenModelFile({
    required this.filename,
    required this.urlPath,
    required this.sizeMB,
  });
}

class QwenModelInfo {
  final String id;
  final String name;
  final String description;
  final String dirName;
  final String size;
  final double sizeMB;
  final ModelType type;
  final List<QwenModelFile> files;

  /// 同一基础模型的量化变体共享同一个 encoder。
  /// 例如 0.6B 的三个变体 encoderGroup 均为 'qwen3-asr-0.6b'，
  /// encoder.onnx / config.json 只在组内首次下载时下载一次，其余变体硬链接复用。
  final String? encoderGroup;

  QwenModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.dirName,
    required this.size,
    required this.sizeMB,
    required this.type,
    required this.files,
    this.encoderGroup,
  });
}

class ModelService {
  static final List<ModelMirror> availableMirrors = [
    ModelMirror(
      id: 'hf-mirror',
      name: 'HF-Mirror 国内镜像站 (推荐)',
      baseUrl: 'https://hf-mirror.com',
      description: '国内高速 CDN 加速镜像，适合国内网络环境',
    ),
    ModelMirror(
      id: 'huggingface',
      name: 'HuggingFace 官方源',
      baseUrl: 'https://huggingface.co',
      description: '官方直连源，海外网络环境推荐',
    ),
    ModelMirror(
      id: 'modelscope',
      name: 'ModelScope 魔搭社区',
      baseUrl: 'https://modelscope.cn',
      description: '阿里魔搭社区镜像节点',
    ),
  ];

  static final List<QwenModelInfo> availableQwenModels = [
    // ===== Qwen3-ASR 0.6B =====
    // 默认: 全量 F16 (未量化，最高精度)
    QwenModelInfo(
      id: 'qwen3-asr-0.6b-f16',
      name: 'Qwen3-ASR 0.6B · 全量 F16',
      description: '官方全精度解码器 · 最高识别精度 (默认推荐)',
      dirName: 'qwen3-asr-0.6b-f16',
      encoderGroup: 'qwen3-asr-0.6b',
      size: '2.15 GB',
      sizeMB: 2150.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/encoder.onnx',
          sizeMB: 711.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.f16.gguf',
          sizeMB: 1439.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'qwen3-asr-0.6b-q8',
      name: 'Qwen3-ASR 0.6B · 量化 Q8_0',
      description: '8-bit 量化 · 高精度与速度均衡 (显存 ≥ 4GB)',
      dirName: 'qwen3-asr-0.6b-q8',
      encoderGroup: 'qwen3-asr-0.6b',
      size: '1.48 GB',
      sizeMB: 1478.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/encoder.onnx',
          sizeMB: 711.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q8_0.gguf',
          sizeMB: 767.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'qwen3-asr-0.6b',
      name: 'Qwen3-ASR 0.6B · 量化 Q4_K_M',
      description: '4-bit 量化 · 极速低内存 (推荐显存 < 4GB)',
      dirName: 'qwen3-asr-0.6b',
      encoderGroup: 'qwen3-asr-0.6b',
      size: '1.17 GB',
      sizeMB: 1173.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/encoder.onnx',
          sizeMB: 711.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q4_K_M.gguf',
          sizeMB: 462.0,
        ),
      ],
    ),
    // ===== Qwen3-ASR 1.7B =====
    QwenModelInfo(
      id: 'qwen3-asr-1.7b-f16',
      name: 'Qwen3-ASR 1.7B · 全量 F16',
      description: '官方全精度解码器 · 最高识别精度 (默认推荐)',
      dirName: 'qwen3-asr-1.7b-f16',
      encoderGroup: 'qwen3-asr-1.7b',
      size: '5.1 GB',
      sizeMB: 5150.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/encoder.onnx',
          sizeMB: 1270.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.f16.gguf',
          sizeMB: 3880.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'qwen3-asr-1.7b-q8',
      name: 'Qwen3-ASR 1.7B · 量化 Q8_0',
      description: '8-bit 量化 · 高精度与速度均衡 (显存 ≥ 6GB)',
      dirName: 'qwen3-asr-1.7b-q8',
      encoderGroup: 'qwen3-asr-1.7b',
      size: '3.3 GB',
      sizeMB: 3335.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/encoder.onnx',
          sizeMB: 1270.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.Q8_0.gguf',
          sizeMB: 2065.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'qwen3-asr-1.7b',
      name: 'Qwen3-ASR 1.7B · 量化 Q6_K',
      description: '6-bit 量化 · 高精度 (显存 ≥ 6GB)',
      dirName: 'qwen3-asr-1.7b',
      encoderGroup: 'qwen3-asr-1.7b',
      size: '2.9 GB',
      sizeMB: 2865.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/encoder.onnx',
          sizeMB: 1270.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.Q6_K.gguf',
          sizeMB: 1595.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'forced-aligner-0.6b',
      name: 'ForcedAligner 0.6B',
      description: '精准时间轴组件 (词/字级精确对齐)',
      dirName: 'forced-aligner-0.6b',
      size: '450 MB',
      sizeMB: 450.0,
      type: ModelType.aligner,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'aligner.onnx',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/encoder.int4.onnx',
          sizeMB: 450.0,
        ),
      ],
    ),
  ];

  Future<Directory> getModelDir() async {
    final appDir = await getApplicationSupportDirectory();
    final modelDir = Directory(p.join(appDir.path, 'models'));
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return modelDir;
  }

  Future<List<String>> getDownloadedModels() async {
    try {
      final dir = await getModelDir();
      final List<String> downloadedDirs = [];
      for (final model in availableQwenModels) {
        if (model.type == ModelType.asr) {
          final path = p.join(dir.path, model.dirName);
          if (await Directory(path).exists()) {
            final isDownloaded = await isModelDownloaded(model.dirName);
            if (isDownloaded) {
              downloadedDirs.add(model.dirName);
            }
          }
        }
      }
      return downloadedDirs;
    } catch (e) {
      return [];
    }
  }

  Future<String> getModelPath(String dirName) async {
    final dir = await getModelDir();
    return p.join(dir.path, dirName);
  }

  Future<bool> isModelDownloaded(String dirName) async {
    final path = await getModelPath(dirName);
    final modelFolder = Directory(path);
    if (!await modelFolder.exists()) return false;

    if (await isModelCorrupted(dirName)) {
      try {
        debugPrint('[ModelService] Auto purging corrupted model directory $path');
        await modelFolder.delete(recursive: true);
      } catch (e) {
        debugPrint('[ModelService] Failed to delete corrupted model dir: $e');
      }
      return false;
    }

    return true;
  }

  Future<bool> isModelCorrupted(String dirName) async {
    final path = await getModelPath(dirName);
    final modelFolder = Directory(path);
    if (!await modelFolder.exists()) return true;

    final modelInfoMatch = availableQwenModels.where((m) => m.dirName == dirName);
    if (modelInfoMatch.isEmpty) {
      return false;
    }

    final files = await modelFolder.list().toList();
    if (files.isEmpty) return true;

    // 关键文件必须存在 (ASR 变体的 encoder 可能是与同组变体共享的硬链接)
    final model = modelInfoMatch.first;
    final criticalFiles = model.type == ModelType.asr
        ? ['encoder.onnx', 'decoder.gguf']
        : ['aligner.onnx'];
    final presentFiles = files.whereType<File>().map((f) => p.basename(f.path)).toSet();
    for (final name in criticalFiles) {
      if (!presentFiles.contains(name)) return true;
    }

    int validFiles = 0;
    int totalBytes = 0;

    for (final entity in files) {
      if (entity is File) {
        final len = await entity.length();
        totalBytes += len;
        if (len > 10 * 1024 * 1024) {
          validFiles++;
        }
      }
    }

    return !(validFiles >= 1 && totalBytes > 50 * 1024 * 1024);
  }

  Future<Map<String, int?>> testMirrorsSpeed() async {
    final dio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 4),
      receiveTimeout: const Duration(seconds: 4),
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
      },
    ));

    final Map<String, int?> results = {};

    await Future.wait(availableMirrors.map((mirror) async {
      final stopwatch = Stopwatch()..start();
      try {
        final testUrl = mirror.id == 'modelscope'
            ? 'https://modelscope.cn'
            : '${mirror.baseUrl}/andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json';
        final response = await dio.head(testUrl);
        stopwatch.stop();
        if (response.statusCode != null && response.statusCode! < 400) {
          results[mirror.id] = stopwatch.elapsedMilliseconds;
          mirror.latencyMs = stopwatch.elapsedMilliseconds;
        } else {
          results[mirror.id] = null;
          mirror.latencyMs = null;
        }
      } catch (e) {
        results[mirror.id] = null;
        mirror.latencyMs = null;
      }
    }));

    return results;
  }

  CancelToken? _cancelToken;

  /// 在同组变体目录中查找一个已存在的有效 encoder 副本。
  /// 找到后即可硬链接复用，避免每个量化变体重复下载/存储 encoder。
  Future<String?> _findGroupEncoderDir(Directory modelsDir, String group, String selfDirName) async {
    for (final m in availableQwenModels) {
      if (m.type != ModelType.asr || m.encoderGroup != group || m.dirName == selfDirName) {
        continue;
      }
      try {
        final f = File(p.join(modelsDir.path, m.dirName, 'encoder.onnx'));
        if (await f.exists() && await f.length() > 100 * 1024 * 1024) {
          return p.join(modelsDir.path, m.dirName);
        }
      } catch (_) {}
    }
    return null;
  }

  /// 硬链接复用大文件 (同卷不占额外磁盘空间)；失败时回退为复制。
  Future<void> _linkOrCopy(File src, File dst) async {
    try {
      if (await dst.exists()) {
        if (await dst.length() > 100 * 1024 * 1024) return;
        await dst.delete();
      }
      if (!_hardLink(src.path, dst.path)) {
        throw StateError('CreateHardLinkW returned 0');
      }
      debugPrint('[ModelService] Hard-linked ${src.path} -> ${dst.path}');
    } catch (e) {
      debugPrint('[ModelService] Hard link failed ($e), falling back to copy');
      try {
        await src.copy(dst.path);
      } catch (e2) {
        debugPrint('[ModelService] Copy failed: $e2');
        rethrow;
      }
    }
  }

  Future<void> _copyFileIfMissing(File src, File dst) async {
    if (await dst.exists()) return;
    await src.copy(dst.path);
  }

  Future<void> downloadModel({
    required QwenModelInfo model,
    ModelMirror? mirror,
    required Function(double progress) onProgress,
    required Function() onSuccess,
    required Function(String error) onFailure,
  }) async {
    Directory? targetDir;
    try {
      final dir = await getModelDir();
      targetDir = Directory(p.join(dir.path, model.dirName));
      if (!await targetDir.exists()) {
        await targetDir.create(recursive: true);
      }

      _cancelToken = CancelToken();

      final selectedMirror = mirror ?? availableMirrors.first;
      final baseUrl = selectedMirror.baseUrl;

      double totalExpectedBytes = model.sizeMB * 1024 * 1024;
      double completedFileBytes = 0.0;

      final dio = Dio(BaseOptions(
        followRedirects: true,
        maxRedirects: 10,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
      ));

      // 同组量化变体共享同一 encoder: 组内已有有效副本则直接复用，不重复下载
      final groupEncoderDir = model.encoderGroup != null
          ? await _findGroupEncoderDir(dir, model.encoderGroup!, model.dirName)
          : null;

      for (int i = 0; i < model.files.length; i++) {
        final fileInfo = model.files[i];
        final targetFile = File(p.join(targetDir.path, fileInfo.filename));
        final fileExpectedBytes = fileInfo.sizeMB * 1024 * 1024;

        final isSharedFile = model.encoderGroup != null &&
            (fileInfo.filename == 'encoder.onnx' || fileInfo.filename == 'config.json');
        if (isSharedFile && groupEncoderDir != null) {
          final src = File(p.join(groupEncoderDir, fileInfo.filename));
          if (fileInfo.filename == 'encoder.onnx') {
            await _linkOrCopy(src, targetFile);
          } else {
            await _copyFileIfMissing(src, targetFile);
          }
          completedFileBytes += fileExpectedBytes;
          onProgress((completedFileBytes / totalExpectedBytes).clamp(0.0, 0.99));
          continue;
        }

        final fileDownloadUrl = mirror?.id == 'modelscope'
            ? 'https://modelscope.cn/api/v1/models/${fileInfo.urlPath}'
            : '$baseUrl/${fileInfo.urlPath}';

        debugPrint('[ModelService] Downloading from $fileDownloadUrl to ${targetFile.path}');

        await dio.download(
          fileDownloadUrl,
          targetFile.path,
          cancelToken: _cancelToken,
          onReceiveProgress: (received, total) {
            final fileTotal = total > 0 ? total.toDouble() : fileExpectedBytes;
            final currentProgressBytes = completedFileBytes + (received.toDouble().clamp(0.0, fileTotal));
            final totalProgress = (currentProgressBytes / totalExpectedBytes).clamp(0.0, 0.99);
            onProgress(totalProgress);
          },
        );

        completedFileBytes += fileExpectedBytes;
      }

      onProgress(1.0);
      onSuccess();
    } catch (e) {
      debugPrint('[ModelService] Download error: $e');
      if (targetDir != null && await targetDir.exists()) {
        final isDownloaded = await isModelDownloaded(model.dirName);
        if (!isDownloaded) {
          try {
            await targetDir.delete(recursive: true);
          } catch (_) {}
        }
      }

      if (e is DioException && CancelToken.isCancel(e)) {
        onFailure('下载已取消');
      } else {
        onFailure('下载失败: ${e.toString()}');
      }
    }
  }

  void cancelDownload() {
    _cancelToken?.cancel();
  }

  Future<void> deleteModel(String dirName) async {
    final path = await getModelPath(dirName);
    final dir = Directory(path);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    } else {
      final file = File(path);
      if (await file.exists()) {
        await file.delete();
      }
    }
  }

  Future<String> prepareDFModel() async {
    final appDir = await getApplicationSupportDirectory();
    final modelDir = Directory(p.join(appDir.path, 'models'));
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    
    final targetPath = p.join(modelDir.path, 'DeepFilterNet3_onnx.tar.gz');
    final targetFile = File(targetPath);
    
    if (await targetFile.exists() && await targetFile.length() > 1024 * 1024) {
      return targetPath;
    }
    
    final data = await rootBundle.load('assets/models/DeepFilterNet3_onnx.tar.gz');
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await targetFile.writeAsBytes(bytes);
    
    return targetPath;
  }

  Future<String> prepareVADModel() async {
    final appDir = await getApplicationSupportDirectory();
    final modelDir = Directory(p.join(appDir.path, 'models'));
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    
    final targetPath = p.join(modelDir.path, 'ggml-silero-v5.1.2.bin');
    final targetFile = File(targetPath);
    
    if (await targetFile.exists() && await targetFile.length() > 1024 * 1024) {
      return targetPath;
    }
    
    final data = await rootBundle.load('assets/models/ggml-silero-v5.1.2.bin');
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await targetFile.writeAsBytes(bytes);
    
    return targetPath;
  }
}
