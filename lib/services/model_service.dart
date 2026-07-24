import 'dart:io';
import 'package:dio/dio.dart';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/services.dart' show rootBundle;

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

  QwenModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.dirName,
    required this.size,
    required this.sizeMB,
    required this.type,
    required this.files,
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
    QwenModelInfo(
      id: 'qwen3-asr-0.6b',
      name: 'Qwen3-ASR 0.6B',
      description: '极速 · 低内存占用 (推荐显存 < 4GB)',
      dirName: 'qwen3-asr-0.6b',
      size: '930 MB',
      sizeMB: 930.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-0.6b-onnx/resolve/main/encoder.int4.onnx',
          sizeMB: 450.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q4_K_M.gguf',
          sizeMB: 480.0,
        ),
      ],
    ),
    QwenModelInfo(
      id: 'qwen3-asr-1.7b',
      name: 'Qwen3-ASR 1.7B',
      description: '高精度 · 推荐 (更高识别准确率)',
      dirName: 'qwen3-asr-1.7b',
      size: '2.42 GB',
      sizeMB: 2420.0,
      type: ModelType.asr,
      files: [
        QwenModelFile(
          filename: 'config.json',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/config.json',
          sizeMB: 0.1,
        ),
        QwenModelFile(
          filename: 'encoder.onnx',
          urlPath: 'andrewleech/qwen3-asr-1.7b-onnx/resolve/main/encoder.int4.onnx',
          sizeMB: 1270.0,
        ),
        QwenModelFile(
          filename: 'decoder.gguf',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.Q4_K_M.gguf',
          sizeMB: 461.0,
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

      for (int i = 0; i < model.files.length; i++) {
        final fileInfo = model.files[i];
        final targetFile = File(p.join(targetDir.path, fileInfo.filename));
        final fileExpectedBytes = fileInfo.sizeMB * 1024 * 1024;

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
