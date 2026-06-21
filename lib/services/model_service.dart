import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/services.dart' show rootBundle;

class WhisperModelInfo {
  final String name;
  final String filename;
  final String size;
  final double sizeMB;
  final String url;
  final String fallbackUrl;

  WhisperModelInfo({
    required this.name,
    required this.filename,
    required this.size,
    required this.sizeMB,
    required this.url,
    required this.fallbackUrl,
  });
}

class ModelService {
  static final List<WhisperModelInfo> availableModels = [
    WhisperModelInfo(
      name: 'Tiny (快 / 适合测试)',
      filename: 'ggml-tiny.bin',
      size: '75 MB',
      sizeMB: 75.0,
      url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
      fallbackUrl: 'https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin',
    ),
    WhisperModelInfo(
      name: 'Base (推荐 / 平衡度高)',
      filename: 'ggml-base.bin',
      size: '140 MB',
      sizeMB: 140.0,
      url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
      fallbackUrl: 'https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-base.bin',
    ),
    WhisperModelInfo(
      name: 'Small (适中 / 精度适中)',
      filename: 'ggml-small.bin',
      size: '460 MB',
      sizeMB: 460.0,
      url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
      fallbackUrl: 'https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-small.bin',
    ),
    WhisperModelInfo(
      name: 'Large V3 Turbo Q8 (稍慢 / 较精确)',
      filename: 'ggml-large-v3-turbo-q8_0.bin',
      size: '834 MB',
      sizeMB: 834.0,
      url: 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin',
      fallbackUrl: 'https://hf-mirror.com/ggerganov/whisper.cpp/resolve/main/ggml-large-v3-turbo-q8_0.bin',
    ),
    WhisperModelInfo(
      name: 'Large V3 Q8 (最慢 / 最精确)',
      filename: 'ggml-large-v3-q8_0.bin',
      size: '1.57 GB',
      sizeMB: 1608.0,
      url: 'https://huggingface.co/adriabama06/whisper-large-v3-ggml/resolve/main/ggml-large-v3-q8_0.bin',
      fallbackUrl: 'https://hf-mirror.com/adriabama06/whisper-large-v3-ggml/resolve/main/ggml-large-v3-q8_0.bin',
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
      final List<String> files = [];
      await for (final entity in dir.list()) {
        if (entity is File && entity.path.endsWith('.bin')) {
          files.add(p.basename(entity.path));
        }
      }
      return files;
    } catch (e) {
      return [];
    }
  }

  Future<String> getModelPath(String filename) async {
    final dir = await getModelDir();
    return p.join(dir.path, filename);
  }

  Future<bool> isModelDownloaded(String filename) async {
    final path = await getModelPath(filename);
    final file = File(path);
    return await file.exists() && await file.length() > 1024 * 1024; // > 1MB
  }

  final Dio _dio = Dio();
  CancelToken? _cancelToken;

  Future<void> downloadModel({
    required WhisperModelInfo model,
    required Function(double progress) onProgress,
    required Function() onSuccess,
    required Function(String error) onFailure,
  }) async {
    try {
      final dir = await getModelDir();
      final savePath = p.join(dir.path, model.filename);
      final tempSavePath = '$savePath.tmp';

      _cancelToken = CancelToken();

      try {
        await _dio.download(
          model.url,
          tempSavePath,
          cancelToken: _cancelToken,
          onReceiveProgress: (received, total) {
            if (total != -1) {
              final progress = received / total;
              onProgress(progress);
            }
          },
        );
      } catch (e) {
        if (CancelToken.isCancel(e as DioException)) {
          rethrow;
        }
        // 主站下载失败，回退到镜像站 CDN
        await _dio.download(
          model.fallbackUrl,
          tempSavePath,
          cancelToken: _cancelToken,
          onReceiveProgress: (received, total) {
            if (total != -1) {
              final progress = received / total;
              onProgress(progress);
            }
          },
        );
      }

      // 下载完成后，将重命名临时文件
      final tempFile = File(tempSavePath);
      if (await tempFile.exists()) {
        await tempFile.rename(savePath);
        onSuccess();
      } else {
        onFailure('下载文件不存在');
      }
    } catch (e) {
      if (CancelToken.isCancel(e as DioException)) {
        onFailure('下载已取消');
      } else {
        onFailure('下载失败: ${e.toString()}');
      }
    }
  }

  void cancelDownload() {
    _cancelToken?.cancel();
  }

  Future<void> deleteModel(String filename) async {
    final path = await getModelPath(filename);
    final file = File(path);
    if (await file.exists()) {
      await file.delete();
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
    
    // 加载模型
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

