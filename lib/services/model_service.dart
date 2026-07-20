import 'dart:convert';
import 'dart:io';
import 'package:dio/dio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:path/path.dart' as p;
import 'package:flutter/services.dart' show rootBundle;

class WhisperModelInfo {
  final String name;
  final String repoId;
  final String folderName;
  final String size;
  final double sizeMB;
  final List<String> files;

  WhisperModelInfo({
    required this.name,
    required this.repoId,
    required this.folderName,
    required this.size,
    required this.sizeMB,
    required this.files,
  });

  String get filename => folderName;
}

class ModelService {
  static final List<WhisperModelInfo> availableModels = [
    WhisperModelInfo(
      name: 'Large V3 Turbo (推荐 / 快速精细)',
      repoId: 'mobiuslabsgmbh/faster-whisper-large-v3-turbo',
      folderName: 'faster-whisper-large-v3-turbo',
      size: '1.62 GB',
      sizeMB: 1658.0,
      files: [
        'config.json',
        'model.bin',
        'preprocessor_config.json',
        'tokenizer.json',
        'vocabulary.json',
      ],
    ),
    WhisperModelInfo(
      name: 'Large V3 (旗舰 / 最高精度)',
      repoId: 'Systran/faster-whisper-large-v3',
      folderName: 'faster-whisper-large-v3',
      size: '约 3 GB',
      sizeMB: 3164.0,
      files: [
        'config.json',
        'model.bin',
        'preprocessor_config.json',
        'tokenizer.json',
        'vocabulary.json',
      ],
    ),
  ];

  String? _customModelDir;

  Future<File> _getConfigFile() async {
    final appDir = await getApplicationSupportDirectory();
    return File(p.join(appDir.path, 'app_config.json'));
  }

  Future<void> init() async {
    try {
      final configFile = await _getConfigFile();
      if (await configFile.exists()) {
        final content = await configFile.readAsString();
        final json = jsonDecode(content) as Map<String, dynamic>;
        if (json.containsKey('customModelDir') && json['customModelDir'] != null) {
          final customPath = json['customModelDir'] as String;
          if (Directory(customPath).existsSync()) {
            _customModelDir = customPath;
          }
        }
      }
    } catch (_) {}
  }

  Future<String?> getCustomModelDir() async {
    if (_customModelDir == null) {
      await init();
    }
    return _customModelDir;
  }

  Future<void> setCustomModelDir(String? path) async {
    _customModelDir = path;
    try {
      final configFile = await _getConfigFile();
      Map<String, dynamic> json = {};
      if (await configFile.exists()) {
        try {
          json = jsonDecode(await configFile.readAsString()) as Map<String, dynamic>;
        } catch (_) {}
      }
      if (path != null && path.isNotEmpty) {
        json['customModelDir'] = path;
      } else {
        json.remove('customModelDir');
      }
      await configFile.writeAsString(jsonEncode(json));
    } catch (_) {}
  }

  Future<Directory> getModelDir() async {
    final customPath = await getCustomModelDir();
    if (customPath != null && customPath.isNotEmpty) {
      final customDir = Directory(customPath);
      if (!await customDir.exists()) {
        await customDir.create(recursive: true);
      }
      return customDir;
    }
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
      final List<String> result = [];
      await for (final entity in dir.list()) {
        if (entity is Directory) {
          final modelBin = File(p.join(entity.path, 'model.bin'));
          if (await modelBin.exists() && await modelBin.length() > 10 * 1024 * 1024) {
            result.add(p.basename(entity.path));
          }
        } else if (entity is File && entity.path.endsWith('.bin')) {
          result.add(p.basename(entity.path));
        }
      }
      return result;
    } catch (e) {
      return [];
    }
  }

  Future<String> getModelPath(String folderOrFilename) async {
    if (p.isAbsolute(folderOrFilename)) {
      return folderOrFilename;
    }
    final dir = await getModelDir();
    return p.join(dir.path, folderOrFilename);
  }

  Future<bool> isModelDownloaded(String folderOrFilename) async {
    if (p.isAbsolute(folderOrFilename)) {
      final modelBin = File(p.join(folderOrFilename, 'model.bin'));
      return await modelBin.exists() && await modelBin.length() > 10 * 1024 * 1024;
    }
    final path = await getModelPath(folderOrFilename);
    final modelBin = File(p.join(path, 'model.bin'));
    if (await modelBin.exists() && await modelBin.length() > 10 * 1024 * 1024) {
      return true;
    }
    final singleFile = File(path);
    return await singleFile.exists() && await singleFile.length() > 10 * 1024 * 1024;
  }

  final Dio _dio = Dio();
  CancelToken? _cancelToken;

  Future<String> _getFastestBaseUrl() async {
    final candidateUrls = [
      'https://hf-mirror.com',
      'https://huggingface.co',
    ];

    final testDio = Dio(BaseOptions(
      connectTimeout: const Duration(seconds: 2),
      receiveTimeout: const Duration(seconds: 2),
    ));

    final futures = candidateUrls.map((baseUrl) async {
      final stopwatch = Stopwatch()..start();
      try {
        final response = await testDio.head(
          baseUrl,
          options: Options(
            followRedirects: false,
            validateStatus: (status) => status != null && status < 500,
          ),
        );
        stopwatch.stop();
        if (response.statusCode != null && response.statusCode! < 500) {
          return MapEntry(baseUrl, stopwatch.elapsedMilliseconds);
        }
      } catch (_) {}
      return MapEntry(baseUrl, 999999);
    });

    final results = await Future.wait(futures);
    int fastestTime = 999999;
    String fastestUrl = 'https://hf-mirror.com';

    for (final entry in results) {
      if (entry.value < fastestTime) {
        fastestTime = entry.value;
        fastestUrl = entry.key;
      }
    }
    return fastestUrl;
  }

  Future<void> downloadModel({
    required WhisperModelInfo model,
    required Function(double progress) onProgress,
    required Function() onSuccess,
    required Function(String error) onFailure,
  }) async {
    try {
      final baseDir = await getModelDir();
      final targetFolder = Directory(p.join(baseDir.path, model.folderName));
      if (!await targetFolder.exists()) {
        await targetFolder.create(recursive: true);
      }

      _cancelToken = CancelToken();

      final fastestBaseUrl = await _getFastestBaseUrl();
      final alternateBaseUrl = fastestBaseUrl == 'https://huggingface.co'
          ? 'https://hf-mirror.com'
          : 'https://huggingface.co';

      final double totalEstimatedBytes = model.sizeMB * 1024 * 1024;
      Map<String, int> fileProgress = {};

      for (int i = 0; i < model.files.length; i++) {
        final fileName = model.files[i];
        final fileSavePath = p.join(targetFolder.path, fileName);
        final tempSavePath = '$fileSavePath.tmp';

        final primaryUrl = '$fastestBaseUrl/${model.repoId}/resolve/main/$fileName';
        final fallbackUrl = '$alternateBaseUrl/${model.repoId}/resolve/main/$fileName';

        try {
          await _dio.download(
            primaryUrl,
            tempSavePath,
            cancelToken: _cancelToken,
            onReceiveProgress: (received, total) {
              fileProgress[fileName] = received;
              final sumReceived = fileProgress.values.fold<int>(0, (a, b) => a + b);
              double overall = sumReceived / totalEstimatedBytes;
              if (overall > 0.99) overall = 0.99;
              onProgress(overall);
            },
          );
        } catch (e) {
          if (CancelToken.isCancel(e as DioException)) {
            rethrow;
          }
          await _dio.download(
            fallbackUrl,
            tempSavePath,
            cancelToken: _cancelToken,
            onReceiveProgress: (received, total) {
              fileProgress[fileName] = received;
              final sumReceived = fileProgress.values.fold<int>(0, (a, b) => a + b);
              double overall = sumReceived / totalEstimatedBytes;
              if (overall > 0.99) overall = 0.99;
              onProgress(overall);
            },
          );
        }

        final tempFile = File(tempSavePath);
        if (await tempFile.exists()) {
          final targetFile = File(fileSavePath);
          if (await targetFile.exists()) {
            await targetFile.delete();
          }
          await tempFile.rename(fileSavePath);
        }
      }

      onProgress(1.0);
      onSuccess();
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

  Future<void> deleteModel(String folderOrFilename) async {
    final path = await getModelPath(folderOrFilename);
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
    
    final targetPath = p.join(modelDir.path, 'silero_vad.onnx');
    final targetFile = File(targetPath);
    
    if (await targetFile.exists() && await targetFile.length() > 1024 * 1024) {
      return targetPath;
    }
    
    final data = await rootBundle.load('assets/models/silero_vad.onnx');
    final bytes = data.buffer.asUint8List(data.offsetInBytes, data.lengthInBytes);
    await targetFile.writeAsBytes(bytes);
    
    return targetPath;
  }
}
