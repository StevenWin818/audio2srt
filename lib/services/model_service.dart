import 'dart:io';
import 'package:archive/archive.dart';
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

/// 单个模型文件描述
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

/// 独立组件模型 (ForcedAligner 等)
class QwenModelInfo {
  final String id;
  final String name;
  final String description;
  final String dirName;
  final String size;
  final double sizeMB;
  final ModelType type;
  final List<QwenModelFile> files;
  /// zip 资产解压后需要存在的文件 (完整性检查; 为空则按 files 检查)
  final List<String>? extractedFiles;

  QwenModelInfo({
    required this.id,
    required this.name,
    required this.description,
    required this.dirName,
    required this.size,
    required this.sizeMB,
    required this.type,
    required this.files,
    this.extractedFiles,
  });
}

/// 解码器量化版本。文件名约定: `decoder.<id>.gguf` (如 `decoder.q4_k_m.gguf`)
class QwenQuantVersion {
  final String id; // 'f16' | 'q8_0' | 'q6_k' | 'q4_k_m'
  final String label; // 显示名，如 '全量 F16'
  final String description;
  final String urlPath; // GGUF 完整 urlPath
  final double sizeMB; // 固定编码 (UI 兜底显示/进度估算)
  final String sizeText;

  QwenQuantVersion({
    required this.id,
    required this.label,
    required this.description,
    required this.urlPath,
    required this.sizeMB,
    required this.sizeText,
  });

  String get ggufName => 'decoder.$id.gguf';
}

/// 一个基础模型目录: encoder + config 共享，多个量化 decoder 并存。
/// 存储结构:
///   models/`<id>`/
///     config.json
///     encoder.onnx
///     decoder.f16.gguf
///     decoder.q8_0.gguf
///     ...
class QwenBaseModel {
  final String id; // 目录名，如 'qwen3-asr-0.6b'
  final String name;
  final String description;
  final List<QwenModelFile> baseFiles; // config.json + encoder.onnx
  final List<QwenQuantVersion> quants;
  final double baseSizeMB; // encoder + config
  final String baseSizeText;

  QwenBaseModel({
    required this.id,
    required this.name,
    required this.description,
    required this.baseFiles,
    required this.quants,
    required this.baseSizeMB,
    required this.baseSizeText,
  });

  QwenQuantVersion? quantById(String id) {
    for (final q in quants) {
      if (q.id == id) return q;
    }
    return null;
  }
}

class ModelService {
  /// 下载源: 官方 + HF-Mirror。另有虚拟的"自动"选项 (autoMirror)，
  /// 下载前自动测速选择最快源。
  static final List<ModelMirror> availableMirrors = [
    ModelMirror(
      id: 'huggingface',
      name: 'HuggingFace 官方源',
      baseUrl: 'https://huggingface.co',
      description: '官方直连源，海外网络环境推荐',
    ),
    ModelMirror(
      id: 'hf-mirror',
      name: 'HF-Mirror 镜像站',
      baseUrl: 'https://hf-mirror.com',
      description: '国内高速 CDN 加速镜像，适合国内网络环境',
    ),
  ];

  /// 虚拟"自动"源: 下载前自动测速选择最快源
  static ModelMirror get autoMirror => ModelMirror(
        id: 'auto',
        name: '自动',
        baseUrl: '',
        description: '每次下载前自动测速，选择官方源 / HF-Mirror 中更快的一个',
      );

  static ModelMirror? mirrorById(String id) {
    if (id == 'auto') return autoMirror;
    for (final m in availableMirrors) {
      if (m.id == id) return m;
    }
    return null;
  }

  static final List<QwenBaseModel> availableBaseModels = [
    QwenBaseModel(
      id: 'qwen3-asr-0.6b',
      name: 'Qwen3-ASR 0.6B',
      description: '极速 · 低内存占用 (推荐显存 < 4GB)',
      baseFiles: [
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
      ],
      baseSizeMB: 711.1,
      baseSizeText: '711 MB',
      quants: [
        QwenQuantVersion(
          id: 'f16',
          label: '全量 F16',
          description: '官方全精度解码器 · 最高识别精度 (默认推荐)',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.f16.gguf',
          sizeMB: 1439.0,
          sizeText: '1.4 GB',
        ),
        QwenQuantVersion(
          id: 'q8_0',
          label: 'Q8_0',
          description: '8-bit 量化 · 高精度与速度均衡 (显存 ≥ 4GB)',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q8_0.gguf',
          sizeMB: 767.0,
          sizeText: '767 MB',
        ),
        QwenQuantVersion(
          id: 'q4_k_m',
          label: 'Q4_K_M',
          description: '4-bit 量化 · 极速低内存 (推荐显存 < 4GB)',
          urlPath: 'mradermacher/Qwen3-ASR-0.6B-GGUF/resolve/main/Qwen3-ASR-0.6B.Q4_K_M.gguf',
          sizeMB: 462.0,
          sizeText: '462 MB',
        ),
      ],
    ),
    QwenBaseModel(
      id: 'qwen3-asr-1.7b',
      name: 'Qwen3-ASR 1.7B',
      description: '高精度 · 推荐 (更高识别准确率)',
      baseFiles: [
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
      ],
      baseSizeMB: 1270.1,
      baseSizeText: '1.3 GB',
      quants: [
        QwenQuantVersion(
          id: 'f16',
          label: '全量 F16',
          description: '官方全精度解码器 · 最高识别精度 (默认推荐)',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.f16.gguf',
          sizeMB: 3880.0,
          sizeText: '3.9 GB',
        ),
        QwenQuantVersion(
          id: 'q8_0',
          label: 'Q8_0',
          description: '8-bit 量化 · 高精度与速度均衡 (显存 ≥ 6GB)',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.Q8_0.gguf',
          sizeMB: 2065.0,
          sizeText: '2.1 GB',
        ),
        QwenQuantVersion(
          id: 'q6_k',
          label: 'Q6_K',
          description: '6-bit 量化 · 高精度 (显存 ≥ 6GB)',
          urlPath: 'mradermacher/Qwen3-ASR-1.7B-GGUF/resolve/main/Qwen3-ASR-1.7B.Q6_K.gguf',
          sizeMB: 1595.0,
          sizeText: '1.6 GB',
        ),
      ],
    ),
  ];

  static final QwenModelInfo alignerModel = QwenModelInfo(
    id: 'forced-aligner-0.6b',
    name: 'ForcedAligner 0.6B',
    description: '精准时间轴组件 (词/字级精确对齐, HaujetZhao GGUF 转换版: frontend/backend ONNX int4 + LLM q4_k)',
    dirName: 'forced-aligner-0.6b',
    size: '505 MB',
    sizeMB: 505.0,
    type: ModelType.aligner,
    files: [
      QwenModelFile(
        filename: 'Qwen3-ForceAligner-0.6B-gguf.zip',
        urlPath:
            'https://github.com/HaujetZhao/Qwen3-ASR-GGUF/releases/download/models/Qwen3-ForceAligner-0.6B-gguf.zip',
        sizeMB: 505.0,
      ),
    ],
    // 解压后 Rust 加载器需要的文件 (完整性检查用)
    extractedFiles: [
      'qwen3_aligner_encoder_frontend.int4.onnx',
      'qwen3_aligner_encoder_backend.int4.onnx',
      'qwen3_aligner_llm.q4_k.gguf',
    ],
  );

  static QwenBaseModel? baseById(String id) {
    for (final m in availableBaseModels) {
      if (m.id == id) return m;
    }
    return null;
  }

  Future<Directory> getModelDir() async {
    final appDir = await getApplicationSupportDirectory();
    final modelDir = Directory(p.join(appDir.path, 'models'));
    if (!await modelDir.exists()) {
      await modelDir.create(recursive: true);
    }
    return modelDir;
  }

  Future<String> getModelPath(String dirName) async {
    final dir = await getModelDir();
    return p.join(dir.path, dirName);
  }

  Future<File> getQuantFile(String baseId, String quantId) async {
    final path = await getModelPath(baseId);
    return File(p.join(path, 'decoder.$quantId.gguf'));
  }

  /// encoder 是否有效: encoder.fp16.onnx (FP16 转换版) 或 encoder.onnx (FP32) 任一存在且 > 100MB
  Future<bool> _encoderOk(String path) async {
    for (final name in ['encoder.fp16.onnx', 'encoder.onnx', 'encoder.int4.onnx']) {
      try {
        final f = File(p.join(path, name));
        if (await f.exists() && await f.length() > 100 * 1024 * 1024) {
          return true;
        }
      } catch (_) {}
    }
    return false;
  }

  /// 基础模型是否完整下载 (encoder + config 就绪)
  Future<bool> isBaseDownloaded(String baseId) async {
    final path = await getModelPath(baseId);
    final dir = Directory(path);
    if (!await dir.exists()) return false;
    final config = File(p.join(path, 'config.json'));
    try {
      return await _encoderOk(path) && await config.exists();
    } catch (_) {
      return false;
    }
  }

  /// 基础模型是否损坏 (encoder 缺失/过小或 config 缺失)
  Future<bool> isBaseCorrupted(String baseId) async {
    final path = await getModelPath(baseId);
    final dir = Directory(path);
    if (!await dir.exists()) return false;
    final config = File(p.join(path, 'config.json'));
    try {
      return !(await _encoderOk(path) && await config.exists());
    } catch (_) {
      return true;
    }
  }

  /// 指定量化 decoder 是否已下载 (文件存在且 > 50MB)
  Future<bool> isQuantDownloaded(String baseId, String quantId) async {
    final f = await getQuantFile(baseId, quantId);
    try {
      return await f.exists() && await f.length() > 50 * 1024 * 1024;
    } catch (_) {
      return false;
    }
  }

  Future<List<String>> getDownloadedBases() async {
    final results = <String>[];
    for (final m in availableBaseModels) {
      if (await isBaseDownloaded(m.id)) {
        results.add(m.id);
      }
    }
    return results;
  }

  /// 基础模型本地实际占用 (encoder + config + 所有已下载 decoder)
  Future<int> getBaseLocalBytes(String baseId) async {
    final path = await getModelPath(baseId);
    final dir = Directory(path);
    if (!await dir.exists()) return 0;
    var total = 0;
    try {
      await for (final entity in dir.list(recursive: true)) {
        if (entity is File) {
          total += await entity.length();
        }
      }
    } catch (_) {}
    return total;
  }

  /// 单个量化 decoder 本地实际大小
  Future<int> getQuantLocalBytes(String baseId, String quantId) async {
    final f = await getQuantFile(baseId, quantId);
    try {
      return await f.exists() ? await f.length() : 0;
    } catch (_) {
      return 0;
    }
  }

  static String formatBytes(int bytes) {
    if (bytes <= 0) return '0 MB';
    final mb = bytes / 1024 / 1024;
    if (mb < 1024) return '${mb.toStringAsFixed(0)} MB';
    return '${(mb / 1024).toStringAsFixed(2)} GB';
  }

  /// 下载前对全部镜像测速，返回最快可用镜像 (全部失败时返回第一个)。
  Future<ModelMirror> pickFastestMirror() async {
    await testMirrorsSpeed();
    ModelMirror? best;
    for (final m in availableMirrors) {
      if (m.latencyMs != null) {
        if (best == null || m.latencyMs! < best.latencyMs!) {
          best = m;
        }
      }
    }
    best ??= availableMirrors.first;
    debugPrint('[ModelService] Auto-picked mirror: ${best.name} (${best.latencyMs}ms)');
    return best;
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
        final testUrl = '${mirror.baseUrl}/andrewleech/qwen3-asr-0.6b-onnx/resolve/main/config.json';
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

  /// 获取远程文件真实大小 (HEAD Content-Length)；失败返回 null (调用方用固定估算)
  Future<int?> _fetchRemoteSize(Dio dio, String url) async {
    try {
      final resp = await dio.head(
        url,
        options: Options(
          followRedirects: true,
          maxRedirects: 10,
        ),
      );
      final len = resp.headers.value(Headers.contentLengthHeader);
      if (len != null) {
        final v = int.tryParse(len);
        if (v != null && v > 0) return v;
      }
    } catch (e) {
      debugPrint('[ModelService] HEAD failed for $url: $e');
    }
    return null;
  }

  /// 下载基础模型的一个量化版本。
  /// - encoder/config 属于基础模型: 目录内已存在则跳过 (同一目录天然共享)
  /// - mirror 为空时自动测速选择最快下载源
  /// - 每个文件下载前 HEAD 获取真实体积，避免进度条错误
  Future<void> downloadQuant({
    required QwenBaseModel model,
    required QwenQuantVersion quant,
    ModelMirror? mirror,
    required Function(double progress) onProgress,
    required Function() onSuccess,
    required Function(String error) onFailure,
  }) async {
    Directory? targetDir;
    // 本次会话待下载文件 (中断时用于清理残片)
    final filesToDownload = <QwenModelFile>[];
    try {
      final dir = await getModelDir();
      targetDir = Directory(p.join(dir.path, model.id));
      if (!await targetDir.exists()) {
        await targetDir.create(recursive: true);
      }

      _cancelToken = CancelToken();

      // 自动测速选择下载源
      final selectedMirror = mirror ?? await pickFastestMirror();
      final baseUrl = selectedMirror.baseUrl;
      debugPrint('[ModelService] Downloading via ${selectedMirror.name} ($baseUrl)');

      final dio = Dio(BaseOptions(
        followRedirects: true,
        maxRedirects: 10,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
      ));

      // 构建待下载文件列表: 基础文件 (缺失才下) + 量化 decoder (缺失才下)
      filesToDownload.clear();
      for (final f in model.baseFiles) {
        final target = File(p.join(targetDir.path, f.filename));
        // encoder 任一有效文件 (fp16 转换版或 fp32) 存在即跳过下载
        final ok = f.filename == 'encoder.onnx'
            ? await _encoderOk(targetDir.path)
            : await target.exists();
        if (!ok) filesToDownload.add(f);
      }
      final quantFile = File(p.join(targetDir.path, quant.ggufName));
      if (!(await quantFile.exists() && await quantFile.length() > 50 * 1024 * 1024)) {
        filesToDownload.add(QwenModelFile(
          filename: quant.ggufName,
          urlPath: quant.urlPath,
          sizeMB: quant.sizeMB,
        ));
      }

      // 动态体积: 先 HEAD 每个文件，拿到真实 Content-Length
      final realSizes = <String, int>{};
      for (final f in filesToDownload) {
        final url = '$baseUrl/${f.urlPath}';
        final size = await _fetchRemoteSize(dio, url);
        realSizes[f.filename] = size ?? (f.sizeMB * 1024 * 1024).round();
      }
      final totalBytes = realSizes.values.fold<int>(0, (a, b) => a + b);
      var completedBytes = 0;

      for (final f in filesToDownload) {
        final targetFile = File(p.join(targetDir.path, f.filename));
        final fileTotal = realSizes[f.filename] ?? (f.sizeMB * 1024 * 1024).round();
        final url = '$baseUrl/${f.urlPath}';

        debugPrint('[ModelService] Downloading $url -> ${targetFile.path} ($fileTotal bytes)');

        await dio.download(
          url,
          targetFile.path,
          cancelToken: _cancelToken,
          onReceiveProgress: (received, total) {
            final fileBytes = total > 0 ? total.toDouble() : fileTotal.toDouble();
            final current = completedBytes + received.toDouble().clamp(0.0, fileBytes);
            final pct = totalBytes > 0 ? (current / totalBytes).clamp(0.0, 0.99) : 0.0;
            onProgress(pct);
          },
        );

        completedBytes += fileTotal;
      }

      onProgress(1.0);
      onSuccess();
    } catch (e) {
      debugPrint('[ModelService] Download error: $e');
      // 清理本次会话下载的文件，防止残片被误判为已下载 (下载中断留下 >50MB 残片时，
      // isQuantDownloaded 会误判，导致下次跳过下载而转写加载损坏文件)
      for (final f in filesToDownload) {
        try {
          final target = File(p.join(targetDir?.path ?? '', f.filename));
          if (await target.exists()) {
            await target.delete();
            debugPrint('[ModelService] Cleaned partial file ${f.filename}');
          }
        } catch (_) {}
      }
      if (e is DioException && CancelToken.isCancel(e)) {
        onFailure('下载已取消');
      } else {
        onFailure('下载失败: ${e.toString()}');
      }
    }
  }

  Future<void> downloadAligner({
    ModelMirror? mirror,
    required Function(double progress) onProgress,
    required Function() onSuccess,
    required Function(String error) onFailure,
  }) async {
    final model = alignerModel;
    Directory? targetDir;
    // 本次会话待下载文件 (中断时用于清理残片)
    final filesToDownload = <QwenModelFile>[];
    try {
      final dir = await getModelDir();
      targetDir = Directory(p.join(dir.path, model.dirName));
      if (!await targetDir.exists()) {
        await targetDir.create(recursive: true);
      }
      _cancelToken = CancelToken();
      final selectedMirror = mirror ?? await pickFastestMirror();
      final baseUrl = selectedMirror.baseUrl;
      final dio = Dio(BaseOptions(
        followRedirects: true,
        maxRedirects: 10,
        headers: {
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        },
      ));

      // 已解压完整则跳过下载 (zip 解压后会删除, 用解压产物判断)
      Future<bool> alreadyExtracted() async {
        final checks = model.extractedFiles ?? model.files.map((f) => f.filename).toList();
        for (final name in checks) {
          if (!await File(p.join(targetDir!.path, name)).exists()) return false;
        }
        return true;
      }

      if (await alreadyExtracted()) {
        onProgress(1.0);
        onSuccess();
        return;
      }

      filesToDownload.clear();
      for (final f in model.files) {
        final target = File(p.join(targetDir.path, f.filename));
        if (!await target.exists()) filesToDownload.add(f);
      }
      final realSizes = <String, int>{};
      for (final f in filesToDownload) {
        // 绝对 URL (如 GitHub release 资产) 不走镜像前缀
        final url = f.urlPath.startsWith('http')
            ? f.urlPath
            : '$baseUrl/${f.urlPath}';
        final size = await _fetchRemoteSize(dio, url);
        realSizes[f.filename] = size ?? (f.sizeMB * 1024 * 1024).round();
      }
      final totalBytes = realSizes.values.fold<int>(0, (a, b) => a + b);
      var completedBytes = 0;

      for (final f in filesToDownload) {
        final targetFile = File(p.join(targetDir.path, f.filename));
        final fileTotal = realSizes[f.filename] ?? (f.sizeMB * 1024 * 1024).round();
        final url = f.urlPath.startsWith('http')
            ? f.urlPath
            : '$baseUrl/${f.urlPath}';
        await dio.download(
          url,
          targetFile.path,
          cancelToken: _cancelToken,
          onReceiveProgress: (received, total) {
            final fileBytes = total > 0 ? total.toDouble() : fileTotal.toDouble();
            final current = completedBytes + received.toDouble().clamp(0.0, fileBytes);
            final pct = totalBytes > 0 ? (current / totalBytes).clamp(0.0, 0.99) : 0.0;
            onProgress(pct);
          },
        );
        completedBytes += fileTotal;
      }

      // 解压 zip 资产 (HaujetZhao GGUF 包), 成功后删除 zip
      for (final f in filesToDownload) {
        if (!f.filename.toLowerCase().endsWith('.zip')) continue;
        final zipFile = File(p.join(targetDir.path, f.filename));
        if (await zipFile.exists()) {
          debugPrint('[ModelService] Extracting ${f.filename} ...');
          await extractZip(zipFile, targetDir!);
          await zipFile.delete();
          debugPrint('[ModelService] Extracted and removed ${f.filename}');
        }
      }

      onProgress(1.0);
      onSuccess();
    } catch (e) {
      debugPrint('[ModelService] Aligner download error: $e');
      // 清理本次会话下载的文件，防止残片被误判为已下载
      for (final f in filesToDownload) {
        try {
          final target = File(p.join(targetDir?.path ?? '', f.filename));
          if (await target.exists()) {
            await target.delete();
            debugPrint('[ModelService] Cleaned partial file ${f.filename}');
          }
        } catch (_) {}
      }
      if (e is DioException && CancelToken.isCancel(e)) {
        onFailure('下载已取消');
      } else {
        onFailure('下载失败: ${e.toString()}');
      }
    }
  }

  /// 解压 zip 到目标目录 (archive 包, 纯 Dart)
  Future<void> extractZip(File zipFile, Directory targetDir) async {
    final bytes = await zipFile.readAsBytes();
    final archive = ZipDecoder().decodeBytes(bytes);
    for (final file in archive) {
      if (file.isFile) {
        final name = file.name.replaceAll('\\', '/');
        // 忽略顶层目录 (zip 内可能有 model/ 前缀)
        final parts = name.split('/');
        final baseName = parts.last;
        if (baseName.isEmpty) continue;
        final out = File(p.join(targetDir.path, baseName));
        await out.create(recursive: true);
        await out.writeAsBytes(file.content as List<int>, flush: true);
      }
    }
  }

  void cancelDownload() {
    _cancelToken?.cancel();
  }

  /// 删除单个量化 decoder 文件
  Future<void> deleteQuant(String baseId, String quantId) async {
    final f = await getQuantFile(baseId, quantId);
    if (await f.exists()) {
      await f.delete();
    }
  }

  /// 修复基础模型: 只删除损坏的基础文件 (encoder 过小/损坏)，保留完好的量化 decoder。
  /// config.json 缺失时无需删除 (下载时自动补齐)。返回是否清理了文件。
  Future<bool> repairBase(String baseId) async {
    final path = await getModelPath(baseId);
    var cleaned = false;
    for (final name in ['encoder.fp16.onnx', 'encoder.onnx', 'encoder.int4.onnx']) {
      try {
        final f = File(p.join(path, name));
        if (await f.exists() && await f.length() <= 100 * 1024 * 1024) {
          await f.delete();
          debugPrint('[ModelService] Repair: removed corrupted $name in $baseId');
          cleaned = true;
        }
      } catch (e) {
        debugPrint('[ModelService] Repair encoder check error: $e');
      }
    }
    return cleaned;
  }

  /// ForcedAligner 组件是否损坏: 校验解压产物是否齐全且体积合理。
  /// 注意: 基础模型的 isBaseCorrupted (encoder/config 逻辑) 不适用于 aligner 目录。
  Future<bool> isAlignerCorrupted(String dirName) async {
    if (dirName != alignerModel.dirName) return false;
    final path = await getModelPath(dirName);
    final dir = Directory(path);
    if (!await dir.exists()) return false; // 未下载不算损坏
    final checks = alignerModel.extractedFiles ?? const [];
    if (checks.isEmpty) return false;
    for (final name in checks) {
      try {
        final f = File(p.join(path, name));
        // 残留的半截解压产物 (小文件) 视为损坏
        if (!await f.exists() || await f.length() < 1024 * 1024) {
          debugPrint('[ModelService] Aligner file corrupted/missing: $name');
          return true;
        }
      } catch (e) {
        debugPrint('[ModelService] Aligner check error for $name: $e');
        return true;
      }
    }
    return false;
  }

  /// 修复 ForcedAligner: 删除整个组件目录 (下次下载时重建)
  Future<void> repairAligner() async {
    final path = await getModelPath(alignerModel.dirName);
    final dir = Directory(path);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
      debugPrint('[ModelService] Repair: removed corrupted aligner dir $path');
    }
  }

  Future<void> deleteAligner() async {
    final path = await getModelPath(alignerModel.dirName);
    final dir = Directory(path);
    if (await dir.exists()) {
      await dir.delete(recursive: true);
    }
  }

  /// 迁移旧版存储布局:
  /// 旧结构每个量化一个目录 (qwen3-asr-0.6b-f16/...)，且 decoder 统一命名为 decoder.gguf。
  /// 新结构: 一模型一目录 + `decoder.<quant>.gguf` 多文件。
  /// 这里把旧目录中已知量化的 decoder.gguf 复制为新命名 (保留原文件，安全)。
  Future<void> migrateLegacyLayout() async {
    try {
      final dir = await getModelDir();
      // 旧命名 decoder.gguf -> 新命名 (按体积推测量化; 仅复制不删除)
      final legacyCopies = [
        ('qwen3-asr-0.6b', 300.0, 600.0, 'q4_k_m'), // ~462MB
        ('qwen3-asr-1.7b', 1200.0, 2000.0, 'q6_k'), // ~1596MB
      ];
      for (final (baseId, minMB, maxMB, quantId) in legacyCopies) {
        final baseDir = Directory(p.join(dir.path, baseId));
        if (!await baseDir.exists()) continue;
        final oldFile = File(p.join(baseDir.path, 'decoder.gguf'));
        if (!await oldFile.exists()) continue;
        final newFile = File(p.join(baseDir.path, 'decoder.$quantId.gguf'));
        if (await newFile.exists()) continue;
        final sizeMB = (await oldFile.length()) / 1024 / 1024;
        if (sizeMB < minMB || sizeMB > maxMB) continue;
        await oldFile.copy(newFile.path);
        debugPrint('[ModelService] Migrated legacy decoder.gguf ($baseId, $sizeMB MB) -> decoder.$quantId.gguf');
      }
      // 旧量化变体目录可安全保留 (不迁移，用户可手动删除)
    } catch (e) {
      debugPrint('[ModelService] Legacy layout migration error: $e');
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
