import 'dart:io';

class FFmpegService {
  String _customFFmpegPath = 'ffmpeg'; // 默认从系统 PATH 查找

  String get ffmpegPath => _customFFmpegPath;

  set ffmpegPath(String path) {
    _customFFmpegPath = path.isEmpty ? 'ffmpeg' : path;
  }

  /// 检查当前的 FFmpeg 路径是否有效
  Future<bool> checkFFmpegAvailable() async {
    try {
      final result = await Process.run(_customFFmpegPath, ['-version']);
      return result.exitCode == 0;
    } catch (e) {
      return false;
    }
  }

  /// 搜索系统常见路径
  Future<String?> findSystemFFmpeg() async {
    // 1. 检查默认 PATH
    if (await checkFFmpegAvailable()) {
      return _customFFmpegPath;
    }

    // 2. 检查 Windows 常见安装位置或当前运行路径
    final commonPaths = [
      './ffmpeg.exe',
      './bin/ffmpeg.exe',
      'C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe',
      'C:\\ffmpeg\\bin\\ffmpeg.exe',
    ];

    for (final path in commonPaths) {
      final file = File(path);
      if (await file.exists()) {
        try {
          final result = await Process.run(path, ['-version']);
          if (result.exitCode == 0) {
            _customFFmpegPath = path;
            return path;
          }
        } catch (_) {}
      }
    }

    return null;
  }
}
