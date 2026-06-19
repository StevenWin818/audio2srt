import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:path_provider/path_provider.dart';
import '../providers/transcription_provider.dart';

class SettingsView extends StatefulWidget {
  const SettingsView({super.key});

  @override
  State<SettingsView> createState() => _SettingsViewState();
}

class _SettingsViewState extends State<SettingsView> {
  final TextEditingController _ffmpegController = TextEditingController();
  String _testStatus = '';
  Color _testStatusColor = Colors.white;
  String _appSupportDir = '加载中...';
  int _cpuThreads = Platform.numberOfProcessors;

  @override
  void initState() {
    super.initState();
    final provider = Provider.of<TranscriptionProvider>(context, listen: false);
    _ffmpegController.text = provider.ffmpegService.ffmpegPath;
    _loadPaths();
  }

  Future<void> _loadPaths() async {
    final dir = await getApplicationSupportDirectory();
    setState(() {
      _appSupportDir = dir.path;
    });
  }

  @override
  void dispose() {
    _ffmpegController.dispose();
    super.dispose();
  }

  Future<void> _testFFmpeg(TranscriptionProvider provider) async {
    setState(() {
      _testStatus = '正在运行测试...';
      _testStatusColor = Colors.white;
    });

    provider.ffmpegService.ffmpegPath = _ffmpegController.text;
    final available = await provider.ffmpegService.checkFFmpegAvailable();

    setState(() {
      if (available) {
        _testStatus = '测试成功：找到可运行的 FFmpeg 命令！';
        _testStatusColor = Colors.green;
      } else {
        _testStatus = '测试失败：无法在此路径执行 FFmpeg，请检查路径。';
        _testStatusColor = Colors.redAccent;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);

    return SingleChildScrollView(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(),
          const SizedBox(height: 24),
          _buildFFmpegConfigCard(provider),
          const SizedBox(height: 24),
          _buildHardwareConfigCard(provider),
          const SizedBox(height: 24),
          _buildSystemInfoCard(),
        ],
      ),
    );
  }

  Widget _buildHardwareConfigCard(TranscriptionProvider provider) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '硬件加速配置',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'GPU 硬件加速 (Vulkan)',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '使用 Vulkan 后端加速模型推理。编译带有 vulkan 特征的软件且安装 Vulkan SDK 时该选项有效。否则会自动安全回退至 CPU 推理。',
                      style: TextStyle(fontSize: 12, color: Colors.grey[400]),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Switch(
                value: provider.useGpu,
                activeColor: const Color(0xFF8B5CF6),
                onChanged: (val) {
                  provider.setUseGpu(val);
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return const Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          '系统设置',
          style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 4),
        Text(
          '配置 FFmpeg 路径及查看系统配置信息',
          style: TextStyle(fontSize: 14, color: Colors.grey),
        ),
      ],
    );
  }

  Widget _buildFFmpegConfigCard(TranscriptionProvider provider) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            'FFmpeg 环境配置',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 16),
          const Text(
            '本应用使用 FFmpeg 对音视频文件进行解码并提取 16kHz WAV。默认会尝试使用全局 PATH 中的 "ffmpeg"。',
            style: TextStyle(fontSize: 13, color: Colors.grey),
          ),
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: TextField(
                  controller: _ffmpegController,
                  decoration: const InputDecoration(
                    labelText: 'FFmpeg 可执行文件路径 (e.g. ffmpeg.exe 或绝对路径)',
                    border: OutlineInputBorder(),
                    contentPadding: EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  ),
                ),
              ),
              const SizedBox(width: 16),
              ElevatedButton(
                onPressed: () => _testFFmpeg(provider),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF8B5CF6),
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
                ),
                child: const Text('检测并保存'),
              ),
            ],
          ),
          if (_testStatus.isNotEmpty) ...[
            const SizedBox(height: 16),
            Text(
              _testStatus,
              style: TextStyle(fontSize: 14, color: _testStatusColor, fontWeight: FontWeight.w600),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildSystemInfoCard() {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text(
            '运行环境信息',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 20),
          _buildInfoRow('操作系统', Platform.operatingSystem.toUpperCase()),
          const Divider(color: Color(0x1FFFFFFF), height: 24),
          _buildInfoRow('CPU 逻辑核心数', '$_cpuThreads 核'),
          const Divider(color: Color(0x1FFFFFFF), height: 24),
          _buildInfoRow('本地存储路径 (AppData)', _appSupportDir),
          const Divider(color: Color(0x1FFFFFFF), height: 24),
          _buildInfoRow('音频解码采样规格', '16,000 Hz, Mono, 16-bit PCM WAV'),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: const TextStyle(fontSize: 14, color: Colors.grey)),
        Expanded(
          child: Text(
            value,
            textAlign: TextAlign.right,
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.w600, fontFamily: 'monospace'),
          ),
        ),
      ],
    );
  }
}
