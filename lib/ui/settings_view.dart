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
          _buildRepetitionControlCard(provider),
          const SizedBox(height: 24),
          _buildSystemInfoCard(),
        ],
      ),
    );
  }

  // 硬件加速配置
  Widget _buildHardwareConfigCard(TranscriptionProvider provider) {
    final hasGpuActive = provider.useGpu && provider.vulkanDevices.isNotEmpty;

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
                      '使用 Vulkan 后端加速模型推理，建议开启。如果您的设备支持 Vulkan，能大幅加快推理速度，否则会自动安全回退至 CPU 推理。',
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
          const SizedBox(height: 16),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
            decoration: BoxDecoration(
              color: const Color(0x05FFFFFF),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0x0FFFFFFF)),
            ),
            child: Row(
              children: [
                Icon(
                  hasGpuActive ? Icons.developer_board : Icons.memory_outlined,
                  size: 16,
                  color: hasGpuActive ? const Color(0xFFA78BFA) : Colors.grey[400],
                ),
                const SizedBox(width: 8),
                Text(
                  '当前计算设备: ',
                  style: TextStyle(fontSize: 13, color: Colors.grey[400]),
                ),
                Expanded(
                  child: Text(
                    provider.activeDeviceName,
                    style: TextStyle(
                      fontSize: 13,
                      fontWeight: FontWeight.w600,
                      color: hasGpuActive ? const Color(0xFFA78BFA) : Colors.white,
                    ),
                    overflow: TextOverflow.ellipsis,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRepetitionControlCard(TranscriptionProvider provider) {
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
            '语音端点检测与防重复设置',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            '针对 Whisper 模型在推理静音段或较长音频时容易陷入幻觉和无限重复句子的优化配置。',
            style: TextStyle(fontSize: 13, color: Colors.grey[400]),
          ),
          const SizedBox(height: 20),
          
          // VAD 开关
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '启用端点检测 (VAD)',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '提取有效人声切片分别推理，自动跳过大片静音以避免幻觉和静音句重复。',
                      style: TextStyle(fontSize: 12, color: Colors.grey[400]),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Switch(
                value: provider.vadEnabled,
                activeColor: const Color(0xFF8B5CF6),
                onChanged: (val) {
                  provider.setVadEnabled(val);
                },
              ),
            ],
          ),
          
          if (provider.vadEnabled) ...[
            const SizedBox(height: 16),
            const Divider(color: Color(0x1FFFFFFF)),
            const SizedBox(height: 12),
            
            // VAD 阈值
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      'VAD 语音概率阈值',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    Text(
                      provider.vadThreshold.toStringAsFixed(2),
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                    ),
                  ],
                ),
                Slider(
                  value: provider.vadThreshold,
                  min: 0.1,
                  max: 0.9,
                  divisions: 80,
                  activeColor: const Color(0xFF8B5CF6),
                  inactiveColor: const Color(0x1FFFFFFF),
                  onChanged: (val) {
                    provider.setVadThreshold(val);
                  },
                ),
                Text(
                  'Silero VAD 判断为语音的概率阈值。值越高判定越严格，能有效过滤杂音，但值过高可能会漏掉微弱人声（推荐默认 0.5）。',
                  style: TextStyle(fontSize: 11, color: Colors.grey[500]),
                ),
              ],
            ),
            const SizedBox(height: 16),
            
            // VAD 最小静音判定
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      '最小静音判定时间',
                      style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                    ),
                    Text(
                      '${provider.vadMinSilenceMs} ms',
                      style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                    ),
                  ],
                ),
                Slider(
                  value: provider.vadMinSilenceMs.toDouble(),
                  min: 200,
                  max: 2000,
                  divisions: 18,
                  activeColor: const Color(0xFF8B5CF6),
                  inactiveColor: const Color(0x1FFFFFFF),
                  onChanged: (val) {
                    provider.setVadMinSilenceMs(val.round());
                  },
                ),
                Text(
                  '判定人声中断并进行切片拆分的连续静音长度。数值越小，切片越多，越能有效避免连续幻觉。',
                  style: TextStyle(fontSize: 11, color: Colors.grey[500]),
                ),
              ],
            ),
          ],
          
          const SizedBox(height: 16),
          const Divider(color: Color(0x1FFFFFFF)),
          const SizedBox(height: 16),
          
          // No Context 开关
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '禁用上下文历史 (No Context)',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '在推理新时间窗口时不参考上一句文本。能彻底阻断由于上一句发生重复而带偏后文的循环死锁。',
                      style: TextStyle(fontSize: 12, color: Colors.grey[400]),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Switch(
                value: provider.noContext,
                activeColor: const Color(0xFF8B5CF6),
                onChanged: (val) {
                  provider.setNoContext(val);
                },
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          const Divider(color: Color(0x1FFFFFFF)),
          const SizedBox(height: 16),

          // No State History 开关
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      '禁用 KV 缓存记忆 (No KV Cache)',
                      style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold),
                    ),
                    const SizedBox(height: 4),
                    Text(
                      '在推理新分段时彻底隔离并重建模型状态。能有效解决音频中更换语言时被强行翻译成上一句语言的现象。',
                      style: TextStyle(fontSize: 12, color: Colors.grey[400]),
                    ),
                  ],
                ),
              ),
              const SizedBox(width: 16),
              Switch(
                value: provider.noStateHistory,
                activeColor: const Color(0xFF8B5CF6),
                onChanged: (val) {
                  provider.setNoStateHistory(val);
                },
              ),
            ],
          ),
          
          const SizedBox(height: 16),
          const Divider(color: Color(0x1FFFFFFF)),
          const SizedBox(height: 12),
          
          // 初始温度 Temperature
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '初始解码温度 (Temperature)',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    provider.temperature.toStringAsFixed(1),
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                  ),
                ],
              ),
              Slider(
                value: provider.temperature,
                min: 0.0,
                max: 1.0,
                divisions: 10,
                activeColor: const Color(0xFF8B5CF6),
                inactiveColor: const Color(0x1FFFFFFF),
                onChanged: (val) {
                  provider.setTemperature(val);
                },
              ),
              Text(
                '0.0 为贪婪解码（最稳定，易陷入死循环）。调高可增加生成随机性与发散度以打破死结，但可能降低精确度。',
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // 运行温度增量 Temperature Increment
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '温度回退增量 (Temperature Increment)',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    provider.temperatureInc.toStringAsFixed(1),
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                  ),
                ],
              ),
              Slider(
                value: provider.temperatureInc,
                min: 0.0,
                max: 1.0,
                divisions: 10,
                activeColor: const Color(0xFF8B5CF6),
                inactiveColor: const Color(0x1FFFFFFF),
                onChanged: (val) {
                  provider.setTemperatureInc(val);
                },
              ),
              Text(
                '推理失败回退并重新评估时，温度的每次递增幅度。默认 0.2。',
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // 熵阈值 Entropy Threshold
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '文本熵判定阈值 (Entropy Threshold)',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    provider.entropyThold.toStringAsFixed(1),
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                  ),
                ],
              ),
              Slider(
                value: provider.entropyThold,
                min: 1.0,
                max: 3.0,
                divisions: 20,
                activeColor: const Color(0xFF8B5CF6),
                inactiveColor: const Color(0x1FFFFFFF),
                onChanged: (val) {
                  provider.setEntropyThold(val);
                },
              ),
              Text(
                '文本生成的压缩率/混沌程度阈值。若生成的熵过高（文字混乱无逻辑），触发降级重试机制。默认 2.4。',
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // 对数概率阈值 Logprob Threshold
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '置信对数概率阈值 (Logprob Threshold)',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    provider.logprobThold.toStringAsFixed(1),
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                  ),
                ],
              ),
              Slider(
                value: provider.logprobThold,
                min: -2.0,
                max: 0.0,
                divisions: 20,
                activeColor: const Color(0xFF8B5CF6),
                inactiveColor: const Color(0x1FFFFFFF),
                onChanged: (val) {
                  provider.setLogprobThold(val);
                },
              ),
              Text(
                '模型输出词的平均对数概率下限阈值。若低于此值（不确信度高），触发回退重试。默认 -1.0。',
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
              ),
            ],
          ),
          const SizedBox(height: 16),
          
          // 静音概率阈值 No Speech Threshold
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    '无声判定阈值 (No Speech Threshold)',
                    style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold),
                  ),
                  Text(
                    provider.noSpeechThold.toStringAsFixed(2),
                    style: const TextStyle(fontSize: 13, fontWeight: FontWeight.bold, color: Color(0xFFA78BFA)),
                  ),
                ],
              ),
              Slider(
                value: provider.noSpeechThold,
                min: 0.0,
                max: 1.0,
                divisions: 20,
                activeColor: const Color(0xFF8B5CF6),
                inactiveColor: const Color(0x1FFFFFFF),
                onChanged: (val) {
                  provider.setNoSpeechThold(val);
                },
              ),
              Text(
                '当模型自身预测此处为静音的概率高于该值时，直接舍弃对应文本。默认 0.60。',
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
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

  // FFmpeg 配置
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

  // 系统信息卡片
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
