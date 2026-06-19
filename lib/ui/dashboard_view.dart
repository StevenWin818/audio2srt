import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:path/path.dart' as p;
import '../providers/transcription_provider.dart';
import '../services/model_service.dart';

class DashboardView extends StatelessWidget {
  const DashboardView({super.key});

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
          _buildImportArea(context, provider),
          const SizedBox(height: 24),
          Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Expanded(flex: 3, child: _buildConfigCard(context, provider)),
              const SizedBox(width: 24),
              Expanded(flex: 2, child: _buildStatusCard(context, provider)),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text(
          '语音识别与字幕生成',
          style: TextStyle(
            fontSize: 28,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 4),
        Text(
          '快速从音频或视频文件中提取语音并本地生成标准 SRT/VTT 字幕',
          style: TextStyle(
            fontSize: 14,
            color: Colors.grey[400],
          ),
        ),
      ],
    );
  }

  Widget _buildImportArea(BuildContext context, TranscriptionProvider provider) {
    final hasFile = provider.inputMediaFile != null;

    return DropTarget(
      onDragDone: (details) {
        if (details.files.isNotEmpty) {
          provider.setInputFile(File(details.files.first.path));
        }
      },
      child: Container(
        height: 200,
        decoration: BoxDecoration(
          color: const Color(0x0CFFFFFF),
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: hasFile ? const Color(0xFF8B5CF6) : const Color(0x1FFFFFFF),
            width: 2,
            style: hasFile ? BorderStyle.solid : BorderStyle.none,
          ),
          boxShadow: const [
            BoxShadow(
              color: Colors.black12,
              blurRadius: 10,
              offset: Offset(0, 4),
            )
          ],
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () async {
            final result = await FilePicker.pickFiles(
              type: FileType.custom,
              allowedExtensions: ['mp3', 'wav', 'mp4', 'mkv', 'm4a', 'aac', 'flac'],
            );
            if (result != null && result.files.single.path != null) {
              provider.setInputFile(File(result.files.single.path!));
            }
          },
          child: Stack(
            children: [
              Positioned.fill(
                child: Opacity(
                  opacity: 0.05,
                  child: Image.network(
                    'https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?q=80&w=600&auto=format&fit=crop',
                    fit: BoxFit.cover,
                    errorBuilder: (_, __, ___) => const SizedBox(),
                  ),
                ),
              ),
              Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: const BoxDecoration(
                        color: Color(0x1A8B5CF6),
                        shape: BoxShape.circle,
                      ),
                      child: Icon(
                        hasFile ? Icons.video_file_outlined : Icons.cloud_upload_outlined,
                        size: 40,
                        color: const Color(0xFF8B5CF6),
                      ),
                    ),
                    const SizedBox(height: 16),
                    Text(
                      hasFile ? p.basename(provider.inputMediaFile!.path) : '拖拽音视频文件到此处，或点击浏览本地文件',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Text(
                      hasFile
                          ? '文件大小: ${(provider.inputMediaFile!.lengthSync() / (1024 * 1024)).toStringAsFixed(2)} MB'
                          : '支持 mp4, mkv, mp3, wav, m4a, flac 等常见格式',
                      style: TextStyle(
                        fontSize: 12,
                        color: Colors.grey[400],
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildConfigCard(BuildContext context, TranscriptionProvider provider) {
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
            '识别与配置选项',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 24),
          
          // 模型选择
          const Text('识别模型 (Whisper GGML)', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: const Color(0x08FFFFFF),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0x1FFFFFFF)),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                isExpanded: true,
                value: provider.selectedModel,
                dropdownColor: const Color(0xFF1E1E2C),
                items: ModelService.availableModels.map((m) {
                  final isDl = provider.downloadedModels.contains(m.filename);
                  return DropdownMenuItem<String>(
                    value: m.filename,
                    child: Row(
                      children: [
                        Text(m.name),
                        const SizedBox(width: 8),
                        Text(
                          '(${m.size})',
                          style: const TextStyle(fontSize: 12, color: Colors.grey),
                        ),
                        const Spacer(),
                        Tooltip(
                          message: isDl ? '模型已就绪 (可前往管理)' : '点击下载',
                          child: GestureDetector(
                            onTap: () {
                              provider.setCurrentTab(2);
                            },
                            child: Padding(
                              padding: const EdgeInsets.symmetric(horizontal: 4, vertical: 8),
                              child: Icon(
                                isDl ? Icons.check_circle_outline : Icons.download_for_offline_outlined,
                                color: isDl ? Colors.green : Colors.grey,
                                size: 18,
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  );
                }).toList(),
                onChanged: (val) {
                  if (val != null) provider.setSelectedModel(val);
                },
              ),
            ),
          ),
          const SizedBox(height: 20),

          // 语言选择
          const Text('音频主语言', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
          const SizedBox(height: 8),
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 12),
            decoration: BoxDecoration(
              color: const Color(0x08FFFFFF),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0x1FFFFFFF)),
            ),
            child: DropdownButtonHideUnderline(
              child: DropdownButton<String>(
                isExpanded: true,
                value: provider.selectedLanguage,
                dropdownColor: const Color(0xFF1E1E2C),
                items: const [
                  DropdownMenuItem(value: 'auto', child: Text('自动检测语言 (Auto Detect)')),
                  DropdownMenuItem(value: 'zh', child: Text('中文 (Chinese)')),
                  DropdownMenuItem(value: 'en', child: Text('英文 (English)')),
                  DropdownMenuItem(value: 'ja', child: Text('日语 (Japanese)')),
                  DropdownMenuItem(value: 'ko', child: Text('韩语 (Korean)')),
                ],
                onChanged: (val) {
                  if (val != null) provider.setSelectedLanguage(val);
                },
              ),
            ),
          ),
          const SizedBox(height: 20),

          // 翻译选项
          Row(
            children: [
              const Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text('英语翻译模式', style: TextStyle(fontSize: 14, fontWeight: FontWeight.w600)),
                    SizedBox(height: 4),
                    Text('开启后识别文本将自动被翻译成英文输出', style: TextStyle(fontSize: 11, color: Colors.grey)),
                  ],
                ),
              ),
              Switch(
                value: provider.translateToEnglish,
                activeColor: const Color(0xFF8B5CF6),
                onChanged: (val) {
                  provider.setTranslate(val);
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildStatusCard(BuildContext context, TranscriptionProvider provider) {
    final isLoading = provider.status == TranscriptionStatus.extractingAudio ||
        provider.status == TranscriptionStatus.transcribing;

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
            '处理控制与状态',
            style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 24),
          
          // 运行阶段显示
          _buildStageIndicator(
            '1. 音频重采样',
            provider.status == TranscriptionStatus.extractingAudio,
            provider.status == TranscriptionStatus.transcribing ||
                provider.status == TranscriptionStatus.completed,
          ),
          const SizedBox(height: 12),
          _buildStageIndicator(
            '2. 神经网络转写',
            provider.status == TranscriptionStatus.transcribing,
            provider.status == TranscriptionStatus.completed,
          ),
          const SizedBox(height: 24),

          // 进度展示
          if (isLoading) ...[
            ClipRRect(
              borderRadius: BorderRadius.circular(4),
              child: LinearProgressIndicator(
                value: provider.status == TranscriptionStatus.transcribing
                    ? provider.progress / 100
                    : null, // 音频提取为不确定进度
                backgroundColor: const Color(0x1FFFFFFF),
                color: const Color(0xFF8B5CF6),
                minHeight: 8,
              ),
            ),
            const SizedBox(height: 12),
            if (provider.status == TranscriptionStatus.transcribing)
              Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text('识别进度', style: TextStyle(fontSize: 12, color: Colors.grey)),
                  Text('${provider.progress}%', style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
                ],
              ),
          ],

          const SizedBox(height: 20),
          Text(
            provider.statusMessage,
            style: TextStyle(
              fontSize: 13,
              color: provider.status == TranscriptionStatus.failed
                  ? Colors.redAccent
                  : provider.status == TranscriptionStatus.completed
                      ? Colors.green
                      : Colors.white,
            ),
          ),
          const SizedBox(height: 24),

          // 开始按钮
          SizedBox(
            width: double.infinity,
            height: 48,
            child: ElevatedButton(
              onPressed: isLoading ? null : () => provider.startTranscription(),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF8B5CF6),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
                disabledBackgroundColor: const Color(0x1F8B5CF6),
              ),
              child: isLoading
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                    )
                  : const Text('开始提取并转写', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStageIndicator(String label, bool isRunning, bool isDone) {
    Color color = Colors.grey[600]!;
    Widget icon = const Icon(Icons.circle_outlined, size: 16, color: Colors.grey);

    if (isRunning) {
      color = const Color(0xFF8B5CF6);
      icon = const SizedBox(
        width: 14,
        height: 14,
        child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF8B5CF6)),
      );
    } else if (isDone) {
      color = Colors.green;
      icon = const Icon(Icons.check_circle, size: 16, color: Colors.green);
    }

    return Row(
      children: [
        icon,
        const SizedBox(width: 12),
        Text(
          label,
          style: TextStyle(
            fontSize: 14,
            fontWeight: isRunning ? FontWeight.bold : FontWeight.normal,
            color: color,
          ),
        ),
      ],
    );
  }
}
