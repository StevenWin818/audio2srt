import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/transcription_provider.dart';
import '../services/model_service.dart';

class ModelsView extends StatelessWidget {
  const ModelsView({super.key});

  Future<void> _deleteModel(BuildContext context, String filename, TranscriptionProvider provider) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除模型文件'),
        content: Text('确定要删除模型 $filename 吗？这将释放磁盘空间，之后如果需要需重新下载。'),
        actions: [
          TextButton(onPressed: () => Navigator.pop(context, false), child: const Text('取消')),
          TextButton(
            onPressed: () => Navigator.pop(context, true),
            style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            child: const Text('删除'),
          ),
        ],
      ),
    );

    if (confirm == true) {
      await provider.deleteModel(filename);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('模型 $filename 已成功删除')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);

    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(),
          const SizedBox(height: 24),
          Expanded(
            child: _buildModelsGrid(context, provider),
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
          '模型管理器',
          style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 4),
        Text(
          '下载并管理用于离线语音识别的 Whisper (GGML 格式) 模型',
          style: TextStyle(fontSize: 14, color: Colors.grey),
        ),
      ],
    );
  }

  Widget _buildModelsGrid(BuildContext context, TranscriptionProvider provider) {
    return GridView.builder(
      gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
        crossAxisCount: 2,
        crossAxisSpacing: 20,
        mainAxisSpacing: 20,
        childAspectRatio: 1.6,
      ),
      itemCount: ModelService.availableModels.length,
      itemBuilder: (context, index) {
        final model = ModelService.availableModels[index];
        final isDownloaded = provider.downloadedModels.contains(model.filename);
        final isDownloading = provider.downloadingModelFile == model.filename;
        final downloadProgress = provider.downloadProgress;

        return Card(
          color: const Color(0x0CFFFFFF),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: BorderSide(
              color: isDownloaded
                  ? const Color(0x3A00FF00)
                  : isDownloading
                      ? const Color(0xFF8B5CF6)
                      : const Color(0x1FFFFFFF),
              width: 1.5,
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Text(
                      model.name.split(' (').first,
                      style: const TextStyle(fontSize: 18, fontWeight: FontWeight.bold),
                    ),
                    const Spacer(),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                      decoration: BoxDecoration(
                        color: isDownloaded ? const Color(0x1A00FF00) : const Color(0x1AFFFFFF),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        isDownloaded ? '已就绪' : '未下载',
                        style: TextStyle(
                          fontSize: 11,
                          color: isDownloaded ? Colors.green : Colors.grey,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text(
                  '文件名: ${model.filename}',
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
                Text(
                  '文件大小: ${model.size}',
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
                const Spacer(),
                if (isDownloading) ...[
                  ClipRRect(
                    borderRadius: BorderRadius.circular(4),
                    child: LinearProgressIndicator(
                      value: downloadProgress,
                      backgroundColor: const Color(0x1FFFFFFF),
                      color: const Color(0xFF8B5CF6),
                      minHeight: 6,
                    ),
                  ),
                  const SizedBox(height: 8),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Text(
                        '正在下载: ${(downloadProgress * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(fontSize: 11, color: Colors.grey),
                      ),
                      TextButton(
                        onPressed: () => provider.cancelDownload(),
                        style: TextButton.styleFrom(
                          padding: EdgeInsets.zero,
                          minimumSize: const Size(40, 24),
                          tapTargetSize: MaterialTapTargetSize.shrinkWrap,
                        ),
                        child: const Text('取消', style: TextStyle(fontSize: 11, color: Colors.redAccent)),
                      ),
                    ],
                  ),
                ] else ...[
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      if (isDownloaded)
                        IconButton(
                          icon: const Icon(Icons.delete_outline, color: Colors.redAccent, size: 20),
                          onPressed: () => _deleteModel(context, model.filename, provider),
                        )
                      else
                        const SizedBox(),
                      ElevatedButton(
                        onPressed: provider.downloadingModelFile != null
                            ? null // 阻止同时下载多个
                            : isDownloaded
                                ? () {
                                    provider.setSelectedModel(model.filename);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text('已默认选择 ${model.filename} 模型')),
                                    );
                                  }
                                : () {
                                    provider.downloadModel(model);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text('开始下载 ${model.filename} 模型...')),
                                    );
                                  },
                        style: ElevatedButton.styleFrom(
                          backgroundColor: isDownloaded ? const Color(0xFF1E293B) : const Color(0xFF8B5CF6),
                          foregroundColor: Colors.white,
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(8),
                          ),
                          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                        ),
                        child: Text(isDownloaded ? '选用模型' : '开始下载'),
                      ),
                    ],
                  ),
                ],
              ],
            ),
          ),
        );
      },
    );
  }
}
