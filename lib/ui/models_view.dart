import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/transcription_provider.dart';
import '../services/model_service.dart';

class ModelsView extends StatelessWidget {
  const ModelsView({super.key});

  Future<void> _deleteModel(BuildContext context, String dirName, TranscriptionProvider provider) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除模型组件'),
        content: Text('确定要删除模型 $dirName 吗？这将释放磁盘空间，之后如果需要需重新下载。'),
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
      await provider.deleteModel(dirName);
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('模型 $dirName 已成功删除')),
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
          const SizedBox(height: 16),
          _buildMirrorSelector(context, provider),
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
          'Qwen3 模型管理器',
          style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
        ),
        SizedBox(height: 4),
        Text(
          '管理 Qwen3-ASR (0.6B/1.7B) 语音识别模型与 ForcedAligner 精准时间轴组件',
          style: TextStyle(fontSize: 14, color: Colors.grey),
        ),
      ],
    );
  }

  Widget _buildMirrorSelector(BuildContext context, TranscriptionProvider provider) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      child: Row(
        children: [
          const Icon(Icons.cloud_download_outlined, color: Color(0xFF8B5CF6)),
          const SizedBox(width: 12),
          const Text('下载源 (镜像站):', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
          const SizedBox(width: 12),
          Expanded(
            child: DropdownButtonHideUnderline(
              child: DropdownButton<ModelMirror>(
                value: provider.selectedMirror,
                isExpanded: true,
                dropdownColor: const Color(0xFF1E1E2C),
                items: ModelService.availableMirrors.map((mirror) {
                  final lat = mirror.latencyMs;
                  final latText = lat != null ? ' (${lat}ms)' : '';
                  return DropdownMenuItem<ModelMirror>(
                    value: mirror,
                    child: Text('${mirror.name}$latText'),
                  );
                }).toList(),
                onChanged: (val) {
                  if (val != null) provider.setSelectedMirror(val);
                },
              ),
            ),
          ),
          const SizedBox(width: 12),
          ElevatedButton.icon(
            onPressed: provider.isTestingMirrors ? null : () => provider.testMirrorsSpeed(),
            icon: provider.isTestingMirrors
                ? const SizedBox(
                    width: 14,
                    height: 14,
                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                  )
                : const Icon(Icons.speed, size: 16),
            label: Text(provider.isTestingMirrors ? '测速中...' : '一键测速'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF8B5CF6),
              foregroundColor: Colors.white,
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
            ),
          ),
        ],
      ),
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
      itemCount: ModelService.availableQwenModels.length,
      itemBuilder: (context, index) {
        final model = ModelService.availableQwenModels[index];
        final isDownloaded = provider.downloadedModels.contains(model.dirName);
        final isDownloading = provider.downloadingModelFile == model.dirName;
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
                      model.name,
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
                const SizedBox(height: 6),
                Text(
                  model.description,
                  style: const TextStyle(fontSize: 12, color: Colors.grey),
                ),
                const SizedBox(height: 4),
                Text(
                  '组件大小: ${model.size}',
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
                  if (provider.downloadError.isNotEmpty) ...[
                    Text(
                      provider.downloadError,
                      style: const TextStyle(fontSize: 11, color: Colors.redAccent),
                      maxLines: 2,
                      overflow: TextOverflow.ellipsis,
                    ),
                    const SizedBox(height: 4),
                  ],
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      if (isDownloaded)
                        IconButton(
                          icon: const Icon(Icons.delete_outline, color: Colors.redAccent, size: 20),
                          onPressed: () => _deleteModel(context, model.dirName, provider),
                        )
                      else
                        const SizedBox(),
                      ElevatedButton(
                        onPressed: provider.downloadingModelFile != null
                            ? null
                            : isDownloaded
                                ? () {
                                    if (model.type == ModelType.asr) {
                                      provider.setSelectedModel(model.dirName);
                                    } else {
                                      provider.setAlignerModel(model.dirName);
                                    }
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text('已成功启用 ${model.name}')),
                                    );
                                  }
                                : () {
                                    provider.downloadQwenModel(model);
                                    ScaffoldMessenger.of(context).showSnackBar(
                                      SnackBar(content: Text('开始从 [${provider.selectedMirror.name}] 下载 ${model.name}...')),
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
                        child: Text(isDownloaded ? '启用组件' : '开始下载'),
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
