import 'dart:async';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/transcription_provider.dart';
import '../services/model_service.dart';
import 'hover_dropdown.dart';

class ModelsView extends StatefulWidget {
  const ModelsView({super.key});

  @override
  State<ModelsView> createState() => _ModelsViewState();
}

class _ModelsViewState extends State<ModelsView> {
  /// 各量化版本的本地实际体积缓存: `<baseId>|<quantId>` -> bytes
  final Map<String, int> _localSizes = {};

  /// 基础模型总占用缓存
  final Map<String, int> _baseSizes = {};

  TranscriptionProvider? _provider;
  Timer? _sizeRefreshTimer;

  @override
  void initState() {
    super.initState();
    _refreshSizes();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        context.read<TranscriptionProvider>().testMirrorsSpeedOnPageOpen();
      }
    });
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final p = Provider.of<TranscriptionProvider>(context);
    if (p != _provider) {
      _provider?.removeListener(_onProviderChanged);
      _provider = p;
      _provider?.addListener(_onProviderChanged);
    }
  }

  @override
  void dispose() {
    _provider?.removeListener(_onProviderChanged);
    _sizeRefreshTimer?.cancel();
    super.dispose();
  }

  /// 下载进度会高频 notify: 用 Timer 防抖，状态稳定 2s 后才刷新一次本地体积，
  /// 避免下载期间反复扫描磁盘。
  void _onProviderChanged() {
    _sizeRefreshTimer?.cancel();
    _sizeRefreshTimer = Timer(const Duration(seconds: 2), () {
      if (mounted) _refreshSizes();
    });
  }

  Future<void> _refreshSizes() async {
    final service = ModelService();
    final updated = <String, int>{};
    for (final m in ModelService.availableBaseModels) {
      for (final q in m.quants) {
        updated['${m.id}|${q.id}'] = await service.getQuantLocalBytes(m.id, q.id);
      }
      updated['base:${m.id}'] = await service.getBaseLocalBytes(m.id);
    }
    updated['base:${ModelService.alignerModel.dirName}'] =
        await service.getBaseLocalBytes(ModelService.alignerModel.dirName);
    if (mounted) {
      setState(() {
        _localSizes
          ..clear()
          ..addAll(updated);
      });
    }
  }

  Future<void> _deleteQuant(
      BuildContext context, String baseId, QwenQuantVersion quant, TranscriptionProvider provider) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除量化版本'),
        content: Text('确定要删除 ${quant.label} (${quant.ggufName}) 吗？这将释放磁盘空间。'),
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
      await provider.deleteQuant(baseId, quant.id);
      await _refreshSizes();
      if (context.mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('已删除 ${quant.label} 量化版本')),
        );
      }
    }
  }

  Future<void> _deleteAligner(BuildContext context, TranscriptionProvider provider) async {
    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('删除 ForcedAligner'),
        content: const Text('确定要删除 ForcedAligner 组件吗？'),
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
      await provider.deleteAlignerModel();
      await _refreshSizes();
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
          _buildMirrorSelector(provider),
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

  /// 下载源选择: 自动 (测速) / 官方 / HF-Mirror，紧凑单行
  Widget _buildMirrorSelector(TranscriptionProvider provider) {
    final mirrors = [ModelService.autoMirror, ...ModelService.availableMirrors];
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      child: Row(
        children: [
          const Icon(Icons.cloud_download_outlined, size: 16, color: Color(0xFF8B5CF6)),
          const SizedBox(width: 8),
          const Text('下载源:', style: TextStyle(fontSize: 13, fontWeight: FontWeight.bold)),
          if (provider.isTestingMirrors) ...[
            const SizedBox(width: 8),
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2, color: Color(0xFF8B5CF6)),
            ),
          ],
          const SizedBox(width: 8),
          Expanded(
            child: HoverDropdown<String>(
              value: provider.selectedMirrorId,
              padding: const EdgeInsets.symmetric(horizontal: 10),
              borderRadius: 8,
              height: 42,
              // 按钮 (选中态) 显示完整标签, 如 "自动（当前：HF-Mirror 国内镜像站）"
              selectedItemBuilder: (context) => [
                for (final mirror in mirrors)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: Text(
                      mirror.id == 'auto' ? provider.selectedMirrorLabel : mirror.name,
                      style: const TextStyle(fontSize: 13, color: Colors.white),
                      overflow: TextOverflow.ellipsis,
                    ),
                  ),
              ],
              items: [
                for (final mirror in mirrors)
                  DropdownMenuItem<String>(
                    value: mirror.id,
                    child: Text(mirror.name),
                  ),
              ],
              onChanged: (val) {
                if (val != null) provider.setSelectedMirrorId(val);
              },
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildModelsGrid(BuildContext context, TranscriptionProvider provider) {
    return ListView(
      padding: const EdgeInsets.only(bottom: 24),
      children: [
        // ===== Qwen3-ASR 基础模型卡片 =====
        for (final model in ModelService.availableBaseModels) ...[
          _buildBaseModelCard(context, provider, model),
          const SizedBox(height: 20),
        ],
        // ===== ForcedAligner 卡片 =====
        _buildAlignerCard(context, provider),
      ],
    );
  }

  Widget _buildBaseModelCard(
      BuildContext context, TranscriptionProvider provider, QwenBaseModel model) {
    final isBaseDl = provider.downloadedModels.contains(model.id);
    final baseLocalBytes = _baseSizes['base:${model.id}'] ?? 0;

    return Card(
      color: const Color(0x0CFFFFFF),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: isBaseDl ? const Color(0x3A00FF00) : const Color(0x1FFFFFFF),
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
                  style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
                ),
                const Spacer(),
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: isBaseDl ? const Color(0x1A00FF00) : const Color(0x1AFFFFFF),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    isBaseDl ? 'Encoder 就绪' : 'Encoder 未下载',
                    style: TextStyle(
                      fontSize: 11,
                      color: isBaseDl ? Colors.green : Colors.grey,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
                if (baseLocalBytes > 0) ...[
                  const SizedBox(width: 8),
                  Text(
                    '本地占用: ${ModelService.formatBytes(baseLocalBytes)}',
                    style: const TextStyle(fontSize: 11, color: Colors.grey),
                  ),
                ],
              ],
            ),
            const SizedBox(height: 4),
            Text(
              model.description,
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
            const SizedBox(height: 12),
            // 量化版本列表
            ...model.quants.map((q) => _buildQuantRow(context, provider, model, q)),
          ],
        ),
      ),
    );
  }

  Widget _buildQuantRow(
      BuildContext context, TranscriptionProvider provider, QwenBaseModel model, QwenQuantVersion quant) {
    final isDownloading = provider.downloadingModelFile == '${model.id}|${quant.id}';
    final isSelected = provider.selectedModelBase == model.id && provider.selectedQuant == quant.id;
    final isDl = (_localSizes['${model.id}|${quant.id}'] ?? 0) > 0;
    final downloadProgress = provider.downloadProgress;
    final localBytes = _localSizes['${model.id}|${quant.id}'] ?? 0;

    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
      decoration: BoxDecoration(
        color: const Color(0x08FFFFFF),
        borderRadius: BorderRadius.circular(10),
        border: Border.all(
          color: isSelected ? const Color(0xFF8B5CF6) : const Color(0x14FFFFFF),
          width: isSelected ? 1.5 : 1,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(
                quant.label,
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                  color: isSelected ? const Color(0xFFA78BFA) : Colors.white,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                '(${quant.sizeText})',
                style: const TextStyle(fontSize: 11, color: Colors.grey),
              ),
              const Spacer(),
              if (localBytes > 0)
                Text(
                  ModelService.formatBytes(localBytes),
                  style: const TextStyle(fontSize: 11, color: Colors.grey),
                ),
              const SizedBox(width: 8),
              if (isDownloading) ...[
                SizedBox(
                  width: 14,
                  height: 14,
                  child: CircularProgressIndicator(
                    strokeWidth: 2,
                    value: downloadProgress,
                    color: const Color(0xFF8B5CF6),
                  ),
                ),
                const SizedBox(width: 8),
                Text(
                  '${(downloadProgress * 100).toStringAsFixed(0)}%',
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
              ] else ...[
                if (isDl)
                  IconButton(
                    icon: const Icon(Icons.delete_outline, color: Colors.redAccent, size: 18),
                    onPressed: provider.downloadingModelFile != null
                        ? null
                        : () => _deleteQuant(context, model.id, quant, provider),
                  ),
                ElevatedButton(
                  onPressed: provider.downloadingModelFile != null
                      ? null
                      : isDl
                          ? () {
                              provider.setSelectedModelBase(model.id);
                              provider.setSelectedQuant(quant.id);
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('已启用 ${model.name} · ${quant.label}')),
                              );
                            }
                          : () {
                              provider.downloadQuant(model, quant);
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(
                                  content: Text('开始下载 ${model.name} · ${quant.label} (自动选择最快下载源)...'),
                                ),
                              );
                            },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: isDl ? const Color(0xFF1E293B) : const Color(0xFF8B5CF6),
                    foregroundColor: Colors.white,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 6),
                  ),
                  child: Text(isDl ? '启用' : '下载'),
                ),
              ],
            ],
          ),
          if (isDownloading && provider.downloadError.isNotEmpty) ...[
            const SizedBox(height: 4),
            Text(
              provider.downloadError,
              style: const TextStyle(fontSize: 11, color: Colors.redAccent),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildAlignerCard(BuildContext context, TranscriptionProvider provider) {
    final model = ModelService.alignerModel;
    final isDl = provider.selectedAlignerModel != null ||
        (_localSizes['base:${model.dirName}'] ?? 0) > 0;
    final isDownloading = provider.downloadingModelFile == model.dirName;
    final downloadProgress = provider.downloadProgress;
    final localBytes = _localSizes['base:${model.dirName}'] ?? 0;

    return Card(
      color: const Color(0x0CFFFFFF),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(16),
        side: BorderSide(
          color: isDl ? const Color(0x3A00FF00) : const Color(0x1FFFFFFF),
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
                Text(model.name, style: const TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                const Spacer(),
                if (localBytes > 0)
                  Text(
                    '本地占用: ${ModelService.formatBytes(localBytes)}',
                    style: const TextStyle(fontSize: 11, color: Colors.grey),
                  ),
              ],
            ),
            const SizedBox(height: 4),
            Text(model.description, style: const TextStyle(fontSize: 12, color: Colors.grey)),
            const SizedBox(height: 12),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                if (isDownloading) ...[
                  Expanded(
                    child: ClipRRect(
                      borderRadius: BorderRadius.circular(4),
                      child: LinearProgressIndicator(
                        value: downloadProgress,
                        backgroundColor: const Color(0x1FFFFFFF),
                        color: const Color(0xFF8B5CF6),
                        minHeight: 6,
                      ),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Text(
                    '${(downloadProgress * 100).toStringAsFixed(1)}%',
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
                ] else ...[
                  if (isDl)
                    IconButton(
                      icon: const Icon(Icons.delete_outline, color: Colors.redAccent, size: 20),
                      onPressed: provider.downloadingModelFile != null
                          ? null
                          : () => _deleteAligner(context, provider),
                    ),
                  ElevatedButton(
                    onPressed: provider.downloadingModelFile != null
                        ? null
                        : isDl
                            ? () {
                                provider.setAlignerModel(model.dirName);
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('已成功启用 ForcedAligner')),
                                );
                              }
                            : () {
                                provider.downloadAligner();
                                ScaffoldMessenger.of(context).showSnackBar(
                                  const SnackBar(content: Text('开始下载 ForcedAligner (自动选择最快下载源)...')),
                                );
                              },
                    style: ElevatedButton.styleFrom(
                      backgroundColor: isDl ? const Color(0xFF1E293B) : const Color(0xFF8B5CF6),
                      foregroundColor: Colors.white,
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    ),
                    child: Text(isDl ? '启用组件' : '开始下载'),
                  ),
                ],
              ],
            ),
          ],
        ),
      ),
    );
  }
}
