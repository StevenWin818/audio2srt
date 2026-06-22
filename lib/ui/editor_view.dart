import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:path/path.dart' as p;
import 'package:path_provider/path_provider.dart';
import '../providers/transcription_provider.dart';

class EditorView extends StatefulWidget {
  const EditorView({super.key});

  @override
  State<EditorView> createState() => _EditorViewState();
}

class _EditorViewState extends State<EditorView> {
  String _searchQuery = '';
  int? _editingIndex;
  final TextEditingController _textEditController = TextEditingController();
  final TextEditingController _startEditController = TextEditingController();
  final TextEditingController _endEditController = TextEditingController();

  bool _isMuxing = false;
  String _muxStatus = '';

  @override
  void dispose() {
    _textEditController.dispose();
    _startEditController.dispose();
    _endEditController.dispose();
    super.dispose();
  }

  void _startEditing(int index, SubtitleItem item) {
    setState(() {
      _editingIndex = index;
      _textEditController.text = item.text;
      _startEditController.text = item.startMs.toString();
      _endEditController.text = item.endMs.toString();
    });
  }

  void _saveEditing(TranscriptionProvider provider) {
    if (_editingIndex == null) return;
    
    final start = int.tryParse(_startEditController.text) ?? 0;
    final end = int.tryParse(_endEditController.text) ?? 0;
    
    provider.updateSubtitleText(_editingIndex!, _textEditController.text);
    provider.updateSubtitleTimes(_editingIndex!, start, end);

    setState(() {
      _editingIndex = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context, listen: false);
    final selectedLanguage = context.select<TranscriptionProvider, String>((p) => p.selectedLanguage);
    final hasInputFile = context.select<TranscriptionProvider, bool>((p) => p.inputMediaFile != null);

    return Selector<TranscriptionProvider, List<SubtitleItem>>(
      selector: (context, provider) => provider.subtitles,
      builder: (context, subs, child) {
        final filteredSubs = subs.asMap().entries.where((entry) {
          return entry.value.text.toLowerCase().contains(_searchQuery.toLowerCase());
        }).toList();

        return Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              _buildHeader(provider, subs, selectedLanguage, hasInputFile),
              const SizedBox(height: 24),
              _buildSearchAndToolbar(provider, subs),
              const SizedBox(height: 16),
              Expanded(
                child: subs.isEmpty
                    ? _buildEmptyState()
                    : _buildSubtitlesList(filteredSubs, provider),
              ),
            ],
          ),
        );
      },
    );
  }

  Widget _buildHeader(
    TranscriptionProvider provider,
    List<SubtitleItem> subs,
    String selectedLanguage,
    bool hasInputFile,
  ) {
    return Row(
      children: [
        const Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '字幕编辑器',
              style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
            ),
            SizedBox(height: 4),
            Text(
              '校对、修改文本和微调时间戳',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
          ],
        ),
        const Spacer(),
        if (subs.isNotEmpty) ...[
          if (selectedLanguage == 'zh' || selectedLanguage == 'auto') ...[
            PopupMenuButton<bool>(
              tooltip: '简繁转换',
              position: PopupMenuPosition.under,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  border: Border.all(color: const Color(0xFF8B5CF6)),
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  children: const [
                    Icon(Icons.g_translate, color: Color(0xFF8B5CF6), size: 18),
                    SizedBox(width: 8),
                    Text('简繁转换', style: TextStyle(color: Color(0xFF8B5CF6), fontWeight: FontWeight.w500)),
                  ],
                ),
              ),
              onSelected: (toSimplified) {
                provider.convertSubtitlesToChinese(toSimplified);
                ScaffoldMessenger.of(context).showSnackBar(
                  SnackBar(content: Text(toSimplified ? '已转换为简体中文' : '已转换为繁体中文')),
                );
              },
              itemBuilder: (context) => [
                const PopupMenuItem(
                  value: true,
                  child: Text('转换为简体中文 (Simplified)'),
                ),
                const PopupMenuItem(
                  value: false,
                  child: Text('转换为繁体中文 (Traditional)'),
                ),
              ],
            ),
            const SizedBox(width: 12),
          ],
          ElevatedButton.icon(
            onPressed: () => _showExportDialog(context, provider),
            icon: const Icon(Icons.download),
            label: const Text('导出字幕'),
            style: ElevatedButton.styleFrom(
              backgroundColor: const Color(0xFF8B5CF6),
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            ),
          ),
          const SizedBox(width: 12),
          OutlinedButton.icon(
            onPressed: () => _showMuxDialog(context, provider),
            icon: const Icon(Icons.movie_creation_outlined),
            label: const Text('字幕压制/集成'),
            style: OutlinedButton.styleFrom(
              foregroundColor: const Color(0xFF8B5CF6),
              side: const BorderSide(color: Color(0xFF8B5CF6)),
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            ),
          ),
        ],
      ],
    );
  }

  Widget _buildSearchAndToolbar(TranscriptionProvider provider, List<SubtitleItem> subs) {
    return Row(
      children: [
        Expanded(
          child: Container(
            height: 40,
            decoration: BoxDecoration(
              color: const Color(0x0CFFFFFF),
              borderRadius: BorderRadius.circular(8),
            ),
            child: TextField(
              decoration: const InputDecoration(
                hintText: '搜索字幕文本...',
                prefixIcon: Icon(Icons.search, size: 18),
                border: InputBorder.none,
                contentPadding: EdgeInsets.symmetric(vertical: 10),
              ),
              onChanged: (val) {
                setState(() {
                  _searchQuery = val;
                });
              },
            ),
          ),
        ),
        const SizedBox(width: 16),
        if (subs.isNotEmpty)
          Text(
            '共 ${subs.length} 条字幕',
            style: const TextStyle(fontSize: 13, color: Colors.grey),
          ),
      ],
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(Icons.edit_note_outlined, size: 64, color: Colors.grey[600]),
          const SizedBox(height: 16),
          const Text(
            '当前无可用字幕',
            style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold),
          ),
          const SizedBox(height: 8),
          Text(
            '请在“主面板”导入并运行转写流程，以生成待编辑字幕。',
            style: TextStyle(fontSize: 13, color: Colors.grey[500]),
          ),
        ],
      ),
    );
  }

  Widget _buildSubtitlesList(
      List<MapEntry<int, SubtitleItem>> items, TranscriptionProvider provider) {
    return ListView.builder(
      itemCount: items.length,
      itemBuilder: (context, index) {
        final globalIndex = items[index].key;
        final item = items[index].value;
        final isEditing = _editingIndex == globalIndex;

        return Card(
          margin: const EdgeInsets.only(bottom: 8),
          color: isEditing ? const Color(0x1F8B5CF6) : const Color(0x0CFFFFFF),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
            side: BorderSide(
              color: isEditing ? const Color(0xFF8B5CF6) : const Color(0x0FFFFFFF),
            ),
          ),
          child: Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
            child: isEditing
                ? _buildRowEditMode(globalIndex, provider)
                : _buildRowDisplayMode(globalIndex, item, provider),
          ),
        );
      },
    );
  }

  Widget _buildRowDisplayMode(int index, SubtitleItem item, TranscriptionProvider provider) {
    final startStr = TranscriptionProvider.formatSrtTimestamp(item.startMs).substring(3, 11);
    final endStr = TranscriptionProvider.formatSrtTimestamp(item.endMs).substring(3, 11);
    final duration = ((item.endMs - item.startMs) / 1000).toStringAsFixed(2);

    return Row(
      children: [
        Container(
          width: 32,
          alignment: Alignment.center,
          child: Text(
            '#${index + 1}',
            style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.grey),
          ),
        ),
        const SizedBox(width: 16),
        Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              '$startStr --> $endStr',
              style: const TextStyle(fontFamily: 'monospace', fontSize: 13, color: Color(0xFF06B6D4)),
            ),
            const SizedBox(height: 2),
            Text(
              '时长: ${duration}s',
              style: const TextStyle(fontSize: 11, color: Colors.grey),
            ),
          ],
        ),
        const SizedBox(width: 24),
        Expanded(
          child: Text(
            item.text,
            style: const TextStyle(fontSize: 15),
          ),
        ),
        IconButton(
          icon: const Icon(Icons.edit_outlined, size: 20, color: Colors.grey),
          onPressed: () => _startEditing(index, item),
        ),
      ],
    );
  }

  Widget _buildRowEditMode(int index, TranscriptionProvider provider) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          children: [
            Text(
              '编辑字幕 #${index + 1}',
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 13, color: Color(0xFF8B5CF6)),
            ),
            const Spacer(),
            TextButton.icon(
              onPressed: () => _saveEditing(provider),
              icon: const Icon(Icons.check, size: 16),
              label: const Text('保存'),
              style: TextButton.styleFrom(foregroundColor: Colors.green),
            ),
            TextButton.icon(
              onPressed: () {
                setState(() {
                  _editingIndex = null;
                });
              },
              icon: const Icon(Icons.close, size: 16),
              label: const Text('取消'),
              style: TextButton.styleFrom(foregroundColor: Colors.redAccent),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: TextField(
                controller: _startEditController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: '起始时间 (ms)',
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                ),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: TextField(
                controller: _endEditController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: '结束时间 (ms)',
                  border: OutlineInputBorder(),
                  contentPadding: EdgeInsets.symmetric(horizontal: 10, vertical: 8),
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        TextField(
          controller: _textEditController,
          maxLines: 2,
          decoration: const InputDecoration(
            labelText: '字幕内容',
            border: OutlineInputBorder(),
            contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 10),
          ),
        ),
      ],
    );
  }

  void _showExportDialog(BuildContext context, TranscriptionProvider provider) {
    showDialog(
      context: context,
      builder: (context) {
        return AlertDialog(
          title: const Text('导出字幕文件'),
          content: const Text('请选择您希望导出的字幕格式格式类型。'),
          actions: [
            TextButton(
              onPressed: () async {
                Navigator.pop(context);
                final srtName = '${p.basenameWithoutExtension(provider.inputMediaFile!.path)}.srt';
                final path = await FilePicker.saveFile(
                  dialogTitle: '保存 SRT 字幕文件',
                  fileName: srtName,
                  type: FileType.custom,
                  allowedExtensions: ['srt'],
                );
                if (path != null) {
                  await provider.exportSubtitles(path, isVtt: false);
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('已成功保存字幕到 $path')),
                    );
                  }
                }
              },
              child: const Text('导出为 .srt (标准格式)'),
            ),
            TextButton(
              onPressed: () async {
                Navigator.pop(context);
                final vttName = '${p.basenameWithoutExtension(provider.inputMediaFile!.path)}.vtt';
                final path = await FilePicker.saveFile(
                  dialogTitle: '保存 VTT 字幕文件',
                  fileName: vttName,
                  type: FileType.custom,
                  allowedExtensions: ['vtt'],
                );
                if (path != null) {
                  await provider.exportSubtitles(path, isVtt: true);
                  if (mounted) {
                    ScaffoldMessenger.of(context).showSnackBar(
                      SnackBar(content: Text('已成功保存字幕到 $path')),
                    );
                  }
                }
              },
              child: const Text('导出为 .vtt (网页格式)'),
            ),
          ],
        );
      },
    );
  }

  void _showMuxDialog(BuildContext context, TranscriptionProvider provider) {
    bool hardBurn = false;

    showDialog(
      context: context,
      builder: (context) {
        return StatefulBuilder(
          builder: (context, setDialogState) {
            return AlertDialog(
              title: const Text('将字幕封装/硬压制到视频中'),
              content: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('此操作将读取当前导出的字幕并与原视频合并。'),
                  const SizedBox(height: 16),
                  Row(
                    children: [
                      Checkbox(
                        value: hardBurn,
                        onChanged: (val) {
                          setDialogState(() {
                            hardBurn = val ?? false;
                          });
                        },
                      ),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text('硬压制字幕 (Hard Burn)', style: TextStyle(fontWeight: FontWeight.bold)),
                            Text('开启后，字幕会被直接烧录进视频画面，无法隐藏。关闭则为软封装字幕轨。',
                                style: TextStyle(fontSize: 11, color: Colors.grey)),
                          ],
                        ),
                      )
                    ],
                  ),
                  if (_isMuxing) ...[
                    const SizedBox(height: 20),
                    const LinearProgressIndicator(),
                    const SizedBox(height: 8),
                    Text(_muxStatus, style: const TextStyle(fontSize: 12, color: Colors.blue)),
                  ],
                ],
              ),
              actions: [
                TextButton(
                  onPressed: _isMuxing ? null : () => Navigator.pop(context),
                  child: const Text('取消'),
                ),
                ElevatedButton(
                  onPressed: _isMuxing
                      ? null
                      : () async {
                          setDialogState(() {
                            _isMuxing = true;
                            _muxStatus = '正在准备字幕文件...';
                          });

                          try {
                            // 1. 创建一个临时的 SRT 缓存文件
                            final tempDir = await getTemporaryDirectory();
                            final tempSrtPath = p.join(tempDir.path, 'temp_subs.srt');
                            await provider.exportSubtitles(tempSrtPath, isVtt: false);

                            // 2. 选择保存最终视频的路径
                            final ext = p.extension(provider.inputMediaFile!.path);
                            final videoName =
                                '${p.basenameWithoutExtension(provider.inputMediaFile!.path)}_with_subs$ext';
                            
                            final outPath = await FilePicker.saveFile(
                              dialogTitle: '保存合成视频',
                              fileName: videoName,
                              type: FileType.custom,
                              allowedExtensions: ext.replaceAll('.', '').isEmpty ? ['mp4', 'mkv'] : [ext.replaceAll('.', '')],
                            );

                            if (outPath != null) {
                              setDialogState(() {
                                _muxStatus = 'FFmpeg 合成中，这需要一些时间，请稍候...';
                              });

                              await provider.muxSubtitlesToVideo(
                                srtPath: tempSrtPath,
                                outputPath: outPath,
                                hardBurn: hardBurn,
                              );

                              if (mounted) {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(content: Text('集成视频合成成功！已保存到 $outPath')),
                                );
                              }
                            }
                          } catch (e) {
                            if (mounted) {
                              ScaffoldMessenger.of(context).showSnackBar(
                                SnackBar(content: Text('合成失败: $e')),
                              );
                            }
                          } finally {
                            setDialogState(() {
                              _isMuxing = false;
                              _muxStatus = '';
                            });
                            Navigator.pop(context);
                          }
                        },
                  style: ElevatedButton.styleFrom(backgroundColor: const Color(0xFF8B5CF6)),
                  child: const Text('开始合成'),
                )
              ],
            );
          },
        );
      },
    );
  }
}
