import 'dart:io';
import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:file_picker/file_picker.dart';
import 'package:desktop_drop/desktop_drop.dart';
import 'package:path/path.dart' as p;
import '../providers/transcription_provider.dart';
import '../services/model_service.dart';
import 'hover_dropdown.dart';
import '../src/rust/api/ffmpeg.dart' as rust_ffmpeg;

class DashboardView extends StatefulWidget {
  const DashboardView({super.key});

  @override
  State<DashboardView> createState() => _DashboardViewState();
}

class _DashboardViewState extends State<DashboardView> with SingleTickerProviderStateMixin {
  late AnimationController _flipController;
  late Animation<double> _flipAnimation;
  bool _isFlipped = false;
  bool _isFlipping = false;
  bool _localShowInterruptConfirm = false;
  TranscriptionProvider? _provider;

  @override
  void initState() {
    super.initState();
    _flipController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 600),
    );
    _flipAnimation = Tween<double>(begin: 0.0, end: 1.0).animate(
      CurvedAnimation(parent: _flipController, curve: Curves.easeInOutBack),
    );
  }

  @override
  void didChangeDependencies() {
    super.didChangeDependencies();
    final newProvider = Provider.of<TranscriptionProvider>(context);
    if (_provider != newProvider) {
      _provider?.removeListener(_onProviderChange);
      _provider = newProvider;
      _provider?.addListener(_onProviderChange);
    }
    _onProviderChange();
  }

  @override
  void dispose() {
    _provider?.removeListener(_onProviderChange);
    _flipController.dispose();
    super.dispose();
  }

  void _onProviderChange() {
    if (_provider == null) return;
    if (_isFlipping) return;
    final status = _provider!.status;
    final isTranscribing = status == TranscriptionStatus.transcribing ||
        status == TranscriptionStatus.extractingAudio;

    if (isTranscribing && !_isFlipped) {
      _isFlipped = true;
      _flipController.forward();
    } else if (!isTranscribing && _isFlipped) {
      if (status == TranscriptionStatus.idle) {
        _isFlipped = false;
        _flipController.reverse();
        setState(() {
          _localShowInterruptConfirm = false;
        });
      }
    }
  }

  String _formatMs(int ms) {
    if (ms <= 0) return '0秒';
    final s = ms ~/ 1000;
    final m = s ~/ 60;
    final sec = s % 60;
    if (m > 0) {
      return '$m分$sec秒';
    }
    return '$sec秒';
  }

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(24.0),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildHeader(),
          const SizedBox(height: 24),
          Expanded(
            child: FlipCard(
              animation: _flipAnimation,
              front: _buildFrontSide(context),
              back: _buildBackSide(context),
            ),
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

  Widget _buildFrontSide(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);
    final inputMediaFile = provider.inputMediaFile;

    return Container(
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1FFFFFFF)),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 上方：拖入和选择区域
          _buildTopImportArea(provider, inputMediaFile),
          const SizedBox(height: 24),
          
          // 两栏配置选项
          Expanded(
            child: SingleChildScrollView(
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  // 左栏
                  Expanded(
                    child: _buildLeftColumn(provider),
                  ),
                  const SizedBox(width: 24),
                  // 右栏
                  Expanded(
                    child: _buildRightColumn(provider),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),

          // 下方：开始按钮
          _buildStartButton(provider),
        ],
      ),
    );
  }

  Widget _buildTopImportArea(TranscriptionProvider provider, File? file) {
    final hasFile = file != null;

    return DropTarget(
      onDragDone: (details) {
        if (details.files.isNotEmpty) {
          provider.setInputFile(File(details.files.first.path));
        }
      },
      child: Container(
        height: 140,
        decoration: BoxDecoration(
          color: const Color(0x05FFFFFF),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: hasFile ? const Color(0xFF8B5CF6) : const Color(0x0FFFFFFF),
            width: 1.5,
          ),
        ),
        child: InkWell(
          borderRadius: BorderRadius.circular(12),
          onTap: () async {
            final result = await FilePicker.pickFiles(
              type: FileType.custom,
              allowedExtensions: ['mp3', 'wav', 'mp4', 'mkv', 'm4a', 'aac', 'flac'],
            );
            if (result != null && result.files.single.path != null) {
              provider.setInputFile(File(result.files.single.path!));
            }
          },
          child: hasFile
              ? Padding(
                  padding: const EdgeInsets.all(16.0),
                  child: Row(
                    children: [
                      // 左侧：文件解析数据
                      Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text(
                              p.basename(file.path),
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 16,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            const SizedBox(height: 6),
                            Text(
                              '路径: ${file.path}',
                              maxLines: 1,
                              overflow: TextOverflow.ellipsis,
                              style: const TextStyle(
                                fontSize: 12,
                                color: Colors.grey,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Row(
                              children: [
                                Container(
                                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                                  decoration: BoxDecoration(
                                    color: const Color(0x1F8B5CF6),
                                    borderRadius: BorderRadius.circular(4),
                                  ),
                                  child: Text(
                                    '时长: ${_formatMs(provider.totalMs)}',
                                    style: const TextStyle(
                                      fontSize: 12,
                                      color: Color(0xFFA78BFA),
                                      fontWeight: FontWeight.bold,
                                    ),
                                  ),
                                ),
                                const SizedBox(width: 12),
                                Text(
                                  '文件大小: ${(file.lengthSync() / (1024 * 1024)).toStringAsFixed(2)} MB',
                                  style: const TextStyle(
                                    fontSize: 12,
                                    color: Colors.grey,
                                  ),
                                ),
                              ],
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(width: 16),
                      // 右侧：音视频封面
                      _buildMediaCover(provider),
                    ],
                  ),
                )
              : Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      const Icon(
                        Icons.cloud_upload_outlined,
                        size: 32,
                        color: Color(0xFF8B5CF6),
                      ),
                      const SizedBox(height: 8),
                      const Text(
                        '拖拽音视频文件到此处，或点击浏览本地文件',
                        style: TextStyle(
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                      const SizedBox(height: 4),
                      Text(
                        '支持 mp4, mkv, mp3, wav, m4a, flac 等常用格式',
                        style: TextStyle(
                          fontSize: 11,
                          color: Colors.grey[400],
                        ),
                      ),
                    ],
                  ),
                ),
        ),
      ),
    );
  }

  Widget _buildMediaCover(TranscriptionProvider provider) {
    if (provider.thumbnailPath != null) {
      return ClipRRect(
        borderRadius: BorderRadius.circular(8),
        child: Container(
          width: 160,
          height: 90,
          decoration: BoxDecoration(
            color: Colors.black26,
            border: Border.all(color: const Color(0x1FFFFFFF)),
          ),
          child: Image.file(
            File(provider.thumbnailPath!),
            fit: BoxFit.cover,
            errorBuilder: (_, _, _) => _buildMusicCoverPlaceholder(),
          ),
        ),
      );
    }

    return _buildMusicCoverPlaceholder();
  }

  Widget _buildMusicCoverPlaceholder() {
    return Container(
      width: 160,
      height: 90,
      decoration: BoxDecoration(
        gradient: const LinearGradient(
          colors: [Color(0xFF2E1A47), Color(0xFF0F071B)],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        borderRadius: BorderRadius.circular(8),
        border: Border.all(color: const Color(0x1F8B5CF6)),
      ),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: const [
            Icon(Icons.music_note_outlined, color: Color(0xFF8B5CF6), size: 28),
            SizedBox(height: 4),
            Text(
              '音频信号',
              style: TextStyle(fontSize: 10, color: Colors.grey, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildLeftColumn(TranscriptionProvider provider) {
    final selectedBase = provider.selectedModelBase;
    final downloadedModels = provider.downloadedModels;
    final showLowPowerWarning = provider.showLowPowerWarning;
    final enableDenoise = provider.enableDenoise;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('识别模型 (Qwen3-ASR)', style:  TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.grey)),
        const SizedBox(height: 8),
        // 模型版本选择
        HoverDropdown<String>(
          value: selectedBase,
          items: ModelService.availableBaseModels.map((m) {
            final isDl = downloadedModels.contains(m.id);
            return DropdownMenuItem<String>(
              value: m.id,
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Text(m.name),
                  const Spacer(),
                  Tooltip(
                    message: isDl ? '模型已就绪' : '点击下载',
                    child: GestureDetector(
                      onTap: () {
                        provider.setCurrentTab(2);
                      },
                      child: Container(
                        alignment: Alignment.center,
                        padding: const EdgeInsets.symmetric(horizontal: 4),
                        child: Icon(
                          isDl ? Icons.check_circle_outline : Icons.download_for_offline_outlined,
                          color: isDl ? Colors.green : Colors.grey,
                          size: 16,
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            );
          }).toList(),
          onChanged: (val) {
            if (val != null) provider.setSelectedModelBase(val);
          },
        ),
        const SizedBox(height: 8),
        // 量化等级: 横向拖动数轴 (左=快速, 右=精确; 未下载档位灰色不可选)
        Builder(builder: (context) {
          final base = ModelService.baseById(selectedBase ?? '');
          if (base == null) return const SizedBox.shrink();
          return _buildQuantSlider(context, provider, base);
        }),
        if (showLowPowerWarning) ...[
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(10),
            decoration: BoxDecoration(
              color: const Color(0x1BFF9800),
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: const Color(0x3BFF9800)),
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Icon(Icons.warning_amber_rounded, color: Color(0xFFFFB74D), size: 18),
                const SizedBox(width: 8),
                Expanded(
                  child: Text(
                    '当前选用较大模型且未开启 GPU 加速，推理耗时可能较长。',
                    style: TextStyle(color: Colors.grey[300], fontSize: 11, height: 1.3),
                  ),
                ),
              ],
            ),
          ),
        ],
        const SizedBox(height: 16),
        Row(
          children: [
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('神经网络降噪预处理', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white)),
                  SizedBox(height: 2),
                  Text('使用 DeepFilterNet3 人声降噪', style: TextStyle(fontSize: 11, color: Colors.grey)),
                ],
              ),
            ),
            Switch(
              value: enableDenoise,
              activeColor: const Color(0xFF8B5CF6),
              onChanged: (val) {
                provider.setEnableDenoise(val);
              },
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildRightColumn(TranscriptionProvider provider) {
    final selectedLanguage = provider.selectedLanguage;
    final translateToEnglish = provider.translateToEnglish;
    final availableTracks = provider.availableTracks;
    final selectedTrack = provider.selectedTrack;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('音频主语言', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.grey)),
        const SizedBox(height: 8),
        HoverDropdown<String>(
          value: selectedLanguage,
          items: const [
            DropdownMenuItem(value: 'auto', child: Text('自动检测语言 (Auto)')),
            DropdownMenuItem(value: 'zh', child: Text('中文 (Chinese)')),
            DropdownMenuItem(value: 'en', child: Text('英文 (English)')),
            DropdownMenuItem(value: 'ja', child: Text('日语 (Japanese)')),
            DropdownMenuItem(value: 'ko', child: Text('韩语 (Korean)')),
          ],
          onChanged: (val) {
            if (val != null) provider.setSelectedLanguage(val);
          },
        ),
        const SizedBox(height: 16),
        Row(
          children: [
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text('英语翻译模式', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.white)),
                  SizedBox(height: 2),
                  Text('自动翻译识别文本为英文', style: TextStyle(fontSize: 11, color: Colors.grey)),
                ],
              ),
            ),
            Switch(
              value: translateToEnglish,
              activeColor: const Color(0xFF8B5CF6),
              onChanged: (val) {
                provider.setTranslate(val);
              },
            ),
          ],
        ),
        if (availableTracks.length > 1) ...[
          const SizedBox(height: 16),
          const Text('选择提取音轨', style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold, color: Colors.grey)),
          const SizedBox(height: 8),
          HoverDropdown<rust_ffmpeg.AudioTrackInfo>(
            value: selectedTrack,
            items: availableTracks.map((track) {
              final lang = track.language ?? '未知';
              final codec = track.codecName;
              final title = track.title != null ? ' - ${track.title}' : '';
              return DropdownMenuItem<rust_ffmpeg.AudioTrackInfo>(
                value: track,
                child: Text('音轨 ${track.index.toInt() + 1}: $lang ($codec)$title', style: const TextStyle(fontSize: 13)),
              );
            }).toList(),
            onChanged: (val) {
              if (val != null) {
                provider.setSelectedTrack(val);
              }
            },
          ),
        ],
      ],
    );
  }

  void _handleStartTranscription(TranscriptionProvider provider) {
    if (_isFlipping || _isFlipped) return;
    setState(() {
      _isFlipping = true;
      _isFlipped = true;
    });
    _flipController.forward().then((_) {
      if (mounted) {
        setState(() {
          _isFlipping = false;
        });
        provider.startTranscription();
      }
    });
  }

  Widget _buildStartButton(TranscriptionProvider provider) {
    final hasFile = provider.inputMediaFile != null;

    return SizedBox(
      width: double.infinity,
      height: 48,
      child: ElevatedButton(
        onPressed: !hasFile ? null : () => _handleStartTranscription(provider),
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFF8B5CF6),
          foregroundColor: Colors.white,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(8),
          ),
          disabledBackgroundColor: const Color(0x1F8B5CF6),
          elevation: 0,
        ),
        child: const Text(
          '开始提取并转写',
          style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
        ),
      ),
    );
  }

  Widget _buildBackSide(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);

    return Container(
      decoration: BoxDecoration(
        color: const Color(0x0CFFFFFF),
        borderRadius: BorderRadius.circular(16),
        border: Border.all(color: const Color(0x1F8B5CF6)),
      ),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // 上方：已选择配置
          _buildBackConfigSummary(provider),
          const SizedBox(height: 16),

          // 中间：实时滚动输出字幕
          Expanded(
            child: Container(
              width: double.infinity,
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: const Color(0x05FFFFFF),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(color: const Color(0x0FFFFFFF)),
              ),
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTap: () {
                  provider.setCurrentTab(1); // 跳转字幕编辑页面
                },
                child: MouseRegion(
                  cursor: SystemMouseCursors.click,
                  // child: Tooltip(
                  //   // message: '点击跳转字幕编辑器',
                  child: ScrollingOutputList(subtitles: provider.subtitles),
                  // ),
                ),
              ),
            ),
          ),
          const SizedBox(height: 16),

          // 下方：实时控制和中断区域
          _buildBackControlBar(provider),
        ],
      ),
    );
  }

  /// 量化等级横向数轴: 左=快速(低精度低体积), 右=精确(全精度)。
  /// 未下载档位灰色显示且不可选中 (拖动/点击被忽略并提示)。
  Widget _buildQuantSlider(BuildContext context, TranscriptionProvider provider, QwenBaseModel base) {
    // 数轴顺序: 快速 -> 精确 (reversed: q4_k_m/q6_k 在左, f16 在右)
    final quants = base.quants.reversed.toList();
    final currentQuant = provider.selectedQuant;
    final currentIdx = quants.indexWhere((q) => q.id == currentQuant);
    final selIdx = currentIdx < 0 ? 0 : currentIdx;

    // thumb 位置: 当前选中档若未下载, 吸附到最近的已下载档
    int thumbIdx = selIdx;
    if (!provider.isQuantReady(base.id, quants[selIdx].id)) {
      final downloaded = <int>[];
      for (int i = 0; i < quants.length; i++) {
        if (provider.isQuantReady(base.id, quants[i].id)) downloaded.add(i);
      }
      if (downloaded.isNotEmpty) {
        thumbIdx = downloaded.reduce((a, b) =>
            (a - selIdx).abs() <= (b - selIdx).abs() ? a : b);
      }
    }

    return LayoutBuilder(builder: (context, constraints) {
      final w = constraints.maxWidth;
      final n = quants.length;
      // 数轴两端内缩 (与标签半宽一致), 让轨道端点 = 首尾档位 = 快速/精确 标记三者绝对对齐
      const labelWidth = 88.0;
      final axisPad = labelWidth / 2; // 44.0
      final trackW = (w - 2 * axisPad).clamp(0.0, double.infinity);
      final slot = n > 1 ? trackW / (n - 1) : 0.0;

      return Semantics(
        container: true,
        label: '模型量化数轴选择器',
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisSize: MainAxisSize.min,
          children: [
            // 1. 首尾两端标记 ("快速" & "精确"): 增强 line-height 与 Chip 美化，与轨道首尾端点 (X=axisPad, X=w-axisPad) 精确居中对齐
            SizedBox(
              height: 26,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  // 左侧 "快速" 标签 (中心点居中在 axisPad)
                  Positioned(
                    left: axisPad - labelWidth / 2,
                    width: labelWidth,
                    child: Container(
                      alignment: Alignment.center,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0x1510B981),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: const Color(0x3010B981)),
                        ),
                        child: const Text(
                          '⚡ 快速',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFF34D399),
                            height: 1.4,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                    ),
                  ),
                  // 右侧 "精确" 标签 (中心点居中在 w - axisPad)
                  Positioned(
                    left: (w - axisPad) - labelWidth / 2,
                    width: labelWidth,
                    child: Container(
                      alignment: Alignment.center,
                      child: Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0x158B5CF6),
                          borderRadius: BorderRadius.circular(6),
                          border: Border.all(color: const Color(0x308B5CF6)),
                        ),
                        child: const Text(
                          '🎯 精确',
                          textAlign: TextAlign.center,
                          style: TextStyle(
                            fontSize: 11,
                            fontWeight: FontWeight.w600,
                            color: Color(0xFFA78BFA),
                            height: 1.4,
                            letterSpacing: 0.5,
                          ),
                        ),
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 8),

            // 2. 数轴 Slider: 使用 _CustomQuantSliderTrackShape 确保轨道左端点 = 0th档位点 = axisPad, 右端点 = (n-1)th档位点 = w - axisPad
            Padding(
              padding: EdgeInsets.symmetric(horizontal: axisPad),
              child: SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  trackShape: const _CustomQuantSliderTrackShape(),
                  trackHeight: 4.0,
                  thumbShape: const RoundSliderThumbShape(enabledThumbRadius: 7.0),
                  overlayShape: const RoundSliderOverlayShape(overlayRadius: 14.0),
                  activeTrackColor: const Color(0xFF8B5CF6),
                  inactiveTrackColor: const Color(0x33FFFFFF),
                  thumbColor: const Color(0xFFA78BFA),
                  overlayColor: const Color(0x228B5CF6),
                ),
                child: Slider(
                  value: thumbIdx.toDouble(),
                  min: 0,
                  max: (n - 1).toDouble(),
                  divisions: n - 1,
                  onChanged: (v) {
                    final idx = v.round();
                    final q = quants[idx];
                    if (provider.isQuantReady(base.id, q.id)) {
                      provider.setSelectedQuant(q.id);
                    } else {
                      ScaffoldMessenger.of(context).showSnackBar(
                        SnackBar(
                          content: Text('${q.label} 尚未下载，请前往模型管理器下载'),
                          duration: const Duration(seconds: 2),
                        ),
                      );
                    }
                  },
                ),
              ),
            ),
            const SizedBox(height: 8),

            // 3. 档位卡片标签: 每档中心 X = axisPad + i * slot, 与 Slider 档位点及首尾标记精确对齐
            SizedBox(
              height: 52,
              child: Stack(
                clipBehavior: Clip.none,
                children: [
                  for (int i = 0; i < n; i++)
                    Positioned(
                      left: axisPad + i * slot - labelWidth / 2,
                      width: labelWidth,
                      child: GestureDetector(
                        onTap: provider.isQuantReady(base.id, quants[i].id)
                            ? () => provider.setSelectedQuant(quants[i].id)
                            : () {
                                ScaffoldMessenger.of(context).showSnackBar(
                                  SnackBar(
                                    content: Text('${quants[i].label} 尚未下载，请前往模型管理器下载'),
                                    duration: const Duration(seconds: 2),
                                  ),
                                );
                              },
                        child: Column(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            Icon(
                              Icons.circle,
                              size: i == selIdx ? 10 : 8,
                              color: provider.isQuantReady(base.id, quants[i].id)
                                  ? (i == selIdx
                                      ? const Color(0xFF8B5CF6)
                                      : Colors.grey.shade300)
                                  : Colors.grey.shade700,
                            ),
                            const SizedBox(height: 4),
                            Text(
                              quants[i].label,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 11,
                                height: 1.4,
                                color: provider.isQuantReady(base.id, quants[i].id)
                                    ? (i == selIdx ? const Color(0xFFA78BFA) : Colors.white)
                                    : Colors.grey.shade600,
                                fontWeight: i == selIdx ? FontWeight.bold : FontWeight.normal,
                              ),
                            ),
                            Text(
                              quants[i].sizeText,
                              textAlign: TextAlign.center,
                              style: TextStyle(
                                fontSize: 9,
                                height: 1.3,
                                color: Colors.grey.shade500,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                ],
              ),
            ),
          ],
        ),
      );
    });
  }

  Widget _buildBackConfigSummary(TranscriptionProvider provider) {
    final base = ModelService.baseById(provider.selectedModelBase ?? '');
    final quant = base?.quantById(provider.selectedQuant);
    final modelName = base == null
        ? '未知模型'
        : quant == null
            ? base.name
            : '${base.name} · ${quant.label}';

    final langLabel = {
      'auto': '自动检测',
      'zh': '中文',
      'en': '英文',
      'ja': '日语',
      'ko': '韩语',
    }[provider.selectedLanguage] ?? '其他';

    final device = provider.activeDeviceName;
    final denoiseStr = provider.enableDenoise ? '神经网络降噪' : '无降噪';

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0x05FFFFFF),
        borderRadius: BorderRadius.circular(10),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    const Icon(Icons.tune, size: 14, color: Colors.grey),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        '配置: $modelName | $langLabel | $denoiseStr',
                        style: const TextStyle(fontSize: 12, color: Colors.grey),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Row(
                  children: [
                    const Icon(Icons.memory_outlined, size: 14, color: Colors.grey),
                    const SizedBox(width: 6),
                    Expanded(
                      child: Text(
                        '执行硬件: $device',
                        style: const TextStyle(fontSize: 12, color: Colors.grey),
                        maxLines: 1,
                        overflow: TextOverflow.ellipsis,
                      ),
                    ),
                  ],
                ),
              ],
            ),
          ),
          const SizedBox(width: 16),
          // 预期剩余时间 / 状态徽章
          _buildEtaBadge(provider),
        ],
      ),
    );
  }

  Widget _buildEtaBadge(TranscriptionProvider provider) {
    final status = provider.status;
    String label = '--';
    Color color = const Color(0xFF8B5CF6);

    if (status == TranscriptionStatus.extractingAudio) {
      label = '音频提取中...';
      color = const Color(0xFF06B6D4);
    } else if (status == TranscriptionStatus.transcribing) {
      label = '预计剩余: ${provider.etaStr}';
    } else if (status == TranscriptionStatus.completed) {
      label = '转写完成';
      color = Colors.green;
    } else if (status == TranscriptionStatus.failed) {
      label = '转写失败';
      color = Colors.redAccent;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.15),
        border: Border.all(color: color.withOpacity(0.3)),
        borderRadius: BorderRadius.circular(6),
      ),
      child: Text(
        label,
        style: TextStyle(
          fontSize: 12,
          fontWeight: FontWeight.bold,
          color: color,
        ),
      ),
    );
  }

  Widget _buildBackControlBar(TranscriptionProvider provider) {
    final status = provider.status;
    final isLoading = status == TranscriptionStatus.extractingAudio ||
        status == TranscriptionStatus.transcribing;
    final isDone = status == TranscriptionStatus.completed;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: const Color(0x1F8B5CF6),
        border: Border.all(color: const Color(0x3F8B5CF6)),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          // 控制和状态输出
          Expanded(
            child: isLoading
                ? ValueListenableBuilder<int>(
                    valueListenable: provider.progressNotifier,
                    builder: (context, progress, child) {
                      return Row(
                        children: [
                          Expanded(
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              children: [
                                ClipRRect(
                                  borderRadius: BorderRadius.circular(3),
                                  child: LinearProgressIndicator(
                                    value: status == TranscriptionStatus.transcribing
                                        ? progress / 100
                                        : null,
                                    backgroundColor: const Color(0x1FFFFFFF),
                                    color: const Color(0xFF8B5CF6),
                                    minHeight: 6,
                                  ),
                                ),
                                const SizedBox(height: 6),
                                Text(
                                  status == TranscriptionStatus.transcribing
                                      ? '神经网络推理中... $progress%'
                                      : 'FFmpeg 提取重采样中...',
                                  style: const TextStyle(fontSize: 12, color: Colors.white70),
                                ),
                              ],
                            ),
                          ),
                        ],
                      );
                    },
                  )
                : isDone
                    ? Row(
                        children: const [
                          Icon(Icons.check_circle_outline, color: Colors.green, size: 20),
                          SizedBox(width: 8),
                          Text('转写全部就绪！', style: TextStyle(color: Colors.green, fontWeight: FontWeight.bold)),
                        ],
                      )
                    : Row(
                        children: [
                          const Icon(Icons.error_outline, color: Colors.redAccent, size: 20),
                          const SizedBox(width: 8),
                          Expanded(
                            child: Text(
                              provider.statusMessage,
                              style: const TextStyle(color: Colors.redAccent, fontSize: 12),
                              maxLines: 2,
                              overflow: TextOverflow.ellipsis,
                            ),
                          ),
                        ],
                      ),
          ),
          const SizedBox(width: 16),
          // 右侧动作按钮
          if (isLoading)
            _buildInterruptButton(provider)
          else
            Row(
              children: [
                if (isDone)
                  TextButton.icon(
                    onPressed: () {
                      provider.setCurrentTab(1); // 转跳到编辑器
                    },
                    icon: const Icon(Icons.edit_note, color: Colors.white),
                    label: const Text('去编辑器', style: TextStyle(color: Colors.white)),
                    style: TextButton.styleFrom(
                      backgroundColor: const Color(0xFF8B5CF6),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    ),
                  ),
                const SizedBox(width: 8),
                TextButton(
                  onPressed: () {
                    provider.cancelTranscription(); // 实际操作是重置任务为 idle，触发翻转回 Front
                  },
                  child: const Text('返回配置', style: TextStyle(color: Colors.white70)),
                ),
              ],
            ),
        ],
      ),
    );
  }

  Widget _buildInterruptButton(TranscriptionProvider provider) {
    if (_localShowInterruptConfirm) {
      return Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          IconButton(
            tooltip: '确认中止',
            icon: const Icon(Icons.check, color: Colors.green, size: 22),
            onPressed: () {
              provider.cancelTranscription();
              setState(() {
                _localShowInterruptConfirm = false;
              });
            },
          ),
          IconButton(
            tooltip: '继续运行',
            icon: const Icon(Icons.close, color: Colors.redAccent, size: 22),
            onPressed: () {
              setState(() {
                _localShowInterruptConfirm = false;
              });
            },
          ),
        ],
      );
    }

    return Tooltip(
      message: '打断识别并强行退出',
      child: IconButton(
        icon: const Icon(
          Icons.stop_circle_outlined,
          color: Color(0xFFEF4444),
          size: 24,
        ),
        onPressed: () {
          setState(() {
            _localShowInterruptConfirm = true;
          });
        },
      ),
    );
  }
}

// ==== 3D Card Flip Animation Helper ====
class FlipCard extends StatelessWidget {
  final Animation<double> animation;
  final Widget front;
  final Widget back;

  const FlipCard({
    super.key,
    required this.animation,
    required this.front,
    required this.back,
  });

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: animation,
      builder: (context, child) {
        final angle = animation.value * 3.1415926535;
        final isFront = angle < 3.1415926535 / 2;

        return Transform(
          transform: Matrix4.identity()
            ..setEntry(3, 2, 0.001) // perspective projection
            ..rotateY(angle),
          alignment: Alignment.center,
          child: isFront
              ? front
              : Transform(
                  transform: Matrix4.identity()..rotateY(3.1415926535),
                  alignment: Alignment.center,
                  child: back,
                ),
        );
      },
    );
  }
}

// ==== Scrolling Output List Helper ====
class ScrollingOutputList extends StatefulWidget {
  final List<SubtitleItem> subtitles;
  const ScrollingOutputList({super.key, required this.subtitles});

  @override
  State<ScrollingOutputList> createState() => _ScrollingOutputListState();
}

class _ScrollingOutputListState extends State<ScrollingOutputList> {
  final ScrollController _scrollController = ScrollController();

  @override
  void didUpdateWidget(ScrollingOutputList oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (widget.subtitles.length != oldWidget.subtitles.length) {
      WidgetsBinding.instance.addPostFrameCallback((_) {
        if (_scrollController.hasClients) {
          _scrollController.animateTo(
            _scrollController.position.maxScrollExtent,
            duration: const Duration(milliseconds: 300),
            curve: Curves.easeOut,
          );
        }
      });
    }
  }

  @override
  void dispose() {
    _scrollController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (widget.subtitles.isEmpty) {
      return const Center(
        child: Text(
          '等待流式语音转写输入...',
          style: TextStyle(color: Colors.grey, fontStyle: FontStyle.italic),
        ),
      );
    }

    return ListView.builder(
      controller: _scrollController,
      itemCount: widget.subtitles.length,
      itemBuilder: (context, index) {
        final item = widget.subtitles[index];
        final startStr = TranscriptionProvider.formatSrtTimestamp(item.startMs).substring(3, 11);
        return Padding(
          padding: const EdgeInsets.symmetric(vertical: 4.0),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                '[$startStr]',
                style: const TextStyle(
                  fontFamily: 'monospace',
                  color: Color(0xFF06B6D4),
                  fontSize: 12,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  item.text,
                  style: const TextStyle(color: Colors.white70, fontSize: 13),
                ),
              ),
            ],
          ),
        );
      },
    );
  }
}

/// 自定义数轴轨道 Shape: 强制轨道从 Slider Widget 的左边缘 (0.0) 精确绘制到右边缘 (width)，
/// 从而使首尾 Slider 档位点、数轴轨道端点与 "快速/精确" 标记完全无死角居中对齐。
class _CustomQuantSliderTrackShape extends RectangularSliderTrackShape {
  const _CustomQuantSliderTrackShape();

  @override
  Rect getPreferredRect({
    required RenderBox parentBox,
    Offset offset = Offset.zero,
    required SliderThemeData sliderTheme,
    bool isEnabled = false,
    bool isDiscrete = false,
  }) {
    final double trackHeight = sliderTheme.trackHeight ?? 4.0;
    final double trackLeft = offset.dx;
    final double trackWidth = parentBox.size.width;
    final double trackTop = offset.dy + (parentBox.size.height - trackHeight) / 2;
    return Rect.fromLTWH(trackLeft, trackTop, trackWidth, trackHeight);
  }
}
