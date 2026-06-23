import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../providers/transcription_provider.dart';

class SidebarItem {
  final IconData icon;
  final String label;

  SidebarItem({required this.icon, required this.label});
}

class Sidebar extends StatelessWidget {
  final int selectedIndex;
  final Function(int) onSelected;
  final List<SidebarItem> items;

  const Sidebar({
    super.key,
    required this.selectedIndex,
    required this.onSelected,
    required this.items,
  });

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);
    final isExpanded = provider.isSidebarExpanded;

    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      width: isExpanded ? 240 : 72,
      decoration: const BoxDecoration(
        color: Color(0xFF131324),
        border: Border(right: BorderSide(color: Color(0x1FFFFFFF), width: 1)),
      ),
      child: ClipRect(
        child: Column(
          children: [
            _buildLogo(isExpanded),
            const SizedBox(height: 32),
            Expanded(
              child: ListView.builder(
                itemCount: items.length,
                itemBuilder: (context, index) {
                  final item = items[index];
                  final isActive = index == selectedIndex;

                  return Padding(
                    padding: EdgeInsets.symmetric(
                      horizontal: isExpanded ? 16.0 : 8.0,
                      vertical: 4.0,
                    ),
                    child: InkWell(
                      onTap: () => onSelected(index),
                      borderRadius: BorderRadius.circular(10),
                      child: Container(
                        height: 48,
                        padding: EdgeInsets.symmetric(
                          horizontal: isExpanded ? 16 : 0,
                          vertical: 12,
                        ),
                        decoration: BoxDecoration(
                          color: isActive
                              ? const Color(0x1F8B5CF6)
                              : Colors.transparent,
                          borderRadius: BorderRadius.circular(10),
                          border: Border.all(
                            color: isActive
                                ? const Color(0x408B5CF6)
                                : Colors.transparent,
                          ),
                        ),
                        child: isExpanded
                            ? OverflowBox(
                                minWidth: 176,
                                maxWidth: 176,
                                minHeight: 24,
                                maxHeight: 24,
                                alignment: Alignment.centerLeft,
                                child: Row(
                                  children: [
                                    Icon(
                                      item.icon,
                                      color: isActive
                                          ? const Color(0xFF8B5CF6)
                                          : Colors.grey[400],
                                      size: 20,
                                    ),
                                    const SizedBox(width: 16),
                                    Expanded(
                                      child: Text(
                                        item.label,
                                        maxLines: 1,
                                        overflow: TextOverflow.clip,
                                        style: TextStyle(
                                          fontSize: 14,
                                          fontWeight: isActive
                                              ? FontWeight.bold
                                              : FontWeight.w500,
                                          color: isActive
                                              ? Colors.white
                                              : Colors.grey[400],
                                        ),
                                      ),
                                    ),
                                  ],
                                ),
                              )
                            : Center(
                                child: Icon(
                                  item.icon,
                                  color: isActive
                                      ? const Color(0xFF8B5CF6)
                                      : Colors.grey[400],
                                  size: 20,
                                ),
                              ),
                      ),
                    ),
                  );
                },
              ),
            ),
            _buildToggleBtn(provider),
            _buildFooter(isExpanded),
          ],
        ),
      ),
    );
  }

  Widget _buildToggleBtn(TranscriptionProvider provider) {
    final isExpanded = provider.isSidebarExpanded;
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: IconButton(
        tooltip: isExpanded ? '收起边栏' : '展开边栏',
        icon: Icon(
          isExpanded ? Icons.chevron_left : Icons.chevron_right,
          color: Colors.grey[400],
        ),
        onPressed: () {
          provider.setSidebarExpanded(!isExpanded);
        },
      ),
    );
  }

  Widget _buildLogo(bool isExpanded) {
    if (!isExpanded) {
      return Container(
        height: 88,
        padding: const EdgeInsets.only(top: 40),
        child: Center(
          child: Image.asset('assets/app_icon.png', width: 48, height: 48),
        ),
      );
    }

    return Container(
      height: 88,
      padding: const EdgeInsets.only(top: 40, left: 24, right: 24),
      child: OverflowBox(
        minWidth: 192,
        maxWidth: 192,
        minHeight: 48,
        maxHeight: 48,
        alignment: Alignment.centerLeft,
        child: Row(
          children: [
            Image.asset('assets/app_icon.png', width: 48, height: 48),
            const SizedBox(width: 16),
            const Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisSize: MainAxisSize.min,
                children: [
                  Text(
                    'Audio2Srt',
                    maxLines: 1,
                    overflow: TextOverflow.clip,
                    style: TextStyle(
                      fontSize: 20,
                      fontWeight: FontWeight.w900,
                      letterSpacing: 0.5,
                    ),
                  ),
                  Text(
                    '本地智能字幕生成',
                    maxLines: 1,
                    overflow: TextOverflow.clip,
                    style: TextStyle(fontSize: 10, color: Colors.grey),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFooter(bool isExpanded) {
    if (!isExpanded) {
      return Container(
        height: 72,
        padding: const EdgeInsets.only(bottom: 24),
        child: const Icon(Icons.security, size: 14, color: Colors.grey),
      );
    }

    return Container(
      height: 72,
      padding: const EdgeInsets.all(24),
      child: OverflowBox(
        minWidth: 192,
        maxWidth: 192,
        minHeight: 24,
        maxHeight: 24,
        alignment: Alignment.centerLeft,
        child: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.security, size: 14, color: Colors.grey),
            const SizedBox(width: 8),
            Expanded(
              child: Text(
                '100% 本地离线处理',
                maxLines: 1,
                overflow: TextOverflow.clip,
                style: TextStyle(fontSize: 11, color: Colors.grey[500]),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
