import 'package:flutter/material.dart';

/// 优雅精美的自定义 Hover 下拉选择器组件。
///
/// 完全解耦 Flutter 原生 DropdownButton 硬编码 _kMenuHorizontalPadding 的缺陷，
/// 1. 闭合按钮与下拉选项统一高度 (38px)，符合桌面端精致紧凑审美；
/// 2. 展开位置精确覆盖/置顶于按钮 (Offset 0, 0)，维持原生 Dropdown 定位体验；
/// 3. 展开菜单与按钮宽度 100% 精确对齐等宽，右侧 0 像素凸出；
/// 4. 悬浮时整个圆角容器 (含左右 padding) 100% 均匀高亮；
/// 5. 自动管理 Overlay 与外部点击收起。
class HoverDropdown<T> extends StatefulWidget {
  final T? value;
  final List<DropdownMenuItem<T>> items;
  final ValueChanged<T?> onChanged;
  final List<Widget> Function(BuildContext)? selectedItemBuilder;
  final Color? dropdownColor;
  final EdgeInsets padding;
  final double borderRadius;
  final double height;

  const HoverDropdown({
    super.key,
    required this.value,
    required this.items,
    required this.onChanged,
    this.selectedItemBuilder,
    this.dropdownColor,
    this.padding = const EdgeInsets.symmetric(horizontal: 12),
    this.borderRadius = 8,
    this.height = 50,
  });

  @override
  State<HoverDropdown<T>> createState() => _HoverDropdownState<T>();
}

class _HoverDropdownState<T> extends State<HoverDropdown<T>> {
  bool _hover = false;
  bool _isOpen = false;
  OverlayEntry? _overlayEntry;
  final LayerLink _layerLink = LayerLink();

  void _toggleDropdown() {
    if (_isOpen) {
      _closeDropdown();
    } else {
      _openDropdown();
    }
  }

  void _openDropdown() {
    final renderBox = context.findRenderObject() as RenderBox?;
    if (renderBox == null) return;
    final size = renderBox.size;

    _overlayEntry = OverlayEntry(
      builder: (context) => Stack(
        children: [
          // 点击非菜单区域自动收起
          GestureDetector(
            onTap: _closeDropdown,
            behavior: HitTestBehavior.translucent,
            child: const SizedBox.expand(),
          ),
          // 展开菜单浮层：位置 Offset(0, 0) 精确覆盖按钮，宽度 100% 与按钮等宽！
          Positioned(
            width: size.width,
            child: CompositedTransformFollower(
              link: _layerLink,
              showWhenUnlinked: false,
              offset: const Offset(0, 0), // 👈 对齐按钮顶边 (0, 0)
              child: Material(
                color: Colors.transparent,
                child: Container(
                  constraints: const BoxConstraints(maxHeight: 260),
                  decoration: BoxDecoration(
                    color: widget.dropdownColor ?? const Color(0xFF1E1E2C),
                    borderRadius: BorderRadius.circular(widget.borderRadius),
                    border: Border.all(color: const Color(0x358B5CF6)),
                    boxShadow: const [
                      BoxShadow(
                        color: Color(0x66000000),
                        blurRadius: 16,
                        offset: Offset(0, 4),
                      ),
                    ],
                  ),
                  child: ClipRRect(
                    borderRadius: BorderRadius.circular(widget.borderRadius),
                    child: SingleChildScrollView(
                      padding: const EdgeInsets.symmetric(vertical: 2),
                      child: Column(
                        mainAxisSize: MainAxisSize.min,
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: widget.items.map((item) {
                          final isSelected = item.value == widget.value;
                          return _DropdownItemTile<T>(
                            item: item,
                            isSelected: isSelected,
                            height: widget.height,
                            onTap: () {
                              _closeDropdown();
                              if (item.value != widget.value) {
                                widget.onChanged(item.value);
                              }
                            },
                          );
                        }).toList(),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );

    Overlay.of(context).insert(_overlayEntry!);
    if (mounted) {
      setState(() => _isOpen = true);
    }
  }

  void _closeDropdown() {
    _overlayEntry?.remove();
    _overlayEntry = null;
    if (mounted) {
      setState(() => _isOpen = false);
    }
  }

  @override
  void dispose() {
    _overlayEntry?.remove();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    // 寻找选中的 Label 视图
    Widget selectedWidget;
    if (widget.selectedItemBuilder != null) {
      final selectedList = widget.selectedItemBuilder!(context);
      final selectedIdx = widget.items.indexWhere((it) => it.value == widget.value);
      if (selectedIdx >= 0 && selectedIdx < selectedList.length) {
        selectedWidget = selectedList[selectedIdx];
      } else {
        selectedWidget = const SizedBox();
      }
    } else {
      final match = widget.items.firstWhere(
        (it) => it.value == widget.value,
        orElse: () => widget.items.isNotEmpty
            ? widget.items.first
            : DropdownMenuItem<T>(value: null, child: const SizedBox()),
      );
      selectedWidget = match.child;
    }

    return Semantics(
      container: true,
      child: CompositedTransformTarget(
        link: _layerLink,
        child: MouseRegion(
          cursor: SystemMouseCursors.click,
          onEnter: (_) => setState(() => _hover = true),
          onExit: (_) => setState(() => _hover = false),
          child: GestureDetector(
            onTap: _toggleDropdown,
            behavior: HitTestBehavior.opaque,
            child: AnimatedContainer(
              duration: const Duration(milliseconds: 120),
              padding: widget.padding,
              height: widget.height,
              alignment: Alignment.centerLeft,
              decoration: BoxDecoration(
                color: (_hover || _isOpen) ? const Color(0x208B5CF6) : const Color(0x05FFFFFF),
                borderRadius: BorderRadius.circular(widget.borderRadius),
                border: Border.all(
                  color: (_hover || _isOpen) ? const Color(0x358B5CF6) : const Color(0x0FFFFFFF),
                ),
              ),
              child: Row(
                crossAxisAlignment: CrossAxisAlignment.center,
                children: [
                  Expanded(
                    child: Align(
                      alignment: Alignment.centerLeft,
                      child: DefaultTextStyle.merge(
                        style: const TextStyle(fontSize: 13, color: Colors.white),
                        overflow: TextOverflow.ellipsis,
                        child: selectedWidget,
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  Icon(
                    _isOpen ? Icons.keyboard_arrow_up_rounded : Icons.keyboard_arrow_down_rounded,
                    size: 18,
                    color: (_hover || _isOpen) ? const Color(0xFFA78BFA) : Colors.white70,
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _DropdownItemTile<T> extends StatefulWidget {
  final DropdownMenuItem<T> item;
  final bool isSelected;
  final double height;
  final VoidCallback onTap;

  const _DropdownItemTile({
    required this.item,
    required this.isSelected,
    required this.height,
    required this.onTap,
  });

  @override
  State<_DropdownItemTile<T>> createState() => _DropdownItemTileState<T>();
}

class _DropdownItemTileState<T> extends State<_DropdownItemTile<T>> {
  bool _hover = false;

  @override
  Widget build(BuildContext context) {
    return MouseRegion(
      cursor: SystemMouseCursors.click,
      onEnter: (_) => setState(() => _hover = true),
      onExit: (_) => setState(() => _hover = false),
      child: GestureDetector(
        onTap: widget.onTap,
        behavior: HitTestBehavior.opaque,
        child: Container(
          height: widget.height,
          padding: const EdgeInsets.symmetric(horizontal: 12),
          alignment: Alignment.centerLeft,
          decoration: BoxDecoration(
            color: widget.isSelected
                ? const Color(0x308B5CF6)
                : (_hover ? const Color(0x208B5CF6) : Colors.transparent),
          ),
          child: DefaultTextStyle.merge(
            style: TextStyle(
              fontSize: 13,
              color: widget.isSelected ? const Color(0xFFA78BFA) : Colors.white,
              fontWeight: widget.isSelected ? FontWeight.bold : FontWeight.normal,
            ),
            child: widget.item.child,
          ),
        ),
      ),
    );
  }
}
