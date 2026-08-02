import 'dart:io';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:provider/provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';
import 'package:window_manager/window_manager.dart';
import 'package:local_notifier/local_notifier.dart';
import 'package:tray_manager/tray_manager.dart';

import 'src/rust/frb_generated.dart';
import 'providers/transcription_provider.dart';
import 'ui/sidebar.dart';
import 'ui/dashboard_view.dart';
import 'ui/editor_view.dart';
import 'ui/models_view.dart';
import 'ui/settings_view.dart';

Future<void> main() async {
  // 必须初始化 Rust 绑定库
  WidgetsFlutterBinding.ensureInitialized();
  
  // 初始化 window_manager 实现顶部沉浸
  try {
    await windowManager.ensureInitialized();
    WindowOptions windowOptions = const WindowOptions(
      size: Size(1200, 800),
      minimumSize: Size(950, 650),
      center: true,
      backgroundColor: Colors.transparent,
      skipTaskbar: false,
      titleBarStyle: TitleBarStyle.hidden,
    );
    windowManager.waitUntilReadyToShow(windowOptions, () async {
      await windowManager.show();
      await windowManager.focus();
      await windowManager.setPreventClose(true);
    });
  } catch (e) {
    debugPrint('window_manager initialization failed: $e');
  }

  // 初始化 local_notifier 本地通知
  try {
    await localNotifier.setup(
      appName: 'Audio2Srt',
      shortcutPolicy: ShortcutPolicy.requireCreate,
    );
  } catch (e) {
    debugPrint('local_notifier initialization failed: $e');
  }
  
  bool isRustInitialized = false;
  String initError = '';
  String initStackTrace = '';
  
  try {
    await RustLib.init();
    isRustInitialized = true;
  } catch (e, stackTrace) {
    debugPrint('Rust initialization failed: $e\n$stackTrace');
    initError = e.toString();
    initStackTrace = stackTrace.toString();
  }

  if (isRustInitialized) {
    runApp(const Audio2SrtApp());
  } else {
    try {
      await windowManager.setTitleBarStyle(TitleBarStyle.normal);
      await windowManager.setPreventClose(false);
    } catch (e) {
      debugPrint('Resetting windowManager failed: $e');
    }
    runApp(InitializationErrorApp(error: initError, stackTrace: initStackTrace));
  }
}

class Audio2SrtApp extends StatelessWidget {
  const Audio2SrtApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => TranscriptionProvider()..init(),
      child: MaterialApp(
        title: 'Audio2Srt - 本地智能字幕生成',
        debugShowCheckedModeBanner: false,
        builder: (context, child) {
          return ExcludeSemantics(
            child: child ?? const SizedBox(),
          );
        },
        locale: const Locale('zh', 'CN'),
        supportedLocales: const [
          Locale('zh', 'CN'),
          Locale('en', 'US'),
        ],
        localizationsDelegates: const [
          GlobalMaterialLocalizations.delegate,
          GlobalWidgetsLocalizations.delegate,
          GlobalCupertinoLocalizations.delegate,
        ],
        theme: ThemeData(
          brightness: Brightness.dark,
          scaffoldBackgroundColor: const Color(0xFF07070F),
          colorScheme: const ColorScheme.dark(
            primary: Color(0xFF8B5CF6),
            secondary: Color(0xFF06B6D4),
            surface: Color(0xFF131324),
          ),
          fontFamily: 'Segoe UI',
          textTheme: ThemeData.dark().textTheme.apply(
            fontFamily: 'Segoe UI',
            fontFamilyFallback: const [
              'Microsoft YaHei',
              'PingFang SC',
              'Heiti SC',
              'sans-serif',
            ],
          ),
          appBarTheme: const AppBarTheme(
            backgroundColor: Color.fromARGB(255, 20, 20, 44),
            elevation: 0,
          ),
        ),
        home: const MainShell(),
      ),
    );
  }
}

class MainShell extends StatefulWidget {
  const MainShell({super.key});

  @override
  State<MainShell> createState() => _MainShellState();
}

class _MainShellState extends State<MainShell> with WindowListener, TrayListener {
  List<SidebarItem> get sidebarItems => [
    SidebarItem(icon: Icons.dashboard_outlined, label: '首页'),
    SidebarItem(icon: Icons.edit_note_outlined, label: '字幕编辑器'),
    SidebarItem(icon: Icons.layers_outlined, label: '模型管理'),
    SidebarItem(icon: Icons.settings_outlined, label: '系统设置'),
  ];

  bool _isTrayInitialized = false;

  @override
  void initState() {
    super.initState();
    windowManager.addListener(this);
    trayManager.addListener(this);
    
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        final provider = Provider.of<TranscriptionProvider>(context, listen: false);
        provider.addListener(_onProviderChanged);
        _onProviderChanged(); // initial check
      }
    });
  }

  @override
  void dispose() {
    windowManager.removeListener(this);
    trayManager.removeListener(this);
    try {
      final provider = Provider.of<TranscriptionProvider>(context, listen: false);
      provider.removeListener(_onProviderChanged);
    } catch (_) {}
    _destroyTray();
    super.dispose();
  }

  void _onProviderChanged() {
    if (!mounted) return;
    final provider = Provider.of<TranscriptionProvider>(context, listen: false);
    if (provider.needsCloseConfirmation) {
      _initTray();
    } else {
      _destroyTray();
    }
  }

  Future<void> _initTray() async {
    if (_isTrayInitialized) return;
    try {
      await trayManager.setIcon(
        Platform.isWindows ? 'assets/app_icon.ico' : 'assets/app_icon.png',
      );
      await trayManager.setToolTip('Audio2Srt');
      
      final menu = Menu(
        items: [
          MenuItem(key: 'show_window', label: '显示主窗口'),
          MenuItem.separator(),
          MenuItem(key: 'exit_app', label: '退出程序'),
        ],
      );
      await trayManager.setContextMenu(menu);
      _isTrayInitialized = true;
    } catch (e) {
      debugPrint('Failed to initialize tray: $e');
    }
  }

  Future<void> _destroyTray() async {
    if (!_isTrayInitialized) return;
    try {
      await trayManager.destroy();
      _isTrayInitialized = false;
    } catch (e) {
      debugPrint('Failed to destroy tray: $e');
    }
  }

  @override
  void onWindowClose() async {
    final provider = Provider.of<TranscriptionProvider>(context, listen: false);
    if (provider.needsCloseConfirmation) {
      _showCloseConfirmationDialog();
    } else {
      await windowManager.hide();
      await _destroyTray();
      await windowManager.destroy();
    }
  }

  // Tray listener overrides
  @override
  void onTrayIconMouseDown() async {
    await windowManager.show();
    await windowManager.focus();
  }

  @override
  void onTrayIconMouseUp() {}

  @override
  void onTrayIconRightMouseDown() {
    trayManager.popUpContextMenu();
  }

  @override
  void onTrayIconRightMouseUp() {}

  @override
  void onTrayMenuItemClick(MenuItem menuItem) async {
    if (menuItem.key == 'show_window') {
      await windowManager.show();
      await windowManager.focus();
    } else if (menuItem.key == 'exit_app') {
      await windowManager.hide();
      await _destroyTray();
      await windowManager.destroy();
    }
  }

  void _showCloseConfirmationDialog() {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) {
        return AlertDialog(
          backgroundColor: const Color(0xFF131324), // matching the dark surface color
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(16),
            side: const BorderSide(color: Color(0x1FFFFFFF), width: 1),
          ),
          title: const Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Color(0xFFEF4444)),
              SizedBox(width: 8),
              Text('确认关闭？', style: TextStyle(color: Colors.white, fontSize: 18, fontWeight: FontWeight.bold)),
            ],
          ),
          content: const Text(
            '当前任务正在运行中，或者生成的字幕尚未导出。\n直接关闭窗口可能会丢失所有未保存的内容。',
            style: TextStyle(color: Colors.white70, fontSize: 14),
          ),
          actionsPadding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
          actions: [
            // 取消
            TextButton(
              onPressed: () => Navigator.of(dialogContext).pop(),
              child: const Text('取消', style: TextStyle(color: Colors.grey)),
            ),
            
            // 直接关闭
            TextButton(
              onPressed: () async {
                Navigator.of(dialogContext).pop();
                await windowManager.hide();
                await _destroyTray();
                await windowManager.destroy();
              },
              child: const Text('直接关闭', style: TextStyle(color: Color(0xFFEF4444))),
            ),
            
            // 最小化挂机
            ElevatedButton(
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFF8B5CF6),
                foregroundColor: Colors.white,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(8),
                ),
              ),
              onPressed: () async {
                Navigator.of(dialogContext).pop();
                await _initTray(); // Ensure tray is initialized
                await windowManager.hide(); // Hides window from taskbar and shows only in tray
              },
              child: const Text('托盘最小化'),
            ),
          ],
        );
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    final provider = Provider.of<TranscriptionProvider>(context);
    final currentIndex = provider.currentTab;

    return Scaffold(
      body: Row(
        children: [
          Sidebar(
            selectedIndex: currentIndex,
            onSelected: (index) {
              provider.setCurrentTab(index);
            },
            items: sidebarItems,
          ),
          Expanded(
            child: Column(
              children: [
                const SizedBox(
                  height: kWindowCaptionHeight,
                  child: WindowCaption(
                    brightness: Brightness.dark,
                    backgroundColor: Colors.transparent,
                  ),
                ),
                Expanded(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 200),
                    transitionBuilder: (child, animation) {
                      return FadeTransition(opacity: animation, child: child);
                    },
                    child: _buildCurrentView(currentIndex),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCurrentView(int index) {
    switch (index) {
      case 0:
        return const DashboardView(key: ValueKey('dashboard'));
      case 1:
        return const EditorView(key: ValueKey('editor'));
      case 2:
        return const ModelsView(key: ValueKey('models'));
      case 3:
        return const SettingsView(key: ValueKey('settings'));
      default:
        return const DashboardView(key: ValueKey('dashboard'));
    }
  }
}

class InitializationErrorApp extends StatelessWidget {
  final String error;
  final String stackTrace;

  const InitializationErrorApp({
    super.key,
    required this.error,
    required this.stackTrace,
  });

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Audio2Srt - 初始化失败',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        brightness: Brightness.dark,
        scaffoldBackgroundColor: const Color(0xFF07070F),
        colorScheme: const ColorScheme.dark(
          primary: Color(0xFF8B5CF6),
          secondary: Color(0xFFEF4444),
          surface: Color(0xFF131324),
        ),
        fontFamily: 'Segoe UI',
      ),
      home: InitializationErrorPage(error: error, stackTrace: stackTrace),
    );
  }
}

class InitializationErrorPage extends StatefulWidget {
  final String error;
  final String stackTrace;

  const InitializationErrorPage({
    super.key,
    required this.error,
    required this.stackTrace,
  });

  @override
  State<InitializationErrorPage> createState() => _InitializationErrorPageState();
}

class _InitializationErrorPageState extends State<InitializationErrorPage> {
  bool _showDetails = false;
  String _copyStatus = '复制详细错误报告';
  Color _copyStatusColor = const Color(0xFF8B5CF6);

  void _downloadRedistributable() {
    Process.run('cmd', ['/c', 'start', 'https://aka.ms/vs/17/release/vc_redist.x64.exe']);
  }

  void _copyToClipboard() {
    final report = 'Audio2Srt Error Report\n'
        'Error: ${widget.error}\n\n'
        'Stack Trace:\n${widget.stackTrace}';
    Clipboard.setData(ClipboardData(text: report));
    setState(() {
      _copyStatus = '复制成功！已存入剪贴板';
      _copyStatusColor = Colors.green;
    });
    Future.delayed(const Duration(seconds: 3), () {
      if (mounted) {
        setState(() {
          _copyStatus = '复制详细错误报告';
          _copyStatusColor = const Color(0xFF8B5CF6);
        });
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Container(
        decoration: const BoxDecoration(
          gradient: LinearGradient(
            begin: Alignment.topLeft,
            end: Alignment.bottomRight,
            colors: [
              Color(0xFF0D0D1E),
              Color(0xFF07070F),
              Color(0xFF1A102F),
            ],
          ),
        ),
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(32.0),
            child: Container(
              constraints: const BoxConstraints(maxWidth: 700),
              padding: const EdgeInsets.all(40.0),
              decoration: BoxDecoration(
                color: const Color(0x0CFFFFFF),
                borderRadius: BorderRadius.circular(24),
                border: Border.all(color: const Color(0x1F8B5CF6)),
                boxShadow: const [
                  BoxShadow(
                    color: Color(0x1A000000),
                    blurRadius: 30,
                    offset: Offset(0, 10),
                  ),
                ],
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: const Color(0x1AEF4444),
                          borderRadius: BorderRadius.circular(16),
                          border: Border.all(color: const Color(0x3FEF4444)),
                        ),
                        child: const Icon(
                          Icons.gpp_maybe_outlined,
                          color: Color(0xFFEF4444),
                          size: 36,
                        ),
                      ),
                      const SizedBox(width: 20),
                      const Expanded(
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '底层组件装载失败',
                              style: TextStyle(
                                fontSize: 24,
                                fontWeight: FontWeight.bold,
                                color: Colors.white,
                              ),
                            ),
                            SizedBox(height: 4),
                            Text(
                              'Audio2Srt Initialization Failed',
                              style: TextStyle(
                                fontSize: 13,
                                color: Colors.grey,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 32),
                  const Text(
                    '可能的原因：',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 12),
                  _buildCauseItem(
                    '1. 缺少微软 Visual C++ 运行库：',
                    '这是最常见的问题。如果这是您的新电脑或纯净系统，可能未安装 MSVC 运行库。请点击下方修复按钮进行一键安装。',
                  ),
                  const SizedBox(height: 10),
                  _buildCauseItem(
                    '2. 显卡 Vulkan 驱动兼容性问题：',
                    '本软件核心使用 Vulkan GPU 加速，核显驱动损坏或过旧会导致库加载闪退。',
                  ),
                  const SizedBox(height: 32),
                  Row(
                    children: [
                      Expanded(
                        child: ElevatedButton.icon(
                          onPressed: _downloadRedistributable,
                          icon: const Icon(Icons.download_for_offline_outlined),
                          label: const Text(
                            '一键修复：下载 VC++ 运行库',
                            style: TextStyle(fontWeight: FontWeight.bold),
                          ),
                          style: ElevatedButton.styleFrom(
                            backgroundColor: const Color(0xFF8B5CF6),
                            foregroundColor: Colors.white,
                            padding: const EdgeInsets.symmetric(vertical: 18),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                            elevation: 0,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  Row(
                    children: [
                      Expanded(
                        child: OutlinedButton.icon(
                          onPressed: _copyToClipboard,
                          icon: const Icon(Icons.copy_outlined),
                          label: Text(
                            _copyStatus,
                            style: TextStyle(
                              color: _copyStatusColor,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          style: OutlinedButton.styleFrom(
                            side: const BorderSide(color: Color(0x3F8B5CF6)),
                            padding: const EdgeInsets.symmetric(vertical: 16),
                            shape: RoundedRectangleBorder(
                              borderRadius: BorderRadius.circular(12),
                            ),
                          ),
                        ),
                      ),
                      const SizedBox(width: 12),
                      OutlinedButton.icon(
                        onPressed: () {
                          try {
                            windowManager.destroy();
                          } catch (_) {
                            exit(0);
                          }
                        },
                        icon: const Icon(Icons.close, color: Color(0xFFEF4444)),
                        label: const Text(
                          '退出程序',
                          style: TextStyle(
                            color: Color(0xFFEF4444),
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        style: OutlinedButton.styleFrom(
                          side: const BorderSide(color: Color(0x3FEF4444)),
                          padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 20),
                          shape: RoundedRectangleBorder(
                            borderRadius: BorderRadius.circular(12),
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  const Divider(color: Color(0x1FFFFFFF)),
                  const SizedBox(height: 12),
                  InkWell(
                    onTap: () {
                      setState(() {
                        _showDetails = !_showDetails;
                      });
                    },
                    borderRadius: BorderRadius.circular(8),
                    child: Padding(
                      padding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 4.0),
                      child: Row(
                        mainAxisAlignment: MainAxisAlignment.spaceBetween,
                        children: [
                          const Text(
                            '查看详细崩溃日志与错误堆栈',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey,
                              fontWeight: FontWeight.bold,
                            ),
                          ),
                          Icon(
                            _showDetails ? Icons.expand_less : Icons.expand_more,
                            color: Colors.grey,
                          ),
                        ],
                      ),
                    ),
                  ),
                  if (_showDetails) ...[
                    const SizedBox(height: 12),
                    Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: const Color(0x05FFFFFF),
                        borderRadius: BorderRadius.circular(12),
                        border: Border.all(color: const Color(0x0FFFFFFF)),
                      ),
                      child: SelectableText(
                        'Error Info:\n${widget.error}\n\nStack Trace:\n${widget.stackTrace}',
                        style: const TextStyle(
                          fontFamily: 'monospace',
                          fontSize: 11,
                          color: Color(0xFFEF4444),
                        ),
                      ),
                    ),
                  ],
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildCauseItem(String title, String desc) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.bold,
            color: Color(0xFFA78BFA),
          ),
        ),
        const SizedBox(height: 2),
        Text(
          desc,
          style: TextStyle(
            fontSize: 12,
            height: 1.5,
            color: Colors.grey[300],
          ),
        ),
      ],
    );
  }
}
