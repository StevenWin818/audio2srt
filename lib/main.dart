import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'package:flutter_localizations/flutter_localizations.dart';

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
  await RustLib.init();

  runApp(const Audio2SrtApp());
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
            backgroundColor: Color(0xFF07070F),
            elevation: 0,
          ),
        ),
        home: const MainShell(),
      ),
    );
  }
}

class MainShell extends StatelessWidget {
  const MainShell({super.key});

  final List<SidebarItem> _sidebarItems = const [
    // Note: We can make SidebarItem const by adding const to its constructor, but let's just make it final
  ];

  List<SidebarItem> get sidebarItems => [
    SidebarItem(icon: Icons.dashboard_outlined, label: '主控制板'),
    SidebarItem(icon: Icons.edit_note_outlined, label: '字幕编辑器'),
    SidebarItem(icon: Icons.layers_outlined, label: '模型管理'),
    SidebarItem(icon: Icons.settings_outlined, label: '系统设置'),
  ];

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
