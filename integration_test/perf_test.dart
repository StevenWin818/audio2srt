import 'dart:io';
import 'package:flutter/widgets.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:provider/provider.dart';
import 'package:integration_test/integration_test.dart';
import 'package:audio2srt/main.dart';
import 'package:audio2srt/providers/transcription_provider.dart';
import 'package:audio2srt/src/rust/frb_generated.dart';

void main() {
  final binding = IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  setUpAll(() async {
    await RustLib.init();
  });

  testWidgets('Performance Profiling Test', (WidgetTester tester) async {
    // 1. Create flutter_frame_perf.csv and write header in append mode
    final file = File('flutter_frame_perf.csv');
    final fileExists = await file.exists();
    final sink = file.openWrite(mode: FileMode.append);
    if (!fileExists) {
      sink.writeln('timestamp_ms,vsync_ms,build_ms,raster_ms,total_ms');
    }

    // 2. Add timings callback to log every frame timing
    final timingsCallback = (List<FrameTiming> timings) {
      for (final timing in timings) {
        final totalSpanMs = timing.totalSpan.inMicroseconds / 1000.0;
        final timestamp = DateTime.now().millisecondsSinceEpoch - timing.totalSpan.inMilliseconds;
        final vsync = timing.vsyncOverhead.inMicroseconds / 1000.0;
        final build = timing.buildDuration.inMicroseconds / 1000.0;
        final raster = timing.rasterDuration.inMicroseconds / 1000.0;

        sink.writeln('$timestamp,$vsync,$build,$raster,$totalSpanMs');
      }
    };
    WidgetsBinding.instance.addTimingsCallback(timingsCallback);

    try {
      // 3. Launch App UI
      await tester.pumpWidget(const Audio2SrtApp());
      await tester.pumpAndSettle();

      // Find the transcription provider
      final provider = Provider.of<TranscriptionProvider>(
        tester.element(find.byType(MainShell)),
        listen: false,
      );

      // Wait for provider to finish initialization (finding FFmpeg, model files, etc.)
      await Future.delayed(const Duration(seconds: 3));

      // 4. Locate input file and model, with fallback support
      const movieFile = r'C:\FFOutput\testmovie.mkv';
      const fallbackFile = r'C:\Projects\audio2srt\testaudio.wav';
      final fileToTest = File(File(movieFile).existsSync() ? movieFile : fallbackFile);
      if (!fileToTest.existsSync()) {
        fail('No test media file found at $movieFile or $fallbackFile');
      }

      // Check models
      const largeModel = 'ggml-large-v3-q8_0.bin';
      const baseModel = 'ggml-base.bin';
      final isLargeAvailable = provider.downloadedModels.contains(largeModel);
      final selectedModel = isLargeAvailable ? largeModel : baseModel;

      print('[PerfTest] Selected Model: $selectedModel');
      print('[PerfTest] Selected Input: ${fileToTest.path}');

      // Set options in provider
      provider.setInputFile(fileToTest);
      provider.setSelectedModel(selectedModel);
      provider.setSelectedLanguage('zh');
      provider.setEnableDenoise(true);
      provider.setUseGpu(true);
      provider.setVadEnabled(true);

      // Start transcription
      print('[PerfTest] Starting transcription...');
      await provider.startTranscription();

      // 5. Keep the test running for 3 minutes (180 seconds) to accumulate performance data
      print('[PerfTest] Running transcription profiling for 3 minutes...');
      for (int i = 0; i < 36; i++) {
        // Pump frame cycles to trigger UI rendering and keep integration test event loop spinning
        await tester.pump(const Duration(seconds: 5));
        print('[PerfTest] Elapsed: ${(i + 1) * 5} seconds. Progress: ${provider.progress}%. Status: ${provider.statusMessage}');
      }

      // 6. Explicitly trigger cancelTranscription() to cleanly stop
      print('[PerfTest] Cancelling transcription...');
      await provider.cancelTranscription();
      await tester.pumpAndSettle();

    } finally {
      // Clean up and close resources
      WidgetsBinding.instance.removeTimingsCallback(timingsCallback);
      await sink.flush();
      await sink.close();
      print('[PerfTest] Performance logs written to flutter_frame_perf.csv. Test complete.');
    }
  });
}
