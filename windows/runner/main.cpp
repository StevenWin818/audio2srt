#include <flutter/dart_project.h>
#include <flutter/flutter_view_controller.h>
#include <windows.h>
#include <vector>
#include <string>

#include "flutter_window.h"
#include "utils.h"

// Test graphics acceleration safety. If only basic display adapters are active, fallback to CPU software rendering to prevent crashes.
void CheckAndSetupSoftwareRendering() {
  DISPLAY_DEVICEW dd;
  dd.cb = sizeof(dd);
  int deviceIndex = 0;
  bool found_real_gpu = false;

  while (::EnumDisplayDevicesW(nullptr, deviceIndex, &dd, 0)) {
    if (dd.StateFlags & DISPLAY_DEVICE_ACTIVE) {
      std::wstring device_string(dd.DeviceString);
      // Check if this is a standard Microsoft Basic driver
      if (device_string.find(L"Basic Display") == std::wstring::npos &&
          device_string.find(L"Basic Render") == std::wstring::npos &&
          device_string.find(L"Driverless") == std::wstring::npos) {
        found_real_gpu = true;
        break;
      }
    }
    deviceIndex++;
  }

  // If no hardware accelerated GPU driver is active, force Flutter to CPU software rendering
  if (!found_real_gpu) {
    ::SetEnvironmentVariableW(L"FLUTTER_ENGINE_SWITCHES", L"--enable-software-rendering");
  }
}

int APIENTRY wWinMain(_In_ HINSTANCE instance, _In_opt_ HINSTANCE prev,
                      _In_ wchar_t *command_line, _In_ int show_command) {
  // Evaluate graphics driver state before loading any Flutter engines
  CheckAndSetupSoftwareRendering();

  // Attach to console when present (e.g., 'flutter run') or create a
  // new console when running with a debugger.
  if (!::AttachConsole(ATTACH_PARENT_PROCESS) && ::IsDebuggerPresent()) {
    CreateAndAttachConsole();
  }

  // Initialize COM, so that it is available for use in the library and/or
  // plugins.
  ::CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);

  flutter::DartProject project(L"data");

  std::vector<std::string> command_line_arguments =
      GetCommandLineArguments();

  project.set_dart_entrypoint_arguments(std::move(command_line_arguments));

  FlutterWindow window(project);
  Win32Window::Point origin(10, 10);
  Win32Window::Size size(1280, 720);
  if (!window.Create(L"audio2srt", origin, size)) {
    return EXIT_FAILURE;
  }
  window.SetQuitOnClose(true);

  ::MSG msg;
  while (::GetMessage(&msg, nullptr, 0, 0)) {
    ::TranslateMessage(&msg);
    ::DispatchMessage(&msg);
  }

  ::CoUninitialize();
  return EXIT_SUCCESS;
}
