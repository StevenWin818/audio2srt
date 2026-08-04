# 为 llama-cpp-sys-2 的 ggml-vulkan 打补丁: 缩短 vulkan-shaders-gen 嵌套构建目录。
#
# 背景: MSVC cl.exe 创建 PDB 有硬性 260 字符路径限制 (Windows 长路径策略无效,
# 实测 LongPathsEnabled=1 时 322 字符 PDB 路径仍报 C1041)。
# vulkan 特性下 ExternalProject 默认构建目录
#   ggml/src/ggml-vulkan/vulkan-shaders-gen-prefix/src/vulkan-shaders-gen-build
# 使编译器探测阶段的 PDB 路径达到 ~266 字符 -> C1041 必现。
# 补丁将 BINARY_DIR 改为 vkgen, PDB 路径降至 ~220 字符。
#
# 该补丁作用于 cargo registry 缓存 (crate 版本由 Cargo.lock 锁定,
# 正常构建不受影响); 仅在 llama-cpp-sys-2 重新解压 (cargo update / 缓存清理)
# 后需要重新执行本脚本。

$ErrorActionPreference = "Stop"

$target = Join-Path $env:USERPROFILE ".cargo\registry\src"
if (-not (Test-Path $target)) {
    Write-Error "cargo registry src not found: $target"
}

$cmakeListsAll = Get-ChildItem $target -Recurse -Filter "CMakeLists.txt" |
    Where-Object { $_.FullName -match "llama-cpp-sys-2-[\d.]+\\llama\.cpp\\ggml\\src\\ggml-vulkan\\CMakeLists\.txt$" }

if (-not $cmakeListsAll) {
    Write-Error "ggml-vulkan CMakeLists.txt not found under llama-cpp-sys-2 in $target"
}

foreach ($cmakeLists in $cmakeListsAll) {
    $content = Get-Content $cmakeLists.FullName -Raw
    if ($content -match "BINARY_DIR.*vkgen") {
        Write-Host "Patch already applied: $($cmakeLists.FullName)"
        continue
    }

    $old = "        SOURCE_DIR `${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders"
    $new = "        SOURCE_DIR `${CMAKE_CURRENT_SOURCE_DIR}/vulkan-shaders`n" +
           "        # 缩短嵌套构建目录: cl.exe 创建 PDB 有硬性 260 字符限制 (长路径策略无效),`n" +
           "        # 默认 -prefix/src/-build 层级会让 PDB 路径超限报 C1041`n" +
           "        BINARY_DIR `${CMAKE_CURRENT_BINARY_DIR}/vkgen"

    if (-not $content.Contains($old)) {
        Write-Error "Unexpected CMakeLists.txt content; patch pattern not found in $($cmakeLists.FullName)"
    }

    $content = $content.Replace($old, $new)
    Set-Content -Path $cmakeLists.FullName -Value $content -Encoding ascii -NoNewline
    Write-Host "Patched: $($cmakeLists.FullName)"
}

Write-Host "Next: 删除失败的 llama-cpp-sys-2 构建残留 (cargo clean -p llama-cpp-sys-2), 然后重新构建"
