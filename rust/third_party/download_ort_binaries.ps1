# =============================================================================
# download_ort_binaries.ps1
#
# 下载并解包项目所需的预编译 ONNX Runtime 二进制 (Git 不追踪这些文件，
# 见根目录 .gitignore 的 rust/third_party/onnxruntime-* 条目)。
#
# 背景:
#   - deep_filter 依赖启用了 ort 的 load-dynamic, ort-sys 构建期不下载二进制;
#     应用在运行时 (ensure_onnxruntime_loaded) 把 exe 目录需要的 DLL 部署好。
#   - rust/third_party/onnxruntime-cuda: CUDA 13 版 onnxruntime 1.28.0 (GPU),
#     含 CUDA EP。运行时还会从系统安装拷贝 cuDNN 9 / cudart / cublas,
#     这部分不需要也不应该放进仓库。
#   - rust/third_party/onnxruntime-dml: DirectML 版 onnxruntime 1.23.0,
#     作为无 cuDNN 环境下的 DX12 后备 (CPU EP 兜底)。
#
# 用法:  powershell -ExecutionPolicy Bypass -File rust/third_party/download_ort_binaries.ps1
# 幂等: 目标文件已存在时跳过下载/拷贝。
# =============================================================================
$ErrorActionPreference = "Stop"

$tmp = Join-Path $env:TEMP "audio2srt-ort-download"
New-Item -ItemType Directory -Force -Path $tmp | Out-Null

function Get-NuGetPackage {
    param(
        [string]$PackageId,   # e.g. "Microsoft.ML.OnnxRuntime.Gpu.Windows"
        [string]$Version,     # e.g. "1.28.0"
        [string]$OutDir,      # 目标目录 (third_party 下)
        [string[]]$Files      # runtimes/win-x64/native 下需要拷贝的文件名
    )
    $idLower = $PackageId.ToLowerInvariant()
    $nupkg = Join-Path $tmp "$idLower.$Version.nupkg"
    if (-not (Test-Path $nupkg)) {
        $url = "https://api.nuget.org/v3-flatcontainer/$idLower/$Version/$idLower.$Version.nupkg"
        Write-Host "Downloading $PackageId $Version ($url) ..."
        Invoke-WebRequest -Uri $url -OutFile $nupkg -UseBasicParsing
    } else {
        Write-Host "Using cached $nupkg"
    }
    $extract = Join-Path $tmp "$idLower.$Version"
    if (-not (Test-Path (Join-Path $extract "runtimes"))) {
        # Expand-Archive 只认 .zip 扩展名, nupkg 本质是 zip
        $zip = "$nupkg.zip"
        Copy-Item -Path $nupkg -Destination $zip -Force
        Expand-Archive -Path $zip -DestinationPath $extract -Force
    }
    New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
    foreach ($f in $Files) {
        $dst = Join-Path $OutDir $f
        if (Test-Path $dst) {
            Write-Host "  skip  $dst (already exists)"
            continue
        }
        $src = Join-Path $extract "runtimes\win-x64\native\$f"
        if (-not (Test-Path $src)) {
            throw "File not found in package: $src"
        }
        Copy-Item -Path $src -Destination $dst -Force
        Write-Host "  -> $dst"
    }
}

$thirdParty = $PSScriptRoot

# CUDA 13 版 (onnxruntime 1.28.0 GPU, 支持 CUDA 13 + cuDNN 9.x)
Get-NuGetPackage `
    -PackageId "Microsoft.ML.OnnxRuntime.Gpu.Windows" `
    -Version "1.28.0" `
    -OutDir (Join-Path $thirdParty "onnxruntime-cuda") `
    -Files @("onnxruntime.dll", "onnxruntime_providers_cuda.dll", "onnxruntime_providers_shared.dll")

# DirectML 版 (DX12 后备, 含 CPU EP)
Get-NuGetPackage `
    -PackageId "Microsoft.ML.OnnxRuntime.DirectML" `
    -Version "1.23.0" `
    -OutDir (Join-Path $thirdParty "onnxruntime-dml") `
    -Files @("onnxruntime.dll", "onnxruntime_providers_shared.dll")

Write-Host ""
Write-Host "Done. ONNX Runtime binaries are ready in:"
Write-Host "  $thirdParty\onnxruntime-cuda"
Write-Host "  $thirdParty\onnxruntime-dml"
Write-Host "Note: cuDNN 9 / cudart / cublas are NOT bundled here; they are copied from the"
Write-Host "system installation (C:\Program Files\NVIDIA\...) at app runtime."
