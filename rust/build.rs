fn main() {
    // CUDA 架构编译分发策略已交由 cargokit.cmake 的 CARGOKIT_ENV 动态控制。

    // 2. 当在 Windows 下启用 cuda 特性时，自动为 MSVC 链接器注入 CUDA Toolkit 的库搜索路径 (lib/x64)
    #[cfg(target_os = "windows")]
    {
        let is_cuda_enabled = std::env::var("CARGO_FEATURE_CUDA").is_ok() 
            || std::env::var("CARGO_FEATURE_QWEN_CUDA").is_ok();
        if is_cuda_enabled {
            if let Ok(cuda_path) = std::env::var("CUDA_PATH") {
                let lib_path = format!("{}\\lib\\x64", cuda_path);
                if std::path::Path::new(&lib_path).exists() {
                    println!("cargo:rustc-link-search=native={}", lib_path);
                }
            }
            if let Ok(cuda_path) = std::env::var("CUDA_PATH_V13_3") {
                let lib_path = format!("{}\\lib\\x64", cuda_path);
                if std::path::Path::new(&lib_path).exists() {
                    println!("cargo:rustc-link-search=native={}", lib_path);
                }
            }
            if std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\lib\x64").exists() {
                println!(r"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\lib\x64");
            }
            if std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64").exists() {
                println!(r"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.0\lib\x64");
            }
            if std::path::Path::new(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64").exists() {
                println!(r"cargo:rustc-link-search=native=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\lib\x64");
            }
        }
    }
}
