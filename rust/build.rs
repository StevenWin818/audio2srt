use std::env;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=src/api/ct2_wrapper.cc");
    println!("cargo:rerun-if-changed=src/api/ct2_wrapper.h");
    println!("cargo:rerun-if-changed=src/api/ctranslate2_bridge.rs");

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let ct2_dir = manifest_dir.join("third_party").join("ctranslate2");
    
    let include_dir = ct2_dir.join("include");
    let lib_dir = ct2_dir.join("lib");

    // Configure cxx
    cxx_build::bridge("src/api/ctranslate2_bridge.rs")
        .file("src/api/ct2_wrapper.cc")
        .include("src/api")
        .include(&include_dir)
        .flag_if_supported("/std:c++17")
        .flag_if_supported("-std=c++17")
        .compile("ct2_wrapper");

    println!("cargo:rustc-link-search=native={}", lib_dir.display());
    println!("cargo:rustc-link-lib=dylib=ctranslate2");

    // Copy DLLs to OUT_DIR so that test/bin execution can find them
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let dll_names = [
        "ctranslate2.dll",
        "libiomp5md.dll",
    ];
    for dll in &dll_names {
        let src = lib_dir.join(dll);
        if src.exists() {
            let dest = out_dir.join(dll);
            let _ = std::fs::copy(&src, &dest);
            
            // Also copy to profile target dir (like target/debug or target/release)
            // OUT_DIR is usually target/debug/build/rust_lib_audio2srt-.../out
            // Go up 3 levels to target/debug
            if let Some(target_dir) = out_dir.parent().and_then(|p| p.parent()).and_then(|p| p.parent()) {
                let dest2 = target_dir.join(dll);
                let _ = std::fs::copy(&src, &dest2);
            }

            // Also copy to Flutter runner output directories if they exist
            let workspace_dir = manifest_dir.parent().unwrap();
            let runner_dirs = [
                workspace_dir.join("build").join("windows").join("x64").join("runner").join("Debug"),
                workspace_dir.join("build").join("windows").join("x64").join("runner").join("Release"),
                workspace_dir.join("build").join("windows").join("x64").join("runner").join("Profile"),
            ];
            for runner_dir in &runner_dirs {
                if runner_dir.exists() {
                    let dest3 = runner_dir.join(dll);
                    let _ = std::fs::copy(&src, &dest3);
                }
            }
        }
    }
}
