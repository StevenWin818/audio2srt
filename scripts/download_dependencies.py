import os
import zipfile
import shutil
import subprocess

def download_and_extract_dependencies():
    output_dir = os.path.join("build", "windows", "x64", "runner", "Release")
    
    # 确保发布目录存在
    os.makedirs(output_dir, exist_ok=True)
    print(f"Target directory: {output_dir}")

    print("\n--- Step 1: 下载CTranslate2 (v4.8.1) ---")
    os.makedirs("temp_ct2", exist_ok=True)
    subprocess.run(['pip', 'download', 'ctranslate2==4.8.1', '--no-deps', '-d', 'temp_ct2'])
    
    ct2_whl = [f for f in os.listdir("temp_ct2") if f.endswith(".whl")][0]
    with zipfile.ZipFile(os.path.join("temp_ct2", ct2_whl), 'r') as z:
        print("Extracting ctranslate2.dll...")
        z.extract("ctranslate2/ctranslate2.dll", "temp_ct2")
        shutil.copy(os.path.join("temp_ct2", "ctranslate2", "ctranslate2.dll"), os.path.join(output_dir, "ctranslate2.dll"))
        
        # 从CTranslate2轮中提取cudnn64_9.dll以确保兼容性
        print("Extracting cudnn64_9.dll...")
        z.extract("ctranslate2/cudnn64_9.dll", "temp_ct2")
        shutil.copy(os.path.join("temp_ct2", "ctranslate2", "cudnn64_9.dll"), os.path.join(output_dir, "cudnn64_9.dll"))
        
        print("Extracting libiomp5md.dll...")
        z.extract("ctranslate2/libiomp5md.dll", "temp_ct2")
        shutil.copy(os.path.join("temp_ct2", "ctranslate2", "libiomp5md.dll"), os.path.join(output_dir, "libiomp5md.dll"))

    print("\n--- Step 2: 下载适用于 CUDA 12 的 cuDNN 9 ---")
    os.makedirs("temp_cudnn", exist_ok=True)
    subprocess.run(['pip', 'download', 'nvidia-cudnn-cu12', '--no-deps', '-d', 'temp_cudnn'])
    
    cudnn_whl = [f for f in os.listdir("temp_cudnn") if f.endswith(".whl")][0]
    # 我们只需要核心推理模块以节省空间（ctranslate2不需要graph/engines/adv）
    required_cudnn_dlls = ['cudnn_cnn64_9.dll', 'cudnn_ops64_9.dll']
    with zipfile.ZipFile(os.path.join("temp_cudnn", cudnn_whl), 'r') as z:
        for item in z.namelist():
            filename = os.path.basename(item)
            if filename in required_cudnn_dlls:
                print(f"Extracting {filename}...")
                z.extract(item, "temp_cudnn")
                shutil.copy(os.path.join("temp_cudnn", item), os.path.join(output_dir, filename))

    print("\n--- Step 3: 下载 CUDA 12 的 cuBLAS ---")
    os.makedirs("temp_cublas", exist_ok=True)
    subprocess.run(['pip', 'download', 'nvidia-cublas-cu12', '--no-deps', '-d', 'temp_cublas'])
    
    cublas_whl = [f for f in os.listdir("temp_cublas") if f.endswith(".whl")][0]
    required_cublas_dlls = ['cublas64_12.dll', 'cublasLt64_12.dll']
    with zipfile.ZipFile(os.path.join("temp_cublas", cublas_whl), 'r') as z:
        for item in z.namelist():
            filename = os.path.basename(item)
            if filename in required_cublas_dlls:
                print(f"Extracting {filename}...")
                z.extract(item, "temp_cublas")
                shutil.copy(os.path.join("temp_cublas", item), os.path.join(output_dir, filename))

    print("\n--- Step 4:清理临时文件 ---")
    shutil.rmtree("temp_ct2")
    shutil.rmtree("temp_cudnn")
    shutil.rmtree("temp_cublas")

    print("\n✅ 所有DLL已成功下载并放置在Release文件夹中！")

if __name__ == "__main__":
    download_and_extract_dependencies()
