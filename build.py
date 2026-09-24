"""打包成免安裝的 exe：dist/Corder-一般版.zip 與 dist/Corder-GPU版.zip。

用法：.venv\\Scripts\\python build.py
"""

import shutil
import sys
import zipfile
from pathlib import Path

import PyInstaller.__main__

ROOT = Path(__file__).resolve().parent
DIST = ROOT / "dist"
NVIDIA = Path(sys.prefix) / "Lib" / "site-packages" / "nvidia"
# ctranslate2 GPU 運算需要的 CUDA 函式庫（cuDNN 只取用得到的部分，省下約 900 MB）
CUDA_DLLS = [
    "cublas/bin/cublas64_12.dll",
    "cublas/bin/cublasLt64_12.dll",
    "cudnn/bin/cudnn64_9.dll",
    "cudnn/bin/cudnn_graph64_9.dll",
    "cudnn/bin/cudnn_ops64_9.dll",
    "cudnn/bin/cudnn_cnn64_9.dll",
    "cudnn/bin/cudnn_ext64_9.dll",
]
# PyInstaller 會收到某個套件附的舊版 VC++ runtime（14.29），onnxruntime 載入時會當掉，換成系統的新版
VC_RUNTIME = ["msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"]

README = """Corder 逐字稿小幫手（{edition}）

使用方式：雙擊資料夾裡的 Corder.exe（不需要安裝 Python）。

- 第一次轉錄時會自動下載語音模型，需要連網路：
  small 約 500 MB，large-v3-turbo 約 1.6 GB。下載完之後可以離線使用。
- 錄音檔會存在 Corder.exe 旁邊的 recordings 資料夾。
- {note}
- Windows 若跳出「Windows 已保護您的電腦」，按「其他資訊」→「仍要執行」。
"""


def main():
    PyInstaller.__main__.run([
        str(ROOT / "corder.py"),
        "--name=Corder",
        "--windowed",
        "--noconfirm",
        f"--distpath={DIST / 'build-out'}",
        f"--workpath={ROOT / 'build'}",
        f"--specpath={ROOT / 'build'}",
        "--collect-data=faster_whisper",  # Silero VAD 模型
        "--collect-binaries=ctranslate2",
        "--collect-binaries=onnxruntime",
        "--collect-data=opencc",
        "--collect-data=soundcard",
        "--hidden-import=hf_xet",  # 加快模型下載
    ])
    built = DIST / "build-out" / "Corder"
    for dll in VC_RUNTIME:
        shutil.copy2(Path(r"C:\Windows\System32") / dll, built / "_internal" / dll)

    editions = {
        "一般版": "這是一般版，任何 Windows 電腦都能用，用 CPU 轉錄。建議模型選 small。",
        "GPU版": "這是 GPU 版，有 NVIDIA 顯卡時會自動加速；沒有顯卡也能用（自動改用 CPU）。",
    }
    for edition, note in editions.items():
        target = DIST / f"Corder-{edition}"
        shutil.rmtree(target, ignore_errors=True)
        shutil.copytree(built, target)
        if edition == "GPU版":
            cuda = target / "cuda"
            cuda.mkdir()
            for dll in CUDA_DLLS:
                shutil.copy2(NVIDIA / dll, cuda)
        (target / "使用說明.txt").write_text(README.format(edition=edition, note=note), encoding="utf-8-sig")

        zip_path = DIST / f"Corder-{edition}.zip"
        print(f"壓縮 {zip_path.name} …")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for f in target.rglob("*"):
                zf.write(f, Path(target.name) / f.relative_to(target))
        print(f"完成：{zip_path}（{zip_path.stat().st_size / 1024**2:.0f} MB）")

    shutil.rmtree(DIST / "build-out", ignore_errors=True)


if __name__ == "__main__":
    main()
