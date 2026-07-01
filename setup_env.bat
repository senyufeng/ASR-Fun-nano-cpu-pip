@echo off
echo ============================================
echo   ASR-Fun-Nano 环境配置 (Windows)
echo ============================================

REM 检查 conda
where conda >nul 2>nul
if %errorlevel% neq 0 (
    echo [ERROR] 未找到 conda，请先安装 Miniconda
    echo 下载地址: https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b 1
)

echo [INFO] 创建 conda 环境 asr-nano (Python 3.11)...
call conda create -n asr-nano -y python=3.11
if %errorlevel% neq 0 (
    echo [ERROR] conda 环境创建失败
    pause
    exit /b 1
)

echo [INFO] 安装 PyTorch CPU 版...
call conda run -n asr-nano pip install torch --index-url https://download.pytorch.org/whl/cpu

echo [INFO] 安装 FunASR 依赖...
call conda run -n asr-nano pip install scipy librosa soundfile numpy PyYAML omegaconf hydra-core modelscope huggingface_hub safetensors transformers tiktoken sentencepiece kaldiio jieba jamo jaconv umap_learn editdistance torch_complex tensorboardX oss2 torchaudio

echo [INFO] 安装 funasr (开发模式)...
call conda run -n asr-nano pip install -e %~dp0.

echo.
echo ============================================
echo   安装完成！
echo   使用方法:
echo     1. 激活环境: conda activate asr-nano
echo     2. 运行推理: python inference.py audio.wav
echo ============================================
pause