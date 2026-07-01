#!/usr/bin/env bash
set -e

echo "============================================"
echo "  ASR-Fun-Nano 环境配置 (Linux/Mac)"
echo "============================================"

# 检查 conda
if ! command -v conda &>/dev/null; then
    echo "[ERROR] 未找到 conda，请先安装 Miniconda"
    echo "下载地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "[INFO] 创建 conda 环境 asr-nano (Python 3.11)..."
conda create -n asr-nano -y python=3.11

echo "[INFO] 安装 PyTorch CPU 版..."
conda run -n asr-nano pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[INFO] 安装 FunASR 依赖..."
conda run -n asr-nano pip install -r "$(dirname "$0")/requirements.txt"

echo "[INFO] 安装 funasr (开发模式)..."
conda run -n asr-nano pip install -e "$(dirname "$0")"

echo ""
echo "============================================"
echo "  安装完成！"
echo "  使用方法:"
echo "     conda activate asr-nano"
echo "     python download_model.py"
echo "     python inference.py audio.wav"
echo "============================================"