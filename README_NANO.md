# ASR-Fun-Nano — CPU 语音识别 + 字符级时间戳

基于 [FunASR](https://github.com/modelscope/FunASR) 的 **Fun-ASR-Nano-2512** 模型，
**CPU-only** 推理，支持 3 种语言（中/英/日），输出**字符级时间戳**。

## 模型信息

| 属性 | 值 |
|---|---|
| 模型 | Fun-ASR-Nano-2512 |
| 参数 | 800M (SenseVoice编码器 + Qwen3-0.6B LLM + CTC解码器) |
| 语言 | 中文、英文、日文 |
| 任务 | 语音识别 + 字符级时间戳 |
| 推理硬件 | CPU (PyTorch), 也支持 GPU/mps |
| 时间戳 | ✅ 字符级 CTC 强制对齐 (60ms 粒度) |
| 标点 | ✅ 原生输出 |
| 热词 | ✅ 支持自定义热词 |

## 环境要求

- Python 3.8+ (推荐 3.11)
- Conda (Miniconda/Anaconda)
- 约 5GB 磁盘空间 (模型文件)

### 快速安装

**Windows:**
```bat
setup_env.bat
```

**Linux / Mac:**
```bash
bash setup_env.sh
```

**手动安装:**
```bash
conda create -n asr-nano python=3.11 -y
conda activate asr-nano
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pip install -e .
```

## 下载模型

```bash
conda activate asr-nano
python download_model.py
```

模型约 4GB，下载到 `models/` 目录。也可直接用 HuggingFace 模型名在线下载（首次推理时自动缓存）。

## 使用

```bash
conda activate asr-nano

# 基本用法 — 输出文本 + 时间戳
python inference.py audio.wav

# 指定输出文件
python inference.py -o result.json audio.wav

# 多文件批量
python inference.py audio1.wav audio2.mp3 audio3.flac

# 英文识别
python inference.py --language 英文 en_audio.wav

# 长音频 + VAD 分段
python inference.py --vad long_audio.wav

# 热词提升特定词汇识别
python inference.py --hotwords 开放时间 会议室 audio.wav

# 使用本地模型路径
python inference.py --model ./models/FunAudioLLM/Fun-ASR-Nano-2512 audio.wav
```

### 输出格式

```json
[
  {
    "key": "audio.wav",
    "text": "开放时间是上午九点到下午五点。",
    "timestamps": [
      {"token": "开", "start_time": 0.78, "end_time": 0.84},
      {"token": "放", "start_time": 1.08, "end_time": 1.14},
      {"token": "时", "start_time": 1.38, "end_time": 1.44},
      {"token": "间", "start_time": 1.56, "end_time": 1.62}
    ]
  }
]
```

## 性能参考 (CPU, Intel i7)

| 音频长度 | 推理时间 | RTF |
|---|---|---|
| 5s | ~3s | 0.56 |
| 30s | ~20s | 0.67 |
| 60s | ~45s | 0.75 |

> 首次加载模型约需 40s (加载到内存)。模型常驻内存后后续推理更快。

## 项目结构

```
ASR-Fun-nano/
├── inference.py          # 推理脚本 (CPU)
├── download_model.py     # 模型下载脚本
├── requirements.txt      # Python 依赖
├── setup_env.bat         # Windows 一键环境配置
├── setup_env.sh          # Linux/Mac 一键环境配置
├── ASR_NANO_README.md    # 本文档
├── funasr/               # FunASR SDK (核心库)
│   └── models/fun_asr_nano/  # Fun-ASR-Nano 模型代码
└── models/               # 模型文件 (下载后)
    └── FunAudioLLM/Fun-ASR-Nano-2512/
```