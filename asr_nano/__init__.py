#!/usr/bin/env python3
"""ASR-Fun-Nano — CPU 语音识别 + 字符级时间戳 + 声纹识别。

公共 API:
    from asr_nano import ASREngine        # 推理引擎
    from asr_nano import VoiceprintDB     # 声纹数据库
    from asr_nano import VoiceprintEngine # 声纹引擎
    from asr_nano import serve            # 启动常驻服务
"""

from asr_nano.engine import ASREngine
from asr_nano.voiceprint import VoiceprintDB, VoiceprintEngine
from asr_nano.server import main as serve

__version__ = "1.0.0"
__all__ = ["ASREngine", "VoiceprintDB", "VoiceprintEngine", "serve"]