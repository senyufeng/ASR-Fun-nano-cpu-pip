#!/usr/bin/env python3
"""ASR-Fun-Nano CPU 推理引擎 — 语音识别 + 字符级时间戳 + 声纹识别。

Python API:
    from asr_nano import ASREngine
    engine = ASREngine(enable_voiceprint=True)
    engine.load()
    results = engine.transcribe(["audio.wav"])
    json_str = ASREngine.to_json(results)
"""

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch

warnings.filterwarnings("ignore")


# ============================================================
# 数据结构
# ============================================================

@dataclass
class CharTimestamp:
    """字符级时间戳"""
    token: str
    start_time: float
    end_time: float


@dataclass
class VoiceprintResult:
    """声纹识别结果"""
    enabled: bool
    matched: bool = False
    speaker_id: Optional[str] = None
    confidence: float = 0.0
    candidates: list = field(default_factory=list)


@dataclass
class SegmentResult:
    """单个语音段"""
    index: int
    start_time: float
    end_time: float
    text: str
    timestamps: list = field(default_factory=list)
    voiceprint: Optional[VoiceprintResult] = None


@dataclass
class ASRResult:
    """完整识别结果"""
    audio_file: str
    audio_duration: float = 0.0
    language: str = "中文"
    text: str = ""
    segments: list = field(default_factory=list)
    voiceprint_enabled: bool = False
    voiceprint_summary: Optional[dict] = None
    elapsed_sec: float = 0.0


# ============================================================
# 推理引擎
# ============================================================

class ASREngine:
    """ASR-Fun-Nano 推理引擎。整合 CPU ASR + VAD + 声纹识别。

    用法:
        engine = ASREngine(
            model="FunAudioLLM/Fun-ASR-Nano-2512",
            device="cpu",
            language="中文",
            enable_vad=False,
            enable_voiceprint=False,
        )
        engine.load()
        results = engine.transcribe(["audio.wav"])
        print(ASREngine.to_json(results))
    """

    def __init__(
        self,
        model: str = "FunAudioLLM/Fun-ASR-Nano-2512",
        device: str | None = None,
        hub: str = "ms",
        language: str = "中文",
        enable_vad: bool = False,
        enable_voiceprint: bool = False,
        voiceprint_db_dir: str | None = None,
        voiceprint_threshold: float = 0.55,
        hotwords: list[str] | None = None,
    ):
        self.model_id = model
        self.device = device or self._auto_device()
        self.hub = hub
        self.language = language
        self.enable_vad = enable_vad
        self.enable_voiceprint = enable_voiceprint
        self.voiceprint_db_dir = voiceprint_db_dir
        self.voiceprint_threshold = voiceprint_threshold
        self.hotwords = hotwords
        self._asr_model = None
        self._vp_engine = None
        self._vp_db = None

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda:0"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    # ---- 加载 ----

    def load(self):
        """加载所有模型。模型常驻内存直到调用 unload()。"""
        self._load_asr()
        if self.enable_voiceprint:
            self._load_voiceprint()

    def unload(self):
        """释放模型内存。"""
        self._asr_model = None
        self._vp_engine = None
        self._vp_db = None

    def _load_asr(self):
        from funasr import AutoModel

        kwargs = {
            "model": self.model_id,
            "trust_remote_code": True,
            "device": self.device,
            "hub": self.hub,
        }
        if self.enable_vad:
            kwargs["vad_model"] = "fsmn-vad"
            kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

        self._asr_model = AutoModel(**kwargs)

    def _load_voiceprint(self):
        from asr_nano.voiceprint import VoiceprintDB, VoiceprintEngine

        self._vp_engine = VoiceprintEngine(device=self.device, hub=self.hub)
        db_dir = Path(self.voiceprint_db_dir) if self.voiceprint_db_dir else None
        self._vp_db = VoiceprintDB(db_dir)

    # ---- 推理 ----

    def transcribe(self, audio_files: list[str]) -> list[ASRResult]:
        """对一批音频/视频文件执行 ASR + 声纹识别。

        Args:
            audio_files: 音频或视频文件路径列表

        Returns:
            [ASRResult, ...]
        """
        if self._asr_model is None:
            self.load()

        gen_kwargs = {
            "input": audio_files,
            "cache": {},
            "batch_size": 1,
            "language": self.language,
            "itn": True,
        }
        if self.hotwords:
            gen_kwargs["hotwords"] = self.hotwords

        t_start = time.time()
        raw_results = self._asr_model.generate(**gen_kwargs)
        total_elapsed = time.time() - t_start

        results = []
        for i, raw in enumerate(raw_results):
            elapsed_per_file = total_elapsed / max(1, len(audio_files))
            results.append(self._build_result(audio_files[i], raw, elapsed_per_file))

        return results

    def _build_result(self, audio_path: str, raw: dict, elapsed: float) -> ASRResult:
        """将 FunASR 原始输出转换为统一 ASRResult。"""
        text = raw.get("text", "")
        ts_raw = raw.get("timestamps", []) or raw.get("timestamp", [])

        # 规范化时间戳
        char_ts = []
        for t in ts_raw:
            if isinstance(t, dict):
                char_ts.append(CharTimestamp(
                    token=t.get("token", ""),
                    start_time=float(t.get("start_time", 0)),
                    end_time=float(t.get("end_time", 0)),
                ))
            elif isinstance(t, (list, tuple)) and len(t) >= 2:
                char_ts.append(CharTimestamp(
                    token="",
                    start_time=float(t[0]),
                    end_time=float(t[1]),
                ))

        seg_start = char_ts[0].start_time if char_ts else 0.0
        seg_end = char_ts[-1].end_time if char_ts else 0.0

        segment = SegmentResult(
            index=0,
            start_time=round(seg_start, 3),
            end_time=round(seg_end, 3),
            text=text,
            timestamps=char_ts,
        )

        # 声纹识别
        voiceprint_summary = None
        if self.enable_voiceprint and self._vp_engine is not None and self._vp_db is not None:
            actual_path = audio_path
            if not os.path.isfile(actual_path):
                actual_path = raw.get("key", actual_path)
            if os.path.isfile(actual_path):
                try:
                    vp_result = self._vp_engine.match(
                        actual_path, self._vp_db, threshold=self.voiceprint_threshold
                    )
                    candidates = [
                        {"speaker_id": c["name"], "confidence": c["score"]}
                        for c in vp_result.get("all_scores", [])
                    ]
                    segment.voiceprint = VoiceprintResult(
                        enabled=True,
                        matched=vp_result.get("matched", False),
                        speaker_id=vp_result.get("name"),
                        confidence=vp_result.get("score", 0.0),
                        candidates=candidates,
                    )
                    if vp_result.get("matched"):
                        voiceprint_summary = {
                            "speaker_id": vp_result["name"],
                            "confidence": vp_result["score"],
                        }
                except Exception:
                    pass

        audio_duration = seg_end - seg_start

        return ASRResult(
            audio_file=os.path.abspath(audio_path),
            audio_duration=round(audio_duration, 3),
            language=self.language,
            text=text,
            segments=[segment],
            voiceprint_enabled=self.enable_voiceprint,
            voiceprint_summary=voiceprint_summary,
            elapsed_sec=round(elapsed, 2),
        )

    # ---- 序列化 ----

    @staticmethod
    def to_json(results: list[ASRResult], indent: int = 2) -> str:
        """将 ASRResult 列表转为 JSON 字符串。"""
        return json.dumps(
            [ASREngine._result_to_dict(r) for r in results],
            ensure_ascii=False,
            indent=indent,
        )

    @staticmethod
    def _result_to_dict(result: ASRResult) -> dict:
        """递归转换 dataclass 为纯 dict，排除 None 值。"""
        out = {
            "audio_file": result.audio_file,
            "audio_duration": result.audio_duration,
            "language": result.language,
            "text": result.text,
            "segments": [],
            "voiceprint_enabled": result.voiceprint_enabled,
            "elapsed_sec": result.elapsed_sec,
        }
        if result.voiceprint_summary:
            out["voiceprint_summary"] = result.voiceprint_summary

        for seg in result.segments:
            seg_dict = {
                "index": seg.index,
                "start_time": seg.start_time,
                "end_time": seg.end_time,
                "text": seg.text,
                "timestamps": [
                    {"token": ts.token, "start_time": ts.start_time, "end_time": ts.end_time}
                    for ts in seg.timestamps
                ],
            }
            if seg.voiceprint:
                vp = seg.voiceprint
                seg_dict["voiceprint"] = {
                    "enabled": vp.enabled,
                    "matched": vp.matched,
                    "speaker_id": vp.speaker_id,
                    "confidence": vp.confidence,
                    "candidates": vp.candidates,
                }
            out["segments"].append(seg_dict)

        return out


# ============================================================
# CLI
# ============================================================

def main():
    # 修复 Windows 终端中文乱码
    if sys.platform == "win32":
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(
        description="ASR-Fun-Nano 推理 — 语音识别 + 时间戳 + 声纹",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python -m asr_nano.engine audio.wav
    python -m asr_nano.engine --voiceprint audio.wav
    python -m asr_nano.engine -o result.json audio.wav
""",
    )

    parser.add_argument("input", nargs="+", help="输入音频/视频文件")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 文件")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512")
    parser.add_argument("--hub", default="ms", choices=["ms", "hf"])
    parser.add_argument("--language", default="中文")
    parser.add_argument("--device", default=None)
    parser.add_argument("--vad", action="store_true")
    parser.add_argument("--hotwords", nargs="*", default=None)
    parser.add_argument("--voiceprint", action="store_true", help="启用声纹识别")
    parser.add_argument("--voiceprint-db", default=None, help="声纹库目录")
    parser.add_argument("--voiceprint-threshold", type=float, default=0.55)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    engine = ASREngine(
        model=args.model,
        device=args.device,
        hub=args.hub,
        language=args.language,
        enable_vad=args.vad,
        enable_voiceprint=args.voiceprint,
        voiceprint_db_dir=args.voiceprint_db,
        voiceprint_threshold=args.voiceprint_threshold,
        hotwords=args.hotwords,
    )

    if not args.quiet:
        print(f"[INFO] 设备: {engine.device}, VAD: {engine.enable_vad}, 声纹: {engine.enable_voiceprint}")

    t0 = time.time()
    engine.load()
    if not args.quiet:
        print(f"[INFO] 模型加载完成 ({time.time() - t0:.1f}s)")

    results = engine.transcribe(args.input)
    json_str = ASREngine.to_json(results)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(json_str)
        if not args.quiet:
            print(f"[INFO] 已保存: {args.output}")
    else:
        print(json_str)

    return 0


if __name__ == "__main__":
    sys.exit(main())