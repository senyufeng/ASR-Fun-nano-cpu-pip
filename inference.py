#!/usr/bin/env python3
"""ASR-Fun-Nano CPU 推理脚本 — 带字符级时间戳输出。

用法:
    python inference.py audio.wav
    python inference.py audio1.wav audio2.mp3
    python inference.py -o result.json audio.wav
    python inference.py --language 英文 audio.wav
    python inference.py --vad audio.wav
"""

import argparse
import json
import sys
import time

import torch


def main():
    parser = argparse.ArgumentParser(
        description="Fun-ASR-Nano CPU 推理 — 语音转文字 + 字符级时间戳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python inference.py audio.wav
    python inference.py -o result.json audio.wav
    python inference.py --language 英文 audio.wav
    python inference.py --vad audio1.wav audio2.wav

语言 (Fun-ASR-Nano-2512): 中文、英文、日文
支持格式: WAV, MP3, FLAC, OGG 等
""",
    )

    parser.add_argument("input", nargs="+", help="输入音频文件路径")
    parser.add_argument("-o", "--output", default=None, help="输出 JSON 文件路径")
    parser.add_argument("--model", default="FunAudioLLM/Fun-ASR-Nano-2512", help="模型名或本地路径")
    parser.add_argument("--hub", default="ms", choices=["ms", "hf"], help="下载源: ms=ModelScope, hf=HuggingFace")
    parser.add_argument("--language", default="中文", help="识别语言")
    parser.add_argument("--device", default=None, help="设备 (自动: cuda > mps > cpu)")
    parser.add_argument("--vad", action="store_true", help="启用 VAD 分段 (长音频推荐)")
    parser.add_argument("--hotwords", nargs="*", default=None, help="热词列表")
    parser.add_argument("--itn", action="store_true", default=True, help="逆文本正则化 (默认开启)")
    parser.add_argument("--no-itn", action="store_false", dest="itn", help="禁用逆文本正则化")
    parser.add_argument("--batch-size", type=int, default=1, help="批处理大小")
    parser.add_argument("--quiet", action="store_true", help="静默模式")

    args = parser.parse_args()

    # 设备选择
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda:0"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    if not args.quiet:
        print(f"[INFO] 设备: {device}")
        print(f"[INFO] 模型: {args.model} (hub={args.hub})")
        print(f"[INFO] 语言: {args.language}")

    # 加载模型
    from funasr import AutoModel

    t0 = time.time()
    model_kwargs = {
        "model": args.model,
        "trust_remote_code": True,
        "device": device,
        "hub": args.hub,
    }
    if args.vad:
        model_kwargs["vad_model"] = "fsmn-vad"
        model_kwargs["vad_kwargs"] = {"max_single_segment_time": 30000}

    model = AutoModel(**model_kwargs)
    if not args.quiet:
        print(f"[INFO] 模型加载完成 ({time.time() - t0:.1f}s)")

    # 推理
    kwargs = {
        "input": args.input,
        "cache": {},
        "batch_size": args.batch_size,
        "language": args.language,
        "itn": args.itn,
    }
    if args.hotwords:
        kwargs["hotwords"] = args.hotwords

    if not args.quiet:
        print(f"[INFO] 推理中 ({len(args.input)} 个文件)...")

    t1 = time.time()
    results = model.generate(**kwargs)
    elapsed = time.time() - t1

    # 处理结果
    output = []
    for i, res in enumerate(results):
        text = res.get("text", "")
        ts_raw = res.get("timestamps", [])
        ts_alt = res.get("timestamp", [])
        timestamps = ts_raw if ts_raw else ts_alt

        item = {
            "key": res.get("key", args.input[i]),
            "text": text,
            "timestamps": timestamps,
        }
        output.append(item)

        if not args.quiet:
            print(f"\n{'=' * 60}")
            print(f"文件: {item['key']}")
            print(f"文本: {text}")
            if timestamps:
                print(f"时间戳 (字符级, {len(timestamps)} 个):")
                for ts in timestamps:
                    if isinstance(ts, dict):
                        token = ts.get("token", "?")
                        s = ts.get("start_time", 0)
                        e = ts.get("end_time", 0)
                        print(f"  [{s:7.3f}s - {e:7.3f}s] {token}")
                    elif isinstance(ts, (list, tuple)) and len(ts) >= 2:
                        print(f"  [{ts[0]:7.3f}s - {ts[1]:7.3f}s]")
            print(f"耗时: {elapsed:.1f}s")

    # 保存
    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        if not args.quiet:
            print(f"\n[INFO] 结果已保存至: {args.output}")
    else:
        print("\n" + json.dumps(output, ensure_ascii=False, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())