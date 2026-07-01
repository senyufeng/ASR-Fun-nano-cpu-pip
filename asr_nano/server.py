#!/usr/bin/env python3
"""ASR-Fun-Nano CPU 常驻服务 — 模型加载后不释放。

模式:
    python -m asr_nano.server --stdin    交互模式
    python -m asr_nano.server --http     HTTP 服务
    python -m asr_nano.server --batch    批量模式

HTTP API:
    POST /transcribe   body: {"files": ["/path/audio.mp4"]}
    GET  /health       -> {"status": "ready"}
"""

import argparse
import json
import os
import sys
import time

from asr_nano.engine import ASREngine


def interactive_mode(engine: ASREngine):
    """交互模式: 逐行输入路径, 逐行输出 JSON。"""
    print("[INFO] 模型已加载, 常驻就绪.")
    print("[INFO] 输入音频/视频文件路径, /q 退出, /h 帮助.\n")
    count = 0
    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue
        if line in ("/q", "/quit", "/exit"):
            break
        if line in ("/h", "/help"):
            print("  输入音频/视频文件路径开始推理")
            print("  /q 退出, /h 帮助")
            continue
        if not os.path.isfile(line):
            print(json.dumps({"error": "file not found", "path": line}, ensure_ascii=False))
            continue
        t0 = time.time()
        results = engine.transcribe([line])
        elapsed = time.time() - t0
        r = results[0]
        r.elapsed_sec = round(elapsed, 2)
        print(ASREngine.to_json(results, indent=None))
        count += 1
    print(f"\n[INFO] 退出. 共处理 {count} 个文件.")


def batch_mode(engine: ASREngine, files: list[str]):
    """批量模式: 一次性处理多个文件。"""
    print(f"[INFO] 批量推理 {len(files)} 个文件...")
    t0 = time.time()
    results = engine.transcribe(files)
    total_elapsed = time.time() - t0
    for i, r in enumerate(results):
        r.elapsed_sec = round(total_elapsed / len(files), 2)
    print(ASREngine.to_json(results))


def http_mode(engine: ASREngine, host: str, port: int):
    """HTTP 常驻服务。"""
    from http.server import HTTPServer, BaseHTTPRequestHandler

    class ASRHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            if self.path != "/transcribe":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                req = json.loads(body)
                files = req.get("files", [])
            except json.JSONDecodeError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error":"invalid json"}')
                return
            t0 = time.time()
            results = engine.transcribe(files)
            elapsed = time.time() - t0
            for i, r in enumerate(results):
                r.elapsed_sec = round(elapsed / max(1, len(files)), 2)
            resp = ASREngine.to_json(results).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", len(resp))
            self.end_headers()
            self.wfile.write(resp)

        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"status":"ready"}')
            else:
                self.send_response(404)
                self.end_headers()

        def log_message(self, fmt, *args):
            pass

    server = HTTPServer((host, port), ASRHandler)
    print(f"[INFO] HTTP 服务: http://{host}:{port}")
    print(f"[INFO] POST /transcribe  GET /health")
    print(f"[INFO] 模型常驻, Ctrl+C 退出.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] 关闭服务.")
        server.shutdown()


def main():
    """CLI entry point for asr_nano.server."""
    parser = argparse.ArgumentParser(description="ASR-Fun-Nano CPU 常驻服务")
    parser.add_argument("--stdin", action="store_true", help="交互模式")
    parser.add_argument("--batch", action="store_true", help="批量模式")
    parser.add_argument("--http", action="store_true", help="HTTP 服务模式")
    parser.add_argument("--host", default="127.0.0.1", help="HTTP 绑定地址")
    parser.add_argument("--port", type=int, default=8899, help="HTTP 端口")
    parser.add_argument("--voiceprint", action="store_true", help="启用声纹")
    parser.add_argument("--voiceprint-db", default=None, help="声纹库目录")
    parser.add_argument("--files", nargs="*", default=None, help="批量模式: 文件列表")
    args = parser.parse_args()

    engine = ASREngine(
        model="FunAudioLLM/Fun-ASR-Nano-2512",
        device="cpu",
        language="中文",
        enable_voiceprint=args.voiceprint,
        voiceprint_db_dir=args.voiceprint_db,
    )

    print("[INFO] 加载模型 (仅一次, 加载后常驻)...")
    t0 = time.time()
    engine.load()
    print(f"[INFO] 加载完成 ({time.time() - t0:.1f}s), 模型常驻内存.\n")

    if args.stdin:
        interactive_mode(engine)
    elif args.http:
        http_mode(engine, args.host, args.port)
    elif args.files:
        batch_mode(engine, args.files)
    else:
        interactive_mode(engine)


if __name__ == "__main__":
    main()