#!/usr/bin/env python3
"""声纹管理 — 注册、存储、比对。

基于 FunASR cam++ 说话人模型，提取 192 维声纹嵌入 (speaker embedding)。

用法:
    # 注册声纹
    python voiceprint.py register --name "张三" audio.wav
    python voiceprint.py register --name "李四" audio2.wav

    # 列出已注册声纹
    python voiceprint.py list

    # 比对: 给定音频，匹配已注册声纹
    python voiceprint.py match audio.wav

    # 删除声纹
    python voiceprint.py delete --name "张三"

    # 1:1 验证 (两段音频是否是同一个人)
    python voiceprint.py verify audio1.wav audio2.wav

声纹存储位置: ./voiceprint_db/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch


# --- 声纹数据库 ---
DEFAULT_DB_DIR = Path(__file__).parent / "voiceprint_db"


class VoiceprintDB:
    """声纹数据库: 存储 name → embedding 映射。"""

    def __init__(self, db_dir: Path = DEFAULT_DB_DIR):
        self.db_dir = Path(db_dir) if db_dir else DEFAULT_DB_DIR
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.db_dir / "index.json"
        self._load()

    def _load(self):
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {}  # name → {embedding_file, created_at, audio_file}

    def _save(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def register(self, name: str, embedding: np.ndarray, audio_file: str = ""):
        """注册声纹。"""
        emb_file = self.db_dir / f"{name}.npy"
        np.save(emb_file, embedding)
        self.index[name] = {
            "embedding_file": str(emb_file),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "audio_file": audio_file,
        }
        self._save()
        return True

    def delete(self, name: str):
        """删除声纹。"""
        if name not in self.index:
            return False
        emb_file = Path(self.index[name]["embedding_file"])
        if emb_file.exists():
            emb_file.unlink()
        del self.index[name]
        self._save()
        return True

    def list_all(self) -> list:
        """列出所有已注册声纹。"""
        return [
            {"name": name, "created_at": info["created_at"], "audio_file": info.get("audio_file", "")}
            for name, info in self.index.items()
        ]

    def get_embedding(self, name: str) -> np.ndarray | None:
        """获取已注册声纹嵌入。"""
        if name not in self.index:
            return None
        emb_file = Path(self.index[name]["embedding_file"])
        if not emb_file.exists():
            return None
        return np.load(emb_file)

    def get_all(self) -> list[tuple[str, np.ndarray]]:
        """获取所有已注册 (name, embedding) 对。"""
        result = []
        for name in self.index:
            emb = self.get_embedding(name)
            if emb is not None:
                result.append((name, emb))
        return result

    def __len__(self):
        return len(self.index)


# --- 声纹引擎 ---
class VoiceprintEngine:
    """声纹提取与比对引擎。"""

    def __init__(self, device: str = "cpu", hub: str = "ms"):
        self.device = device
        self.hub = hub
        self.model = None
        self._load_model()

    def _load_model(self):
        from funasr import AutoModel

        self.model = AutoModel(
            model="cam++",
            device=self.device,
            hub=self.hub,
        )

    def extract(self, audio_path: str) -> np.ndarray:
        """从音频中提取声纹嵌入 (192 维)。"""
        res = self.model.generate(
            input=[audio_path],
            cache={},
            batch_size=1,
            device=self.device,
        )
        if not res or "spk_embedding" not in res[0]:
            raise RuntimeError(f"无法提取声纹: {audio_path}")
        emb = res[0]["spk_embedding"].detach().cpu().numpy()
        if emb.ndim > 1:
            emb = emb.flatten()
        return emb

    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """余弦相似度。范围 [-1, 1], 越高越相似。"""
        dot = float(np.dot(emb1, emb2))
        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot / (norm1 * norm2)

    def match(self, audio_path: str, db: VoiceprintDB, threshold: float = 0.55) -> dict:
        """匹配音频到已注册声纹。

        Args:
            audio_path: 输入音频
            db: 声纹数据库
            threshold: 匹配阈值 (0.55 适合 cam++)

        Returns:
            {"matched": bool, "name": str, "score": float, "all_scores": [...]}
        """
        emb = self.extract(audio_path)
        all_entries = db.get_all()

        if not all_entries:
            return {"matched": False, "name": None, "score": 0.0, "all_scores": []}

        scores = []
        for name, stored_emb in all_entries:
            score = self.cosine_similarity(emb, stored_emb)
            scores.append({"name": name, "score": round(float(score), 4)})

        scores.sort(key=lambda x: x["score"], reverse=True)
        best = scores[0]

        return {
            "matched": best["score"] >= threshold,
            "name": best["name"] if best["score"] >= threshold else None,
            "score": best["score"],
            "all_scores": scores,
        }

    def verify(self, audio1: str, audio2: str, threshold: float = 0.55) -> dict:
        """1:1 验证两段音频是否为同一说话人。"""
        emb1 = self.extract(audio1)
        emb2 = self.extract(audio2)
        score = self.cosine_similarity(emb1, emb2)
        return {
            "same_speaker": score >= threshold,
            "score": round(float(score), 4),
            "threshold": threshold,
        }


# --- CLI ---
def main():
    parser = argparse.ArgumentParser(
        description="声纹管理 — 注册、存储、比对",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python voiceprint.py register --name "张三" audio.wav
    python voiceprint.py list
    python voiceprint.py match audio.wav
    python voiceprint.py verify audio1.wav audio2.wav
    python voiceprint.py delete --name "张三"
""",
    )

    sub = parser.add_subparsers(dest="command", help="操作类型")

    # register
    p_reg = sub.add_parser("register", help="注册声纹")
    p_reg.add_argument("--name", required=True, help="说话人姓名/ID")
    p_reg.add_argument("audio", help="注册音频文件 (>=3秒)")
    p_reg.add_argument("--db-dir", default=None, help="声纹库目录")

    # match
    p_match = sub.add_parser("match", help="匹配说话人")
    p_match.add_argument("audio", help="待识别音频文件")
    p_match.add_argument("--threshold", type=float, default=0.55, help="匹配阈值 (默认: 0.55)")
    p_match.add_argument("--db-dir", default=None, help="声纹库目录")

    # verify
    p_ver = sub.add_parser("verify", help="1:1 验证两段音频是否同一人")
    p_ver.add_argument("audio1", help="音频 1")
    p_ver.add_argument("audio2", help="音频 2")
    p_ver.add_argument("--threshold", type=float, default=0.55, help="阈值 (默认: 0.55)")

    # list
    p_list = sub.add_parser("list", help="列出已注册声纹")
    p_list.add_argument("--db-dir", default=None, help="声纹库目录")

    # update (重新注册覆盖)
    p_upd = sub.add_parser("update", help="更新声纹 (重新注册)")
    p_upd.add_argument("--name", required=True, help="说话人姓名/ID")
    p_upd.add_argument("audio", help="新的注册音频")
    p_upd.add_argument("--db-dir", default=None, help="声纹库目录")

    # delete
    p_del = sub.add_parser("delete", help="删除声纹")
    p_del.add_argument("--name", required=True, help="说话人姓名/ID")
    p_del.add_argument("--db-dir", default=None, help="声纹库目录")

    # info (查看单个声纹详情)
    p_info = sub.add_parser("info", help="查看声纹详情")
    p_info.add_argument("--name", required=True, help="说话人姓名/ID")
    p_info.add_argument("--db-dir", default=None, help="声纹库目录")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    db_dir = Path(args.db_dir) if getattr(args, "db_dir", None) else DEFAULT_DB_DIR
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    engine = VoiceprintEngine(device=device)

    if args.command == "register":
        print(f"[INFO] 提取声纹: {args.audio}")
        t0 = time.time()
        emb = engine.extract(args.audio)
        print(f"[INFO] 声纹提取完成 ({time.time() - t0:.1f}s), 嵌入维度: {emb.shape}")

        db = VoiceprintDB(db_dir)
        db.register(args.name, emb, audio_file=os.path.abspath(args.audio))
        print(f"[OK] 已注册: {args.name}")
        print(f"[INFO] 当前声纹库共 {len(db)} 人")

    elif args.command == "match":
        print(f"[INFO] 匹配: {args.audio}")
        db = VoiceprintDB(db_dir)
        if len(db) == 0:
            print("[WARN] 声纹库为空，请先注册: python voiceprint.py register --name NAME audio.wav")
            return 1
        t0 = time.time()
        result = engine.match(args.audio, db, threshold=args.threshold)
        elapsed = time.time() - t0
        print(f"[INFO] 匹配完成 ({elapsed:.1f}s)")
        print(f"\n{'=' * 50}")
        if result["matched"]:
            print(f"  匹配成功: {result['name']} (相似度: {result['score']:.4f})")
        else:
            print(f"  未匹配 (最高相似度: {result['score']:.4f} < 阈值 {args.threshold})")
        print(f"\n  所有候选:")
        for s in result["all_scores"]:
            flag = " ✅" if s["score"] >= args.threshold else ""
            print(f"    {s['name']:20s}  {s['score']:.4f}{flag}")
        print(f"{'=' * 50}")

    elif args.command == "verify":
        print(f"[INFO] 验证: {args.audio1} vs {args.audio2}")
        t0 = time.time()
        result = engine.verify(args.audio1, args.audio2, threshold=args.threshold)
        elapsed = time.time() - t0
        print(f"[INFO] 验证完成 ({elapsed:.1f}s)")
        print(f"\n{'=' * 50}")
        if result["same_speaker"]:
            print(f"  ✅ 同一说话人 (相似度: {result['score']:.4f})")
        else:
            print(f"  ❌ 不同说话人 (相似度: {result['score']:.4f})")
        print(f"{'=' * 50}")

    elif args.command == "list":
        db = VoiceprintDB(db_dir)
        entries = db.list_all()
        if not entries:
            print("[INFO] 声纹库为空")
        else:
            print(f"\n已注册声纹 ({len(entries)} 人):")
            print(f"{'姓名':<20} {'注册时间':<22} {'注册音频'}")
            print("-" * 70)
            for e in entries:
                print(f"{e['name']:<20} {e['created_at']:<22} {e['audio_file']}")

    elif args.command == "update":
        print(f"[INFO] 更新声纹: {args.name} <- {args.audio}")
        t0 = time.time()
        emb = engine.extract(args.audio)
        print(f"[INFO] 声纹提取完成 ({time.time() - t0:.1f}s)")

        db = VoiceprintDB(db_dir)
        if args.name not in db.index:
            print(f"[WARN] 不存在: {args.name}, 将作为新注册")
        db.register(args.name, emb, audio_file=os.path.abspath(args.audio))
        print(f"[OK] 已更新: {args.name}")

    elif args.command == "info":
        db = VoiceprintDB(db_dir)
        if args.name not in db.index:
            print(f"[WARN] 未找到: {args.name}")
            return 1
        info = db.index[args.name]
        emb = db.get_embedding(args.name)
        print(f"\n声纹信息: {args.name}")
        print(f"  注册时间: {info['created_at']}")
        print(f"  注册音频: {info.get('audio_file', 'N/A')}")
        print(f"  嵌入文件: {info['embedding_file']}")
        print(f"  嵌入维度: {emb.shape}")
        print(f"  嵌入范数: {float(np.linalg.norm(emb)):.4f}")

    elif args.command == "delete":
        db = VoiceprintDB(db_dir)
        if db.delete(args.name):
            print(f"[OK] 已删除: {args.name}")
        else:
            print(f"[WARN] 未找到: {args.name}")

    return 0


if __name__ == "__main__":
    sys.exit(main())