import json
import os
import time
from pathlib import Path
from cs336_basics.bpe import train_bpe

# ==========================================
# 1. 确定项目根目录
# ==========================================
def get_project_root() -> Path:
    """基于当前文件的位置查找项目根目录（含有 pyproject.toml 或 .git）"""
    current = Path(__file__).resolve().parent
    # 如果当前文件就在根目录，直接返回
    if (current / "pyproject.toml").exists() or (current / ".git").exists():
        return current
    for parent in current.parents:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    # 保底：假设脚本位于根目录
    return current

PROJECT_ROOT = get_project_root()

# ==========================================
# 2. 固定的配置区域（路径使用相对于根目录的写法）
# ==========================================
OUTPUT_DIR = PROJECT_ROOT / "data" / "bpe_outputs"   # 输出目录也基于根目录

TASKS = [
    {
        "name": "TinyStories",
        "input_relpath": "/root/autodl-tmp/data/TinyStoriesV2-GPT4-train.txt",   # 相对于根目录
        "vocab_size": 10000,
        "special_tokens": ["<|endoftext|>"]
    },
    {
        "name": "OpenWebText",
        "input_relpath": "/root/autodl-tmp/data/owt_valid.txt",
        "vocab_size": 32000,
        "special_tokens": ["<|endoftext|>"]
    }
]

# ==========================================
# 3. 辅助函数（保持不变）
# ==========================================
def save_vocab_json(vocab: dict[int, bytes], filepath: Path):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    json_vocab = {str(k): v.decode("latin-1") for k, v in vocab.items()}
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(json_vocab, f, ensure_ascii=False, indent=2)

def save_merges_json(merges: list[tuple[bytes, bytes]], filepath: Path):
    """将 merges 保存为 JSON 文件，每个合并规则存储为 [a_str, b_str]"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    # 将 bytes 解码为 latin-1 字符串（保证可逆）
    merges_list = [[a.decode("latin-1"), b.decode("latin-1")] for a, b in merges]
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(merges_list, f, ensure_ascii=False, indent=2)
# ==========================================
# 4. 主训练程序
# ==========================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Output directory: {OUTPUT_DIR}")

    for task in TASKS:
        input_path = Path(task["input_relpath"])
        vocab_size = task["vocab_size"]
        special_tokens = task["special_tokens"]
        name = task["name"]

        if not input_path.exists():
            print(f"[!] Skipping {name}: Input file '{input_path}' not found.")
            continue

        print(f"\n================ Training {name} ================")
        print(f"[*] Input Path: {input_path}")
        print(f"[*] Vocab Size: {vocab_size}")

        start_time = time.time()
        vocab, merges = train_bpe(
            input_path=str(input_path),   # train_bpe 目前接受字符串路径
            vocab_size=vocab_size,
            special_tokens=special_tokens
        )
        elapsed = time.time() - start_time
        print(f"[*] Training finished in {elapsed:.2f} seconds.")
        print(f"[*] Resulting vocab size: {len(vocab)}")
        print(f"[*] Number of merges: {len(merges)}")

        vocab_out = OUTPUT_DIR / f"{name.lower()}_vocab.json"
        merges_out = OUTPUT_DIR / f"{name.lower()}_merges.json"

        save_vocab_json(vocab, vocab_out)
        print(f"[*] Saved vocabulary JSON to {vocab_out}")

        save_merges_json(merges, merges_out)
        print(f"[*] Saved merges JSON to {merges_out}")

if __name__ == "__main__":
    main()