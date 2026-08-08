import os
import json
import time
import numpy as np
from pathlib import Path
from bpe import Tokenizer
from bpe import parallel_encode_file, load_vocab_json, load_merges_json
# ==========================================
# 配置区域
# ==========================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OWT_INPUT = "/root/autodl-tmp/data/owt_train.txt"


OWT_VOCAB = os.path.join(BASE_DIR, "data/bpe_outputs", "openwebtext_vocab.json")
OWT_MERGES = os.path.join(BASE_DIR, "data/bpe_outputs", "openwebtext_merges.json")

OUTPUT_DIR = "/root/autodl-tmp/data"


# ==========================================
# 实验主环节
# ==========================================
def main():
    print("="*60)
    print("      CS336 Assignment 1 - encoder")
    print("="*60)

    #OpenWebText 很大，改用多进程并行编码 + 批量写入二进制文件
    print("[*] OpenWebText 数据集较大，采用多进程并行编码并保存为二进制文件...")
    owt_bin_path = os.path.join(OUTPUT_DIR, "owt_encoded_train.bin")
    txt_bytes = os.path.getsize(OWT_INPUT)
    start_t = time.time()
    total_tokens = parallel_encode_file(
        input_path=OWT_INPUT,
        vocab_json_path=OWT_VOCAB,
        merges_json_path=OWT_MERGES,
        output_bin_path=owt_bin_path,
        special_tokens=["<|endoftext|>"],
        num_processes=6,
        chunk_target_size=100 * 1024 * 1024   # 每块 100MB
    )
    end_t = time.time()
    print(f"    -> 已保存 {total_tokens} 个 token 至 {owt_bin_path} (二进制格式，每个 token 占 2 字节)")
    print("    (如需加载为 NumPy 数组，可使用 np.fromfile('openwebtext_encoded.bin', dtype=np.uint16))")
    elapsed = end_t - start_t
    throughput_bps = txt_bytes / elapsed
    throughput_mbs = throughput_bps / (1024 * 1024)

    print(f"[*] 测算多进程分词吞吐量: {throughput_bps:.2f} Bytes/s  (约为 {throughput_mbs:.2f} MB/s)")
    print(f"[*] 处理 825GB The Pile 预估时间: {825 * 1024 / throughput_mbs / 3600:.2f} 小时")

    


if __name__ == '__main__':
    main()