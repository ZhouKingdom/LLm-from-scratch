#!/bin/bash
set -euo pipefail

# Use the project's uv environment (Python >=3.11, torch 2.6) - required for
# the cs336_basics package to be importable and for bf16/FlashAttention support.
#
# 模型规模：方案 B（从 91.6M 扩到 217M）
#   d_model=1024, num_layers=12, num_heads=16, d_ff=2752（= 8/3*d_model 向上取 64 倍数）
#   实测峰值显存 ~19.7GB（24GB 卡安全）
#
# RESUME:
#   --resume <路径>   # 指定检查点续训
#   --resume-latest   # 自动用最新检查点
#   ⚠️ 架构参数已改变（91M → 217M），旧 checkpoint 不兼容，必须从头训练！
#      当前为全新架构训练（未开启 resume）。之后要续训时，把下面这行取消注释：
#      --resume-latest
uv run python -m cs336_basics.train \
--train-data /root/autodl-tmp/data/owt_train_encoded.bin \
--val-data /root/autodl-tmp/data/owt_val_encoded.bin \
--vocab-size 32000 \
--context-length 1024 \
--d-model 1024 \
--d-ff 2752 \
--num-layers 12 \
--num-heads 16 \
--batch-size 16 \
--gradient-accumulation-steps 4 \
--total-steps 10000 \
--max-lr 6e-4 \
--min-lr 6e-5 \
--warmup-iters 500 \
--gradient-clip 1.0 \
--save-dir /root/autodl-tmp/data/checkpoints/owt \
--device cuda \
--amp \
--attn-impl fused \
--loss-chunks 8 \
--resume-latest