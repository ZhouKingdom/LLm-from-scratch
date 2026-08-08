# LLM from Scratch — 从零实现大语言模型训练与推理系统

> 一个不依赖任何高层 Transformer 库，从零实现的大语言模型（LLM）完整训练与推理系统。
> 覆盖 **BPE 分词 → Transformer 建模 → 训练优化 → 文本生成** 全链路。

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-orange.svg)]()
[![Tests](https://img.shields.io/badge/tests-47%20passed-brightgreen.svg)]()

---

## ✨ 项目亮点

- 🧱 **完全从零实现**：不依赖 `transformers` 等高层库，手写 Transformer 每一个组件
- 📊 **真实训练成果**：217M 参数模型训练 10,000 步，验证困惑度降至 **29.1**
- 🚀 **显存优化**：bf16 + FlashAttention + 分块交叉熵，峰值显存从 16.3GB 降至 **5.5GB**
- 🧪 **测试严谨**：47 个单元测试，与 PyTorch 参考实现数值对齐
- 💬 **可生成文本**：自回归生成连贯英文，支持 KV Cache 加速（~120 token/s）

---

## 🏗️ 项目结构

```
.
├── cs336_basics/              # 核心包
│   ├── bpe.py                 # BPE 分词器（训练 / 编解码 / 多进程）
│   ├── data.py                # 数据批处理 get_batch
│   ├── model.py               # Transformer 模型类
│   ├── nn_utils.py            # 注意力 / 交叉熵等工具函数
│   ├── optimizer.py           # AdamW / 学习率调度 / 梯度裁剪
│   ├── encoder.py             # 大语料并行编码（文本 → token）
│   ├── generate.py            # 文本生成（KV Cache + Top-p 采样）
│   ├── train.py               # 训练主循环
│   └── train_bpe_model.py     # BPE 词表训练入口
├── data/bpe_outputs/          # 训练好的词表 + 合并规则
├── tests/                     # 47 个单元测试
├── docs/                      # 技术文档
├── train.sh                   # 训练一键脚本
├── generate.sh                # 生成一键脚本
└── pyproject.toml             # 项目配置（uv 管理）
```

### 分层架构

```
┌────────────────────────────────────────────┐
│  推理层  generate.py（自回归生成 + KV Cache）│
├────────────────────────────────────────────┤
│  训练层  train.py（训练循环 / 断点续训）     │
├────────────────────────────────────────────┤
│  优化层  optimizer.py（AdamW / 调度 / 裁剪）│
├────────────────────────────────────────────┤
│  模型层  model.py + nn_utils.py（Transformer）│
├────────────────────────────────────────────┤
│  数据层  bpe.py + data.py（分词 / 批处理）  │
└────────────────────────────────────────────┘
```

---

## 🧠 从零实现的组件

| 组件 | 说明 |
|------|------|
| **BPE 分词器** | 字节级词表训练、多进程并行、编解码、流式编码 |
| **RMSNorm** | 均方根归一化（LLaMA 同款） |
| **RoPE** | 旋转位置编码，天然相对位置，无需可学习参数 |
| **SwiGLU FFN** | 门控前馈网络 |
| **多头因果注意力** | 支持 FlashAttention（fused）融合加速 |
| **预归一化残差块** | Pre-norm Transformer Block |
| **AdamW** | 解耦权重衰减优化器 |
| **余弦学习率调度** | 线性预热 + 余弦退火 |
| **梯度裁剪** | 全局 L2 范数裁剪 |

---

## ⚡ 显存优化（实测数据）

| 优化手段 | 作用 |
|---------|------|
| bf16 混合精度（AMP） | 激活值减半 |
| FlashAttention（fused） | 注意力显存 O(n²) → O(n) |
| 分块交叉熵损失 | 避免 materialize 完整 logits 张量 |
| 梯度累积 | 小批量凑大有效批量 |

**实测**：91.6M 模型峰值显存从 16.3GB 降至 5.5GB（节省约 66%），使 217M 参数模型可在单张 24GB 显卡上训练。

---

## 🚀 快速开始

### 1. 环境准备

```bash
# 安装 uv（项目依赖管理）
pip install uv

# 安装依赖
uv sync
```

### 2. 训练 BPE 词表

```bash
uv run python -m cs336_basics.train_bpe_model
```

### 3. 编码语料（文本 → token 二进制）

```bash
uv run python -m cs336_basics.encoder
```

### 4. 训练模型

```bash
./train.sh
```

`train.sh` 默认配置（方案 B：217M 参数）：

```
d_model=1024, num_layers=12, num_heads=16, d_ff=2752, vocab=32000
batch_size=16, gradient_accumulation=4, total_steps=10000
bf16 混合精度 + FlashAttention + 分块交叉熵
```

### 5. 生成文本

```bash
./generate.sh
```

---

## 📊 训练成果（真实数据）

217M 参数模型，OpenWebText 子集，10,000 步训练：

| 步数 | 验证损失 | 困惑度 |
|-----:|--------:|-------:|
| 3,000 | 3.78 | 43.7 |
| 5,000 | 3.56 | 35.3 |
| 7,000 | 3.44 | 31.3 |
| 9,000 | 3.40 | 29.8 |
| **10,000** | **3.37** | **29.1** |

### 生成效果示例

> **Prompt**：*"LOUISVILLE, Ky. — A few unflattering reviews are to be expected with any hotel, particularly one whose rates start at $49 per night..."*

> **模型续写**：*"...about as widespread as a lot of hotel rooms across the country. The hotel industry is a little more complex. With the long hours, the hotel industry has become more..."*

模型学会了新闻评论文体，能产出语法正确、语义连贯的英文。

---

## 🧪 测试

```bash
uv run pytest
```

- 47 个单元测试全部通过
- 通过快照（snapshot）与 PyTorch 参考实现做数值对比（容差 ~1e-6）
- 分词器与 tiktoken 参考实现对比

---

## 📚 文档

- [项目技术讲解文档](./docs/PROJECT_TECHNICAL_DOCUMENTATION.md)
- [训练规模参照表](./docs/TRAINING_SCALE_REFERENCE.md)
- [知名大模型公开数据参照](./docs/FAMOUS_MODEL_REFERENCE.md)

---

## 🛠️ 技术栈

- **Python** ≥ 3.11
- **PyTorch** 2.6（自动微分、FlashAttention、bf16）
- **NumPy**（数据、内存映射）
- **uv**（环境与依赖管理）

---

## 📄 License

MIT

---

*本项目基于斯坦福 CS336 课程作业 1（Basics）实现，作为个人大模型原理学习与工程实践项目。*

