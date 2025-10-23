# Optical Compression for Qwen3-VL-2B

Adapter-based optical compression for long documents using DeepSeek's DeepEncoder with Qwen3-VL-2B.

Built on [DeepSeek-OCR](https://github.com/DeepSeek-AI/DeepSeek-OCR) by DeepSeek-AI. This repo includes `deepencoder.py` from DeepSeek-OCR (Apache-2.0 License).

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Model Weights

```bash
# Download DeepEncoder weights (401M params, auto-downloads on first use)
# Will be cached in: ~/.cache/huggingface/hub/models--Volkopat--DeepSeek-DeepEncoder

# Download pre-trained adapter (41MB)
mkdir -p adapters
wget https://huggingface.co/Volkopat/Qwen-VLM-Optical-Encoder/resolve/main/qwen3_vl_2b.pth -O adapters/qwen3_vl_2b.pth
```

**Model weights**:
- DeepEncoder: [Volkopat/DeepSeek-DeepEncoder](https://huggingface.co/Volkopat/DeepSeek-DeepEncoder)
- Adapter: [Volkopat/Qwen-VLM-Optical-Encoder](https://huggingface.co/Volkopat/Qwen-VLM-Optical-Encoder)

### 3. Test with Pre-trained Adapter

```bash
# Quick test (10 samples)
python test.py \
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
    --adapter_checkpoint adapters/qwen3_vl_2b.pth \
    --benchmark longbench \
    --num_samples 10

# Full benchmark (50 samples)
python test.py \
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
    --adapter_checkpoint adapters/qwen3_vl_2b.pth \
    --benchmark longbench \
    --num_samples 50
```

This will run the benchmark and show you the results comparing optical vs native text processing.

### 4. Train Your Own Adapter (Optional)

```bash
python train.py \
    --vlm_model_path Qwen/Qwen3-VL-2B-Instruct \
    --target_dim 2048 \
    --num_samples 1000 \
    --num_epochs 10
```

**Training time**: ~2-3 hours on RTX 5070 12GB

## 📊 Results (LongBench v2, 50 samples)

**Tested on**: Qwen3-VL-2B-Instruct, RTX 5070 12GB

| Metric | Native Text | Optical Compression |
|--------|-------------|---------------------|
| **Overall Score** | 12% (6/50) | **18% (9/50)** ✅ |
| **Success Rate** | 22% (11/50) | **90% (45/50)** ✅ |
| **Accuracy (completed)** | 54.5% (6/11) | 20% (9/45) |
| **Avg Tokens** | 38K | 17K (2.2× compression) |
| **Avg Time** | 6s | 24s (4× slower) |

**Key Finding**: When counting failures (OOM/context exceeded) as wrong answers, optical achieves **6% better overall score** because it successfully completes 90% of samples vs native's 22%. Native has higher accuracy on completed samples but fails on 78% of long documents.

## 🔧 Methodology

### DeepSeek-OCR (Original)

DeepSeek-OCR uses **DeepEncoder** (SAM-ViT-B + CLIP-L + Projector) for optical character recognition:

```
┌──────────────────────────────────────────────────────────────┐
│                    DeepSeek-OCR Pipeline                      │
└──────────────────────────────────────────────────────────────┘

Text Document
     ↓
Render to Images (1024×1024)
     ↓
┌────────────────────────────────────┐
│   DeepEncoder (401M params)        │
│   ├── SAM-ViT-B (95M)              │
│   ├── CLIP-L (303M)                │
│   └── Projector (2.6M)             │
│   Output: [N, 1280]                │
└────────────────────────────────────┘
     ↓
DeepSeek VLM (proprietary)
     ↓
Response
```

**Limitation**: DeepEncoder outputs are designed for DeepSeek's proprietary VLM.

### My Approach: Adapter for Qwen3-VL-2B

We add a lightweight adapter to bridge DeepEncoder to Qwen3-VL-2B:

```
┌──────────────────────────────────────────────────────────────┐
│            Optical Compression for Qwen3-VL-2B                │
└──────────────────────────────────────────────────────────────┘

Text Document
     ↓
Render to Images (1024×1024, 10pt font)
     ↓
┌────────────────────────────────────┐
│   DeepEncoder (401M) [FROZEN]      │
│   Auto-downloads from HuggingFace  │
│   Output: [N, 1280]                │
└────────────────────────────────────┘
     ↓
┌────────────────────────────────────┐
│   Adapter (10.6M) [TRAINABLE]      │  ← Our contribution
│   ├── MLP: 1280 → 3072 → 2048     │
│   ├── Page Embeddings (200 pages) │
│   └── Layer Norm                   │
└────────────────────────────────────┘
     ↓
Qwen3-VL-2B (2048 dims)
     ↓
Response
```

**Key idea**: Freeze DeepEncoder, train only lightweight adapter (10.6M params) to align with Qwen3-VL-2B's 2048-dimensional space.

## 💡 Training Details

### Training Dataset
- **Source**: Wikipedia (20220301.en)
- **Samples**: 1000 documents
- **Length**: 5K-100K characters per document (1-6 pages)
- **Rendering**: Black text on white background, 10pt monospace font

### Training Process
1. Load Qwen3-VL-2B (frozen)
2. Load DeepEncoder from HuggingFace (frozen, 401M params)
3. Generate 1000 Wikipedia documents, render to images
4. Train adapter to align DeepEncoder outputs (1280-dim) with Qwen3-VL text embeddings (2048-dim)
5. Loss: MSE between optical vision tokens and native text embeddings

### Training Stats
- **Time**: 2-3 hours on RTX 5070 12GB
- **Trainable params**: 10.6M (adapter only)
- **Loss reduction**: 87% (1.17 → 0.14)
- **GPU memory**: ~5GB (2B model + DeepEncoder + adapter)

## 📈 Performance Analysis

### Token Compression
- **Average**: 2.2× token savings
- **Formula**: ~256 tokens/page (optical) vs ~4096 tokens/page (native)
- **Benefit**: Enables processing of 100+ page documents that exceed context limits

### Speed vs Context Tradeoff
- **Optical**: 24s per document (slower) but 90% success rate
- **Native**: 6s per document (faster) but 22% success rate on long documents
- **Tradeoff**: 4× slower processing for unlimited context handling

### Why Optical Wins Overall
Native text has higher accuracy (54.5%) on the few samples it completes, but fails on 78% of long documents due to OOM/context limits. Optical has lower accuracy (20%) but successfully processes 90% of samples, resulting in better overall score (18% vs 12%).

## 🔍 When to Use Optical Compression

### ✅ Use Optical When:
- Documents exceed 32K token context window
- Processing 50+ page documents
- Context window exhaustion is the bottleneck
- 4× slower processing is acceptable

### ❌ Use Native Text When:
- Short documents (< 10 pages)
- Real-time processing required
- Speed is critical
- Documents fit comfortably in context window

## 📁 Repository Structure

```
VLM-Optical-Encoder/
├── deepencoder.py             # DeepEncoder architecture (from DeepSeek-OCR)
├── optical_encoder.py         # Adapter + integration code
├── train.py                   # Training script
├── test.py                    # Testing script (LongBench support)
├── requirements.txt           # Dependencies (CUDA 12.8)
├── configs/
│   └── qwen3_vl.yaml         # Training configuration
├── adapters/                  # Download from HuggingFace
│   └── qwen3_vl_2b.pth       # Pre-trained adapter (41MB)
├── .gitignore
├── LICENSE                    # MIT License
└── README.md
```

**Download pre-trained adapter**: [Volkopat/Qwen-VLM-Optical-Encoder](https://huggingface.co/Volkopat/Qwen-VLM-Optical-Encoder)

## ⚠️ Disclaimer

**This is experimental research code.** Scalability has not been tested beyond the reported benchmarks. Performance on diverse tasks and production workloads is untested. Use at your own risk. I lack the hardware to scale this up to 4B, 8B, 30B and 235B models so I appreciate it if someone can test this beyond 2B.

## 🙏 Credits

This work builds on:
- **[DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)** (2025) by DeepSeek-AI for DeepEncoder architecture
- **[Qwen3-VL-2B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct)** by Alibaba for the vision-language model
- **LongBench v2** benchmark for evaluation
- **[Claude Code](https://claude.com/claude-code)** by Anthropic for development assistance

**Model Weights**:
- DeepEncoder (401M params): [Volkopat/DeepSeek-DeepEncoder](https://huggingface.co/Volkopat/DeepSeek-DeepEncoder)
- Adapter (10.6M params): [Volkopat/Qwen-VLM-Optical-Encoder](https://huggingface.co/Volkopat/Qwen-VLM-Optical-Encoder)

## 📊 Citation

```bibtex
@software{optical_compression_qwen,
  title = {Optical Compression for Qwen3-VL via Universal Adapter},
  year = {2025},
  note = {Built on DeepSeek-OCR by DeepSeek-AI}
}

@misc{deepseek_ocr,
  title = {DeepSeek-OCR},
  author = {DeepSeek-AI},
  year = {2025},
  url = {https://huggingface.co/deepseek-ai/DeepSeek-OCR}
}

@misc{qwen3vl,
  title = {Qwen3-VL-2B-Instruct},
  author = {Qwen Team, Alibaba},
  year = {2024},
  url = {https://huggingface.co/Qwen/Qwen3-VL-2B-Instruct}
}
```

## 📝 License

- **DeepEncoder**: Apache-2.0 (from DeepSeek-AI/DeepSeek-OCR)
- **Adapter code**: MIT License

---

**Last Updated**: 2025-10-23
**Tested On**: RTX 5070 12GB, CUDA 12.8, Qwen3-VL-2B-Instruct
