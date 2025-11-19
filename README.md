# StoryHash: Lightweight Multi-Modal Fusion for Robust Video Retrieval

[![Paper](https://img.shields.io/badge/Paper-arXiv-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)
[![Conference](https://img.shields.io/badge/CVPR-2026-blue.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **When Action Features Boost Confidence, Not Detection**

Official implementation of our paper on multi-modal video retrieval with systematic robustness evaluation and ablation studies.

---

## 📄 Paper

**Title**: Lightweight Multi-Modal Fusion for Robust Video Retrieval: When Action Features Boost Confidence, Not Detection

**Authors**: Denis Billi

**Abstract**: Video content retrieval systems must handle realistic perturbations like cropping, compression, and speed changes while maintaining semantic accuracy. We present **StoryHash**, a lightweight multi-modal fusion architecture combining visual (CLIP), temporal (action features), and structural (scene graph) embeddings into a 527-dimensional vector. Through systematic evaluation on 89 DAVIS videos with 14 transformation types (1,242 tests), we achieve **98.6% Recall@1** and **0.9748 average similarity**. Our ablation study reveals a surprising dichotomy: **action features boost confidence (+28% similarity) but minimally impact detection rate (+0.4% Recall@1)**. This finding validates efficiency-focused designs where lightweight temporal features (15D, 2.8% overhead) dramatically improve match confidence without requiring expensive temporal models. We identify a CLIP vertical flip bias (84.3% detection) and show rotation is the only transformation where action features aid detection (-2.3% degradation without).

**Paper PDF**: [main.pdf](paper/main.pdf) (single-column, 19 pages) | [main_two_column.pdf](paper/main_two_column.pdf) (two-column, 14 pages)

**arXiv**: https://arxiv.org/abs/XXXX.XXXXX *(coming soon)*

---

## 🔑 Key Findings

1. **Detection vs Confidence Dichotomy**: Action features are "confidence boosters" (+28% similarity) not "detection enablers" (+0.4% Recall@1)
2. **CLIP Vertical Flip Bias**: 84.3% detection on vertical flip vs 100% horizontal (16pp gap)
3. **Rotation Special Case**: Only transformation where action features aid detection (-2.3%)
4. **Efficiency Validation**: 15D action features (2.8% overhead) provide 28% confidence gain
5. **Robustness**: 98.6% Recall@1 across 14 realistic transformations (crop, rotate, brightness, compression, speed, flip, watermark, combined)

---

## 📊 Results

### Robustness Benchmark (14 Transformations)

| Configuration | Recall@1 | Recall@3 | Avg Similarity | Min Sim | Max Sim |
|--------------|----------|----------|----------------|---------|---------|
| **Fusion (CLIP + Action + Graph)** | **98.6%** | **99.7%** | **0.9748** | 0.8764 | 0.9957 |
| CLIP-only (512D) | 98.2% | 99.7% | 0.6996 | 0.3314 | 0.9881 |
| Action-only (15D) | 1.1% | 2.2% | 0.7480 | 0.2825 | 0.9954 |

### Per-Transformation Performance

| Transform | Fusion R@1 | CLIP R@1 | Action R@1 |
|-----------|------------|----------|------------|
| crop_10 | 100% | 100% | 0% |
| crop_20 | 100% | 100% | 0% |
| rotate_5 | 100% | 98.9% | 0% |
| rotate_10 | 98.9% | 96.6% | 0% |
| brightness_20 | 100% | 100% | 0% |
| brightness_40 | 100% | 100% | 0% |
| compress_50 | 100% | 100% | 0% |
| compress_25 | 98.9% | 98.9% | 0% |
| speed_0.9 | 100% | 100% | 0% |
| speed_1.1 | 100% | 100% | 0% |
| flip_h | 100% | 100% | 0% |
| **flip_v** | **84.3%** | **84.3%** | 0% |
| watermark | 98.9% | 98.9% | 0% |
| combined | 98.9% | 98.9% | 11.2% |

**Key Insight**: Action features improve confidence (0.97 vs 0.70 similarity) without improving detection rate, except for rotation (-2.3% without action).

---

## 🏗️ Architecture

```
Input Video (89 DAVIS 2017 videos, 2-5s, 480p)
    │
    ├─→ [CLIP ViT-B/32]          → 512D visual features (97.2% contribution)
    │   └─ Median pooling @ 2fps
    │
    ├─→ [SAM2 Tracking]           → 15D action features (2.8% contribution)
    │   ├─ Velocity (mean, std, max) [3D]
    │   ├─ Acceleration (mean, std, max) [3D]
    │   ├─ Direction entropy [1D]
    │   ├─ Spatial spread (mean, std, max) [3D]
    │   └─ Interaction density (collisions, proximity, cluster) [5D]
    │
    └─→ [Scene Graph]             → 8D structural features (<0.1% contribution)
        ├─ Spatial relations histogram [4D]
        └─ Temporal relations histogram [4D]
        
    → Concatenate → L2 Normalize → 527D Embedding
    → FAISS HNSW (M=16, efConstruction=200)
    → Query: <0.01ms median, 0.32ms P99
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/denisbilli/storyhash-paper.git
cd storyhash-paper

# Create environment
conda create -n storyhash python=3.10
conda activate storyhash

# Install dependencies
pip install -r requirements.txt
```

### Extract Features

```bash
# Extract CLIP features
python scripts/extract_clip_features.py \
    --video_dir data/davis/JPEGImages/480p \
    --output embeddings/clip_features.npy

# Extract action features (requires SAM2 tracking)
python scripts/extract_action_features.py \
    --video_dir data/davis/JPEGImages/480p \
    --tracks_dir data/davis/sam2_tracks \
    --output embeddings/action_features.npy

# Extract scene graph features
python scripts/extract_graph_features.py \
    --video_dir data/davis/JPEGImages/480p \
    --output embeddings/graph_features.npy

# Concatenate and normalize
python scripts/fuse_embeddings.py \
    --clip embeddings/clip_features.npy \
    --action embeddings/action_features.npy \
    --graph embeddings/graph_features.npy \
    --output embeddings/fusion_527d.npy
```

### Build Index

```bash
# Create FAISS HNSW index
python scripts/build_index.py \
    --embeddings embeddings/fusion_527d.npy \
    --output indices/storyhash.index \
    --M 16 \
    --efConstruction 200
```

### Query

```bash
# Retrieve similar videos
python scripts/query.py \
    --index indices/storyhash.index \
    --query_video path/to/video.mp4 \
    --top_k 5
```

### Robustness Benchmark

```bash
# Apply 14 transformations and evaluate
python scripts/benchmark_robustness.py \
    --video_dir data/davis/JPEGImages/480p \
    --index indices/storyhash.index \
    --output results/robustness_benchmark.json

# Generate per-transformation table
python scripts/generate_results_table.py \
    --benchmark results/robustness_benchmark.json \
    --output paper/figures/table1_per_transform.tex
```

### Ablation Study

```bash
# Test CLIP-only (512D)
python scripts/ablation_clip_only.py \
    --embeddings embeddings/clip_features.npy \
    --output results/ablation_clip.json

# Test Action-only (15D)
python scripts/ablation_action_only.py \
    --embeddings embeddings/action_features.npy \
    --output results/ablation_action.json

# Compare configurations
python scripts/compare_ablations.py \
    --fusion results/robustness_benchmark.json \
    --clip results/ablation_clip.json \
    --action results/ablation_action.json \
    --output paper/figures/fig2_ablation_summary.pdf
```

---

## 📁 Repository Structure

```
storyhash-paper/
├── paper/                      # LaTeX sources and compiled PDF
│   ├── main.tex               # Single-column (19 pages)
│   ├── main_two_column.tex    # Two-column (14 pages)
│   ├── main_body.tex          # Shared content sections
│   ├── references.bib         # Bibliography (14 citations)
│   ├── figures/               # Paper figures (5 PDFs + 1 table)
│   ├── main.pdf              # Compiled single-column
│   └── main_two_column.pdf   # Compiled two-column
│
├── scripts/                    # Implementation code
│   ├── extract_clip_features.py
│   ├── extract_action_features.py
│   ├── extract_graph_features.py
│   ├── fuse_embeddings.py
│   ├── build_index.py
│   ├── query.py
│   ├── benchmark_robustness.py
│   ├── ablation_clip_only.py
│   ├── ablation_action_only.py
│   ├── generate_results_table.py
│   ├── compare_ablations.py
│   └── generate_paper_figures.py  # Reproduce all figures
│
├── requirements.txt            # Python dependencies
├── LICENSE                     # MIT License
└── README.md                   # This file
```

---

## 🎯 Reproduce Paper Results

All experiments in the paper can be reproduced using the provided scripts:

```bash
# 1. Download DAVIS 2017 dataset
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
unzip DAVIS-2017-trainval-480p.zip -d data/davis

# 2. Extract all features
bash scripts/extract_all_features.sh

# 3. Run robustness benchmark (Table 1)
python scripts/benchmark_robustness.py

# 4. Run ablation studies (Figure 2, 3)
bash scripts/run_ablations.sh

# 5. Generate all figures
python scripts/generate_paper_figures.py

# 6. Compile paper
cd paper && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```

**Expected Runtime**: ~4 hours on M2 Pro (feature extraction dominates)

---

## 📦 Dependencies

- Python 3.10+
- PyTorch 2.1.0+
- transformers 4.35.0+ (CLIP)
- segment-anything-2 (SAM2 tracking)
- faiss-cpu 1.7.4+ (indexing)
- opencv-python 4.8.0+ (video processing)
- ffmpeg (transformations)

Full list in `requirements.txt`.

---

## 📝 Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{billi2025storyhash,
  title={Lightweight Multi-Modal Fusion for Robust Video Retrieval: When Action Features Boost Confidence, Not Detection},
  author={Billi, Denis},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025}
}
```

---

## 🙏 Acknowledgments

We thank the open-source community for:
- [CLIP](https://github.com/openai/CLIP) (OpenAI)
- [SAM2](https://github.com/facebookresearch/segment-anything-2) (Meta AI)
- [FAISS](https://github.com/facebookresearch/faiss) (Meta AI)
- [DAVIS 2017](https://davischallenge.org/) (ETH Zurich)

This work was conducted independently without external funding.

---

## 📧 Contact

**Denis Billi**  
Email: denis@denisbilli.it  
GitHub: [@denisbilli](https://github.com/denisbilli)

---

## 📜 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 Links

- **Paper PDF**: [main.pdf](paper/main.pdf)
- **arXiv**: https://arxiv.org/abs/XXXX.XXXXX *(coming soon)*
- **Project Page**: https://github.com/denisbilli/storyhash-paper
- **Main Development Repo**: https://github.com/denisbilli/StoryHash *(private)*
