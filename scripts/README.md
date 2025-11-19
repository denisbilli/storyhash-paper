# Scripts Directory

Implementation scripts for StoryHash paper reproduction.

## Overview

This directory contains all code necessary to reproduce the experiments in our paper:

1. **Feature Extraction**: Extract CLIP, action, and scene graph features
2. **Indexing**: Build FAISS HNSW index
3. **Evaluation**: Robustness benchmark and ablation studies
4. **Visualization**: Generate paper figures and tables

## Feature Extraction Scripts

### `extract_clip_features.py`
Extract CLIP ViT-B/32 visual features (512D) from video frames.

**Usage**:
```bash
python extract_clip_features.py \\
    --video_dir data/davis/JPEGImages/480p \\
    --output embeddings/clip_features.npy \\
    --fps 2
```

**Output**: `(N, 512)` numpy array where N = number of videos

---

### `extract_action_features.py`
Extract 15D motion features from SAM2 object tracking.

**Usage**:
```bash
python extract_action_features.py \\
    --video_dir data/davis/JPEGImages/480p \\
    --tracks_dir data/davis/sam2_tracks \\
    --output embeddings/action_features.npy
```

**Requires**: Pre-computed SAM2 tracks (run `run_sam2_tracking.py` first)

**Output**: `(N, 15)` numpy array
- Velocity: mean, std, max [3D]
- Acceleration: mean, std, max [3D]
- Direction entropy [1D]
- Spatial spread: mean, std, max [3D]
- Interaction density: collisions, proximity, clustering [5D]

---

### `extract_graph_features.py`
Extract 8D scene graph features (spatial + temporal relations).

**Usage**:
```bash
python extract_graph_features.py \\
    --video_dir data/davis/JPEGImages/480p \\
    --output embeddings/graph_features.npy
```

**Output**: `(N, 8)` numpy array
- Spatial relations histogram [4D]: above, below, left, right
- Temporal relations histogram [4D]: before, after, overlap, no-overlap

---

### `fuse_embeddings.py`
Concatenate and L2-normalize multi-modal features.

**Usage**:
```bash
python fuse_embeddings.py \\
    --clip embeddings/clip_features.npy \\
    --action embeddings/action_features.npy \\
    --graph embeddings/graph_features.npy \\
    --output embeddings/fusion_527d.npy
```

**Output**: `(N, 527)` normalized numpy array

---

## Indexing Scripts

### `build_index.py`
Create FAISS HNSW index for efficient retrieval.

**Usage**:
```bash
python build_index.py \\
    --embeddings embeddings/fusion_527d.npy \\
    --output indices/storyhash.index \\
    --M 16 \\
    --efConstruction 200
```

**Parameters**:
- `M`: HNSW graph connectivity (paper uses 16)
- `efConstruction`: Build-time search depth (paper uses 200)

---

### `query.py`
Retrieve similar videos given query.

**Usage**:
```bash
python query.py \\
    --index indices/storyhash.index \\
    --query_video path/to/video.mp4 \\
    --top_k 5 \\
    --show_results
```

**Output**: Top-K similar videos with similarity scores

---

## Evaluation Scripts

### `benchmark_robustness.py`
Evaluate retrieval performance across 14 transformations (Table 1).

**Usage**:
```bash
python benchmark_robustness.py \\
    --video_dir data/davis/JPEGImages/480p \\
    --index indices/storyhash.index \\
    --transformations crop_10 crop_20 rotate_5 rotate_10 \\
                     brightness_20 brightness_40 \\
                     compress_50 compress_25 \\
                     speed_0.9 speed_1.1 \\
                     flip_h flip_v watermark combined \\
    --output results/robustness_benchmark.json
```

**Output**: JSON with per-transform Recall@1/3/5 and similarity stats

---

### `ablation_clip_only.py`
Evaluate CLIP-only configuration (512D).

**Usage**:
```bash
python ablation_clip_only.py \\
    --embeddings embeddings/clip_features.npy \\
    --video_dir data/davis/JPEGImages/480p \\
    --output results/ablation_clip.json
```

---

### `ablation_action_only.py`
Evaluate Action-only configuration (15D).

**Usage**:
```bash
python ablation_action_only.py \\
    --embeddings embeddings/action_features.npy \\
    --video_dir data/davis/JPEGImages/480p \\
    --output results/ablation_action.json
```

---

### `compare_ablations.py`
Generate ablation comparison figures (Figure 2, 3).

**Usage**:
```bash
python compare_ablations.py \\
    --fusion results/robustness_benchmark.json \\
    --clip results/ablation_clip.json \\
    --action results/ablation_action.json \\
    --output_recall paper/figures/fig2_ablation_summary.pdf \\
    --output_similarity paper/figures/fig3_similarity_delta.pdf
```

---

## Visualization Scripts

### `generate_paper_figures.py`
Reproduce all figures in the paper.

**Usage**:
```bash
python generate_paper_figures.py \\
    --results_dir results/ \\
    --output_dir paper/figures/
```

**Generates**:
- `fig1_pipeline_placeholder.pdf`: Architecture diagram
- `fig2_ablation_summary.pdf`: Recall@1 comparison
- `fig3_similarity_delta.pdf`: Similarity comparison
- `fig4_flip_bias.pdf`: Horizontal vs vertical flip
- `fig5_architecture.pdf`: 527D composition breakdown
- `table1_per_transform.tex`: Per-transformation results

---

### `generate_results_table.py`
Create LaTeX table from benchmark results.

**Usage**:
```bash
python generate_results_table.py \\
    --benchmark results/robustness_benchmark.json \\
    --output paper/figures/table1_per_transform.tex
```

---

## Helper Scripts

### `extract_all_features.sh`
Run all feature extraction sequentially.

**Usage**:
```bash
bash extract_all_features.sh
```

Runs:
1. SAM2 tracking
2. CLIP feature extraction
3. Action feature extraction
4. Scene graph extraction
5. Feature fusion

**Runtime**: ~4 hours on M2 Pro

---

### `run_ablations.sh`
Run all ablation studies.

**Usage**:
```bash
bash run_ablations.sh
```

Runs:
1. Fusion benchmark
2. CLIP-only ablation
3. Action-only ablation
4. Comparison plots

---

### `run_sam2_tracking.py`
Generate SAM2 object tracks (prerequisite for action features).

**Usage**:
```bash
python run_sam2_tracking.py \\
    --video_dir data/davis/JPEGImages/480p \\
    --output_dir data/davis/sam2_tracks
```

**Note**: Requires SAM2 checkpoint (`sam2_hiera_large.pt`)

---

## Expected Outputs

After running all scripts, you should have:

```
embeddings/
├── clip_features.npy          # (89, 512)
├── action_features.npy        # (89, 15)
├── graph_features.npy         # (89, 8)
└── fusion_527d.npy           # (89, 527)

indices/
└── storyhash.index           # FAISS HNSW index

results/
├── robustness_benchmark.json  # Fusion results
├── ablation_clip.json        # CLIP-only results
└── ablation_action.json      # Action-only results

paper/figures/
├── fig2_ablation_summary.pdf
├── fig3_similarity_delta.pdf
├── fig4_flip_bias.pdf
├── fig5_architecture.pdf
└── table1_per_transform.tex
```

---

## Troubleshooting

**ImportError: No module named 'clip'**
```bash
pip install git+https://github.com/openai/CLIP.git
```

**SAM2 not found**
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```

**FAISS installation issues**
```bash
# CPU version
pip install faiss-cpu

# GPU version (requires CUDA)
pip install faiss-gpu
```

**Out of memory during feature extraction**
- Reduce `--batch_size` in `extract_clip_features.py`
- Process videos sequentially instead of batched

---

## Performance Notes

**Feature extraction runtime** (M2 Pro, 89 videos):
- CLIP: ~15 minutes
- SAM2 tracking: ~3 hours
- Action features: ~5 minutes (after tracking)
- Scene graphs: ~10 minutes

**Index building**: <1 second (89 videos)

**Query latency**: <0.01ms median, 0.32ms P99

---

## Paper Reproduction Checklist

- [ ] Download DAVIS 2017 dataset
- [ ] Run `setup.sh` to create environment
- [ ] Extract CLIP features
- [ ] Run SAM2 tracking
- [ ] Extract action features
- [ ] Extract scene graph features
- [ ] Fuse embeddings
- [ ] Build FAISS index
- [ ] Run robustness benchmark
- [ ] Run ablation studies
- [ ] Generate figures
- [ ] Compile LaTeX paper

**Expected results**:
- Fusion R@1: 98.6%
- CLIP-only R@1: 98.2%
- Action-only R@1: 1.1%
- flip_v R@1: 84.3%

If results differ by >1%, check:
1. DAVIS dataset version (should be 2017)
2. CLIP model checkpoint (ViT-B/32)
3. SAM2 tracking quality
4. Random seed (for reproducibility, set `np.random.seed(42)`)
