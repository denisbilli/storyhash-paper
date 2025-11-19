#!/usr/bin/env python3
"""
Ablation study: CLIP-only baseline (512D).

Evaluates retrieval performance using only CLIP visual features
without action or graph features.

Usage:
    python ablation_clip_only.py \
        --embeddings embeddings/clip_features.npy \
        --output results/ablation_clip.json

Output:
    JSON file with same metrics as robustness benchmark for comparison.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='CLIP-only ablation study')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to CLIP features (.npy)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for results JSON')
    parser.add_argument('--video_dir', type=str, default='data/davis/JPEGImages/480p',
                        help='Directory containing original videos')
    return parser.parse_args()


def run_ablation(clip_features, video_dir):
    """
    Run ablation study with CLIP-only features.
    
    Returns:
        results: dict with per-transformation metrics
    """
    # TODO: Implement CLIP-only ablation
    # 1. Build CLIP-only index (512D)
    # 2. Run same 14 transformations as main benchmark
    # 3. Evaluate with same metrics
    raise NotImplementedError("CLIP-only ablation not yet implemented")


def main():
    args = parse_args()
    
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings not found: {embeddings_path}")
    
    logger.info(f"Loading CLIP features from {embeddings_path}")
    clip_features = np.load(embeddings_path)
    
    logger.info(f"CLIP features shape: {clip_features.shape}")
    logger.info("Running CLIP-only ablation study...")
    
    # Run ablation
    results = run_ablation(clip_features, args.video_dir)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Saved CLIP-only results to {output_path}")
    logger.info(f"Overall Recall@1: {results['aggregate']['recall@1']:.1%}")
    logger.info(f"Avg similarity: {results['aggregate']['avg_similarity']:.4f}")


if __name__ == '__main__':
    main()
