#!/usr/bin/env python3
"""
Extract 15D action features from SAM2 tracking trajectories.

Features:
- Velocity statistics (mean, std, max) [3D]
- Acceleration statistics (mean, std, max) [3D]
- Direction entropy [1D]
- Spatial spread (mean, std, max) [3D]
- Interaction density (collisions, proximity, cluster) [5D]

Usage:
    python extract_action_features.py \
        --video_dir data/davis/JPEGImages/480p \
        --tracks_dir data/davis/sam2_tracks \
        --output embeddings/action_features.npy

Output:
    - action_features.npy: (N, 15) float32 array
    - action_metadata.json: Video names and statistics
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Extract action features from SAM2 tracks')
    parser.add_argument('--video_dir', type=str, required=True,
                        help='Directory containing video frames')
    parser.add_argument('--tracks_dir', type=str, required=True,
                        help='Directory containing SAM2 tracking outputs')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for action features (.npy)')
    parser.add_argument('--fps', type=int, default=2,
                        help='Sampling rate for feature extraction')
    return parser.parse_args()


def extract_features(video_dir: Path, tracks_dir: Path, fps: int = 2):
    """
    Extract 15D action features from tracking trajectories.
    
    Returns:
        features: (N, 15) ndarray
        metadata: dict with video names and statistics
    """
    # TODO: Implement feature extraction
    # 1. Load SAM2 tracking results
    # 2. Compute velocity/acceleration
    # 3. Calculate direction entropy
    # 4. Compute spatial spread
    # 5. Calculate interaction metrics
    raise NotImplementedError("Feature extraction not yet implemented")


def main():
    args = parse_args()
    
    video_dir = Path(args.video_dir)
    tracks_dir = Path(args.tracks_dir)
    output_path = Path(args.output)
    
    if not video_dir.exists():
        raise ValueError(f"Video directory not found: {video_dir}")
    if not tracks_dir.exists():
        raise ValueError(f"Tracks directory not found: {tracks_dir}")
    
    logger.info(f"Extracting action features from {video_dir}")
    logger.info(f"Using tracking data from {tracks_dir}")
    
    # Extract features
    features, metadata = extract_features(video_dir, tracks_dir, args.fps)
    
    # Save features
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    
    # Save metadata
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved {len(features)} action features to {output_path}")
    logger.info(f"Feature shape: {features.shape}")
    logger.info(f"Metadata saved to {meta_path}")


if __name__ == '__main__':
    main()
