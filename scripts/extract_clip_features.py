#!/usr/bin/env python3
"""
Extract CLIP ViT-B/32 features from videos.

Usage:
    python extract_clip_features.py \\
        --video_dir data/davis/JPEGImages/480p \\
        --output embeddings/clip_features.npy \\
        --batch_size 32 \\
        --fps 2
"""

import argparse
import numpy as np
import torch
import clip
from pathlib import Path
from tqdm import tqdm
import cv2


def extract_clip_features(video_dir, output_path, batch_size=32, fps=2):
    """
    Extract CLIP features from video frames.
    
    Args:
        video_dir: Directory containing video folders (DAVIS format)
        output_path: Output .npy file path
        batch_size: Batch size for CLIP inference
        fps: Frames per second to sample
    
    Returns:
        features: (N, 512) numpy array
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    
    video_folders = sorted(Path(video_dir).iterdir())
    all_features = []
    
    for video_folder in tqdm(video_folders, desc="Processing videos"):
        if not video_folder.is_dir():
            continue
            
        # Load frames at specified fps
        frames = load_frames(video_folder, fps=fps)
        
        # Preprocess frames
        inputs = torch.stack([preprocess(frame) for frame in frames]).to(device)
        
        # Extract features
        with torch.no_grad():
            features = model.encode_image(inputs)
            features = features.cpu().numpy()
        
        # Median pooling
        video_feature = np.median(features, axis=0)
        
        # L2 normalize
        video_feature = video_feature / np.linalg.norm(video_feature)
        
        all_features.append(video_feature)
    
    # Save
    features_array = np.stack(all_features)
    np.save(output_path, features_array)
    print(f"Saved {len(features_array)} features to {output_path}")
    print(f"Shape: {features_array.shape}")
    
    return features_array


def load_frames(video_folder, fps=2):
    """Load video frames at specified fps."""
    frame_files = sorted(video_folder.glob("*.jpg"))
    
    # Sample frames based on fps
    total_frames = len(frame_files)
    # Assuming original is 30fps, sample every N frames for target fps
    stride = max(1, int(30 / fps))
    sampled_indices = list(range(0, total_frames, stride))
    
    frames = []
    for idx in sampled_indices:
        if idx < len(frame_files):
            frame = cv2.imread(str(frame_files[idx]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
    
    return frames


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract CLIP features")
    parser.add_argument("--video_dir", type=str, required=True,
                       help="Directory containing video folders")
    parser.add_argument("--output", type=str, required=True,
                       help="Output .npy file path")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for inference")
    parser.add_argument("--fps", type=int, default=2,
                       help="Frames per second to sample")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # Extract features
    extract_clip_features(
        args.video_dir,
        args.output,
        batch_size=args.batch_size,
        fps=args.fps
    )
