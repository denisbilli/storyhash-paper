#!/usr/bin/env python3
"""
Query FAISS index for similar videos.

Features:
- Extract features from query video
- Search FAISS index
- Return top-k matches with similarities

Usage:
    python query.py \
        --index indices/storyhash.index \
        --query_video path/to/video.mp4 \
        --top_k 5

Output:
    Prints ranked results with video names and similarity scores.
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Query FAISS index for similar videos')
    parser.add_argument('--index', type=str, required=True,
                        help='Path to FAISS index file')
    parser.add_argument('--query_video', type=str, required=True,
                        help='Path to query video file')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of results to return')
    parser.add_argument('--metadata', type=str, default=None,
                        help='Path to metadata JSON (optional, for video names)')
    return parser.parse_args()


def extract_query_features(video_path: Path):
    """
    Extract 527D features from query video.
    
    Returns:
        features: (1, 527) ndarray
    """
    # TODO: Implement feature extraction
    # 1. Extract CLIP features
    # 2. Extract action features (if SAM2 tracks available)
    # 3. Extract graph features
    # 4. Fuse and normalize
    raise NotImplementedError("Query feature extraction not yet implemented")


def search_index(index, query_features, k):
    """
    Search FAISS index.
    
    Returns:
        distances: (1, k) ndarray (L2 distances)
        indices: (1, k) ndarray (video IDs)
    """
    # TODO: Implement search
    # 1. Load FAISS index
    # 2. Search for top-k
    # 3. Convert L2 to cosine similarity
    raise NotImplementedError("Index search not yet implemented")


def main():
    args = parse_args()
    
    index_path = Path(args.index)
    query_path = Path(args.query_video)
    
    if not index_path.exists():
        raise ValueError(f"Index not found: {index_path}")
    if not query_path.exists():
        raise ValueError(f"Query video not found: {query_path}")
    
    logger.info(f"Extracting features from {query_path}")
    query_features = extract_query_features(query_path)
    
    logger.info(f"Searching index {index_path}")
    distances, indices = search_index(index_path, query_features, args.top_k)
    
    # Load metadata if available
    metadata = None
    if args.metadata:
        meta_path = Path(args.metadata)
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
    
    # Print results
    print(f"\nTop {args.top_k} matches for {query_path.name}:")
    print("-" * 60)
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), 1):
        similarity = 1.0 - dist  # Assuming normalized vectors
        video_name = metadata['video_names'][idx] if metadata else f"video_{idx}"
        print(f"{rank}. {video_name} (similarity: {similarity:.4f})")


if __name__ == '__main__':
    main()
