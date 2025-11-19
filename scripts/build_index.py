#!/usr/bin/env python3
"""
Build FAISS HNSW index from 527D embeddings.

Configuration:
- Index type: HNSW (Hierarchical Navigable Small World)
- M: 16 (connections per node)
- efConstruction: 200 (search quality during build)
- Metric: L2 (Euclidean distance on normalized vectors = cosine)

Usage:
    python build_index.py \
        --embeddings embeddings/fusion_527d.npy \
        --output indices/storyhash.index \
        --M 16 \
        --efConstruction 200

Output:
    - storyhash.index: FAISS index file
    - storyhash_metadata.json: Build statistics
"""

import argparse
import json
import logging
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Build FAISS HNSW index')
    parser.add_argument('--embeddings', type=str, required=True,
                        help='Path to embeddings (.npy)')
    parser.add_argument('--output', type=str, required=True,
                        help='Output path for index file')
    parser.add_argument('--M', type=int, default=16,
                        help='HNSW parameter: connections per node')
    parser.add_argument('--efConstruction', type=int, default=200,
                        help='HNSW parameter: search quality during construction')
    return parser.parse_args()


def build_index(embeddings, M, efConstruction):
    """
    Build FAISS HNSW index.
    
    Returns:
        index: FAISS index
        metadata: dict with build statistics (time, ntotal, etc.)
    """
    # TODO: Implement index building
    # 1. Create IndexHNSWFlat(d=527, M=16)
    # 2. Set efConstruction=200
    # 3. Add embeddings
    # 4. Measure build time
    raise NotImplementedError("Index building not yet implemented")


def main():
    args = parse_args()
    
    embeddings_path = Path(args.embeddings)
    output_path = Path(args.output)
    
    if not embeddings_path.exists():
        raise ValueError(f"Embeddings not found: {embeddings_path}")
    
    logger.info(f"Loading embeddings from {embeddings_path}")
    embeddings = np.load(embeddings_path)
    
    logger.info(f"Embeddings shape: {embeddings.shape}")
    logger.info(f"Building HNSW index (M={args.M}, efConstruction={args.efConstruction})")
    
    # Build index
    index, metadata = build_index(embeddings, args.M, args.efConstruction)
    
    # Save index
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # faiss.write_index(index, str(output_path))
    
    # Save metadata
    meta_path = output_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Saved index to {output_path}")
    logger.info(f"Index contains {metadata['ntotal']} vectors")
    logger.info(f"Build time: {metadata['build_time_sec']:.2f}s")


if __name__ == '__main__':
    main()
