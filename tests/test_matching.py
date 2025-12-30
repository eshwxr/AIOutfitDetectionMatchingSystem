"""Tests for matching module"""

import pytest
import numpy as np
from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher


def test_clip_encoder():
    """Test CLIP encoder initialization"""
    encoder = CLIPEncoder()
    assert encoder.device in ["cuda", "cpu"]


def test_faiss_index():
    """Test FAISS index"""
    index = FAISSIndex(embedding_dim=512)
    
    # Create dummy embeddings
    embeddings = np.random.rand(10, 512).astype(np.float32)
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    
    # Create dummy metadata
    metadata = [{"product_id": f"prod_{i}", "product_name": f"Product {i}", "image_url": ""} 
                for i in range(10)]
    
    # Build index
    index.build_index(embeddings, metadata)
    assert index.is_built
    
    # Test search
    query = np.random.rand(512).astype(np.float32)
    query = query / (np.linalg.norm(query) + 1e-8)
    results = index.search(query, top_k=5)
    
    assert len(results) <= 5
    assert all(isinstance(r[0], dict) for r in results)
    assert all(isinstance(r[1], float) for r in results)


def test_product_matcher():
    """Test product matcher"""
    # Create dummy index
    index = FAISSIndex(embedding_dim=512)
    embeddings = np.random.rand(10, 512).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / (norms + 1e-8)
    metadata = [{"product_id": f"prod_{i}", "product_name": f"Product {i}", "image_url": ""} 
                for i in range(10)]
    index.build_index(embeddings, metadata)
    
    # Create encoder (will use actual CLIP, so may be slow)
    encoder = CLIPEncoder()
    
    # Create matcher
    matcher = ProductMatcher(
        faiss_index=index,
        clip_encoder=encoder,
        exact_threshold=0.9,
        similar_threshold=0.75,
        top_k=5
    )
    
    assert matcher.exact_threshold == 0.9
    assert matcher.similar_threshold == 0.75


