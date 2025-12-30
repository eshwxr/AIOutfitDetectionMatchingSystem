"""Tests for classification module"""

import pytest
from src.classification import VibeClassifier, VIBE_TAXONOMY
from src.matching import CLIPEncoder


def test_vibe_taxonomy():
    """Test vibe taxonomy"""
    assert len(VIBE_TAXONOMY) > 0
    assert "Coquette" in VIBE_TAXONOMY


def test_vibe_classifier():
    """Test vibe classifier initialization"""
    encoder = CLIPEncoder()
    classifier = VibeClassifier(
        clip_encoder=encoder,
        top_k=3,
        min_confidence=0.3
    )
    
    assert classifier.top_k == 3
    assert classifier.min_confidence == 0.3
    assert len(classifier.vibe_embeddings) == len(VIBE_TAXONOMY)
    
    # Test classification
    results = classifier.classify(caption="coquette summer vibes", hashtags="#coquette #summer")
    assert isinstance(results, list)
    assert all(isinstance(r, tuple) and len(r) == 2 for r in results)


