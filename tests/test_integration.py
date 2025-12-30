"""Integration tests (require models to be loaded)"""

import pytest
import numpy as np
from pathlib import Path


def test_imports():
    """Test that all main modules can be imported"""
    try:
        from src.detection import FrameExtractor, YOLODetector, crop_detection
        from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher
        from src.classification import VibeClassifier, VIBE_TAXONOMY
        from src.utils import ColorExtractor, VideoProcessor
        from src.api.schemas import VideoProcessRequest, VideoProcessResponse
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_vibe_taxonomy_completeness():
    """Test that vibe taxonomy has all expected vibes"""
    from src.classification import VIBE_TAXONOMY
    
    expected_vibes = [
        "Coquette",
        "Clean Girl",
        "Cottagecore",
        "Streetcore",
        "Y2K",
        "Boho",
        "Party Glam"
    ]
    
    for vibe in expected_vibes:
        assert vibe in VIBE_TAXONOMY, f"Missing vibe: {vibe}"


def test_config_loading():
    """Test that config file can be loaded"""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        assert 'models' in config
        assert 'detection' in config
        assert 'matching' in config
        assert 'vibes' in config

