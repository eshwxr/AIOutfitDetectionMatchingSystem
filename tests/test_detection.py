"""Tests for detection module"""

import pytest
import numpy as np
from src.detection import FrameExtractor, YOLODetector, crop_detection


def test_frame_extractor():
    """Test frame extraction"""
    extractor = FrameExtractor(fps=1.0)
    # Note: Requires actual video file for full test
    assert extractor.fps == 1.0


def test_bbox_utils():
    """Test bounding box utilities"""
    # Create dummy frame
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Test crop
    bbox = (10, 10, 50, 50)
    cropped = crop_detection(frame, bbox, expand=False)
    
    assert cropped.shape[0] > 0
    assert cropped.shape[1] > 0


def test_yolo_detector():
    """Test YOLO detector initialization"""
    detector = YOLODetector(confidence_threshold=0.5)
    assert detector.confidence_threshold == 0.5
    assert detector.device in ["cuda", "cpu"]


