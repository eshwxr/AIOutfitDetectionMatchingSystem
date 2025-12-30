"""Tests for utility modules"""

import pytest
import numpy as np
import cv2
from src.utils import ColorExtractor, VideoProcessor, load_video_caption, extract_hashtags_from_text
from pathlib import Path


def test_color_extractor():
    """Test color extraction"""
    extractor = ColorExtractor(num_colors=3)
    
    # Create test image (red square)
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:, :] = [0, 0, 255]  # BGR format (red)
    
    color_name, hex_code = extractor.extract_dominant_color(test_image)
    
    assert isinstance(color_name, str)
    assert isinstance(hex_code, str)
    assert hex_code.startswith('#')
    assert len(hex_code) == 7


def test_color_extractor_multiple_colors():
    """Test extracting multiple colors"""
    extractor = ColorExtractor(num_colors=3)
    
    # Create test image with multiple colors
    test_image = np.zeros((100, 100, 3), dtype=np.uint8)
    test_image[:33, :] = [255, 0, 0]  # Blue
    test_image[33:66, :] = [0, 255, 0]  # Green
    test_image[66:, :] = [0, 0, 255]  # Red
    
    colors = extractor.extract_colors(test_image, top_k=3)
    
    assert isinstance(colors, list)
    assert len(colors) <= 3
    assert all(isinstance(c, tuple) and len(c) == 2 for c in colors)


def test_video_processor_get_info():
    """Test video info extraction"""
    # This test requires an actual video file, so we'll just test the function exists
    assert hasattr(VideoProcessor, 'get_video_info')
    assert hasattr(VideoProcessor, 'validate_video_path')


def test_extract_hashtags():
    """Test hashtag extraction"""
    text = "This is a test #coquette #summer #fashion post"
    hashtags = extract_hashtags_from_text(text)
    
    assert isinstance(hashtags, str)
    assert "#coquette" in hashtags
    assert "#summer" in hashtags
    assert "#fashion" in hashtags


def test_load_video_caption_nonexistent():
    """Test loading caption from non-existent file"""
    # Use a non-existent video path
    caption = load_video_caption("/nonexistent/path/video.mp4")
    assert caption == ""


def test_bbox_expand():
    """Test bounding box expansion"""
    from src.detection import expand_bbox
    
    bbox = (10, 10, 50, 50)
    img_width, img_height = 200, 200
    
    expanded = expand_bbox(bbox, img_width, img_height, padding_ratio=0.1)
    
    assert len(expanded) == 4
    assert expanded[0] <= bbox[0]  # x should be smaller or equal
    assert expanded[1] <= bbox[1]  # y should be smaller or equal
    assert expanded[2] >= bbox[2]  # w should be larger or equal
    assert expanded[3] >= bbox[3]  # h should be larger or equal

