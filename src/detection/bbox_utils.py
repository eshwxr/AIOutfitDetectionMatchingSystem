"""Bounding box utilities for cropping and validation"""

import numpy as np
from typing import Tuple, Optional


def validate_bbox(bbox: Tuple[float, float, float, float], 
                  img_width: int, 
                  img_height: int) -> Tuple[float, float, float, float]:
    """
    Validate and clamp bounding box coordinates to image dimensions
    
    Args:
        bbox: (x, y, w, h) bounding box
        img_width: Image width
        img_height: Image height
    
    Returns:
        Validated bounding box (x, y, w, h)
    """
    x, y, w, h = bbox
    
    # Clamp coordinates
    x = max(0, min(x, img_width - 1))
    y = max(0, min(y, img_height - 1))
    w = max(1, min(w, img_width - x))
    h = max(1, min(h, img_height - y))
    
    return (x, y, w, h)


def expand_bbox(bbox: Tuple[float, float, float, float],
                img_width: int,
                img_height: int,
                padding_ratio: float = 0.1) -> Tuple[float, float, float, float]:
    """
    Expand bounding box with padding for better matching
    
    Args:
        bbox: (x, y, w, h) bounding box
        img_width: Image width
        img_height: Image height
        padding_ratio: Padding ratio (default: 0.1 = 10%)
    
    Returns:
        Expanded bounding box (x, y, w, h)
    """
    x, y, w, h = bbox
    
    # Calculate padding
    pad_w = w * padding_ratio
    pad_h = h * padding_ratio
    
    # Expand bbox
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(img_width - x, w + 2 * pad_w)
    h = min(img_height - y, h + 2 * pad_h)
    
    return validate_bbox((x, y, w, h), img_width, img_height)


def crop_detection(frame: np.ndarray, 
                  bbox: Tuple[float, float, float, float],
                  expand: bool = True,
                  padding_ratio: float = 0.1) -> np.ndarray:
    """
    Crop detection from frame using bounding box
    
    Args:
        frame: Input frame as numpy array
        bbox: (x, y, w, h) bounding box
        expand: Whether to expand bbox with padding
        padding_ratio: Padding ratio if expand=True
    
    Returns:
        Cropped image as numpy array
    """
    img_height, img_width = frame.shape[:2]
    
    # Validate and optionally expand bbox
    if expand:
        bbox = expand_bbox(bbox, img_width, img_height, padding_ratio)
    else:
        bbox = validate_bbox(bbox, img_width, img_height)
    
    x, y, w, h = bbox
    
    # Convert to integers for slicing
    x1 = int(x)
    y1 = int(y)
    x2 = int(x + w)
    y2 = int(y + h)
    
    # Crop image
    cropped = frame[y1:y2, x1:x2]
    
    return cropped


def bbox_to_xyxy(bbox: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert (x, y, w, h) to (x1, y1, x2, y2) format"""
    x, y, w, h = bbox
    return (x, y, x + w, y + h)


def xyxy_to_bbox(xyxy: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    """Convert (x1, y1, x2, y2) to (x, y, w, h) format"""
    x1, y1, x2, y2 = xyxy
    return (x1, y1, x2 - x1, y2 - y1)


