"""Video processing utilities"""

import os
from typing import List, Tuple
import numpy as np


class VideoProcessor:
    """Utility class for video processing operations"""
    
    @staticmethod
    def get_video_info(video_path: str) -> dict:
        """
        Get video metadata
        
        Args:
            video_path: Path to video file
        
        Returns:
            Dictionary with video info (fps, duration, frame_count, resolution)
        """
        import cv2
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "duration": duration,
            "width": width,
            "height": height,
        }
    
    @staticmethod
    def validate_video_path(video_path: str) -> bool:
        """
        Validate that video file exists and is readable
        
        Args:
            video_path: Path to video file
        
        Returns:
            True if valid, raises exception otherwise
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video file: {video_path}")
        
        cap.release()
        return True


