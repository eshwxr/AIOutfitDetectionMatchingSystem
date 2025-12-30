"""Frame extraction from videos using OpenCV"""

import cv2
import os
from typing import List, Tuple
import numpy as np


class FrameExtractor:
    """Extract keyframes from video files at specified intervals"""
    
    def __init__(self, fps: float = 1.0):
        """
        Initialize frame extractor
        
        Args:
            fps: Frames per second to extract (default: 1.0)
        """
        self.fps = fps
    
    def extract_frames(self, video_path: str, output_dir: str = None) -> List[Tuple[np.ndarray, int]]:
        """
        Extract frames from video at specified FPS
        
        Args:
            video_path: Path to input video file
            output_dir: Optional directory to save frames
        
        Returns:
            List of tuples (frame_array, frame_number)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = int(video_fps / self.fps) if self.fps > 0 else 1
        
        frames = []
        frame_number = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract frame at specified interval
            if frame_number % frame_interval == 0:
                frames.append((frame.copy(), frame_number))
                
                # Save frame if output directory is provided
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                    frame_filename = os.path.join(
                        output_dir, 
                        f"frame_{frame_number:06d}.jpg"
                    )
                    cv2.imwrite(frame_filename, frame)
                
                extracted_count += 1
            
            frame_number += 1
        
        cap.release()
        
        return frames
    
    def extract_frame_at_time(self, video_path: str, timestamp: float) -> np.ndarray:
        """
        Extract a single frame at specific timestamp
        
        Args:
            video_path: Path to input video file
            timestamp: Timestamp in seconds
        
        Returns:
            Frame array as numpy array
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * video_fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError(f"Could not extract frame at timestamp {timestamp}s")
        
        return frame


