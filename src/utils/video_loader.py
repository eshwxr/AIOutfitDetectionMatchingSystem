"""Utility to load video metadata and captions"""

import os
from pathlib import Path
from typing import Tuple, Optional


def load_video_caption(video_path: str) -> str:
    """
    Load caption from .txt file associated with video
    
    Args:
        video_path: Path to video file (.mp4)
    
    Returns:
        Caption text from .txt file, or empty string if not found
    """
    video_path = Path(video_path)
    
    # Try to find associated .txt file
    txt_path = video_path.with_suffix('.txt')
    
    if txt_path.exists():
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                caption = f.read().strip()
            return caption
        except Exception as e:
            print(f"Warning: Could not read caption from {txt_path}: {e}")
            return ""
    
    return ""


def extract_hashtags_from_text(text: str) -> str:
    """
    Extract hashtags from text
    
    Args:
        text: Input text
    
    Returns:
        String of hashtags separated by spaces
    """
    import re
    hashtags = re.findall(r'#\w+', text)
    return ' '.join(hashtags)


def extract_caption_without_hashtags(text: str) -> str:
    """
    Extract caption without hashtags
    
    Args:
        text: Input text
    
    Returns:
        Caption text without hashtags
    """
    import re
    # Remove hashtags
    text = re.sub(r'#\w+', '', text)
    # Clean up extra whitespace
    text = ' '.join(text.split())
    return text.strip()


def load_video_metadata(video_path: str) -> Tuple[str, str]:
    """
    Load caption and hashtags from video's associated files
    
    Args:
        video_path: Path to video file
    
    Returns:
        Tuple of (caption, hashtags)
    """
    caption_text = load_video_caption(video_path)
    
    if not caption_text:
        return "", ""
    
    # Extract hashtags and caption
    hashtags = extract_hashtags_from_text(caption_text)
    caption = extract_caption_without_hashtags(caption_text)
    
    return caption, hashtags

