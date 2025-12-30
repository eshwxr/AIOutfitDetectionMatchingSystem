"""Vibe taxonomy definitions"""

import json
from pathlib import Path
from typing import List


def load_vibes_from_json(json_path: str = None) -> List[str]:
    """
    Load vibes from JSON file
    
    Args:
        json_path: Path to vibeslist.json file
    
    Returns:
        List of vibe names
    """
    if json_path and Path(json_path).exists():
        try:
            with open(json_path, 'r') as f:
                vibes = json.load(f)
            return vibes if isinstance(vibes, list) else VIBE_TAXONOMY
        except Exception as e:
            print(f"Warning: Could not load vibes from {json_path}: {e}")
            return VIBE_TAXONOMY
    return VIBE_TAXONOMY


# Supported fashion vibes (default)
VIBE_TAXONOMY = [
    "Coquette",
    "Clean Girl",
    "Cottagecore",
    "Streetcore",
    "Y2K",
    "Boho",
    "Party Glam",
]


def get_vibe_prompts() -> dict:
    """
    Get prompt templates for each vibe
    
    Returns:
        Dictionary mapping vibe names to prompt templates
    """
    return {
        "Coquette": "A photo of a coquette style outfit with feminine, romantic, and playful aesthetic",
        "Clean Girl": "A photo of a clean girl style outfit with minimal, fresh, and effortless aesthetic",
        "Cottagecore": "A photo of a cottagecore style outfit with rustic, natural, and vintage aesthetic",
        "Streetcore": "A photo of a streetcore style outfit with urban, edgy, and casual aesthetic",
        "Y2K": "A photo of a Y2K style outfit with 2000s fashion, nostalgic, and trendy aesthetic",
        "Boho": "A photo of a boho style outfit with bohemian, free-spirited, and eclectic aesthetic",
        "Party Glam": "A photo of a party glam style outfit with glamorous, elegant, and festive aesthetic",
    }


def get_vibe_keywords() -> dict:
    """
    Get keywords associated with each vibe
    
    Returns:
        Dictionary mapping vibe names to keyword lists
    """
    return {
        "Coquette": ["coquette", "feminine", "romantic", "playful", "lace", "ribbon", "bow", "pink", "frilly"],
        "Clean Girl": ["clean", "minimal", "fresh", "effortless", "neutral", "white", "beige", "simple", "classic"],
        "Cottagecore": ["cottagecore", "rustic", "natural", "vintage", "floral", "earth", "pastel", "cottage"],
        "Streetcore": ["street", "urban", "edgy", "casual", "sneakers", "hoodie", "streetwear", "cool"],
        "Y2K": ["y2k", "2000s", "nostalgic", "trendy", "butterfly", "low-rise", "mini", "bling", "retro"],
        "Boho": ["boho", "bohemian", "free-spirited", "eclectic", "flowy", "fringe", "pattern", "hippie"],
        "Party Glam": ["party", "glam", "glamorous", "elegant", "festive", "sparkle", "sequin", "evening", "dressy"],
    }


