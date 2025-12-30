"""CLIP-Text zero-shot vibe classification"""

import numpy as np
from typing import List, Tuple, Dict
from .vibe_taxonomy import VIBE_TAXONOMY, get_vibe_prompts, load_vibes_from_json
from ..matching.clip_encoder import CLIPEncoder


class VibeClassifier:
    """Zero-shot vibe classifier using CLIP-Text"""
    
    def __init__(self, clip_encoder: CLIPEncoder, top_k: int = 3, min_confidence: float = 0.3):
        """
        Initialize vibe classifier
        
        Args:
            clip_encoder: CLIPEncoder instance (shared with image encoder)
            top_k: Number of top vibes to return (default: 3)
            min_confidence: Minimum confidence threshold (default: 0.3)
        """
        self.clip_encoder = clip_encoder
        self.top_k = top_k
        self.min_confidence = min_confidence
        self.vibe_taxonomy = VIBE_TAXONOMY
        self.vibe_prompts = get_vibe_prompts()
        
        # Pre-encode vibe prompts for efficiency
        self.vibe_embeddings = self._encode_vibe_prompts()
    
    def _encode_vibe_prompts(self) -> Dict[str, np.ndarray]:
        """
        Pre-encode all vibe prompts
        
        Returns:
            Dictionary mapping vibe names to embeddings
        """
        embeddings = {}
        for vibe in self.vibe_taxonomy:
            prompt = self.vibe_prompts.get(vibe, f"A photo of a {vibe.lower()} style outfit")
            embedding = self.clip_encoder.encode_text(prompt)
            embeddings[vibe] = embedding
        
        return embeddings
    
    def classify(self, caption: str = "", hashtags: str = "") -> List[Tuple[str, float]]:
        """
        Classify vibe from caption and hashtags
        
        Args:
            caption: Video caption text
            hashtags: Hashtags string (e.g., "#coquette #summer")
        
        Returns:
            List of (vibe_name, confidence_score) tuples, sorted by confidence
        """
        # Combine caption and hashtags
        text_input = f"{caption} {hashtags}".strip()
        
        if not text_input:
            return []
        
        # Encode input text
        text_embedding = self.clip_encoder.encode_text(text_input)
        
        # Compute cosine similarity with all vibe embeddings
        similarities = []
        for vibe, vibe_embedding in self.vibe_embeddings.items():
            # Cosine similarity (dot product for normalized vectors)
            similarity = np.dot(text_embedding, vibe_embedding)
            similarities.append((vibe, float(similarity)))
        
        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Filter by minimum confidence and return top-k
        filtered = [(vibe, conf) for vibe, conf in similarities if conf >= self.min_confidence]
        
        return filtered[:self.top_k]
    
    def classify_batch(self, texts: List[str]) -> List[List[Tuple[str, float]]]:
        """
        Classify vibes for multiple text inputs
        
        Args:
            texts: List of text inputs (caption + hashtags)
        
        Returns:
            List of classification results, one per input
        """
        results = []
        for text in texts:
            # Split text into caption and hashtags (simple heuristic)
            parts = text.split()
            hashtags = " ".join([p for p in parts if p.startswith("#")])
            caption = " ".join([p for p in parts if not p.startswith("#")])
            
            result = self.classify(caption, hashtags)
            results.append(result)
        
        return results


