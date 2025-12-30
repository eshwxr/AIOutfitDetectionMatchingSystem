"""Product matching logic with similarity thresholds"""

from typing import List, Dict, Tuple
from .faiss_index import FAISSIndex
from .clip_encoder import CLIPEncoder
import numpy as np


class ProductMatcher:
    """Match detected fashion items to catalog products"""
    
    def __init__(self,
                 faiss_index: FAISSIndex,
                 clip_encoder: CLIPEncoder,
                 exact_threshold: float = 0.9,
                 similar_threshold: float = 0.75,
                 top_k: int = 5):
        """
        Initialize product matcher
        
        Args:
            faiss_index: FAISSIndex instance
            clip_encoder: CLIPEncoder instance
            exact_threshold: Similarity threshold for exact match (default: 0.9)
            similar_threshold: Similarity threshold for similar match (default: 0.75)
            top_k: Number of top matches to retrieve (default: 5)
        """
        self.faiss_index = faiss_index
        self.clip_encoder = clip_encoder
        self.exact_threshold = exact_threshold
        self.similar_threshold = similar_threshold
        self.top_k = top_k
    
    def match_product(self, cropped_image: np.ndarray) -> List[Dict]:
        """
        Match a cropped detection to catalog products
        
        Args:
            cropped_image: Cropped detection image as numpy array
        
        Returns:
            List of match dictionaries with:
            - matched_product_id: Product ID
            - confidence: Similarity score
            - match_type: "exact", "similar", or "no_match"
            - product_name: Product name (optional)
        """
        # Encode cropped image
        query_embedding = self.clip_encoder.encode_image(cropped_image)
        
        # Search for similar products
        results = self.faiss_index.search(query_embedding, top_k=self.top_k)
        
        matches = []
        for metadata, similarity in results:
            # Classify match type
            if similarity >= self.exact_threshold:
                match_type = "exact"
            elif similarity >= self.similar_threshold:
                match_type = "similar"
            else:
                match_type = "no_match"
            
            match = {
                "matched_product_id": metadata.get("product_id", ""),
                "confidence": float(similarity),
                "match_type": match_type,
                "product_name": metadata.get("product_name", ""),
            }
            
            matches.append(match)
        
        return matches
    
    def match_products_batch(self, cropped_images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Match multiple cropped detections to catalog
        
        Args:
            cropped_images: List of cropped detection images
        
        Returns:
            List of match lists, one per detection
        """
        all_matches = []
        
        for cropped_image in cropped_images:
            matches = self.match_product(cropped_image)
            all_matches.append(matches)
        
        return all_matches


