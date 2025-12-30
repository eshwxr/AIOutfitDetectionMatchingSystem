"""CLIP image encoder for generating embeddings"""

import torch
import open_clip
from PIL import Image
import numpy as np
import cv2
from typing import Union, List
import requests
from io import BytesIO


class CLIPEncoder:
    """CLIP encoder for image embeddings"""
    
    def __init__(self, 
                 model_name: str = "ViT-B-32",
                 pretrained: str = "openai",
                 device: str = None):
        """
        Initialize CLIP encoder
        
        Args:
            model_name: CLIP model name (e.g., "ViT-B-32")
            pretrained: Pretrained weights (e.g., "openai")
            device: Device to run on ('cuda' or 'cpu'), auto-detected if None
        """
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load CLIP model
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
            device=self.device
        )
        self.model.eval()
        
        # Get tokenizer for text encoding (if needed)
        self.tokenizer = open_clip.get_tokenizer(model_name)
    
    def encode_image(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Encode image to CLIP embedding
        
        Args:
            image: Input image as numpy array (BGR) or PIL Image (RGB)
        
        Returns:
            Normalized embedding vector (512-dim for ViT-B-32)
        """
        # Convert numpy array to PIL Image if needed
        if isinstance(image, np.ndarray):
            # Convert BGR to RGB
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                image = Image.fromarray(image)
        
        # Preprocess and encode
        with torch.no_grad():
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            embedding = self.model.encode_image(image_tensor)
            
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()
    
    def encode_image_from_url(self, url: str) -> np.ndarray:
        """
        Encode image from URL
        
        Args:
            url: Image URL
        
        Returns:
            Normalized embedding vector
        """
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return self.encode_image(image)
        except Exception as e:
            raise ValueError(f"Failed to load image from URL {url}: {str(e)}")
    
    def encode_images_batch(self, images: List[Union[np.ndarray, Image.Image]]) -> np.ndarray:
        """
        Encode multiple images in batch
        
        Args:
            images: List of images
        
        Returns:
            Array of normalized embeddings (N x embedding_dim)
        """
        # Convert to PIL if needed
        pil_images = []
        for img in images:
            if isinstance(img, np.ndarray):
                if len(img.shape) == 3 and img.shape[2] == 3:
                    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                else:
                    img = Image.fromarray(img)
            pil_images.append(img)
        
        # Preprocess batch
        image_tensors = torch.stack([
            self.preprocess(img) for img in pil_images
        ]).to(self.device)
        
        # Encode batch
        with torch.no_grad():
            embeddings = self.model.encode_image(image_tensors)
            # Normalize for cosine similarity
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            
            return embeddings.cpu().numpy()
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text to CLIP embedding (for vibe classification)
        
        Args:
            text: Input text
        
        Returns:
            Normalized embedding vector
        """
        with torch.no_grad():
            text_tokens = self.tokenizer([text]).to(self.device)
            embedding = self.model.encode_text(text_tokens)
            
            # Normalize for cosine similarity
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            
            return embedding.cpu().numpy().flatten()

