"""Dominant color extraction from images"""

import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
import cv2
from typing import Tuple, List


class ColorExtractor:
    """Extract dominant colors from images"""
    
    # Color name mapping (RGB to color names)
    COLOR_NAMES = {
        (0, 0, 0): "black",
        (255, 255, 255): "white",
        (255, 0, 0): "red",
        (0, 0, 255): "blue",
        (0, 255, 0): "green",
        (255, 255, 0): "yellow",
        (255, 165, 0): "orange",
        (255, 192, 203): "pink",
        (128, 0, 128): "purple",
        (165, 42, 42): "brown",
        (128, 128, 128): "gray",
        (245, 245, 220): "beige",
        (0, 0, 128): "navy",
        (128, 0, 0): "maroon",
        (0, 128, 128): "teal",
    }
    
    def __init__(self, num_colors: int = 3):
        """
        Initialize color extractor
        
        Args:
            num_colors: Number of dominant colors to extract (default: 3)
        """
        self.num_colors = num_colors
    
    def rgb_to_color_name(self, rgb: Tuple[int, int, int]) -> str:
        """
        Map RGB value to closest color name
        
        Args:
            rgb: RGB tuple
        
        Returns:
            Color name string
        """
        min_dist = float('inf')
        closest_color = "unknown"
        
        for color_rgb, color_name in self.COLOR_NAMES.items():
            # Calculate Euclidean distance in RGB space
            dist = np.sqrt(
                (rgb[0] - color_rgb[0]) ** 2 +
                (rgb[1] - color_rgb[1]) ** 2 +
                (rgb[2] - color_rgb[2]) ** 2
            )
            
            if dist < min_dist:
                min_dist = dist
                closest_color = color_name
        
        return closest_color
    
    def extract_dominant_color(self, image: np.ndarray) -> Tuple[str, str]:
        """
        Extract dominant color from image
        
        Args:
            image: Input image as numpy array (BGR or RGB)
        
        Returns:
            Tuple of (color_name, hex_code)
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3:
            # Check if it's BGR (OpenCV format)
            if image.shape[2] == 3:
                # Assume BGR, convert to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            # Grayscale, convert to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Reshape image to 2D array of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Remove very dark/light pixels (likely background)
        # Filter out pixels that are too dark or too light
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 220)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            # Fallback to all pixels
            filtered_pixels = pixels
        
        # Use K-means to find dominant colors
        if len(filtered_pixels) < self.num_colors:
            # Not enough pixels, use mean color
            dominant_rgb = np.mean(filtered_pixels, axis=0).astype(int)
        else:
            kmeans = KMeans(n_clusters=min(self.num_colors, len(filtered_pixels)), 
                          random_state=42, n_init=10)
            kmeans.fit(filtered_pixels)
            
            # Get the most frequent cluster (dominant color)
            labels = kmeans.labels_
            cluster_counts = np.bincount(labels)
            dominant_cluster = np.argmax(cluster_counts)
            dominant_rgb = kmeans.cluster_centers_[dominant_cluster].astype(int)
        
        # Clamp RGB values
        dominant_rgb = np.clip(dominant_rgb, 0, 255)
        
        # Convert to color name
        color_name = self.rgb_to_color_name(tuple(dominant_rgb))
        
        # Convert to hex
        hex_code = f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}"
        
        return color_name, hex_code
    
    def extract_colors(self, image: np.ndarray, top_k: int = 1) -> List[Tuple[str, str]]:
        """
        Extract top-k dominant colors
        
        Args:
            image: Input image as numpy array
            top_k: Number of colors to return
        
        Returns:
            List of (color_name, hex_code) tuples
        """
        # Convert BGR to RGB if needed
        if len(image.shape) == 3:
            if image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Reshape image to 2D array of pixels
        pixels = image_rgb.reshape(-1, 3)
        
        # Filter pixels
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 220)
        filtered_pixels = pixels[mask]
        
        if len(filtered_pixels) == 0:
            filtered_pixels = pixels
        
        if len(filtered_pixels) < top_k:
            # Not enough pixels, return mean color
            dominant_rgb = np.mean(filtered_pixels, axis=0).astype(int)
            color_name = self.rgb_to_color_name(tuple(dominant_rgb))
            hex_code = f"#{dominant_rgb[0]:02x}{dominant_rgb[1]:02x}{dominant_rgb[2]:02x}"
            return [(color_name, hex_code)]
        
        # Use K-means
        kmeans = KMeans(n_clusters=min(top_k, len(filtered_pixels)), 
                       random_state=42, n_init=10)
        kmeans.fit(filtered_pixels)
        
        # Get top-k clusters by frequency
        labels = kmeans.labels_
        cluster_counts = np.bincount(labels)
        top_clusters = np.argsort(cluster_counts)[-top_k:][::-1]
        
        colors = []
        for cluster_idx in top_clusters:
            rgb = kmeans.cluster_centers_[cluster_idx].astype(int)
            rgb = np.clip(rgb, 0, 255)
            color_name = self.rgb_to_color_name(tuple(rgb))
            hex_code = f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
            colors.append((color_name, hex_code))
        
        return colors


