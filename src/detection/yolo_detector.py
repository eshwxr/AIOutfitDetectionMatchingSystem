"""YOLOv8 fashion item detection"""

from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
import torch


class YOLODetector:
    """YOLOv8 detector for fashion items"""
    
    # COCO class names that are relevant for fashion
    FASHION_CLASSES = {
        0: "person",  # Will detect person, can infer clothing
        24: "handbag",
        25: "umbrella",
        26: "backpack",
        27: "tie",
        28: "suitcase",
    }
    
    # Map to fashion categories
    CLASS_TO_FASHION_TYPE = {
        "person": ["top", "bottom", "dress", "jacket"],
        "handbag": ["bag"],
        "backpack": ["bag"],
        "umbrella": ["accessory"],
        "tie": ["accessory"],
        "suitcase": ["bag"],
    }
    
    def __init__(self, 
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 device: str = None):
        """
        Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model weights
            confidence_threshold: Minimum confidence for detections
            device: Device to run on ('cuda' or 'cpu'), auto-detected if None
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        
        # Auto-detect device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model.to(self.device)
    
    def detect(self, frame: np.ndarray, frame_number: int = 0) -> List[Dict]:
        """
        Detect fashion items in a frame
        
        Args:
            frame: Input frame as numpy array (BGR format)
            frame_number: Frame number for tracking
        
        Returns:
            List of detections, each with:
            - class_name: Detected class name
            - bbox: (x, y, w, h) bounding box
            - confidence: Confidence score
            - frame_number: Frame number
            - fashion_type: Inferred fashion category
        """
        # Run YOLO inference
        results = self.model(frame, conf=self.confidence_threshold, verbose=False)
        
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            
            for box in boxes:
                # Get class ID and confidence
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                
                # Skip if not a fashion-relevant class
                if class_id not in self.FASHION_CLASSES:
                    continue
                
                class_name = self.FASHION_CLASSES[class_id]
                
                # Get bounding box (xyxy format)
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = xyxy
                
                # Convert to (x, y, w, h) format
                bbox = (float(x1), float(y1), float(x2 - x1), float(y2 - y1))
                
                # Get fashion type
                fashion_types = self.CLASS_TO_FASHION_TYPE.get(class_name, ["unknown"])
                
                # Create detection entry
                detection = {
                    "class_name": class_name,
                    "bbox": bbox,
                    "confidence": confidence,
                    "frame_number": frame_number,
                    "fashion_type": fashion_types[0],  # Use first type
                }
                
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, frames: List[Tuple[np.ndarray, int]]) -> List[List[Dict]]:
        """
        Detect fashion items in multiple frames
        
        Args:
            frames: List of (frame_array, frame_number) tuples
        
        Returns:
            List of detection lists, one per frame
        """
        all_detections = []
        
        for frame, frame_number in frames:
            detections = self.detect(frame, frame_number)
            all_detections.append(detections)
        
        return all_detections


