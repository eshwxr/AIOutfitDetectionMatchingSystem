"""Standalone script for processing videos without API"""

import sys
import yaml
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.detection import FrameExtractor, YOLODetector, crop_detection
from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher
from src.classification import VibeClassifier
from src.utils import ColorExtractor
import json


def load_config():
    """Load configuration"""
    config_path = Path(__file__).parent.parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_video(video_path: str, video_id: str, caption: str = "", hashtags: str = ""):
    """Process a single video"""
    config = load_config()
    
    # Initialize components
    frame_extractor = FrameExtractor(fps=config['detection']['frame_extraction_fps'])
    yolo_detector = YOLODetector(
        model_path=config['models']['yolo']['model_path'],
        confidence_threshold=config['models']['yolo']['confidence_threshold'],
        device=config['models']['clip']['device']
    )
    clip_encoder = CLIPEncoder(
        model_name=config['models']['clip']['model_name'],
        pretrained=config['models']['clip']['pretrained'],
        device=config['models']['clip']['device']
    )
    
    # Load FAISS index
    faiss_index = FAISSIndex(embedding_dim=512)
    index_path = Path(config['matching']['faiss_index_path'])
    metadata_path = Path(config['matching']['catalog_metadata_path'])
    
    if not index_path.exists() or not metadata_path.exists():
        raise FileNotFoundError("FAISS index not found. Build index first.")
    
    faiss_index.load(str(index_path), str(metadata_path))
    
    product_matcher = ProductMatcher(
        faiss_index=faiss_index,
        clip_encoder=clip_encoder,
        exact_threshold=config['matching']['similarity_thresholds']['exact_match'],
        similar_threshold=config['matching']['similarity_thresholds']['similar_match'],
        top_k=config['matching']['top_k']
    )
    
    vibe_classifier = VibeClassifier(
        clip_encoder=clip_encoder,
        top_k=config['vibes']['top_k_vibes'],
        min_confidence=config['vibes']['min_confidence']
    )
    
    color_extractor = ColorExtractor(num_colors=config['color']['num_colors'])
    
    # Process video
    print(f"Processing video: {video_id}")
    
    # Extract frames
    print("Extracting frames...")
    frames = frame_extractor.extract_frames(video_path)
    
    if not frames:
        raise ValueError("No frames extracted")
    
    # Detect fashion items
    print("Detecting fashion items...")
    all_detections = []
    for frame, frame_number in frames:
        detections = yolo_detector.detect(frame, frame_number)
        all_detections.extend(detections)
    
    if not all_detections:
        print("No fashion items detected")
        return {
            "video_id": video_id,
            "vibes": [],
            "products": []
        }
    
    # Process detections
    print(f"Processing {len(all_detections)} detections...")
    products = []
    
    for detection in all_detections:
        frame = frames[detection['frame_number']][0]
        cropped = crop_detection(frame, detection['bbox'], expand=True)
        
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            continue
        
        # Extract color
        color_name, _ = color_extractor.extract_dominant_color(cropped)
        
        # Match product
        matches = product_matcher.match_product(cropped)
        
        if matches and matches[0]['match_type'] != 'no_match':
            best_match = matches[0]
            products.append({
                "type": detection['fashion_type'],
                "color": color_name,
                "match_type": best_match['match_type'],
                "matched_product_id": best_match['matched_product_id'],
                "confidence": best_match['confidence'],
                "product_name": best_match.get('product_name', '')
            })
    
    # Classify vibes
    print("Classifying vibes...")
    vibe_results = vibe_classifier.classify(caption=caption, hashtags=hashtags)
    vibes = [vibe for vibe, _ in vibe_results]
    
    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process video for fashion detection")
    parser.add_argument("--video-path", required=True, help="Path to video file")
    parser.add_argument("--video-id", required=True, help="Video ID")
    parser.add_argument("--caption", default="", help="Video caption")
    parser.add_argument("--hashtags", default="", help="Hashtags")
    parser.add_argument("--output", help="Output JSON file path")
    
    args = parser.parse_args()
    
    result = process_video(
        video_path=args.video_path,
        video_id=args.video_id,
        caption=args.caption,
        hashtags=args.hashtags
    )
    
    # Output result
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"Result saved to {args.output}")
    else:
        print(json.dumps(result, indent=2))


