"""Test script to process a single video from AI Hackathon dataset"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.detection import FrameExtractor, YOLODetector, crop_detection
from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher
from src.classification import VibeClassifier
from src.utils import ColorExtractor, load_video_metadata
import yaml


def test_video_processing(video_path: str):
    """Test video processing pipeline"""
    print("=" * 60)
    print("FLICKD Video Processing Test")
    print("=" * 60)
    
    # Load config
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    video_path = Path(video_path)
    video_id = video_path.stem
    
    print(f"\n📹 Video: {video_path.name}")
    print(f"🆔 Video ID: {video_id}")
    
    # Load caption and hashtags
    print("\n📝 Loading caption and hashtags...")
    caption, hashtags = load_video_metadata(str(video_path))
    print(f"   Caption: {caption[:100] if caption else 'None'}...")
    print(f"   Hashtags: {hashtags}")
    
    # Initialize components
    print("\n🔧 Initializing components...")
    
    print("   - Frame Extractor...")
    frame_extractor = FrameExtractor(fps=config['detection']['frame_extraction_fps'])
    
    print("   - YOLO Detector (this may take a moment to download model on first run)...")
    try:
        yolo_detector = YOLODetector(
            model_path=config['models']['yolo']['model_path'],
            confidence_threshold=config['models']['yolo']['confidence_threshold'],
            device="cpu"  # Use CPU for testing
        )
        print("      ✓ YOLO Detector ready")
    except Exception as e:
        print(f"      ✗ YOLO Detector failed: {e}")
        print("      Make sure ultralytics is installed: pip install ultralytics")
        return None
    
    print("   - CLIP Encoder (this may take a moment to download model on first run)...")
    try:
        clip_encoder = CLIPEncoder(
            model_name=config['models']['clip']['model_name'],
            pretrained=config['models']['clip']['pretrained'],
            device="cpu"  # Use CPU for testing
        )
        print("      ✓ CLIP Encoder ready")
    except Exception as e:
        print(f"      ✗ CLIP Encoder failed: {e}")
        print("      Make sure open-clip-torch is installed: pip install open-clip-torch")
        return None
    
    print("   - Color Extractor...")
    color_extractor = ColorExtractor(num_colors=config['color']['num_colors'])
    
    print("   - Vibe Classifier...")
    vibe_classifier = VibeClassifier(
        clip_encoder=clip_encoder,
        top_k=config['vibes']['top_k_vibes'],
        min_confidence=config['vibes']['min_confidence']
    )
    
    # Check if FAISS index exists
    index_path = Path(config['matching']['faiss_index_path'])
    metadata_path = Path(config['matching']['catalog_metadata_path'])
    
    faiss_index = None
    product_matcher = None
    
    if index_path.exists() and metadata_path.exists():
        print("   - Loading FAISS Index...")
        faiss_index = FAISSIndex(embedding_dim=512)
        faiss_index.load(str(index_path), str(metadata_path))
        
        product_matcher = ProductMatcher(
            faiss_index=faiss_index,
            clip_encoder=clip_encoder,
            exact_threshold=config['matching']['similarity_thresholds']['exact_match'],
            similar_threshold=config['matching']['similarity_thresholds']['similar_match'],
            top_k=config['matching']['top_k']
        )
        print(f"   ✓ Index loaded with {len(faiss_index.metadata)} products")
    else:
        print("   ⚠️  FAISS index not found. Product matching will be skipped.")
        print(f"      Build index first: python build_index.py --catalog-path <catalog.csv>")
    
    # Process video
    print("\n🎬 Processing video...")
    
    # Step 1: Extract frames
    print("   1. Extracting frames...")
    frames = frame_extractor.extract_frames(str(video_path))
    print(f"      ✓ Extracted {len(frames)} frames")
    
    if not frames:
        print("   ❌ No frames extracted!")
        return None
    
    # Step 2: Detect fashion items
    print("   2. Detecting fashion items...")
    all_detections = []
    for frame, frame_number in frames:
        detections = yolo_detector.detect(frame, frame_number)
        all_detections.extend(detections)
    
    print(f"      ✓ Found {len(all_detections)} detections")
    
    if not all_detections:
        print("   ⚠️  No fashion items detected in video")
        # Still classify vibes
        print("\n   3. Classifying vibes...")
        vibe_results = vibe_classifier.classify(caption=caption, hashtags=hashtags)
        vibes = [vibe for vibe, _ in vibe_results]
        
        result = {
            "video_id": video_id,
            "vibes": vibes,
            "products": []
        }
        
        print(f"\n✅ Results:")
        print(json.dumps(result, indent=2))
        return result
    
    # Step 3: Process detections
    print("   3. Processing detections...")
    products = []
    
    for i, detection in enumerate(all_detections, 1):
        print(f"      Processing detection {i}/{len(all_detections)}...")
        
        # Crop detection
        frame = frames[detection['frame_number']][0]
        cropped = crop_detection(frame, detection['bbox'], expand=True)
        
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            print(f"         ⚠️  Skipped (too small)")
            continue
        
        # Extract color
        color_name, hex_code = color_extractor.extract_dominant_color(cropped)
        print(f"         Color: {color_name}")
        
        # Match product (if index available)
        if product_matcher:
            matches = product_matcher.match_product(cropped)
            if matches and matches[0]['match_type'] != 'no_match':
                best_match = matches[0]
                product = {
                    "type": detection['fashion_type'],
                    "color": color_name,
                    "match_type": best_match['match_type'],
                    "matched_product_id": best_match['matched_product_id'],
                    "confidence": best_match['confidence'],
                    "product_name": best_match.get('product_name', '')
                }
                products.append(product)
                print(f"         ✓ Matched: {best_match['match_type']} (confidence: {best_match['confidence']:.2f})")
            else:
                print(f"         ⚠️  No match found")
        else:
            # Without index, just report detection
            product = {
                "type": detection['fashion_type'],
                "color": color_name,
                "match_type": "no_index",
                "matched_product_id": "",
                "confidence": detection['confidence'],
                "product_name": ""
            }
            products.append(product)
            print(f"         ✓ Detected: {detection['fashion_type']} (confidence: {detection['confidence']:.2f})")
    
    # Step 4: Classify vibes
    print("\n   4. Classifying vibes...")
    vibe_results = vibe_classifier.classify(caption=caption, hashtags=hashtags)
    vibes = [vibe for vibe, _ in vibe_results]
    print(f"      ✓ Detected vibes: {', '.join(vibes) if vibes else 'None'}")
    
    # Compile results
    result = {
        "video_id": video_id,
        "vibes": vibes,
        "products": products
    }
    
    print("\n" + "=" * 60)
    print("✅ FINAL RESULTS")
    print("=" * 60)
    print(json.dumps(result, indent=2))
    print("=" * 60)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test video processing")
    parser.add_argument(
        "--video-path",
        default="/Users/eshwarbhadhavath/Downloads/AI Hackathon/videos/2025-05-27_13-46-16_UTC.mp4",
        help="Path to video file"
    )
    parser.add_argument(
        "--output",
        help="Output JSON file path"
    )
    
    args = parser.parse_args()
    
    result = test_video_processing(args.video_path)
    
    if args.output and result:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n💾 Results saved to {args.output}")

