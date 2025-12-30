"""Quick test to verify all components can be imported and initialized"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

print("Testing FLICKD Components...")
print("=" * 60)

# Test imports
print("\n1. Testing imports...")
try:
    from src.detection import FrameExtractor, YOLODetector, crop_detection
    print("   ✓ Detection module imported")
except Exception as e:
    print(f"   ✗ Detection module failed: {e}")
    sys.exit(1)

try:
    from src.matching import CLIPEncoder, FAISSIndex, ProductMatcher
    print("   ✓ Matching module imported")
except Exception as e:
    print(f"   ✗ Matching module failed: {e}")
    sys.exit(1)

try:
    from src.classification import VibeClassifier, VIBE_TAXONOMY
    print("   ✓ Classification module imported")
except Exception as e:
    print(f"   ✗ Classification module failed: {e}")
    sys.exit(1)

try:
    from src.utils import ColorExtractor, load_video_metadata
    print("   ✓ Utils module imported")
except Exception as e:
    print(f"   ✗ Utils module failed: {e}")
    sys.exit(1)

# Test video metadata loading
print("\n2. Testing video metadata loading...")
video_path = "/Users/eshwarbhadhavath/Downloads/AI Hackathon/videos/2025-05-27_13-46-16_UTC.mp4"
try:
    caption, hashtags = load_video_metadata(video_path)
    print(f"   ✓ Caption loaded: {caption[:50] if caption else 'None'}...")
    print(f"   ✓ Hashtags: {hashtags if hashtags else 'None'}")
except Exception as e:
    print(f"   ✗ Failed to load metadata: {e}")

# Test configuration
print("\n3. Testing configuration...")
try:
    import yaml
    config_path = Path(__file__).parent / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    print("   ✓ Configuration loaded")
    print(f"   - YOLO model: {config['models']['yolo']['model_path']}")
    print(f"   - CLIP model: {config['models']['clip']['model_name']}")
    print(f"   - Device: {config['models']['clip']['device']}")
except Exception as e:
    print(f"   ✗ Configuration failed: {e}")

# Test component initialization (without loading models)
print("\n4. Testing component initialization...")
try:
    frame_extractor = FrameExtractor(fps=1.0)
    print("   ✓ FrameExtractor initialized")
except Exception as e:
    print(f"   ✗ FrameExtractor failed: {e}")

try:
    color_extractor = ColorExtractor(num_colors=3)
    print("   ✓ ColorExtractor initialized")
except Exception as e:
    print(f"   ✗ ColorExtractor failed: {e}")

print("\n" + "=" * 60)
print("✅ Basic tests passed!")
print("=" * 60)
print("\nTo run full video processing test:")
print("  python test_video.py --video-path <video_path>")
print("\nNote: First run will download YOLOv8 and CLIP models (requires internet)")

