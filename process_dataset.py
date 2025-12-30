"""Script to process all videos from the AI Hackathon dataset"""

import sys
import json
import os
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).parent))

from src.process_video import process_video
from src.utils import load_video_metadata


def find_video_files(dataset_dir: str) -> List[Dict[str, str]]:
    """
    Find all video files in the dataset directory
    
    Args:
        dataset_dir: Path to dataset directory
    
    Returns:
        List of dicts with video_id, video_path, caption, hashtags
    """
    dataset_path = Path(dataset_dir)
    videos_dir = dataset_path / "videos"
    
    if not videos_dir.exists():
        raise FileNotFoundError(f"Videos directory not found: {videos_dir}")
    
    video_files = []
    
    # Find all .mp4 files
    for mp4_file in videos_dir.glob("*.mp4"):
        video_id = mp4_file.stem  # Get filename without extension
        
        # Load caption and hashtags
        caption, hashtags = load_video_metadata(str(mp4_file))
        
        video_files.append({
            "video_id": video_id,
            "video_path": str(mp4_file),
            "caption": caption,
            "hashtags": hashtags
        })
    
    return video_files


def process_all_videos(dataset_dir: str, output_dir: str = "results"):
    """
    Process all videos from the dataset
    
    Args:
        dataset_dir: Path to AI Hackathon dataset directory
        output_dir: Directory to save results
    """
    # Find all videos
    print(f"Scanning for videos in {dataset_dir}...")
    video_files = find_video_files(dataset_dir)
    
    print(f"Found {len(video_files)} videos to process")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    results = []
    
    # Process each video
    for i, video_info in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] Processing {video_info['video_id']}...")
        
        try:
            result = process_video(
                video_path=video_info['video_path'],
                video_id=video_info['video_id'],
                caption=video_info['caption'],
                hashtags=video_info['hashtags']
            )
            
            # Save individual result
            output_file = output_path / f"{video_info['video_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            results.append(result)
            print(f"✓ Saved to {output_file}")
        
        except Exception as e:
            print(f"✗ Error processing {video_info['video_id']}: {e}")
            results.append({
                "video_id": video_info['video_id'],
                "error": str(e),
                "vibes": [],
                "products": []
            })
    
    # Save combined results
    combined_output = output_path / "all_results.json"
    with open(combined_output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ All results saved to {combined_output}")
    print(f"Processed {len(results)} videos")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process all videos from AI Hackathon dataset")
    parser.add_argument(
        "--dataset-dir",
        required=True,
        help="Path to AI Hackathon dataset directory"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results"
    )
    
    args = parser.parse_args()
    
    process_all_videos(args.dataset_dir, args.output_dir)

