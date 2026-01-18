#!/usr/bin/env python3
"""
PRE-EXTRACT LANDMARKS FOR ALL VIDEOS
Run this locally before uploading to Kaggle to speed up training 10-50x.

Usage:
    python preprocess_landmarks.py --data_dir data/wlasl --output_dir data/wlasl/landmarks
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import urllib.request


def download_mediapipe_models():
    """Download required MediaPipe task models."""
    models = {
        'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    
    for filename, url in models.items():
        if not Path(filename).exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Downloaded {filename}")


def main():
    parser = argparse.ArgumentParser(description='Pre-extract landmarks from videos')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to WLASL data directory (containing videos/)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for landmark .npy files')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract per video')
    parser.add_argument('--subset', type=str, default=None,
                        help='Optional subset file (e.g., nslt_100.json) to limit videos')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos that already have extracted landmarks')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LANDMARK PRE-EXTRACTION")
    print("="*70)
    print()
    
    # Download MediaPipe models if needed
    print("Step 1: Checking MediaPipe models...")
    download_mediapipe_models()
    print()
    
    # Import after download check
    from hybrid_asl_model import MediaPipeLandmarkExtractor
    
    # Setup paths
    data_dir = Path(args.data_dir)
    videos_dir = data_dir / 'videos'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Step 2: Collecting videos...")
    print(f"  Videos directory: {videos_dir}")
    print(f"  Output directory: {output_dir}")
    
    # Get video list
    if args.subset:
        # Load subset file to get specific video IDs
        subset_path = data_dir / args.subset
        if not subset_path.exists():
            raise FileNotFoundError(f"Subset file not found: {subset_path}")
        
        with open(subset_path, 'r') as f:
            subset_data = json.load(f)
        
        video_paths = []
        for video_id in subset_data.keys():
            video_path = videos_dir / f"{video_id}.mp4"
            if video_path.exists():
                video_paths.append(video_path)
        
        print(f"  Using subset: {args.subset}")
    else:
        # Get all videos in directory
        video_paths = list(videos_dir.glob('*.mp4'))
    
    print(f"  Found {len(video_paths)} videos")
    
    # Filter existing if requested
    if args.skip_existing:
        original_count = len(video_paths)
        video_paths = [
            vp for vp in video_paths 
            if not (output_dir / f"{vp.stem}_landmarks.npy").exists()
        ]
        skipped = original_count - len(video_paths)
        print(f"  Skipping {skipped} videos with existing landmarks")
        print(f"  Processing {len(video_paths)} remaining videos")
    
    if len(video_paths) == 0:
        print("\n✓ All videos already processed!")
        return
    
    print()
    print("Step 3: Initializing MediaPipe extractor...")
    
    extractor = MediaPipeLandmarkExtractor(
        hand_model_path='hand_landmarker.task',
        pose_model_path='pose_landmarker.task'
    )
    print("✓ Extractor ready")
    print()
    
    print("Step 4: Extracting landmarks...")
    print("="*70)
    
    success_count = 0
    failed_videos = []
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        try:
            # Extract landmarks
            landmarks = extractor.extract_video_landmarks(
                video_path, 
                max_frames=args.num_frames
            )
            
            # Save as .npy
            output_path = output_dir / f"{video_path.stem}_landmarks.npy"
            np.save(output_path, landmarks)
            
            success_count += 1
            
        except Exception as e:
            failed_videos.append((video_path.name, str(e)))
            continue
    
    # Cleanup
    extractor.close()
    
    print()
    print("="*70)
    print("EXTRACTION COMPLETE")
    print("="*70)
    print(f"✓ Successfully processed: {success_count}/{len(video_paths)} videos")
    print(f"✓ Landmarks saved to: {output_dir}")
    
    if failed_videos:
        print(f"\n⚠ Failed videos ({len(failed_videos)}):")
        for video_name, error in failed_videos[:10]:  # Show first 10
            print(f"  - {video_name}: {error}")
        if len(failed_videos) > 10:
            print(f"  ... and {len(failed_videos) - 10} more")
        
        # Save failed videos list
        failed_path = output_dir / 'failed_videos.txt'
        with open(failed_path, 'w') as f:
            for video_name, error in failed_videos:
                f.write(f"{video_name}: {error}\n")
        print(f"\n✓ Failed videos list saved to: {failed_path}")
    
    # Print summary of output
    npy_files = list(output_dir.glob('*_landmarks.npy'))
    total_size_mb = sum(f.stat().st_size for f in npy_files) / (1024 * 1024)
    print(f"\n📁 Output summary:")
    print(f"   Files: {len(npy_files)} .npy files")
    print(f"   Size: {total_size_mb:.1f} MB")
    print(f"   Shape per file: ({args.num_frames}, 162)")


if __name__ == "__main__":
    main()
