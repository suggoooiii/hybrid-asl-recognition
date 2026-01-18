"""Analyze available WLASL subsets and their video counts."""

import json
import os
from pathlib import Path
from collections import defaultdict


def analyze_subsets(data_dir: str = "data/wlasl"):
    """Analyze all available subsets."""
    data_path = Path(data_dir)

    # Find all subset JSON files
    subset_files = list(data_path.glob("nslt_*.json"))

    print("=" * 70)
    print("WLASL SUBSET ANALYSIS")
    print("=" * 70)

    # Check for missing videos
    missing_file = data_path / "missing.txt"
    missing_videos = set()
    if missing_file.exists():
        with open(missing_file) as f:
            missing_videos = set(line.strip() for line in f if line.strip())
        print(f"\nMissing videos (from missing.txt): {len(missing_videos)}")

    # Check available videos
    videos_dir = data_path / "videos"
    available_videos = set()
    if videos_dir.exists():
        available_videos = set(
            f.stem for f in videos_dir.glob("*.mp4")
        )
        print(f"Available video files: {len(available_videos)}")

    # Check pre-extracted features
    gemma_dir = data_path / "gemma_features"
    landmarks_dir = data_path / "landmarks"

    gemma_features = set()
    if gemma_dir.exists():
        gemma_features = set(
            f.stem.replace("_gemma", "") for f in gemma_dir.glob("*_gemma.npy")
        )
        print(f"Pre-extracted Gemma features: {len(gemma_features)}")

    landmark_features = set()
    if landmarks_dir.exists():
        landmark_features = set(
            f.stem.replace("_landmarks", "") for f in landmarks_dir.glob("*_landmarks.npy")
        )
        print(f"Pre-extracted landmark features: {len(landmark_features)}")

    # Videos with both features (ready for training)
    ready_for_training = gemma_features & landmark_features
    print(
        f"Videos ready for training (both features): {len(ready_for_training)}")

    print("\n" + "=" * 70)
    print("SUBSET BREAKDOWN")
    print("=" * 70)

    results = []

    for subset_file in sorted(subset_files):
        subset_name = subset_file.stem

        with open(subset_file) as f:
            data = json.load(f)

        # Count videos and classes
        total_videos = len(data)
        classes = set()
        train_count = 0
        val_count = 0
        test_count = 0
        available_count = 0
        ready_count = 0

        for video_id, info in data.items():
            if info and 'action' in info and len(info['action']) >= 1:
                classes.add(info['action'][0])

            subset_type = info.get('subset', 'train') if info else 'train'
            if subset_type == 'train':
                train_count += 1
            elif subset_type == 'val':
                val_count += 1
            elif subset_type == 'test':
                test_count += 1

            if video_id in available_videos:
                available_count += 1

            if video_id in ready_for_training:
                ready_count += 1

        num_classes = len(classes)
        videos_per_class = total_videos / num_classes if num_classes > 0 else 0
        available_per_class = available_count / num_classes if num_classes > 0 else 0
        ready_per_class = ready_count / num_classes if num_classes > 0 else 0

        results.append({
            'name': subset_name,
            'classes': num_classes,
            'total': total_videos,
            'available': available_count,
            'ready': ready_count,
            'train': train_count,
            'val': val_count,
            'test': test_count,
            'per_class': videos_per_class,
            'available_per_class': available_per_class,
            'ready_per_class': ready_per_class,
        })

        print(f"\n{subset_name.upper()}")
        print("-" * 50)
        print(f"  Classes:              {num_classes}")
        print(f"  Total videos:         {total_videos}")
        print(
            f"  Available videos:     {available_count} ({100*available_count/total_videos:.1f}%)")
        print(
            f"  Ready for training:   {ready_count} ({100*ready_count/total_videos:.1f}%)")
        print(
            f"  Train/Val/Test split: {train_count}/{val_count}/{test_count}")
        print(f"  Avg videos/class:     {videos_per_class:.1f} (total)")
        print(f"                        {available_per_class:.1f} (available)")
        print(f"                        {ready_per_class:.1f} (ready)")

    # Recommendation
    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)

    for r in results:
        if r['ready_per_class'] >= 5:
            status = "✓ RECOMMENDED"
        elif r['ready_per_class'] >= 2:
            status = "⚠ MARGINAL"
        else:
            status = "✗ TOO SPARSE"

        print(f"  {r['name']:12} - {r['classes']:4} classes, "
              f"{r['ready']:5} ready videos, "
              f"{r['ready_per_class']:.1f} videos/class - {status}")

    print("\nFor meaningful training, aim for at least 5+ videos per class.")
    print("Use --subset nslt_100.json for best results with current data.")

    return results


if __name__ == "__main__":
    analyze_subsets()
