"""Quick check of feature file naming."""

import os
from pathlib import Path

data_dir = Path("data/wlasl")

# Check gemma features
gemma_dir = data_dir / "gemma_features"
if gemma_dir.exists():
    files = list(gemma_dir.glob("*.npy"))[:10]
    print("Gemma feature files (first 10):")
    for f in files:
        print(f"  {f.name}")
    print(f"  Total: {len(list(gemma_dir.glob('*.npy')))} files")

# Check landmarks
landmarks_dir = data_dir / "landmarks"
if landmarks_dir.exists():
    files = list(landmarks_dir.glob("*.npy"))[:10]
    print("\nLandmark files (first 10):")
    for f in files:
        print(f"  {f.name}")
    print(f"  Total: {len(list(landmarks_dir.glob('*.npy')))} files")

# Check video IDs
videos_dir = data_dir / "videos"
if videos_dir.exists():
    files = list(videos_dir.glob("*.mp4"))[:10]
    print("\nVideo files (first 10):")
    for f in files:
        print(f"  {f.name}")
