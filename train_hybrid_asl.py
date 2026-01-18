#!/usr/bin/env python3
"""
TRAINING SCRIPT FOR HYBRID ASL MODEL
Usage: python train_hybrid_asl.py --data_dir /path/to/wlasl
"""

import argparse
import torch
from torch.utils.data import DataLoader, random_split
from transformers import VideoMAEImageProcessor
from pathlib import Path
import json

from hybrid_asl_model import (
    HybridASLModel,
    HybridASLDataset,
    HybridASLTrainer,
    MediaPipeLandmarkExtractor
)


def download_mediapipe_models():
    """Download required MediaPipe task models."""
    import urllib.request
    
    models = {
        'hand_landmarker. task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    
    for filename, url in models.items():
        if not Path(filename).exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"✓ Downloaded {filename}")


def load_wlasl_dataset(data_dir, subset_file='nslt_100.json', split=None):
    """
    Load WLASL dataset using flat video structure with JSON annotations.
    
    Args:
        data_dir: Path to wlasl data directory (containing videos/ and JSON files)
        subset_file: Which subset to use ('nslt_100.json', 'nslt_300.json', etc.)
        split: Optional filter for 'train', 'val', or 'test'. None = all splits.
    
    Returns:
        video_paths: List of video file paths
        labels: List of class indices
        idx_to_gloss: Dict mapping class index to gloss name
    """
    data_dir = Path(data_dir)
    videos_dir = data_dir / 'videos'
    
    # Load subset config (video_id -> class mapping)
    subset_path = data_dir / subset_file
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    
    with open(subset_path, 'r') as f:
        subset_data = json.load(f)
    
    # Load main annotations (for gloss names)
    wlasl_path = data_dir / 'WLASL_v0.3.json'
    if not wlasl_path.exists():
        raise FileNotFoundError(f"WLASL annotations not found: {wlasl_path}")
    
    with open(wlasl_path, 'r') as f:
        wlasl_data = json.load(f)
    
    # Build video_id -> gloss mapping from WLASL_v0.3.json
    video_id_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id_to_gloss[instance['video_id']] = gloss
    
    # Build class_idx -> gloss mapping (inferred from subset data)
    idx_to_gloss = {}
    for video_id, info in subset_data.items():
        class_idx = info['action'][0]
        if video_id in video_id_to_gloss:
            gloss = video_id_to_gloss[video_id]
            if class_idx not in idx_to_gloss:
                idx_to_gloss[class_idx] = gloss
    
    video_paths = []
    labels = []
    skipped = 0
    
    for video_id, info in subset_data.items():
        # Filter by split if specified
        if split is not None and info['subset'] != split:
            continue
        
        video_path = videos_dir / f"{video_id}.mp4"
        
        # Only include videos that exist
        if not video_path.exists():
            skipped += 1
            continue
        
        class_idx = info['action'][0]
        
        video_paths.append(str(video_path))
        labels.append(class_idx)
    
    print(f"Found {len(video_paths)} videos")
    print(f"Found {len(idx_to_gloss)} classes")
    if skipped > 0:
        print(f"Skipped {skipped} missing videos")
    if split:
        print(f"Using split: {split}")
    
    # Save label mapping (idx -> gloss for inference)
    with open('label_mapping.json', 'w') as f:
        # Convert int keys to strings for JSON
        json.dump({str(k): v for k, v in idx_to_gloss.items()}, f, indent=2)
    
    return video_paths, labels, idx_to_gloss


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid ASL Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to WLASL dataset')
    parser.add_argument('--subset', type=str, default='nslt_100.json',
                        choices=['nslt_100.json', 'nslt_300.json', 'nslt_1000.json', 'nslt_2000.json'],
                        help='Subset file to use (vocabulary size)')
    parser.add_argument('--split', type=str, default=None,
                        choices=['train', 'val', 'test'],
                        help='Use specific split from JSON (default: use all with random split)')
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_type', type=str, default='concat', 
                        choices=['concat', 'attention', 'gated'])
    parser.add_argument('--freeze_videomae', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print("="*70)
    print("HYBRID ASL MODEL TRAINING")
    print("="*70)
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Download MediaPipe models
    # ─────────────────────────────────────────────────────────────────
    print("Step 1: Checking MediaPipe models...")
    download_mediapipe_models()
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Load dataset
    # ─────────────────────────────────────────────────────────────────
    print("Step 2: Loading WLASL dataset...")
    print(f"  Subset: {args.subset}")
    print(f"  Split filter: {args.split or 'None (using random 80/20 split)'}")
    video_paths, labels, idx_to_gloss = load_wlasl_dataset(
        args.data_dir, 
        subset_file=args.subset,
        split=args.split
    )
    num_classes = len(idx_to_gloss)
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 3. Initialize extractors and processors
    # ─────────────────────────────────────────────────────────────────
    print("Step 3: Initializing feature extractors...")
    
    landmark_extractor = MediaPipeLandmarkExtractor(
        hand_model_path='hand_landmarker.task',
        pose_model_path='pose_landmarker. task'
    )
    
    videomae_processor = VideoMAEImageProcessor. from_pretrained('MCG-NJU/videomae-base')
    
    print("✓ MediaPipe landmark extractor ready")
    print("✓ VideoMAE processor ready")
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Create dataset and dataloaders
    # ─────────────────────────────────────────────────────────────────
    print("Step 4: Creating datasets...")
    
    full_dataset = HybridASLDataset(
        video_paths=video_paths,
        labels=labels,
        landmark_extractor=landmark_extractor,
        videomae_processor=videomae_processor,
        num_frames=args.num_frames
    )
    
    # Split into train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    print(f"✓ Train samples: {len(train_dataset)}")
    print(f"✓ Val samples: {len(val_dataset)}")
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Create model
    # ─────────────────────────────────────────────────────────────────
    print("Step 5: Creating hybrid model...")
    
    model = HybridASLModel(
        num_classes=num_classes,
        videomae_model='MCG-NJU/videomae-base',
        hidden_dim=args.hidden_dim,
        freeze_videomae=args.freeze_videomae,
        fusion_type=args.fusion_type,
        dropout=0.3
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p. numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Fusion type: {args.fusion_type}")
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 6. Train
    # ─────────────────────────────────────────────────────────────────
    print("Step 6: Starting training...")
    print("="*70)
    print()
    
    trainer = HybridASLTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate
    )
    
    best_acc = trainer.train(
        num_epochs=args.num_epochs,
        save_path='best_hybrid_asl_model. pth'
    )
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to: best_hybrid_asl_model. pth")
    print(f"Label mapping saved to: label_mapping. json")
    
    # Cleanup
    landmark_extractor.close()


if __name__ == "__main__":
    main()