#!/usr/bin/env python3
"""
FAST TRAINING SCRIPT USING PRE-EXTRACTED GEMMA FEATURES

Train the simplified hybrid model using pre-extracted features.
Much faster than training with VideoMAE since no large model forward pass.

Usage:
    python train_simple.py \
        --data_dir data/wlasl \
        --gemma_features_dir data/wlasl/gemma_features \
        --landmarks_dir data/wlasl/landmarks \
        --batch_size 32 \
        --epochs 50

Key Benefits:
    - 10-50x faster training (no Gemma forward pass)
    - Can use larger batch sizes (32-64+)
    - Only ~2M trainable parameters
    - Lower GPU memory requirements
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from pathlib import Path
import json
from tqdm import tqdm

from hybrid_asl_model_simple import HybridASLModelSimple, PreextractedDataset


def load_wlasl_metadata(data_dir, subset_file='nslt_100.json'):
    """
    Load WLASL metadata for training.
    
    Returns:
        video_ids: List of video IDs
        labels: List of class labels
        idx_to_gloss: Dict mapping class index to gloss name
    """
    data_dir = Path(data_dir)
    
    # Load missing videos
    missing_videos = set()
    missing_path = data_dir / 'missing.txt'
    if missing_path.exists():
        with open(missing_path, 'r') as f:
            missing_videos = {line.strip() for line in f if line.strip()}
    
    # Load subset config
    subset_path = data_dir / subset_file
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")
    
    with open(subset_path, 'r') as f:
        subset_data = json.load(f)
    
    # Load main annotations for gloss names
    wlasl_path = data_dir / 'WLASL_v0.3.json'
    if not wlasl_path.exists():
        raise FileNotFoundError(f"WLASL annotations not found: {wlasl_path}")
    
    with open(wlasl_path, 'r') as f:
        wlasl_data = json.load(f)
    
    # Build video_id -> gloss mapping
    video_id_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id_to_gloss[instance['video_id']] = gloss
    
    # Build class_idx -> gloss mapping
    idx_to_gloss = {}
    for video_id, info in subset_data.items():
        class_idx = info['action'][0]
        if video_id in video_id_to_gloss:
            gloss = video_id_to_gloss[video_id]
            if class_idx not in idx_to_gloss:
                idx_to_gloss[class_idx] = gloss
    
    video_ids = []
    labels = []
    skipped = 0
    
    for video_id, info in subset_data.items():
        # Skip missing videos
        if video_id in missing_videos:
            skipped += 1
            continue
        
        video_path = data_dir / 'videos' / f"{video_id}.mp4"
        if not video_path.exists():
            skipped += 1
            continue
        
        class_idx = info['action'][0]
        video_ids.append(video_id)
        labels.append(class_idx)
    
    print(f"Found {len(video_ids)} videos")
    print(f"Found {len(idx_to_gloss)} classes")
    if skipped > 0:
        print(f"Skipped {skipped} missing videos")
    
    return video_ids, labels, idx_to_gloss


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        gemma_features = batch['gemma_features'].to(device)
        landmarks = batch['landmarks'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        logits = model(gemma_features, landmarks)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return total_loss / len(train_loader), 100. * correct / total


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss = 0
    correct = 0
    correct_top5 = 0
    total = 0
    
    pbar = tqdm(val_loader, desc="Validating", leave=False)
    for batch in pbar:
        gemma_features = batch['gemma_features'].to(device)
        landmarks = batch['landmarks'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(gemma_features, landmarks)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        
        # Top-1 accuracy
        _, predicted = logits.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Top-5 accuracy
        _, top5_pred = logits.topk(5, dim=1)
        correct_top5 += sum(labels[i] in top5_pred[i] for i in range(labels.size(0)))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    top1_acc = 100. * correct / total
    top5_acc = 100. * correct_top5 / total
    return total_loss / len(val_loader), top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train simplified hybrid model with pre-extracted features')
    
    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WLASL dataset')
    parser.add_argument('--gemma_features_dir', type=str, required=True,
                        help='Directory with pre-extracted Gemma features')
    parser.add_argument('--landmarks_dir', type=str, default=None,
                        help='Directory with pre-extracted landmarks (optional)')
    parser.add_argument('--subset', type=str, default='nslt_100.json',
                        help='Subset file (nslt_100.json, nslt_300.json, etc.)')
    
    # Model config
    parser.add_argument('--gemma_feature_dim', type=int, default=None,
                        help='Gemma feature dimension (auto-detected from metadata)')
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_type', type=str, default='concat',
                        choices=['concat', 'attention', 'gated'])
    
    # Training config
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--device', type=str, default='cuda')
    
    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_simple',
                        help='Directory to save checkpoints')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    print("="*70)
    print("SIMPLIFIED HYBRID MODEL TRAINING")
    print("="*70)
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 1. Load metadata and detect feature dimension
    # ─────────────────────────────────────────────────────────────────
    print("Step 1: Loading dataset metadata...")
    
    video_ids, labels, idx_to_gloss = load_wlasl_metadata(
        args.data_dir,
        subset_file=args.subset
    )
    num_classes = len(idx_to_gloss)
    
    # Detect Gemma feature dimension from metadata
    gemma_metadata_path = Path(args.gemma_features_dir) / 'extraction_metadata.json'
    if gemma_metadata_path.exists():
        with open(gemma_metadata_path, 'r') as f:
            metadata = json.load(f)
            gemma_feature_dim = metadata.get('feature_dim', 2048)
            print(f"✓ Detected Gemma feature dimension: {gemma_feature_dim}")
    else:
        gemma_feature_dim = args.gemma_feature_dim or 2048
        print(f"⚠ Using default Gemma feature dimension: {gemma_feature_dim}")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 2. Create datasets
    # ─────────────────────────────────────────────────────────────────
    print("Step 2: Creating datasets...")
    
    # If landmarks_dir not specified, use default from data_dir
    landmarks_dir = args.landmarks_dir
    if landmarks_dir is None:
        landmarks_dir = Path(args.data_dir) / 'landmarks'
        if not landmarks_dir.exists():
            raise ValueError(
                f"Landmarks directory not found: {landmarks_dir}\n"
                "Please run preprocess_landmarks.py first or specify --landmarks_dir"
            )
    
    full_dataset = PreextractedDataset(
        video_ids=video_ids,
        labels=labels,
        gemma_features_dir=args.gemma_features_dir,
        landmarks_dir=landmarks_dir,
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
    # 3. Create model
    # ─────────────────────────────────────────────────────────────────
    print("Step 3: Creating model...")
    
    model = HybridASLModelSimple(
        num_classes=num_classes,
        gemma_feature_dim=gemma_feature_dim,
        hidden_dim=args.hidden_dim,
        fusion_type=args.fusion_type,
        dropout=0.3
    ).to(args.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    print(f"✓ Fusion type: {args.fusion_type}")
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 4. Setup training
    # ─────────────────────────────────────────────────────────────────
    print("Step 4: Setup training...")
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6
    )
    
    criterion = nn.CrossEntropyLoss()
    
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Optimizer: AdamW (lr={args.learning_rate})")
    print(f"✓ Scheduler: CosineAnnealingLR")
    print(f"✓ Checkpoint dir: {checkpoint_dir}")
    print()
    
    # ─────────────────────────────────────────────────────────────────
    # 5. Train
    # ─────────────────────────────────────────────────────────────────
    print("Step 5: Starting training...")
    print("="*70)
    print()
    
    best_val_acc = 0
    
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, args.device
        )
        
        val_loss, val_acc, val_top5 = evaluate(
            model, val_loader, criterion, args.device
        )
        
        scheduler.step()
        
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Top-1: {val_acc:.2f}%, Val Top-5: {val_top5:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_simple_model.pth')
            print(f"  ✓ New best model saved! ({val_acc:.2f}%)")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_acc': best_val_acc,
                'val_acc': val_acc,
            }
            checkpoint_path = checkpoint_dir / f'checkpoint_epoch_{epoch+1}.pth'
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Checkpoint saved: {checkpoint_path}")
        
        print()
    
    # Save final model
    torch.save(model.state_dict(), 'final_simple_model.pth')
    
    print("="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"✓ Best validation accuracy: {best_val_acc:.2f}%")
    print(f"✓ Best model saved to: best_simple_model.pth")
    print(f"✓ Final model saved to: final_simple_model.pth")
    print(f"✓ Checkpoints saved to: {checkpoint_dir}/")
    
    # Save label mapping
    with open('label_mapping.json', 'w') as f:
        json.dump({str(k): v for k, v in idx_to_gloss.items()}, f, indent=2)
    print(f"✓ Label mapping saved to: label_mapping.json")
    print()


if __name__ == '__main__':
    main()
