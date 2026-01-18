#!/usr/bin/env python3
"""
MODEL EVALUATION SCRIPT
Computes Top-1, Top-5 accuracy, per-class accuracy, and confusion matrix.

Usage:
    python evaluate_model.py --checkpoint best_hybrid_asl_model.pth --data_dir data/wlasl
"""

import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import VideoMAEImageProcessor
from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tqdm import tqdm

from hybrid_asl_model import (
    HybridASLModel,
    HybridASLDataset,
    MediaPipeLandmarkExtractor
)


def download_mediapipe_models():
    """Download required MediaPipe task models."""
    import urllib.request
    
    models = {
        'hand_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
        'pose_landmarker.task': 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task'
    }
    
    for filename, url in models.items():
        if not Path(filename).exists():
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)


def load_wlasl_dataset(data_dir, subset_file='nslt_100.json', split='test'):
    """Load WLASL dataset for specific split with frame trimming info."""
    data_dir = Path(data_dir)
    videos_dir = data_dir / 'videos'
    
    # Load missing videos list for faster filtering
    missing_videos = set()
    missing_path = data_dir / 'missing.txt'
    if missing_path.exists():
        with open(missing_path, 'r') as f:
            missing_videos = {line.strip() for line in f if line.strip()}
    
    with open(data_dir / subset_file, 'r') as f:
        subset_data = json.load(f)
    
    with open(data_dir / 'WLASL_v0.3.json', 'r') as f:
        wlasl_data = json.load(f)
    
    video_id_to_gloss = {}
    for entry in wlasl_data:
        gloss = entry['gloss']
        for instance in entry['instances']:
            video_id_to_gloss[instance['video_id']] = gloss
    
    idx_to_gloss = {}
    for video_id, info in subset_data.items():
        class_idx = info['action'][0]
        if video_id in video_id_to_gloss:
            gloss = video_id_to_gloss[video_id]
            if class_idx not in idx_to_gloss:
                idx_to_gloss[class_idx] = gloss
    
    video_paths = []
    labels = []
    frame_info = []  # (start_frame, end_frame) for each video
    skipped = 0
    
    for video_id, info in subset_data.items():
        if split is not None and info['subset'] != split:
            continue
        
        # Quick check: skip if in missing.txt
        if video_id in missing_videos:
            skipped += 1
            continue
        
        video_path = videos_dir / f"{video_id}.mp4"
        if not video_path.exists():
            skipped += 1
            continue
        
        # action format: [class_idx, start_frame, end_frame]
        video_paths.append(str(video_path))
        labels.append(info['action'][0])
        frame_info.append((info['action'][1], info['action'][2]))
    
    print(f"Loaded {len(video_paths)} videos for split '{split}'")
    if skipped > 0:
        print(f"Skipped {skipped} missing videos")
    print(f"Frame trimming: enabled")
    
    return video_paths, labels, idx_to_gloss, frame_info


@torch.no_grad()
def evaluate_model(model, dataloader, device, num_classes):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_probs = []
    
    # Per-class tracking
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        pixel_values = batch['pixel_values'].to(device)
        landmarks = batch['landmarks'].to(device)
        labels = batch['label'].to(device)
        
        logits = model(pixel_values, landmarks)
        probs = F.softmax(logits, dim=-1)
        
        _, predicted = logits.max(1)
        
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        
        # Track per-class accuracy
        for pred, label in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
            class_total[label] += 1
            if pred == label:
                class_correct[label] += 1
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.vstack(all_probs)
    
    # Compute Top-1 accuracy
    top1_correct = (all_predictions == all_labels).sum()
    top1_accuracy = 100. * top1_correct / len(all_labels)
    
    # Compute Top-5 accuracy
    top5_pred = np.argsort(-all_probs, axis=1)[:, :5]
    top5_correct = sum(all_labels[i] in top5_pred[i] for i in range(len(all_labels)))
    top5_accuracy = 100. * top5_correct / len(all_labels)
    
    # Per-class accuracy
    per_class_acc = {}
    for class_idx in class_total:
        if class_total[class_idx] > 0:
            per_class_acc[class_idx] = 100. * class_correct[class_idx] / class_total[class_idx]
        else:
            per_class_acc[class_idx] = 0.0
    
    return {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'per_class_accuracy': per_class_acc,
        'predictions': all_predictions,
        'labels': all_labels,
        'class_total': dict(class_total),
        'class_correct': dict(class_correct)
    }


def plot_confusion_matrix(predictions, labels, idx_to_gloss, output_path='confusion_matrix.png', max_classes=50):
    """Plot and save confusion matrix."""
    from sklearn.metrics import confusion_matrix
    
    # Get unique classes
    unique_classes = sorted(set(labels) | set(predictions))
    
    # Limit to max_classes for readability
    if len(unique_classes) > max_classes:
        print(f"Limiting confusion matrix to top {max_classes} classes")
        # Get most frequent classes
        class_counts = defaultdict(int)
        for label in labels:
            class_counts[label] += 1
        top_classes = sorted(class_counts.keys(), key=lambda x: class_counts[x], reverse=True)[:max_classes]
        
        # Filter to only include these classes
        mask = np.isin(labels, top_classes)
        predictions = predictions[mask]
        labels = labels[mask]
        unique_classes = sorted(set(labels) | set(predictions))
    
    # Compute confusion matrix
    cm = confusion_matrix(labels, predictions, labels=unique_classes)
    
    # Normalize
    cm_normalized = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-10)
    
    # Get class names
    class_names = [idx_to_gloss.get(c, str(c))[:10] for c in unique_classes]
    
    # Plot
    fig_size = max(10, len(unique_classes) // 3)
    plt.figure(figsize=(fig_size, fig_size))
    
    sns.heatmap(cm_normalized, annot=False, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Normalized)')
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0, fontsize=6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Confusion matrix saved to: {output_path}")


def plot_per_class_accuracy(per_class_acc, idx_to_gloss, output_path='per_class_accuracy.png'):
    """Plot per-class accuracy bar chart."""
    # Sort by accuracy
    sorted_classes = sorted(per_class_acc.keys(), key=lambda x: per_class_acc[x])
    
    # Limit to top/bottom 30 for readability
    if len(sorted_classes) > 60:
        bottom_30 = sorted_classes[:30]
        top_30 = sorted_classes[-30:]
        selected_classes = bottom_30 + top_30
    else:
        selected_classes = sorted_classes
    
    class_names = [idx_to_gloss.get(c, str(c))[:15] for c in selected_classes]
    accuracies = [per_class_acc[c] for c in selected_classes]
    
    # Color by accuracy
    colors = ['red' if acc < 50 else 'orange' if acc < 75 else 'green' for acc in accuracies]
    
    plt.figure(figsize=(12, max(8, len(selected_classes) // 4)))
    plt.barh(class_names, accuracies, color=colors)
    plt.xlabel('Accuracy (%)')
    plt.ylabel('Class')
    plt.title('Per-Class Accuracy')
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✓ Per-class accuracy plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hybrid ASL Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint (.pth file)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WLASL dataset')
    parser.add_argument('--subset', type=str, default='nslt_100.json',
                        help='Subset file to use')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'val', 'test', 'all'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--fusion_type', type=str, default='concat')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--landmarks_dir', type=str, default=None,
                        help='Directory with pre-extracted landmarks')
    parser.add_argument('--label_mapping', type=str, default='label_mapping.json',
                        help='Path to label mapping JSON')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                        help='Directory to save evaluation outputs')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("HYBRID ASL MODEL EVALUATION")
    print("="*70)
    print()
    
    # Load label mapping
    print("Step 1: Loading label mapping...")
    if Path(args.label_mapping).exists():
        with open(args.label_mapping, 'r') as f:
            idx_to_gloss = {int(k): v for k, v in json.load(f).items()}
        print(f"✓ Loaded {len(idx_to_gloss)} classes from {args.label_mapping}")
    else:
        print(f"⚠ Label mapping not found, will extract from dataset")
        idx_to_gloss = None
    print()
    
    # Load dataset
    print("Step 2: Loading dataset...")
    split = None if args.split == 'all' else args.split
    video_paths, labels, dataset_idx_to_gloss, frame_info = load_wlasl_dataset(
        args.data_dir, 
        subset_file=args.subset,
        split=split
    )
    
    if idx_to_gloss is None:
        idx_to_gloss = dataset_idx_to_gloss
    
    num_classes = len(idx_to_gloss)
    print(f"Number of classes: {num_classes}")
    print()
    
    # Download MediaPipe models if needed
    need_mediapipe = args.landmarks_dir is None
    if need_mediapipe:
        print("Step 3: Downloading MediaPipe models...")
        download_mediapipe_models()
    else:
        print("Step 3: Using pre-extracted landmarks")
    print()
    
    # Initialize extractors
    print("Step 4: Initializing extractors...")
    
    landmark_extractor = None
    if need_mediapipe:
        landmark_extractor = MediaPipeLandmarkExtractor(
            hand_model_path='hand_landmarker.task',
            pose_model_path='pose_landmarker.task'
        )
        print("✓ MediaPipe extractor ready")
    
    videomae_processor = VideoMAEImageProcessor.from_pretrained('MCG-NJU/videomae-base')
    print("✓ VideoMAE processor ready")
    print()
    
    # Create dataset and dataloader
    print("Step 5: Creating dataloader...")
    dataset = HybridASLDataset(
        video_paths=video_paths,
        labels=labels,
        landmark_extractor=landmark_extractor,
        videomae_processor=videomae_processor,
        num_frames=args.num_frames,
        landmarks_dir=args.landmarks_dir,
        frame_info=frame_info  # Pass frame trimming info
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    print(f"✓ Dataset size: {len(dataset)}")
    print()
    
    # Load model
    print("Step 6: Loading model...")
    model = HybridASLModel(
        num_classes=num_classes,
        videomae_model='MCG-NJU/videomae-base',
        hidden_dim=args.hidden_dim,
        fusion_type=args.fusion_type,
        dropout=0.3
    )
    
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Handle both full checkpoint and state_dict only
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(args.device)
    model.eval()
    print(f"✓ Model loaded from {args.checkpoint}")
    print()
    
    # Evaluate
    print("Step 7: Evaluating model...")
    print("="*70)
    results = evaluate_model(model, dataloader, args.device, num_classes)
    
    print()
    print("="*70)
    print("EVALUATION RESULTS")
    print("="*70)
    print(f"Top-1 Accuracy: {results['top1_accuracy']:.2f}%")
    print(f"Top-5 Accuracy: {results['top5_accuracy']:.2f}%")
    print(f"Total samples: {len(results['labels'])}")
    print()
    
    # Per-class statistics
    per_class_acc = results['per_class_accuracy']
    accuracies = list(per_class_acc.values())
    
    print("Per-Class Accuracy Statistics:")
    print(f"  Mean: {np.mean(accuracies):.2f}%")
    print(f"  Std: {np.std(accuracies):.2f}%")
    print(f"  Min: {np.min(accuracies):.2f}%")
    print(f"  Max: {np.max(accuracies):.2f}%")
    print()
    
    # Show worst performing classes
    print("Worst 10 Classes:")
    sorted_classes = sorted(per_class_acc.keys(), key=lambda x: per_class_acc[x])
    for class_idx in sorted_classes[:10]:
        gloss = idx_to_gloss.get(class_idx, str(class_idx))
        acc = per_class_acc[class_idx]
        total = results['class_total'].get(class_idx, 0)
        print(f"  {gloss}: {acc:.1f}% ({total} samples)")
    print()
    
    # Show best performing classes
    print("Best 10 Classes:")
    for class_idx in sorted_classes[-10:]:
        gloss = idx_to_gloss.get(class_idx, str(class_idx))
        acc = per_class_acc[class_idx]
        total = results['class_total'].get(class_idx, 0)
        print(f"  {gloss}: {acc:.1f}% ({total} samples)")
    print()
    
    # Save visualizations
    print("Step 8: Generating visualizations...")
    
    plot_confusion_matrix(
        results['predictions'],
        results['labels'],
        idx_to_gloss,
        output_path=output_dir / 'confusion_matrix.png'
    )
    
    plot_per_class_accuracy(
        per_class_acc,
        idx_to_gloss,
        output_path=output_dir / 'per_class_accuracy.png'
    )
    
    # Save detailed results to JSON
    results_json = {
        'top1_accuracy': results['top1_accuracy'],
        'top5_accuracy': results['top5_accuracy'],
        'total_samples': len(results['labels']),
        'num_classes': num_classes,
        'per_class_accuracy': {str(k): v for k, v in per_class_acc.items()},
        'class_counts': {str(k): v for k, v in results['class_total'].items()},
        'split': args.split,
        'checkpoint': args.checkpoint,
    }
    
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Results saved to: {output_dir / 'evaluation_results.json'}")
    print()
    
    print("="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    
    # Cleanup
    if landmark_extractor:
        landmark_extractor.close()


if __name__ == "__main__":
    main()
