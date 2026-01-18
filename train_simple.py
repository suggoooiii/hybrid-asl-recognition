#!/usr/bin/env python3
"""
Simple training script for Hybrid ASL Recognition using pre-extracted features.
Loads pre-extracted Gemma and landmark features from .npy files for fast training.

This is the PRIMARY training pipeline - optimized for fast iteration with:
- Pre-extracted Gemma features (frozen visual encoder)
- Pre-extracted MediaPipe landmarks
- ~2.8M trainable parameters
- 2-4 GB GPU memory requirement
"""

import argparse
import json
import logging
import math
import os
import random
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from hybrid_asl_model_simple import SimpleHybridASLModel


def setup_logging(log_dir: str = "logs", verbose: bool = False) -> tuple:
    """Setup logging to both file and console."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Create logger
    logger = logging.getLogger("train_simple")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []  # Clear existing handlers

    # File handler (DEBUG level - capture everything)
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler (INFO or DEBUG based on verbose)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    return logger, log_file


def compute_class_weights(labels: list, num_classes: int, smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute inverse-frequency class weights with smoothing.
    
    Args:
        labels: List of class labels
        num_classes: Total number of classes
        smoothing: Smoothing factor to prevent division by zero
        
    Returns:
        torch.FloatTensor of class weights
    """
    counts = Counter(labels)
    counts_array = np.array([counts.get(i, 0) for i in range(num_classes)])
    counts_array = counts_array + smoothing  # Prevent division by zero
    weights = 1.0 / counts_array
    weights = weights * (num_classes / weights.sum())  # Normalize
    weights = np.clip(weights, 0.1, 10.0)  # Clip extreme values
    return torch.FloatTensor(weights)


def verify_feature_dimensions(gemma_features_dir: str, expected_dim: int) -> int:
    """
    Verify pre-extracted features match expected dimensions.
    
    Args:
        gemma_features_dir: Directory containing Gemma feature .npy files
        expected_dim: Expected feature dimension
        
    Returns:
        Actual feature dimension found (or expected_dim if verified)
    """
    logger = logging.getLogger("train_simple")
    gemma_dir = Path(gemma_features_dir)
    sample_files = list(gemma_dir.glob("*_gemma.npy"))[:5]
    
    if not sample_files:
        raise FileNotFoundError(f"No .npy files found in {gemma_features_dir}")
    
    for f in sample_files:
        features = np.load(f)
        actual_dim = features.shape[-1]
        if actual_dim != expected_dim:
            logger.warning(f"Feature dimension mismatch! Expected {expected_dim}, got {actual_dim}")
            logger.warning(f"Auto-adjusting gemma_feature_dim to {actual_dim}")
            return actual_dim
    
    logger.info(f"✓ Feature dimensions verified: {expected_dim}")
    return expected_dim


def get_cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int, steps_per_epoch: int):
    """
    Cosine annealing with linear warmup.
    
    Args:
        optimizer: Optimizer to schedule
        warmup_epochs: Number of warmup epochs
        total_epochs: Total number of training epochs
        steps_per_epoch: Number of steps per epoch
        
    Returns:
        LambdaLR scheduler
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class FeatureAugmentation:
    """Data augmentation for pre-extracted features."""

    def __init__(
        self,
        temporal_jitter: bool = True,
        feature_dropout: float = 0.1,
        feature_noise: float = 0.05,
        temporal_mask_prob: float = 0.1,
        landmark_jitter: float = 0.02,
        random_temporal_crop: bool = True,
        mixup_alpha: float = 0.0,  # Set > 0 to enable mixup
    ):
        """
        Initialize augmentation parameters.

        Args:
            temporal_jitter: Randomly shift/shuffle temporal order slightly
            feature_dropout: Probability of dropping feature dimensions
            feature_noise: Std of Gaussian noise to add to features
            temporal_mask_prob: Probability of masking entire frames
            landmark_jitter: Std of jitter to add to landmark coordinates
            random_temporal_crop: Randomly crop temporal dimension
            mixup_alpha: Alpha parameter for mixup (0 = disabled)
        """
        self.temporal_jitter = temporal_jitter
        self.feature_dropout = feature_dropout
        self.feature_noise = feature_noise
        self.temporal_mask_prob = temporal_mask_prob
        self.landmark_jitter = landmark_jitter
        self.random_temporal_crop = random_temporal_crop
        self.mixup_alpha = mixup_alpha

    def __call__(
        self,
        gemma_features: np.ndarray,
        landmark_features: np.ndarray,
        training: bool = True
    ) -> tuple:
        """
        Apply augmentations to features.

        Args:
            gemma_features: (num_frames, feature_dim) visual features
            landmark_features: (num_frames, 162) landmark features
            training: Whether in training mode (augmentations only applied if True)

        Returns:
            Augmented (gemma_features, landmark_features)
        """
        if not training:
            return gemma_features, landmark_features

        gemma_features = gemma_features.copy()
        landmark_features = landmark_features.copy()

        num_frames = gemma_features.shape[0]

        # 1. Temporal jitter - small random shifts in frame order
        if self.temporal_jitter and num_frames > 2 and random.random() < 0.5:
            # Swap adjacent frames randomly
            for i in range(num_frames - 1):
                if random.random() < 0.2:
                    gemma_features[[i, i+1]] = gemma_features[[i+1, i]]
                    landmark_features[[i, i+1]] = landmark_features[[i+1, i]]

        # 2. Random temporal crop and resize
        if self.random_temporal_crop and num_frames > 4 and random.random() < 0.3:
            crop_ratio = random.uniform(0.7, 1.0)
            crop_frames = max(4, int(num_frames * crop_ratio))
            start_idx = random.randint(0, num_frames - crop_frames)

            gemma_cropped = gemma_features[start_idx:start_idx + crop_frames]
            landmark_cropped = landmark_features[start_idx:start_idx + crop_frames]

            # Resize back to original length using linear interpolation
            indices = np.linspace(0, crop_frames - 1, num_frames)
            gemma_features = np.array([
                gemma_cropped[int(i)] * (1 - (i % 1)) +
                gemma_cropped[min(int(i) + 1, crop_frames - 1)] * (i % 1)
                for i in indices
            ])
            landmark_features = np.array([
                landmark_cropped[int(i)] * (1 - (i % 1)) +
                landmark_cropped[min(int(i) + 1, crop_frames - 1)] * (i % 1)
                for i in indices
            ])

        # 3. Temporal masking - randomly mask entire frames
        if self.temporal_mask_prob > 0:
            mask = np.random.random(num_frames) > self.temporal_mask_prob
            # Ensure at least half the frames are kept
            if mask.sum() < num_frames // 2:
                keep_indices = np.random.choice(
                    num_frames, num_frames // 2, replace=False)
                mask[keep_indices] = True

            # Zero out masked frames
            gemma_features[~mask] = 0
            landmark_features[~mask] = 0

        # 4. Feature dropout - randomly zero feature dimensions
        if self.feature_dropout > 0:
            gemma_mask = np.random.random(
                gemma_features.shape[1]) > self.feature_dropout
            gemma_features = gemma_features * gemma_mask

            landmark_mask = np.random.random(
                landmark_features.shape[1]) > self.feature_dropout
            landmark_features = landmark_features * landmark_mask

        # 5. Gaussian noise on Gemma features
        if self.feature_noise > 0:
            noise = np.random.normal(
                0, self.feature_noise, gemma_features.shape)
            gemma_features = gemma_features + noise

        # 6. Landmark jitter - add small noise to coordinates
        if self.landmark_jitter > 0:
            jitter = np.random.normal(
                0, self.landmark_jitter, landmark_features.shape)
            landmark_features = landmark_features + jitter

        return gemma_features.astype(np.float32), landmark_features.astype(np.float32)


class PreExtractedFeaturesDataset(Dataset):
    """Dataset that loads pre-extracted Gemma and landmark features."""

    def __init__(
        self,
        video_ids: list,
        labels: list,
        gemma_features_dir: str,
        landmarks_dir: str,
        num_frames: int = 16,
        augmentation: Optional[FeatureAugmentation] = None,
        training: bool = True,
    ):
        """
        Initialize the dataset.

        Args:
            video_ids: List of video IDs
            labels: List of corresponding labels
            gemma_features_dir: Directory containing Gemma feature .npy files
            landmarks_dir: Directory containing landmark .npy files
            num_frames: Number of frames to use (will pad/truncate)
            augmentation: Optional augmentation to apply
            training: Whether in training mode
        """
        self.video_ids = video_ids
        self.labels = labels
        self.gemma_features_dir = Path(gemma_features_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.num_frames = num_frames
        self.augmentation = augmentation
        self.training = training

        # Verify files exist and filter out missing ones
        self.valid_indices = []
        for i, vid in enumerate(video_ids):
            gemma_path = self.gemma_features_dir / f"{vid}_gemma.npy"
            landmark_path = self.landmarks_dir / f"{vid}_landmarks.npy"
            if gemma_path.exists() and landmark_path.exists():
                self.valid_indices.append(i)

        logging.getLogger("train_simple").info(
            f"Dataset: {len(self.valid_indices)}/{len(video_ids)} videos have both features"
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        video_id = self.video_ids[actual_idx]
        label = self.labels[actual_idx]

        # Load pre-extracted features
        gemma_path = self.gemma_features_dir / f"{video_id}_gemma.npy"
        landmark_path = self.landmarks_dir / f"{video_id}_landmarks.npy"

        gemma_features = np.load(gemma_path)  # (num_frames, feature_dim)
        landmark_features = np.load(landmark_path)  # (num_frames, 162)

        # Ensure consistent number of frames
        gemma_features = self._adjust_frames(gemma_features, self.num_frames)
        landmark_features = self._adjust_frames(
            landmark_features, self.num_frames)

        # Apply augmentation
        if self.augmentation is not None:
            gemma_features, landmark_features = self.augmentation(
                gemma_features, landmark_features, training=self.training
            )

        return (
            torch.FloatTensor(gemma_features),
            torch.FloatTensor(landmark_features),
            torch.LongTensor([label])[0]
        )

    def _adjust_frames(self, features: np.ndarray, target_frames: int) -> np.ndarray:
        """Adjust feature array to have exactly target_frames frames."""
        current_frames = features.shape[0]

        if current_frames == target_frames:
            return features
        elif current_frames > target_frames:
            # Uniformly sample frames
            indices = np.linspace(0, current_frames - 1,
                                  target_frames, dtype=int)
            return features[indices]
        else:
            # Pad with zeros
            padding = np.zeros(
                (target_frames - current_frames, features.shape[1]))
            return np.vstack([features, padding])

    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for balanced sampling."""
        valid_labels = [self.labels[i] for i in self.valid_indices]
        class_counts = np.bincount(valid_labels)
        # Avoid division by zero
        class_counts = np.maximum(class_counts, 1)
        weights = 1.0 / class_counts
        sample_weights = weights[valid_labels]
        return torch.FloatTensor(sample_weights)


def load_wlasl_data(data_dir: str, subset: str = "nslt_100.json"):
    """Load WLASL dataset annotations."""
    logger = logging.getLogger("train_simple")

    data_path = Path(data_dir)
    subset_path = data_path / subset

    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")

    with open(subset_path) as f:
        data = json.load(f)

    video_ids = []
    labels = []

    # Get unique classes
    classes = set()
    for video_id, info in data.items():
        if info and 'action' in info and len(info['action']) >= 1:
            classes.add(info['action'][0])

    num_classes = len(classes)
    logger.info(f"Found {num_classes} classes in {subset}")

    # Load video info
    for video_id, info in data.items():
        if info and 'action' in info and len(info['action']) >= 1:
            video_ids.append(video_id)
            labels.append(info['action'][0])

    logger.info(f"Loaded {len(video_ids)} videos from {subset}")

    return video_ids, labels, num_classes


def create_label_mapping(data_dir: str, subset: str = "nslt_100.json") -> dict:
    """Create mapping from class index to gloss name."""
    data_path = Path(data_dir)

    # Try to load main WLASL annotations for gloss names
    main_json = data_path / "WLASL_v0.3.json"
    if main_json.exists():
        with open(main_json) as f:
            wlasl_data = json.load(f)

        # Build gloss mapping
        gloss_map = {}
        for entry in wlasl_data:
            gloss = entry.get('gloss', '')
            for instance in entry.get('instances', []):
                video_id = instance.get('video_id', '')
                gloss_map[video_id] = gloss

        # Load subset to get class indices
        subset_path = data_path / subset
        with open(subset_path) as f:
            subset_data = json.load(f)

        # Map class index to gloss
        class_to_gloss = {}
        for video_id, info in subset_data.items():
            if info and 'action' in info and len(info['action']) >= 1:
                class_idx = info['action'][0]
                if video_id in gloss_map and class_idx not in class_to_gloss:
                    class_to_gloss[class_idx] = gloss_map[video_id]

        return class_to_gloss

    return {}


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    logger: logging.Logger,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    step_scheduler_per_batch: bool = False,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)

    for batch_idx, (gemma_feat, landmark_feat, labels) in enumerate(pbar):
        gemma_feat = gemma_feat.to(device)
        landmark_feat = landmark_feat.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(gemma_feat, landmark_feat)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Step scheduler per batch if using warmup
        if step_scheduler_per_batch and scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    avg_loss = total_loss / len(dataloader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
) -> tuple:
    """Validate the model."""
    model.eval()
    total_loss = 0
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for gemma_feat, landmark_feat, labels in tqdm(dataloader, desc="Validating", leave=False):
            gemma_feat = gemma_feat.to(device)
            landmark_feat = landmark_feat.to(device)
            labels = labels.to(device)

            outputs = model(gemma_feat, landmark_feat)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            # Top-1 accuracy
            _, predicted = outputs.max(1)
            correct_top1 += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, dim=1)
            correct_top5 += sum(labels[i] in top5_pred[i]
                                for i in range(labels.size(0)))

            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    top1_acc = 100. * correct_top1 / total
    top5_acc = 100. * correct_top5 / total

    return avg_loss, top1_acc, top5_acc


def main():
    parser = argparse.ArgumentParser(
        description="Train Hybrid ASL Model with pre-extracted features")

    # Data arguments
    parser.add_argument("--data_dir", type=str,
                        default="data/wlasl", help="Data directory")
    parser.add_argument("--gemma_features_dir", type=str, default="data/wlasl/gemma_features",
                        help="Directory with pre-extracted Gemma features")
    parser.add_argument("--landmarks_dir", type=str, default="data/wlasl/landmarks",
                        help="Directory with pre-extracted landmarks")
    parser.add_argument("--subset", type=str,
                        default="nslt_100.json", help="Subset to use")

    # Model arguments
    parser.add_argument("--gemma_feature_dim", type=int, default=1152,
                        help="Dimension of Gemma features (1152 for PaliGemma, 2048 for Gemma 3)")
    parser.add_argument("--hidden_dim", type=int,
                        default=512, help="Hidden dimension")
    parser.add_argument("--num_frames", type=int,
                        default=16, help="Number of frames")
    parser.add_argument("--fusion_type", type=str, default="concat",
                        choices=["concat", "attention", "gated"], help="Fusion type")
    parser.add_argument("--dropout", type=float,
                        default=0.5, help="Dropout rate")

    # Training arguments
    parser.add_argument("--batch_size", type=int,
                        default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float,
                        default=1e-4, help="Weight decay")
    parser.add_argument("--device", type=str,
                        default="cuda", help="Device to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val_split", type=float,
                        default=0.2, help="Validation split ratio")
    parser.add_argument("--label_smoothing", type=float,
                        default=0.1, help="Label smoothing for loss")
    parser.add_argument("--warmup_epochs", type=int,
                        default=5, help="Number of warmup epochs")
    parser.add_argument("--use_class_weights", action="store_true", default=False,
                        help="Use class weights in loss function (recommended for imbalanced datasets)")
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false",
                        help="Disable class weights in loss function")

    # Augmentation arguments
    parser.add_argument("--no_augmentation", action="store_true",
                        help="Disable data augmentation")
    parser.add_argument("--temporal_jitter", type=bool,
                        default=True, help="Enable temporal jitter")
    parser.add_argument("--feature_dropout", type=float,
                        default=0.1, help="Feature dropout rate")
    parser.add_argument("--feature_noise", type=float,
                        default=0.05, help="Feature noise std")
    parser.add_argument("--temporal_mask_prob", type=float,
                        default=0.1, help="Temporal mask probability")
    parser.add_argument("--landmark_jitter", type=float,
                        default=0.02, help="Landmark jitter std")

    # Sampling arguments
    parser.add_argument("--balanced_sampling",
                        action="store_true", help="Use class-balanced sampling")

    # Output arguments
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_simple",
                        help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str,
                        default="logs", help="Log directory")
    parser.add_argument("--verbose", "-v",
                        action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Setup logging
    logger, log_file = setup_logging(args.log_dir, args.verbose)

    logger.info("=" * 70)
    logger.info("HYBRID ASL TRAINING (Pre-extracted Features)")
    logger.info("=" * 70)

    # Log all arguments
    logger.info("\nConfiguration:")
    for key, value in vars(args).items():
        logger.info(f"  {key}: {value}")

    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    logger.info(f"\nUsing device: {device}")

    if device.type == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Load data
    logger.info(f"\nLoading data from {args.data_dir}")
    logger.info(f"  Subset: {args.subset}")
    video_ids, labels, num_classes = load_wlasl_data(
        args.data_dir, args.subset)

    # Create label mapping
    label_mapping = create_label_mapping(args.data_dir, args.subset)
    if label_mapping:
        mapping_path = "label_mapping.json"
        with open(mapping_path, "w") as f:
            json.dump({str(k): v for k, v in label_mapping.items()}, f, indent=2)
        logger.info(f"  Saved label mapping to {mapping_path}")

    # Split into train/val
    indices = list(range(len(video_ids)))
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - args.val_split))

    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_video_ids = [video_ids[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    val_video_ids = [video_ids[i] for i in val_indices]
    val_labels = [labels[i] for i in val_indices]

    logger.info(f"\nData split:")
    logger.info(f"  Train: {len(train_video_ids)} videos")
    logger.info(f"  Val: {len(val_video_ids)} videos")

    # Create augmentation
    augmentation = None
    if not args.no_augmentation:
        augmentation = FeatureAugmentation(
            temporal_jitter=args.temporal_jitter,
            feature_dropout=args.feature_dropout,
            feature_noise=args.feature_noise,
            temporal_mask_prob=args.temporal_mask_prob,
            landmark_jitter=args.landmark_jitter,
            random_temporal_crop=True,
        )
        logger.info("\nData augmentation ENABLED:")
        logger.info(f"  Temporal jitter: {args.temporal_jitter}")
        logger.info(f"  Feature dropout: {args.feature_dropout}")
        logger.info(f"  Feature noise: {args.feature_noise}")
        logger.info(f"  Temporal mask prob: {args.temporal_mask_prob}")
        logger.info(f"  Landmark jitter: {args.landmark_jitter}")
        logger.info(f"  Random temporal crop: True")
    else:
        logger.info("\nData augmentation DISABLED")

    # Create datasets
    train_dataset = PreExtractedFeaturesDataset(
        train_video_ids, train_labels,
        args.gemma_features_dir, args.landmarks_dir,
        num_frames=args.num_frames,
        augmentation=augmentation,
        training=True,
    )

    val_dataset = PreExtractedFeaturesDataset(
        val_video_ids, val_labels,
        args.gemma_features_dir, args.landmarks_dir,
        num_frames=args.num_frames,
        augmentation=None,  # No augmentation for validation
        training=False,
    )

    # Create sampler for balanced training (optional)
    train_sampler = None
    shuffle_train = True
    if args.balanced_sampling:
        sample_weights = train_dataset.get_class_weights()
        train_sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        shuffle_train = False
        logger.info("\nUsing balanced sampling (WeightedRandomSampler)")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=shuffle_train, sampler=train_sampler,
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"\nDataloaders created:")
    logger.info(f"  Train batches: {len(train_loader)}")
    logger.info(f"  Val batches: {len(val_loader)}")

    # Verify feature dimensions
    logger.info(f"\nVerifying feature dimensions...")
    verified_gemma_dim = verify_feature_dimensions(
        args.gemma_features_dir, args.gemma_feature_dim
    )
    if verified_gemma_dim != args.gemma_feature_dim:
        args.gemma_feature_dim = verified_gemma_dim

    # Create model
    model = SimpleHybridASLModel(
        num_classes=num_classes,
        gemma_feature_dim=args.gemma_feature_dim,
        hidden_dim=args.hidden_dim,
        num_frames=args.num_frames,
        fusion_type=args.fusion_type,
        dropout=args.dropout,
    ).to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel()
                           for p in model.parameters() if p.requires_grad)
    logger.info(f"\nModel created:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Fusion type: {args.fusion_type}")

    # Setup training
    # Compute class weights if enabled
    class_weights = None
    if args.use_class_weights:
        logger.info(f"\nComputing class weights for {num_classes} classes...")
        class_weights = compute_class_weights(train_labels, num_classes)
        class_weights = class_weights.to(device)
        logger.info(f"  Class weights computed (min: {class_weights.min():.2f}, max: {class_weights.max():.2f})")
    
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Use warmup scheduler if warmup_epochs > 0
    if args.warmup_epochs > 0:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            warmup_epochs=args.warmup_epochs,
            total_epochs=args.epochs,
            steps_per_epoch=len(train_loader)
        )
        scheduler_type = f"CosineAnnealingLR with {args.warmup_epochs} warmup epochs"
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=1e-6
        )
        scheduler_type = f"CosineAnnealingLR (T_max={args.epochs})"

    logger.info(f"\nTraining setup:")
    logger.info(
        f"  Optimizer: AdamW (lr={args.lr}, weight_decay={args.weight_decay})")
    logger.info(f"  Scheduler: {scheduler_type}")
    logger.info(f"  Loss: CrossEntropyLoss (label_smoothing={args.label_smoothing}, class_weights={args.use_class_weights})")

    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_acc = 0
    step_scheduler_per_batch = args.warmup_epochs > 0
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING START")
    logger.info("=" * 70)

    for epoch in range(1, args.epochs + 1):
        logger.info(f"\nEpoch {epoch}/{args.epochs}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger,
            scheduler=scheduler if step_scheduler_per_batch else None,
            step_scheduler_per_batch=step_scheduler_per_batch
        )

        # Validate
        val_loss, val_top1, val_top5 = validate(
            model, val_loader, criterion, device, logger
        )

        # Update scheduler (only if not stepping per batch)
        if not step_scheduler_per_batch:
            scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log results
        logger.info(
            f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(
            f"  Val Loss: {val_loss:.4f}, Val Top-1: {val_top1:.2f}%, Val Top-5: {val_top5:.2f}%")
        logger.info(f"  Learning Rate: {current_lr:.2e}")

        # Save best model
        if val_top1 > best_val_acc:
            best_val_acc = val_top1
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_top1,
                'val_top5': val_top5,
                'args': vars(args),
            }, "best_simple_model.pth")
            logger.info(f"  ✓ New best model saved! (Top-1: {val_top1:.2f}%)")

        # Save checkpoint every 5 epochs
        if epoch % 5 == 0:
            checkpoint_path = os.path.join(
                args.checkpoint_dir, f"checkpoint_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_top1,
                'val_top5': val_top5,
                'args': vars(args),
            }, checkpoint_path)
            logger.info(f"  ✓ Checkpoint saved: {checkpoint_path}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'args': vars(args),
    }, "final_simple_model.pth")

    logger.info("\n" + "=" * 70)
    logger.info("TRAINING COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✓ Best validation Top-1 accuracy: {best_val_acc:.2f}%")
    logger.info(f"✓ Best model saved to: best_simple_model.pth")
    logger.info(f"✓ Final model saved to: final_simple_model.pth")
    logger.info(f"✓ Checkpoints saved to: {args.checkpoint_dir}/")
    logger.info(f"✓ Training log saved to: {log_file}")


if __name__ == "__main__":
    main()
