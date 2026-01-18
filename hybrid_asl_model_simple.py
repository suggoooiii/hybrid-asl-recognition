#!/usr/bin/env python3
"""
SIMPLIFIED HYBRID ASL MODEL USING PRE-EXTRACTED GEMMA FEATURES

Architecture:
    Pre-extracted Gemma Features (.npy) → [Visual Projection] ─┐
                                                                ├─→ Fusion → Classifier
    Pre-extracted Landmarks (.npy) → [Landmark Encoder] ────────┘

Key Features:
    - Uses pre-extracted Gemma visual features (no Gemma forward pass during training)
    - Reuses LandmarkEncoder from hybrid_asl_model.py
    - ~2M trainable parameters (vs ~88M in full VideoMAE model)
    - 10-50x faster training due to pre-extraction
    - SignGemma-ready: Just re-run feature extraction with new model

Trainable Components:
    1. Visual projection: Projects Gemma features to hidden_dim
    2. Landmark encoder: Transformer encoder for landmarks (reused)
    3. Fusion layer: Combines visual + landmark features
    4. Classifier: Final classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

# Import LandmarkEncoder from existing code
from hybrid_asl_model import LandmarkEncoder


# ═══════════════════════════════════════════════════════════════════════════════
# VISUAL PROJECTION (for pre-extracted Gemma features)
# ═══════════════════════════════════════════════════════════════════════════════

class VisualProjection(nn.Module):
    """
    Project pre-extracted Gemma visual features to hidden dimension.
    
    Input: (batch, num_frames, gemma_feature_dim)
    Output: (batch, hidden_dim)
    """
    
    def __init__(self,
                 gemma_feature_dim: int = 2048,
                 hidden_dim: int = 256,
                 dropout: float = 0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Frame-level projection
        self.frame_projection = nn.Sequential(
            nn.Linear(gemma_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Temporal encoding (positional encoding for frames)
        self.temporal_encoding = nn.Parameter(
            torch.randn(1, 64, hidden_dim) * 0.02
        )
        
        # Temporal attention (lightweight)
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_frames, gemma_feature_dim) pre-extracted Gemma features
        
        Returns:
            (batch, hidden_dim) projected visual features
        """
        batch_size, num_frames, _ = x.shape
        
        # Project each frame
        x = self.frame_projection(x)  # (batch, num_frames, hidden_dim)
        
        # Add temporal encoding
        x = x + self.temporal_encoding[:, :num_frames, :]
        
        # Temporal attention (self-attention over frames)
        x, _ = self.temporal_attention(x, x, x)
        
        # Global average pooling over time
        x = x.mean(dim=1)  # (batch, hidden_dim)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# SIMPLIFIED HYBRID MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HybridASLModelSimple(nn.Module):
    """
    Simplified hybrid model using pre-extracted Gemma features.
    
    Architecture:
        Gemma Features → Visual Projection (256) ─┐
                                                   ├─→ Fusion (512) → Classifier
        Landmarks → Landmark Encoder (256) ────────┘
    
    Args:
        num_classes: Number of sign classes
        gemma_feature_dim: Dimension of pre-extracted Gemma features
        hidden_dim: Hidden dimension for both branches
        dropout: Dropout rate
        fusion_type: 'concat', 'attention', or 'gated'
    """
    
    def __init__(self,
                 num_classes: int,
                 gemma_feature_dim: int = 2048,
                 hidden_dim: int = 256,
                 dropout: float = 0.3,
                 fusion_type: str = 'concat'):
        super().__init__()
        
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        
        # ─────────────────────────────────────────────────────────────
        # Stream 1: Visual Projection (for Gemma features)
        # ─────────────────────────────────────────────────────────────
        self.visual_projection = VisualProjection(
            gemma_feature_dim=gemma_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout
        )
        
        # ─────────────────────────────────────────────────────────────
        # Stream 2: Landmark Encoder (reused from hybrid_asl_model.py)
        # ─────────────────────────────────────────────────────────────
        self.landmark_encoder = LandmarkEncoder(
            input_dim=162,
            hidden_dim=hidden_dim,
            num_heads=4,
            num_layers=2,
            dropout=dropout
        )
        
        # ─────────────────────────────────────────────────────────────
        # Fusion Layer
        # ─────────────────────────────────────────────────────────────
        if fusion_type == 'concat':
            fusion_input_dim = hidden_dim * 2
            self.fusion = nn.Sequential(
                nn.Linear(fusion_input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'attention':
            self.fusion_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        elif fusion_type == 'gated':
            self.gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.Sigmoid()
            )
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
        else:
            raise ValueError(f"Unknown fusion_type: {fusion_type}")
        
        # ─────────────────────────────────────────────────────────────
        # Classifier Head
        # ─────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, gemma_features: torch.Tensor, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            gemma_features: (batch, num_frames, gemma_feature_dim) pre-extracted features
            landmarks: (batch, seq_len, 162) landmark features
        
        Returns:
            logits: (batch, num_classes)
        """
        # Encode both streams
        visual_features = self.visual_projection(gemma_features)      # (B, hidden)
        landmark_features = self.landmark_encoder(landmarks)          # (B, hidden)
        
        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([visual_features, landmark_features], dim=-1)
            fused = self.fusion(fused)
        
        elif self.fusion_type == 'attention':
            # Cross-attention between streams
            v_expanded = visual_features.unsqueeze(1)      # (B, 1, hidden)
            l_expanded = landmark_features.unsqueeze(1)    # (B, 1, hidden)
            
            attended_v, _ = self.fusion_attention(v_expanded, l_expanded, l_expanded)
            attended_l, _ = self.fusion_attention(l_expanded, v_expanded, v_expanded)
            
            fused = torch.cat([attended_v.squeeze(1), attended_l.squeeze(1)], dim=-1)
            fused = self.fusion(fused)
        
        elif self.fusion_type == 'gated':
            combined = torch.cat([visual_features, landmark_features], dim=-1)
            gate = self.gate(combined)
            fused = gate * visual_features + (1 - gate) * landmark_features
            fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def predict(self, gemma_features: torch.Tensor, landmarks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get predictions with confidence scores."""
        logits = self.forward(gemma_features, landmarks)
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)
        return predictions, confidence, probs


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET FOR PRE-EXTRACTED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

class PreextractedDataset(torch.utils.data.Dataset):
    """
    Dataset that loads pre-extracted Gemma features and landmarks.
    
    This is much faster than extracting features on-the-fly.
    
    Args:
        video_ids: List of video IDs
        labels: List of class labels
        gemma_features_dir: Directory containing {video_id}_gemma.npy files
        landmarks_dir: Directory containing {video_id}_landmarks.npy files
        num_frames: Number of frames to use
    """
    
    def __init__(self,
                 video_ids: list,
                 labels: list,
                 gemma_features_dir: str,
                 landmarks_dir: str,
                 num_frames: int = 16):
        
        self.video_ids = video_ids
        self.labels = labels
        self.gemma_features_dir = Path(gemma_features_dir)
        self.landmarks_dir = Path(landmarks_dir)
        self.num_frames = num_frames
        
        # Pre-check which features are available
        self.available_indices = []
        for idx, video_id in enumerate(video_ids):
            gemma_path = self.gemma_features_dir / f"{video_id}_gemma.npy"
            landmark_path = self.landmarks_dir / f"{video_id}_landmarks.npy"
            
            if gemma_path.exists() and landmark_path.exists():
                self.available_indices.append(idx)
        
        print(f"  Found {len(self.available_indices)}/{len(video_ids)} complete feature pairs")
        
        if len(self.available_indices) == 0:
            raise ValueError("No videos with both Gemma features and landmarks found!")
    
    def __len__(self):
        return len(self.available_indices)
    
    def __getitem__(self, idx):
        # Map to original index
        orig_idx = self.available_indices[idx]
        video_id = self.video_ids[orig_idx]
        label = self.labels[orig_idx]
        
        # Load Gemma features
        gemma_path = self.gemma_features_dir / f"{video_id}_gemma.npy"
        gemma_features = np.load(gemma_path)
        
        # Load landmarks
        landmark_path = self.landmarks_dir / f"{video_id}_landmarks.npy"
        landmarks = np.load(landmark_path)
        
        # Ensure correct number of frames (pad or truncate)
        if len(gemma_features) < self.num_frames:
            padding = np.tile(gemma_features[-1:], (self.num_frames - len(gemma_features), 1))
            gemma_features = np.vstack([gemma_features, padding])
        gemma_features = gemma_features[:self.num_frames]
        
        if len(landmarks) < self.num_frames:
            padding = np.tile(landmarks[-1:], (self.num_frames - len(landmarks), 1))
            landmarks = np.vstack([landmarks, padding])
        landmarks = landmarks[:self.num_frames]
        
        return {
            'gemma_features': torch.tensor(gemma_features, dtype=torch.float32),
            'landmarks': torch.tensor(landmarks, dtype=torch.float32),
            'label': torch.tensor(label, dtype=torch.long)
        }


if __name__ == '__main__':
    """Test the simplified model."""
    print("="*70)
    print("SIMPLIFIED HYBRID MODEL TEST")
    print("="*70)
    
    # Create dummy model
    model = HybridASLModelSimple(
        num_classes=100,
        gemma_feature_dim=2048,
        hidden_dim=256,
        fusion_type='concat'
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"✓ Model created")
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    batch_size = 4
    num_frames = 16
    gemma_dim = 2048
    
    gemma_features = torch.randn(batch_size, num_frames, gemma_dim)
    landmarks = torch.randn(batch_size, num_frames, 162)
    
    logits = model(gemma_features, landmarks)
    
    print(f"\n✓ Forward pass successful")
    print(f"✓ Input shapes:")
    print(f"    Gemma features: {gemma_features.shape}")
    print(f"    Landmarks: {landmarks.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    
    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
