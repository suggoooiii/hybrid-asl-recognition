"""
Simple Hybrid ASL Model using pre-extracted features.

This is the PRIMARY model for fast training with:
- Pre-extracted Gemma/PaliGemma features (frozen visual encoder)
- Pre-extracted MediaPipe landmarks
- ~2.8M trainable parameters
- 2-4 GB GPU memory requirement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LandmarkEncoder(nn.Module):
    """Encodes MediaPipe landmark sequences."""

    def __init__(
        self,
        input_dim: int = 162,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_frames, 162) landmark features

        Returns:
            (batch, hidden_dim) encoded features
        """
        # Project input
        x = self.input_projection(x)  # (batch, num_frames, hidden_dim)

        # LSTM encoding
        # lstm_out: (batch, num_frames, hidden_dim*2)
        lstm_out, (h_n, _) = self.lstm(x)

        # Use last hidden states from both directions
        # h_n: (num_layers*2, batch, hidden_dim)
        forward_hidden = h_n[-2]  # Last forward layer
        backward_hidden = h_n[-1]  # Last backward layer
        # (batch, hidden_dim*2)
        combined = torch.cat([forward_hidden, backward_hidden], dim=1)

        # Project to output dimension
        output = self.output_projection(combined)  # (batch, hidden_dim)

        return output


class GemmaVisualEncoder(nn.Module):
    """Encodes pre-extracted Gemma/PaliGemma visual features."""

    def __init__(
        self,
        input_dim: int = 1152,  # PaliGemma SigLIP dimension
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Temporal attention to aggregate frames
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.output_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_frames, input_dim) pre-extracted Gemma features

        Returns:
            (batch, hidden_dim) encoded features
        """
        batch_size, num_frames, _ = x.shape

        # Project features
        x = self.projection(x)  # (batch, num_frames, hidden_dim)

        # Self-attention over temporal dimension
        attn_out, _ = self.temporal_attention(
            x, x, x)  # (batch, num_frames, hidden_dim)

        # Mean pooling over frames
        output = attn_out.mean(dim=1)  # (batch, hidden_dim)

        return output


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(self, visual_dim: int, landmark_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        self.fusion = nn.Sequential(
            nn.Linear(visual_dim + landmark_dim, output_dim * 2),
            nn.LayerNorm(output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(self, visual_features: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([visual_features, landmark_features], dim=-1)
        return self.fusion(combined)


class AttentionFusion(nn.Module):
    """Cross-attention fusion between visual and landmark features."""

    def __init__(self, visual_dim: int, landmark_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        # Project both to same dimension
        self.visual_proj = nn.Linear(visual_dim, output_dim)
        self.landmark_proj = nn.Linear(landmark_dim, output_dim)

        # Cross attention: visual attends to landmark
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        self.output_projection = nn.Sequential(
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(self, visual_features: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        # Project to same dimension
        visual_proj = self.visual_proj(visual_features).unsqueeze(
            1)  # (batch, 1, output_dim)
        landmark_proj = self.landmark_proj(
            landmark_features).unsqueeze(1)  # (batch, 1, output_dim)

        # Cross attention
        attn_out, _ = self.cross_attention(
            visual_proj, landmark_proj, landmark_proj)
        attn_out = attn_out.squeeze(1)  # (batch, output_dim)

        # Combine with visual features
        combined = torch.cat([attn_out, visual_proj.squeeze(1)], dim=-1)
        return self.output_projection(combined)


class GatedFusion(nn.Module):
    """Gated fusion with learnable weighting between streams."""

    def __init__(self, visual_dim: int, landmark_dim: int, output_dim: int, dropout: float = 0.3):
        super().__init__()

        # Project both to same dimension
        self.visual_proj = nn.Sequential(
            nn.Linear(visual_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        self.landmark_proj = nn.Sequential(
            nn.Linear(landmark_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + landmark_dim, output_dim),
            nn.Sigmoid(),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.output_dim = output_dim

    def forward(self, visual_features: torch.Tensor, landmark_features: torch.Tensor) -> torch.Tensor:
        # Project features
        visual_proj = self.visual_proj(visual_features)  # (batch, output_dim)
        landmark_proj = self.landmark_proj(
            landmark_features)  # (batch, output_dim)

        # Compute gate
        gate_input = torch.cat([visual_features, landmark_features], dim=-1)
        gate = self.gate(gate_input)  # (batch, output_dim)

        # Gated combination
        fused = gate * visual_proj + (1 - gate) * landmark_proj

        return self.output_projection(fused)


class SimpleHybridASLModel(nn.Module):
    """
    Simple Hybrid ASL Recognition Model using pre-extracted features.

    This model is optimized for fast training with pre-extracted Gemma/PaliGemma
    visual features and MediaPipe landmark features.
    """

    def __init__(
        self,
        num_classes: int,
        gemma_feature_dim: int = 1152,  # PaliGemma SigLIP dimension
        landmark_dim: int = 162,
        hidden_dim: int = 256,
        num_frames: int = 16,
        fusion_type: str = "concat",
        dropout: float = 0.3,
    ):
        """
        Initialize the model.

        Args:
            num_classes: Number of sign classes
            gemma_feature_dim: Dimension of pre-extracted Gemma features (1152 for PaliGemma)
            landmark_dim: Dimension of landmark features (162)
            hidden_dim: Hidden dimension for encoders
            num_frames: Number of frames in input
            fusion_type: Type of fusion ("concat", "attention", "gated")
            dropout: Dropout rate
        """
        super().__init__()

        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.fusion_type = fusion_type

        # Visual encoder for pre-extracted Gemma features
        self.visual_encoder = GemmaVisualEncoder(
            input_dim=gemma_feature_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Landmark encoder
        self.landmark_encoder = LandmarkEncoder(
            input_dim=landmark_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        # Fusion module
        if fusion_type == "concat":
            self.fusion = ConcatFusion(
                visual_dim=hidden_dim,
                landmark_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_type == "attention":
            self.fusion = AttentionFusion(
                visual_dim=hidden_dim,
                landmark_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
        elif fusion_type == "gated":
            self.fusion = GatedFusion(
                visual_dim=hidden_dim,
                landmark_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
            )
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        gemma_features: torch.Tensor,
        landmark_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            gemma_features: (batch, num_frames, gemma_feature_dim) pre-extracted visual features
            landmark_features: (batch, num_frames, 162) landmark features

        Returns:
            (batch, num_classes) classification logits
        """
        # Encode visual features
        visual_encoded = self.visual_encoder(
            gemma_features)  # (batch, hidden_dim)

        # Encode landmark features
        landmark_encoded = self.landmark_encoder(
            landmark_features)  # (batch, hidden_dim)

        # Fuse features
        # (batch, hidden_dim)
        fused = self.fusion(visual_encoded, landmark_encoded)

        # Classify
        logits = self.classifier(fused)  # (batch, num_classes)

        return logits

    def get_feature_importance(
        self,
        gemma_features: torch.Tensor,
        landmark_features: torch.Tensor,
    ) -> dict:
        """
        Get feature importance scores for analysis.

        Returns dict with visual and landmark contribution scores.
        """
        with torch.no_grad():
            visual_encoded = self.visual_encoder(gemma_features)
            landmark_encoded = self.landmark_encoder(landmark_features)

            # Compute norms as proxy for importance
            visual_norm = visual_encoded.norm(dim=-1).mean().item()
            landmark_norm = landmark_encoded.norm(dim=-1).mean().item()
            total_norm = visual_norm + landmark_norm

            return {
                "visual_importance": visual_norm / total_norm,
                "landmark_importance": landmark_norm / total_norm,
                "visual_norm": visual_norm,
                "landmark_norm": landmark_norm,
            }


# Alias for backward compatibility
HybridASLModelSimple = SimpleHybridASLModel
