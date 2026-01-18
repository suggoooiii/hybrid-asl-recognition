#!/usr/bin/env python3
"""
HYBRID ASL RECOGNITION SYSTEM
Combines VideoMAE visual features with MediaPipe landmark features
for improved sign language recognition. 

Architecture:
    Video → [VideoMAE Branch] → Visual Features ─┐
                                                 ├─→ Fusion → Classifier → Text
    Video → [MediaPipe Branch] → Landmark Features┘

Supports LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor
import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path

# Optional: LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# LORA CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

def get_lora_config(rank=8, alpha=16, dropout=0.1, target_modules=None):
    """
    Create LoRA configuration for VideoMAE.

    Args:
        rank: LoRA rank (lower = fewer params, higher = more capacity)
        alpha: LoRA alpha scaling factor (typically 2x rank)
        dropout: LoRA dropout for regularization
        target_modules: Which modules to apply LoRA to (None = auto-detect)

    Returns:
        LoraConfig object
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library not installed. Run: pip install peft")

    # Target the attention layers in VideoMAE's ViT encoder
    if target_modules is None:
        target_modules = [
            "attention.attention.query",
            "attention.attention.key",
            "attention.attention.value",
            "attention.output.dense",
        ]

    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        modules_to_save=None,  # Don't save any modules outside LoRA
    )


def apply_lora_to_videomae(videomae_model, lora_config):
    """
    Apply LoRA adapters to a VideoMAE model.

    Args:
        videomae_model: Pre-trained VideoMAEModel
        lora_config: LoraConfig object

    Returns:
        VideoMAE model with LoRA adapters
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library not installed. Run: pip install peft")

    # Apply PEFT/LoRA to the model
    lora_model = get_peft_model(videomae_model, lora_config)

    # Print trainable parameters
    trainable_params = sum(p.numel()
                           for p in lora_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in lora_model.parameters())
    print(
        f"  LoRA trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    return lora_model


# ═══════════════════════════════════════════════════════════════════════════════
# MEDIAPIPE LANDMARK EXTRACTOR (New Tasks API)
# ═══════════════════════════════════════════════════════════════════════════════

class MediaPipeLandmarkExtractor:
    """
    Extract hand and pose landmarks using the NEW MediaPipe Tasks API. 

    Output features per frame:
        - Left hand:   21 landmarks × 3 coords = 63
        - Right hand: 21 landmarks × 3 coords = 63
        - Pose (upper body): 12 landmarks × 3 coords = 36
        - Total: 162 features per frame
    """

    def __init__(self,
                 hand_model_path='hand_landmarker.task',
                 pose_model_path='pose_landmarker.task'):

        # Initialize MediaPipe Tasks API
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp.tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Hand landmarker options
        hand_options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=hand_model_path),
            running_mode=VisionRunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Pose landmarker options
        pose_options = PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=pose_model_path),
            running_mode=VisionRunningMode.VIDEO,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        self.hand_landmarker = HandLandmarker.create_from_options(hand_options)
        self.pose_landmarker = PoseLandmarker.create_from_options(pose_options)

        self.feature_dim = 162  # 63 + 63 + 36

        # Store options for reset
        self._hand_options = hand_options
        self._pose_options = pose_options
        self._HandLandmarker = HandLandmarker
        self._PoseLandmarker = PoseLandmarker

    def reset(self):
        """Reset landmarkers for new video (clears internal timestamp state)."""
        self.hand_landmarker.close()
        self.pose_landmarker.close()
        self.hand_landmarker = self._HandLandmarker.create_from_options(
            self._hand_options)
        self.pose_landmarker = self._PoseLandmarker.create_from_options(
            self._pose_options)

    def extract_frame_landmarks(self, frame_rgb, timestamp_ms):
        """Extract landmarks from a single frame."""

        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

        # Detect hands
        hand_result = self.hand_landmarker.detect_for_video(
            mp_image, timestamp_ms)

        # Detect pose
        pose_result = self.pose_landmarker.detect_for_video(
            mp_image, timestamp_ms)

        features = []

        # ─────────────────────────────────────────────────────────────
        # LEFT HAND (63 features)
        # ─────────────────────────────────────────────────────────────
        left_hand_features = [0.0] * 63
        for idx, handedness in enumerate(hand_result.handedness):
            if handedness[0].category_name == 'Left':
                landmarks = hand_result.hand_landmarks[idx]
                left_hand_features = []
                for lm in landmarks:
                    left_hand_features.extend([lm.x, lm.y, lm.z])
                break
        features.extend(left_hand_features)

        # ─────────────────────────────────────────────────────────────
        # RIGHT HAND (63 features)
        # ─────────────────────────────────────────────────────────────
        right_hand_features = [0.0] * 63
        for idx, handedness in enumerate(hand_result.handedness):
            if handedness[0].category_name == 'Right':
                landmarks = hand_result.hand_landmarks[idx]
                right_hand_features = []
                for lm in landmarks:
                    right_hand_features.extend([lm.x, lm.y, lm.z])
                break
        features.extend(right_hand_features)

        # ─────────────────────────────────────────────────────────────
        # POSE - Upper body (36 features: landmarks 11-22)
        # ─────────────────────────────────────────────────────────────
        pose_features = [0.0] * 36
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            pose_features = []
            for i in range(11, 23):  # Shoulders, elbows, wrists, hips
                lm = pose_landmarks[i]
                pose_features.extend([lm.x, lm.y, lm.z])
        features.extend(pose_features)

        return np.array(features, dtype=np.float32)

    def extract_video_landmarks(self, video_path, max_frames=16, start_frame=None, end_frame=None):
        """
        Extract landmarks from entire video with optional frame trimming.

        Args:
            video_path: Path to video file
            max_frames: Number of frames to extract
            start_frame: First frame to include (1-indexed as per WLASL annotations)
            end_frame: Last frame to include (1-indexed, inclusive)
        """
        # Reset landmarkers to clear internal timestamp state for new video
        self.reset()

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply frame trimming if specified (convert 1-indexed to 0-indexed)
        if start_frame is not None and end_frame is not None:
            frame_start = max(0, start_frame - 1)
            frame_end = min(total_frames - 1, end_frame - 1)
            segment_length = frame_end - frame_start + 1
        else:
            frame_start = 0
            frame_end = total_frames - 1
            segment_length = total_frames

        # Calculate which frames to sample (uniform sampling within segment)
        if segment_length <= max_frames:
            sample_indices = [frame_start + i for i in range(segment_length)]
        else:
            sample_indices = np.linspace(
                frame_start, frame_end, max_frames, dtype=int).tolist()

        frame_features = []

        # Use monotonically increasing timestamps (required by MediaPipe VIDEO mode)
        # We use a simple counter: 0ms, 33ms, 66ms, etc. (assuming ~30fps spacing)
        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                # Pad with zeros if frame read fails
                frame_features.append(
                    np.zeros(self.feature_dim, dtype=np.float32))
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Use monotonically increasing timestamp based on sample index (not frame index)
            # This ensures timestamps always increase regardless of which frames we sample
            # +1 to avoid 0ms on first frame
            timestamp_ms = int((i + 1) * 1000 / fps)

            # Extract landmarks
            features = self.extract_frame_landmarks(frame_rgb, timestamp_ms)
            frame_features.append(features)

        cap.release()

        # Padding if needed
        if len(frame_features) < max_frames:
            last_frame = frame_features[-1] if frame_features else np.zeros(
                self.feature_dim)
            while len(frame_features) < max_frames:
                frame_features.append(last_frame.copy())

        return np.array(frame_features[:max_frames], dtype=np.float32)

    def close(self):
        self.hand_landmarker.close()
        self.pose_landmarker.close()


# ═══════════════════════════════════════════════════════════════════════════════
# LANDMARK ENCODER (Transformer-based)
# ═══════════════════════════════════════════════════════════════════════════════

class LandmarkEncoder(nn.Module):
    """
    Transformer encoder for landmark sequences. 

    Input: (batch, seq_len, 162)
    Output: (batch, hidden_dim)
    """

    def __init__(self,
                 input_dim=162,
                 hidden_dim=256,
                 num_heads=4,
                 num_layers=2,
                 dropout=0.3):
        super().__init__()

        # Project landmarks to hidden dimension
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 64, hidden_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.hidden_dim = hidden_dim

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, 162) landmark features
        Returns:
            (batch, hidden_dim) encoded features
        """
        batch_size, seq_len, _ = x.shape

        # Project to hidden dim
        x = self.input_projection(x)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        x = self.transformer(x)

        # Global average pooling
        x = x.mean(dim=1)

        # Output projection
        x = self.output_projection(x)

        return x


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEOMAE ENCODER (Visual features) - With LoRA support
# ═══════════════════════════════════════════════════════════════════════════════

class VideoMAEEncoder(nn.Module):
    """
    VideoMAE encoder for visual features. 

    Input: (batch, num_frames, channels, height, width)
    Output: (batch, hidden_dim)

    Supports LoRA for parameter-efficient fine-tuning.
    """

    def __init__(self,
                 model_name='MCG-NJU/videomae-base',
                 hidden_dim=256,
                 freeze_backbone=False,
                 dropout=0.3,
                 use_lora=False,
                 lora_rank=8,
                 lora_alpha=16,
                 lora_dropout=0.1):
        super().__init__()

        self.use_lora = use_lora

        # Load pretrained VideoMAE
        self.videomae = VideoMAEModel.from_pretrained(model_name)

        # Apply LoRA if requested (mutually exclusive with freeze)
        if use_lora:
            if not PEFT_AVAILABLE:
                raise ImportError(
                    "peft library required for LoRA. Run: pip install peft")

            print(f"  Applying LoRA (rank={lora_rank}, alpha={lora_alpha})")
            lora_config = get_lora_config(
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout
            )
            self.videomae = apply_lora_to_videomae(self.videomae, lora_config)

        elif freeze_backbone:
            # Only freeze if not using LoRA
            for param in self.videomae.parameters():
                param.requires_grad = False

        # Get VideoMAE hidden size
        videomae_hidden = self.videomae.config.hidden_size  # Usually 768

        # Project to our hidden dimension
        self.projection = nn.Sequential(
            nn.Linear(videomae_hidden, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.hidden_dim = hidden_dim

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (batch, num_frames, channels, height, width)
        Returns:
            (batch, hidden_dim) encoded features
        """
        # VideoMAE forward pass
        outputs = self.videomae(pixel_values=pixel_values)

        # Get [CLS] token or mean pool
        # Using mean pooling over sequence
        hidden_states = outputs.last_hidden_state  # (batch, seq, hidden)
        pooled = hidden_states.mean(dim=1)

        # Project to hidden dimension
        features = self.projection(pooled)

        return features


# ═══════════════════════════════════════════════════════════════════════════════
# HYBRID FUSION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class HybridASLModel(nn.Module):
    """
    Hybrid model combining VideoMAE and MediaPipe landmarks.

    Architecture:
        VideoMAE → Visual Features (256) ─┐
                                          ├─→ Fusion (512) → Classifier → Logits
        Landmarks → Landmark Features (256)┘

    Supports LoRA for parameter-efficient fine-tuning of VideoMAE.
    """

    def __init__(self,
                 num_classes,
                 videomae_model='MCG-NJU/videomae-base',
                 hidden_dim=256,
                 freeze_videomae=False,
                 dropout=0.3,
                 fusion_type='concat',  # 'concat', 'attention', 'gated'
                 use_lora=False,
                 lora_rank=8,
                 lora_alpha=16,
                 lora_dropout=0.1):
        super().__init__()

        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        self.use_lora = use_lora

        # ─────────────────────────────────────────────────────────────
        # Stream 1: VideoMAE (Visual) - with optional LoRA
        # ─────────────────────────────────────────────────────────────
        self.visual_encoder = VideoMAEEncoder(
            model_name=videomae_model,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_videomae,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )

        # ─────────────────────────────────────────────────────────────
        # Stream 2: Landmark Encoder
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

        # ─────────────────────────────────────────────────────────────
        # Classifier Head
        # ─────────────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.num_classes = num_classes

    def forward(self, pixel_values, landmarks):
        """
        Args:
            pixel_values: (batch, num_frames, C, H, W) for VideoMAE
            landmarks: (batch, seq_len, 162) landmark features
        Returns:
            logits: (batch, num_classes)
        """
        # Encode both streams
        visual_features = self.visual_encoder(pixel_values)    # (B, hidden)
        landmark_features = self.landmark_encoder(landmarks)    # (B, hidden)

        # Fusion
        if self.fusion_type == 'concat':
            fused = torch.cat([visual_features, landmark_features], dim=-1)
            fused = self.fusion(fused)

        elif self.fusion_type == 'attention':
            # Cross-attention between streams
            v_expanded = visual_features.unsqueeze(1)      # (B, 1, hidden)
            l_expanded = landmark_features.unsqueeze(1)    # (B, 1, hidden)

            attended_v, _ = self.fusion_attention(
                v_expanded, l_expanded, l_expanded)
            attended_l, _ = self.fusion_attention(
                l_expanded, v_expanded, v_expanded)

            fused = torch.cat(
                [attended_v.squeeze(1), attended_l.squeeze(1)], dim=-1)
            fused = self.fusion(fused)

        elif self.fusion_type == 'gated':
            combined = torch.cat([visual_features, landmark_features], dim=-1)
            gate = self.gate(combined)
            fused = gate * visual_features + (1 - gate) * landmark_features
            fused = self.fusion(fused)

        # Classification
        logits = self.classifier(fused)

        return logits

    def predict(self, pixel_values, landmarks):
        """Get predictions with confidence scores."""
        logits = self.forward(pixel_values, landmarks)
        probs = F.softmax(logits, dim=-1)
        confidence, predictions = probs.max(dim=-1)
        return predictions, confidence, probs


# ═══════════════════════════════════════════════════════════════════════════════
# DATASET CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class HybridASLDataset(torch.utils.data.Dataset):
    """
    Dataset for hybrid ASL model.
    Loads both video frames (for VideoMAE) and landmarks (for transformer).

    Supports:
        - Frame trimming using start_frame/end_frame from WLASL annotations
        - Pre-extracted landmarks for 10-50x faster training:
            - If landmarks_dir is provided, loads {video_id}_landmarks.npy
            - Otherwise, extracts landmarks on-the-fly using landmark_extractor
        - Data augmentation via video_augment and landmark_augment callables
    """

    def __init__(self,
                 video_paths,
                 labels,
                 landmark_extractor=None,
                 videomae_processor=None,
                 num_frames=16,
                 image_size=224,
                 landmarks_dir=None,
                 mode='hybrid',  # 'hybrid', 'videomae_only', 'landmark_only'
                 frame_info=None,  # List of (start_frame, end_frame) tuples
                 video_augment=None,  # VideoAugmentation object or callable
                 landmark_augment=None):  # LandmarkAugmentation object or callable

        self.video_paths = video_paths
        self.labels = labels
        self.landmark_extractor = landmark_extractor
        self.videomae_processor = videomae_processor
        self.num_frames = num_frames
        self.image_size = image_size
        self.landmarks_dir = Path(landmarks_dir) if landmarks_dir else None
        self.mode = mode
        self.frame_info = frame_info  # (start_frame, end_frame) for each video
        self.video_augment = video_augment
        self.landmark_augment = landmark_augment

        # Validate mode requirements
        if mode in ['hybrid', 'videomae_only'] and videomae_processor is None:
            raise ValueError(f"videomae_processor required for mode='{mode}'")
        if mode in ['hybrid', 'landmark_only']:
            if landmarks_dir is None and landmark_extractor is None:
                raise ValueError(
                    f"Either landmarks_dir or landmark_extractor required for mode='{mode}'")

        # Pre-check which landmarks are available
        if self.landmarks_dir:
            self.cached_landmarks = {}
            for vp in video_paths:
                video_id = Path(vp).stem
                landmark_path = self.landmarks_dir / \
                    f"{video_id}_landmarks.npy"
                if landmark_path.exists():
                    self.cached_landmarks[video_id] = str(landmark_path)
            print(
                f"  Found {len(self.cached_landmarks)}/{len(video_paths)} pre-extracted landmarks")

    def __len__(self):
        return len(self.video_paths)

    def load_video_frames(self, video_path, start_frame=None, end_frame=None):
        """
        Load and preprocess video frames for VideoMAE.

        Args:
            video_path: Path to video file
            start_frame: First frame to include (1-indexed as per WLASL annotations)
            end_frame: Last frame to include (1-indexed, inclusive)
        """
        cap = cv2.VideoCapture(str(video_path))
        frames = []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Apply frame trimming if specified (convert 1-indexed to 0-indexed)
        if start_frame is not None and end_frame is not None:
            # WLASL uses 1-indexed frames, convert to 0-indexed
            frame_start = max(0, start_frame - 1)
            frame_end = min(total_frames - 1, end_frame - 1)
            segment_length = frame_end - frame_start + 1
        else:
            frame_start = 0
            frame_end = total_frames - 1
            segment_length = total_frames

        # Sample frames uniformly within the trimmed segment
        if segment_length <= self.num_frames:
            indices = [frame_start + i for i in range(segment_length)]
        else:
            indices = np.linspace(frame_start, frame_end,
                                  self.num_frames, dtype=int).tolist()

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)

        cap.release()

        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(
                frames[-1] if frames else np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def load_landmarks(self, video_path):
        """Load landmarks from cache or extract on-the-fly."""
        video_id = Path(video_path).stem

        # Try to load from cache
        if self.landmarks_dir and video_id in self.cached_landmarks:
            landmarks = np.load(self.cached_landmarks[video_id])
            # Ensure correct shape
            if len(landmarks) < self.num_frames:
                # Pad with last frame
                padding = np.tile(
                    landmarks[-1:], (self.num_frames - len(landmarks), 1))
                landmarks = np.vstack([landmarks, padding])
            return landmarks[:self.num_frames]

        # Extract on-the-fly
        if self.landmark_extractor:
            return self.landmark_extractor.extract_video_landmarks(
                video_path,
                max_frames=self.num_frames
            )

        # Return zeros if no extractor (shouldn't happen with proper validation)
        return np.zeros((self.num_frames, 162), dtype=np.float32)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Get frame trimming info if available
        start_frame, end_frame = None, None
        if self.frame_info is not None:
            start_frame, end_frame = self.frame_info[idx]

        result = {
            'label': torch.tensor(label, dtype=torch.long)
        }

        # Load video frames for VideoMAE (if needed)
        if self.mode in ['hybrid', 'videomae_only']:
            frames = self.load_video_frames(video_path, start_frame, end_frame)

            # Apply video augmentation if provided
            if self.video_augment is not None:
                frames = self.video_augment(frames)

            pixel_values = self.videomae_processor(
                list(frames),
                return_tensors="pt"
            ).pixel_values.squeeze(0)
            result['pixel_values'] = pixel_values
        else:
            # Dummy tensor for landmark-only mode
            result['pixel_values'] = torch.zeros(self.num_frames, 3, 224, 224)

        # Load landmarks (if needed)
        if self.mode in ['hybrid', 'landmark_only']:
            landmarks = self.load_landmarks(video_path)

            # Apply landmark augmentation if provided
            if self.landmark_augment is not None:
                landmarks = self.landmark_augment(landmarks)

            result['landmarks'] = torch.tensor(landmarks, dtype=torch.float32)
        else:
            # Dummy tensor for videomae-only mode
            result['landmarks'] = torch.zeros(self.num_frames, 162)

        return result


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class HybridASLTrainer:
    """Training manager for the hybrid model with checkpoint support."""

    def __init__(self,
                 model,
                 train_loader,
                 val_loader,
                 device='cuda',
                 learning_rate=1e-4,
                 weight_decay=0.01,
                 checkpoint_dir='checkpoints'):

        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer with different learning rates
        # Lower LR for pretrained VideoMAE, higher for new layers
        videomae_params = list(model.visual_encoder.videomae.parameters())
        other_params = [p for n, p in model.named_parameters()
                        if 'videomae' not in n]

        self.optimizer = torch.optim.AdamW([
            {'params': videomae_params, 'lr': learning_rate * 0.1},
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )

        self.criterion = nn.CrossEntropyLoss()

        # Track training state
        self.start_epoch = 0
        self.best_val_acc = 0

    def save_checkpoint(self, epoch, val_acc, filename=None):
        """Save training checkpoint."""
        if filename is None:
            filename = f'checkpoint_epoch_{epoch+1}.pth'

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'val_acc': val_acc,
        }

        checkpoint_path = self.checkpoint_dir / filename
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path):
        """Load training checkpoint and resume training."""
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_val_acc = checkpoint['best_val_acc']

        print(
            f"✓ Resumed from epoch {self.start_epoch}, best val acc: {self.best_val_acc:.2f}%")
        return self.start_epoch

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch in self.train_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label'].to(self.device)

            self.optimizer.zero_grad()

            logits = self.model(pixel_values, landmarks)
            loss = self.criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        return total_loss / len(self.train_loader), 100. * correct / total

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        correct_top5 = 0
        total = 0

        for batch in self.val_loader:
            pixel_values = batch['pixel_values'].to(self.device)
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label'].to(self.device)

            logits = self.model(pixel_values, landmarks)
            loss = self.criterion(logits, labels)

            total_loss += loss.item()

            # Top-1 accuracy
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Top-5 accuracy
            _, top5_pred = logits.topk(5, dim=1)
            correct_top5 += sum(labels[i] in top5_pred[i]
                                for i in range(labels.size(0)))

        top1_acc = 100. * correct / total
        top5_acc = 100. * correct_top5 / total
        return total_loss / len(self.val_loader), top1_acc, top5_acc

    def train(self, num_epochs, save_path='hybrid_asl_model.pth', checkpoint_every=5):
        """
        Train the model with checkpoint support.

        Args:
            num_epochs: Total number of epochs to train
            save_path: Path to save the best model
            checkpoint_every: Save checkpoint every N epochs
        """
        for epoch in range(self.start_epoch, num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, val_top5 = self.evaluate()
            self.scheduler.step()

            print(f"Epoch {epoch+1}/{num_epochs}")
            print(
                f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(
                f"  Val Loss: {val_loss:.4f}, Val Top-1: {val_acc:.2f}%, Val Top-5: {val_top5:.2f}%")

            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"  ✓ New best model saved! ({val_acc:.2f}%)")

            # Save periodic checkpoint
            if (epoch + 1) % checkpoint_every == 0:
                ckpt_path = self.save_checkpoint(epoch, val_acc)
                print(f"  ✓ Checkpoint saved: {ckpt_path}")

            print()

        # Save final checkpoint
        self.save_checkpoint(num_epochs - 1, val_acc, 'checkpoint_final.pth')

        return self.best_val_acc
