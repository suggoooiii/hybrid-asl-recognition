#!/usr/bin/env python3
"""
HYBRID ASL RECOGNITION SYSTEM
Combines VideoMAE visual features with MediaPipe landmark features
for improved sign language recognition. 

Architecture:
    Video → [VideoMAE Branch] → Visual Features ─┐
                                                 ├─→ Fusion → Classifier → Text
    Video → [MediaPipe Branch] → Landmark Features┘
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import VideoMAEModel, VideoMAEImageProcessor
import mediapipe as mp
import numpy as np
import cv2
from pathlib import Path


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
        HandLandmarkerOptions = mp. tasks.vision.HandLandmarkerOptions
        PoseLandmarker = mp. tasks.vision.PoseLandmarker
        PoseLandmarkerOptions = mp.tasks. vision.PoseLandmarkerOptions
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
        
    def extract_frame_landmarks(self, frame_rgb, timestamp_ms):
        """Extract landmarks from a single frame."""
        
        # Convert to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Detect hands
        hand_result = self.hand_landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Detect pose
        pose_result = self.pose_landmarker. detect_for_video(mp_image, timestamp_ms)
        
        features = []
        
        # ─────────────────────────────────────────────────────────────
        # LEFT HAND (63 features)
        # ──────────────────────────────────────��──────────────────────
        left_hand_features = [0. 0] * 63
        for idx, handedness in enumerate(hand_result. handedness):
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
        # POSE - Upper body (36 features:  landmarks 11-22)
        # ─────────────��───────────────────────────────────────────────
        pose_features = [0.0] * 36
        if pose_result.pose_landmarks:
            pose_landmarks = pose_result.pose_landmarks[0]
            pose_features = []
            for i in range(11, 23):  # Shoulders, elbows, wrists, hips
                lm = pose_landmarks[i]
                pose_features. extend([lm.x, lm.y, lm.z])
        features.extend(pose_features)
        
        return np.array(features, dtype=np.float32)
    
    def extract_video_landmarks(self, video_path, max_frames=16):
        """Extract landmarks from entire video."""
        
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        
        frame_features = []
        frame_idx = 0
        
        while cap.isOpened() and len(frame_features) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Calculate timestamp
            timestamp_ms = int(frame_idx * 1000 / fps)
            
            # Extract landmarks
            features = self.extract_frame_landmarks(frame_rgb, timestamp_ms)
            frame_features.append(features)
            
            frame_idx += 1
        
        cap.release()
        
        # Padding if needed
        if len(frame_features) < max_frames:
            last_frame = frame_features[-1] if frame_features else np.zeros(self.feature_dim)
            while len(frame_features) < max_frames:
                frame_features.append(last_frame. copy())
        
        return np.array(frame_features[: max_frames], dtype=np. float32)
    
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
        self.pos_encoding = nn.Parameter(torch. randn(1, 64, hidden_dim) * 0.02)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
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
        x = x. mean(dim=1)
        
        # Output projection
        x = self.output_projection(x)
        
        return x


# ═══════════════════════════════════════════════════════════════════════════════
# VIDEOMAE ENCODER (Visual features)
# ═══════════════════════════════════════════════════════════════════════════════

class VideoMAEEncoder(nn.Module):
    """
    VideoMAE encoder for visual features. 
    
    Input: (batch, num_frames, channels, height, width)
    Output: (batch, hidden_dim)
    """
    
    def __init__(self, 
                 model_name='MCG-NJU/videomae-base',
                 hidden_dim=256,
                 freeze_backbone=False,
                 dropout=0.3):
        super().__init__()
        
        # Load pretrained VideoMAE
        self.videomae = VideoMAEModel. from_pretrained(model_name)
        
        # Optionally freeze backbone
        if freeze_backbone: 
            for param in self.videomae.parameters():
                param. requires_grad = False
        
        # Get VideoMAE hidden size
        videomae_hidden = self.videomae.config. hidden_size  # Usually 768
        
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
            pixel_values:  (batch, num_frames, channels, height, width)
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
    """
    
    def __init__(self,
                 num_classes,
                 videomae_model='MCG-NJU/videomae-base',
                 hidden_dim=256,
                 freeze_videomae=False,
                 dropout=0.3,
                 fusion_type='concat'):  # 'concat', 'attention', 'gated'
        super().__init__()
        
        self.fusion_type = fusion_type
        self.hidden_dim = hidden_dim
        
        # ─────────────────────────────────────────────────────────────
        # Stream 1: VideoMAE (Visual)
        # ─────────────────────────────────────────────────────────────
        self. visual_encoder = VideoMAEEncoder(
            model_name=videomae_model,
            hidden_dim=hidden_dim,
            freeze_backbone=freeze_videomae,
            dropout=dropout
        )
        
        # ─────────────────────────────────────────────────────────────
        # Stream 2: Landmark Encoder
        # ─────────────────────────────────────────────────────────────
        self. landmark_encoder = LandmarkEncoder(
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
                nn. Linear(fusion_input_dim, hidden_dim),
                nn. LayerNorm(hidden_dim),
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
                nn. Sigmoid()
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
        self. classifier = nn.Sequential(
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
        visual_features = self. visual_encoder(pixel_values)    # (B, hidden)
        landmark_features = self. landmark_encoder(landmarks)    # (B, hidden)
        
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
            
            fused = torch.cat([attended_v. squeeze(1), attended_l.squeeze(1)], dim=-1)
            fused = self. fusion(fused)
            
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
    """
    
    def __init__(self, 
                 video_paths,
                 labels,
                 landmark_extractor,
                 videomae_processor,
                 num_frames=16,
                 image_size=224):
        
        self.video_paths = video_paths
        self.labels = labels
        self.landmark_extractor = landmark_extractor
        self.videomae_processor = videomae_processor
        self.num_frames = num_frames
        self.image_size = image_size
        
    def __len__(self):
        return len(self.video_paths)
    
    def load_video_frames(self, video_path):
        """Load and preprocess video frames for VideoMAE."""
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample frames uniformly
        if total_frames <= self.num_frames:
            indices = list(range(total_frames))
        else:
            indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        for idx in indices: 
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames. append(frame_rgb)
        
        cap.release()
        
        # Pad if needed
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((self.image_size, self. image_size, 3), dtype=np.uint8))
        
        return frames[: self.num_frames]
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load video frames for VideoMAE
        frames = self.load_video_frames(video_path)
        
        # Process for VideoMAE
        pixel_values = self.videomae_processor(
            list(frames), 
            return_tensors="pt"
        ).pixel_values.squeeze(0)
        
        # Extract landmarks
        landmarks = self.landmark_extractor. extract_video_landmarks(
            video_path, 
            max_frames=self.num_frames
        )
        landmarks = torch.tensor(landmarks, dtype=torch.float32)
        
        return {
            'pixel_values': pixel_values,
            'landmarks': landmarks,
            'label': torch.tensor(label, dtype=torch.long)
        }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

class HybridASLTrainer: 
    """Training manager for the hybrid model."""
    
    def __init__(self, 
                 model, 
                 train_loader, 
                 val_loader,
                 device='cuda',
                 learning_rate=1e-4,
                 weight_decay=0.01):
        
        self.model = model. to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Optimizer with different learning rates
        # Lower LR for pretrained VideoMAE, higher for new layers
        videomae_params = list(model.visual_encoder.videomae.parameters())
        other_params = [p for n, p in model.named_parameters() 
                       if 'videomae' not in n]
        
        self.optimizer = torch.optim. AdamW([
            {'params': videomae_params, 'lr': learning_rate * 0.1},
            {'params': other_params, 'lr': learning_rate}
        ], weight_decay=weight_decay)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-6
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
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
            
            total_loss += loss. item()
            _, predicted = logits.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self. train_loader), 100. * correct / total
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in self.val_loader:
            pixel_values = batch['pixel_values'].to(self. device)
            landmarks = batch['landmarks'].to(self.device)
            labels = batch['label']. to(self.device)
            
            logits = self.model(pixel_values, landmarks)
            loss = self.criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = logits. max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        return total_loss / len(self.val_loader), 100. * correct / total
    
    def train(self, num_epochs, save_path='hybrid_asl_model.pth'):
        best_val_acc = 0
        
        for epoch in range(num_epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            self.scheduler.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model. state_dict(), save_path)
                print(f"  ✓ New best model saved!  ({val_acc:.2f}%)")
            
            print()
        
        return best_val_acc