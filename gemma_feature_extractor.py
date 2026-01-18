#!/usr/bin/env python3
"""
PALIGEMMA VISUAL FEATURE EXTRACTOR
Extracts visual features from video frames using frozen PaliGemma vision encoder.

Key Design:
    - All PaliGemma weights are FROZEN (no training required)
    - Only used for feature extraction during preprocessing
    - Returns feature vectors that can be saved and reused
    - SignGemma-ready: Easy to swap models when SignGemma is released

Usage:
    extractor = GemmaFeatureExtractor(model_name='google/paligemma-3b-pt-224')
    features = extractor.extract_video_features(video_path, num_frames=16)
    # features: (num_frames, feature_dim) numpy array
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoProcessor, AutoModel, PaliGemmaForConditionalGeneration
from PIL import Image
import warnings


class GemmaFeatureExtractor:
    """
    Extract visual features from video frames using frozen PaliGemma vision encoder.

    Architecture:
        Video Frames → [PaliGemma Vision Encoder (FROZEN)] → Feature Vectors

    Args:
        model_name: HuggingFace model ID (default: 'google/paligemma-3b-pt-224')
        device: 'cuda', 'cpu', or 'mps'
        cache_dir: Optional cache directory for model weights
        use_flash_attention: Enable flash attention for faster inference (requires GPU)
    """

    def __init__(self,
                 model_name: str = 'google/paligemma-3b-pt-224',
                 device: str = 'cuda',
                 cache_dir: Optional[str] = None,
                 use_flash_attention: bool = False):

        self.device = device
        self.model_name = model_name

        print(f"Loading PaliGemma model: {model_name}")
        print(f"Device: {device}")

        # Load processor (handles image preprocessing)
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )

        # Load model with optimizations
        model_kwargs = {
            'cache_dir': cache_dir,
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        }

        if use_flash_attention and device == 'cuda':
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            print("✓ Using Flash Attention 2 for faster inference")

        # Load PaliGemma model
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)

        # Freeze all parameters (no training)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Get the vision tower for direct feature extraction
        self.vision_tower = self.model.vision_tower

        # Get feature dimension from vision encoder
        # For Gemma 3, this is typically from the vision tower
        self.feature_dim = self._get_feature_dim()

        print(f"✓ PaliGemma model loaded (frozen)")
        print(f"✓ Feature dimension: {self.feature_dim}")

    def _get_feature_dim(self) -> int:
        """Infer feature dimension from the vision encoder."""
        # PaliGemma's SigLIP vision tower has a known hidden size
        try:
            config = self.vision_tower.config
            if hasattr(config, 'hidden_size'):
                return config.hidden_size
        except:
            pass
        # Default for PaliGemma SigLIP (224x224 model)
        return 1152

    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single RGB frame.

        Args:
            frame: RGB image as numpy array (H, W, 3)

        Returns:
            Feature vector as numpy array (feature_dim,)
        """
        if frame is None:
            raise ValueError("Frame is None")

        # Convert numpy array to PIL Image for processor
        pil_image = Image.fromarray(frame)

        with torch.no_grad():
            # Process image - PaliGemma processor needs <image> token in text
            inputs = self.processor(
                images=pil_image,
                text="<image>",  # Required image token for PaliGemma
                return_tensors="pt"
            )

            # Get pixel values and move to device
            pixel_values = inputs['pixel_values'].to(
                self.device,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )

            # Extract features directly from vision tower (SigLIP)
            vision_outputs = self.vision_tower(pixel_values)

            # Get features from vision tower output
            if hasattr(vision_outputs, 'last_hidden_state') and vision_outputs.last_hidden_state is not None:
                features = vision_outputs.last_hidden_state
            elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                return vision_outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
            elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                features = vision_outputs[0]
            else:
                features = vision_outputs

            # Mean pool over spatial/patch dimension
            # Shape: (1, num_patches, hidden_size) -> (hidden_size,)
            pooled = features.mean(dim=1).squeeze(0)

            return pooled.cpu().numpy().astype(np.float32)

    def extract_video_features(self,
                               video_path: str,
                               num_frames: int = 16,
                               start_frame: Optional[int] = None,
                               end_frame: Optional[int] = None) -> np.ndarray:
        """
        Extract features from video with optional frame trimming.

        Args:
            video_path: Path to video file
            num_frames: Number of frames to extract
            start_frame: First frame to include (1-indexed as per WLASL annotations)
            end_frame: Last frame to include (1-indexed, inclusive)

        Returns:
            Feature array of shape (num_frames, feature_dim)
        """
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Handle empty or corrupted videos
        if total_frames <= 0:
            cap.release()
            raise RuntimeError(
                f"Video has no frames or is corrupted: {video_path}")

        # Apply frame trimming if specified (convert 1-indexed to 0-indexed)
        if start_frame is not None and end_frame is not None:
            frame_start = max(0, start_frame - 1)
            frame_end = min(total_frames - 1, end_frame - 1)
            segment_length = frame_end - frame_start + 1
        else:
            frame_start = 0
            frame_end = total_frames - 1
            segment_length = total_frames

        # Handle invalid frame ranges
        if segment_length <= 0:
            cap.release()
            raise RuntimeError(
                f"Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}")

        # Calculate which frames to sample (uniform sampling within segment)
        if segment_length <= num_frames:
            sample_indices = [frame_start + i for i in range(segment_length)]
        else:
            sample_indices = np.linspace(
                frame_start, frame_end, num_frames, dtype=int).tolist()

        frame_features = []

        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Pad with zeros if frame read fails
                frame_features.append(
                    np.zeros(self.feature_dim, dtype=np.float32))
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract features
            features = self.extract_frame_features(frame_rgb)
            frame_features.append(features)

        cap.release()

        # Handle case where no frames were successfully read
        if len(frame_features) == 0:
            raise RuntimeError(
                f"Could not read any frames from video: {video_path}")

        # Padding if needed
        if len(frame_features) < num_frames:
            last_frame = frame_features[-1] if frame_features else np.zeros(
                self.feature_dim, dtype=np.float32)
            while len(frame_features) < num_frames:
                frame_features.append(last_frame.copy())

        return np.array(frame_features[:num_frames], dtype=np.float32)

    def extract_batch_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of frames (more efficient).

        Args:
            frames: List of RGB frames as numpy arrays

        Returns:
            Feature array of shape (len(frames), feature_dim)
        """
        # Convert to PIL images
        pil_images = [Image.fromarray(f) for f in frames]

        with torch.no_grad():
            # Process all frames at once - PaliGemma needs <image> token
            inputs = self.processor(
                images=pil_images,
                text=["<image>"] * len(pil_images),
                return_tensors="pt",
                padding=True
            )

            pixel_values = inputs['pixel_values'].to(
                self.device,
                dtype=torch.float16 if self.device == 'cuda' else torch.float32
            )

            # Extract features from vision tower
            vision_outputs = self.vision_tower(pixel_values)

            if hasattr(vision_outputs, 'last_hidden_state') and vision_outputs.last_hidden_state is not None:
                features = vision_outputs.last_hidden_state
            elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                return vision_outputs.pooler_output.cpu().numpy().astype(np.float32)
            elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                features = vision_outputs[0]
            else:
                features = vision_outputs

            # Pool features (mean pooling over sequence dimension)
            features = features.mean(dim=1)  # (batch, feature_dim)

            return features.cpu().numpy().astype(np.float32)


if __name__ == '__main__':
    """Test the feature extractor."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Gemma Feature Extractor')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to test video')
    parser.add_argument('--model_name', type=str, default='google/paligemma-3b-pt-224',
                        help='PaliGemma model name')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--num_frames', type=int, default=16)

    args = parser.parse_args()

    print("="*70)
    print("PALIGEMMA FEATURE EXTRACTOR TEST")
    print("="*70)

    # Initialize extractor
    extractor = GemmaFeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )

    # Extract features
    print(f"\nExtracting features from: {args.video_path}")
    features = extractor.extract_video_features(
        args.video_path,
        num_frames=args.num_frames
    )

    print(f"✓ Features shape: {features.shape}")
    print(f"✓ Feature dimension: {features.shape[1]}")
    print(f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")
    print(f"✓ Feature mean: {features.mean():.3f}")
    print(f"✓ Feature std: {features.std():.3f}")

    print("\n" + "="*70)
    print("TEST COMPLETE!")
    print("="*70)
