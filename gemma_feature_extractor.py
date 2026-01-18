#!/usr/bin/env python3
"""
GEMMA 3 VISUAL FEATURE EXTRACTOR
Extracts visual features from video frames using frozen Gemma 3 vision encoder.

Key Design:
    - All Gemma weights are FROZEN (no training required)
    - Only used for feature extraction during preprocessing
    - Returns feature vectors that can be saved and reused
    - SignGemma-ready: Easy to swap models when SignGemma is released

Usage:
    extractor = GemmaFeatureExtractor(model_name='google/gemma-3-4b-it')
    features = extractor.extract_video_features(video_path, num_frames=16)
    # features: (num_frames, feature_dim) numpy array
"""

import torch
import torch.nn as nn
import numpy as np
import cv2
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoProcessor, AutoModel
import warnings


class GemmaFeatureExtractor:
    """
    Extract visual features from video frames using frozen Gemma 3 vision encoder.
    
    Architecture:
        Video Frames → [Gemma 3 Vision Encoder (FROZEN)] → Feature Vectors
        
    Args:
        model_name: HuggingFace model ID (e.g., 'google/gemma-3-4b-it')
        device: 'cuda', 'cpu', or 'mps'
        cache_dir: Optional cache directory for model weights
        use_flash_attention: Enable flash attention for faster inference (requires GPU)
    """
    
    def __init__(self,
                 model_name: str = 'google/gemma-3-4b-it',
                 device: str = 'cuda',
                 cache_dir: Optional[str] = None,
                 use_flash_attention: bool = False):
        
        self.device = device
        self.model_name = model_name
        
        print(f"Loading Gemma model: {model_name}")
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
        
        self.model = AutoModel.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
        
        # Freeze all parameters (no training)
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Get feature dimension from vision encoder
        # For Gemma 3, this is typically from the vision tower
        self.feature_dim = self._get_feature_dim()
        
        print(f"✓ Gemma model loaded (frozen)")
        print(f"✓ Feature dimension: {self.feature_dim}")
    
    def _get_feature_dim(self) -> int:
        """Infer feature dimension from the vision encoder."""
        # Create dummy input to determine output dimension
        dummy_image = torch.zeros(1, 3, 224, 224, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            try:
                # Try to access vision tower directly
                if hasattr(self.model, 'vision_tower'):
                    outputs = self.model.vision_tower(dummy_image)
                    if hasattr(outputs, 'last_hidden_state'):
                        return outputs.last_hidden_state.shape[-1]
                    elif isinstance(outputs, torch.Tensor):
                        return outputs.shape[-1]
                
                # Fallback: Process through full model
                inputs = self.processor(images=dummy_image, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                outputs = self.model(**inputs, output_hidden_states=True)
                
                # Try to get vision features
                if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states is not None:
                    return outputs.vision_hidden_states[-1].shape[-1]
                elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                    return outputs.hidden_states[-1].shape[-1]
                else:
                    # Default dimension for Gemma models
                    return 2048
                    
            except Exception as e:
                warnings.warn(f"Could not infer feature dimension: {e}. Using default 2048.")
                return 2048
    
    def extract_frame_features(self, frame: np.ndarray) -> np.ndarray:
        """
        Extract features from a single RGB frame.
        
        Args:
            frame: RGB image as numpy array (H, W, 3)
            
        Returns:
            Feature vector as numpy array (feature_dim,)
        """
        with torch.no_grad():
            # Preprocess frame
            inputs = self.processor(images=frame, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get vision features (try multiple paths)
            features = None
            
            # Path 1: Direct vision hidden states
            if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states is not None:
                features = outputs.vision_hidden_states[-1]
            
            # Path 2: General hidden states
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                features = outputs.hidden_states[-1]
            
            # Path 3: Last hidden state
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            
            if features is None:
                raise RuntimeError("Could not extract features from model output")
            
            # Pool features (mean pooling over sequence dimension)
            features = features.mean(dim=1).squeeze(0)
            
            return features.cpu().numpy()
    
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
        if segment_length <= num_frames:
            sample_indices = [frame_start + i for i in range(segment_length)]
        else:
            sample_indices = np.linspace(
                frame_start, frame_end, num_frames, dtype=int).tolist()
        
        frame_features = []
        
        for frame_idx in sample_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                # Pad with zeros if frame read fails
                frame_features.append(np.zeros(self.feature_dim, dtype=np.float32))
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Extract features
            features = self.extract_frame_features(frame_rgb)
            frame_features.append(features)
        
        cap.release()
        
        # Padding if needed
        if len(frame_features) < num_frames:
            last_frame = frame_features[-1] if frame_features else np.zeros(self.feature_dim, dtype=np.float32)
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
        with torch.no_grad():
            # Process all frames at once
            inputs = self.processor(images=frames, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Extract features
            outputs = self.model(**inputs, output_hidden_states=True)
            
            # Get vision features
            if hasattr(outputs, 'vision_hidden_states') and outputs.vision_hidden_states is not None:
                features = outputs.vision_hidden_states[-1]
            elif hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
                features = outputs.hidden_states[-1]
            elif hasattr(outputs, 'last_hidden_state'):
                features = outputs.last_hidden_state
            else:
                raise RuntimeError("Could not extract features from model output")
            
            # Pool features (mean pooling over sequence dimension)
            features = features.mean(dim=1)  # (batch, feature_dim)
            
            return features.cpu().numpy()


if __name__ == '__main__':
    """Test the feature extractor."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test Gemma Feature Extractor')
    parser.add_argument('--video_path', type=str, required=True,
                        help='Path to test video')
    parser.add_argument('--model_name', type=str, default='google/gemma-3-4b-it',
                        help='Gemma model name')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'])
    parser.add_argument('--num_frames', type=int, default=16)
    
    args = parser.parse_args()
    
    print("="*70)
    print("GEMMA FEATURE EXTRACTOR TEST")
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
