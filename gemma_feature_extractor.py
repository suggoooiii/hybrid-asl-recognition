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
import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple
from transformers import AutoProcessor, AutoModel, PaliGemmaForConditionalGeneration
from PIL import Image
import warnings

# Configure logging
logger = logging.getLogger(__name__)


def setup_logging(level=logging.INFO, log_file: Optional[str] = None, log_dir: str = "logs"):
    """
    Setup logging configuration for feature extraction.

    Args:
        level: Logging level (e.g., logging.INFO, logging.DEBUG)
        log_file: Optional log filename. If None, auto-generates timestamped filename.
        log_dir: Directory for log files (default: 'logs')

    Returns:
        Logger instance and path to log file
    """
    # Create log directory if it doesn't exist
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    # Generate log filename with timestamp if not provided
    if log_file is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"gemma_extraction_{timestamp}.log"

    log_file_path = log_path / log_file

    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )

    # Get root logger and clear existing handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Console handler (INFO and above)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (DEBUG and above - captures everything)
    file_handler = logging.FileHandler(
        log_file_path, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)  # Always capture debug in file
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Log the setup
    module_logger = logging.getLogger(__name__)
    module_logger.info(f"Logging initialized - file: {log_file_path}")

    return module_logger, str(log_file_path)


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

        logger.info("=" * 60)
        logger.info("INITIALIZING PALIGEMMA FEATURE EXTRACTOR")
        logger.info("=" * 60)
        logger.info(f"Model: {model_name}")
        logger.info(f"Device: {device}")
        logger.info(f"Cache dir: {cache_dir or 'default'}")
        logger.info(f"Flash attention: {use_flash_attention}")

        # Load processor (handles image preprocessing)
        logger.info("Loading processor...")
        load_start = time.time()
        self.processor = AutoProcessor.from_pretrained(
            model_name,
            cache_dir=cache_dir
        )
        logger.info(f"Processor loaded in {time.time() - load_start:.1f}s")

        # Load model with optimizations
        model_kwargs = {
            'cache_dir': cache_dir,
            'torch_dtype': torch.float16 if device == 'cuda' else torch.float32,
        }

        if use_flash_attention and device == 'cuda':
            model_kwargs['attn_implementation'] = 'flash_attention_2'
            logger.info("Using Flash Attention 2 for faster inference")

        # Load PaliGemma model
        logger.info("Loading PaliGemma model (this may take a while)...")
        load_start = time.time()
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
        logger.info(f"Model loaded in {time.time() - load_start:.1f}s")

        # Freeze all parameters (no training)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.eval()

        # Get the vision tower for direct feature extraction
        self.vision_tower = self.model.vision_tower

        # Get feature dimension from vision encoder
        # For Gemma 3, this is typically from the vision tower
        self.feature_dim = self._get_feature_dim()

        # Log memory usage if on GPU
        if device == 'cuda' and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(
                f"GPU memory: {memory_allocated:.2f}GB allocated, {memory_reserved:.2f}GB reserved")

        logger.info("=" * 60)
        logger.info(f"✓ EXTRACTOR READY | Feature dim: {self.feature_dim}")
        logger.info("=" * 60)

    def _get_feature_dim(self) -> int:
        """Infer feature dimension from the vision encoder."""
        # PaliGemma's SigLIP vision tower has a known hidden size
        try:
            config = self.vision_tower.config
            if hasattr(config, 'hidden_size'):
                logger.debug(
                    f"Feature dim from config.hidden_size: {config.hidden_size}")
                return config.hidden_size
        except Exception as e:
            logger.debug(f"Could not get feature dim from config: {e}")
        # Default for PaliGemma SigLIP (224x224 model)
        logger.debug("Using default feature dim: 1152")
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
            logger.error("Received None frame")
            raise ValueError("Frame is None")

        logger.debug(
            f"Processing frame: shape={frame.shape}, dtype={frame.dtype}")

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
            logger.debug(f"Pixel values shape: {pixel_values.shape}")

            # Extract features directly from vision tower (SigLIP)
            vision_outputs = self.vision_tower(pixel_values)

            # Get features from vision tower output
            if hasattr(vision_outputs, 'last_hidden_state') and vision_outputs.last_hidden_state is not None:
                features = vision_outputs.last_hidden_state
                logger.debug(f"Using last_hidden_state: {features.shape}")
            elif hasattr(vision_outputs, 'pooler_output') and vision_outputs.pooler_output is not None:
                logger.debug(
                    f"Using pooler_output: {vision_outputs.pooler_output.shape}")
                return vision_outputs.pooler_output.squeeze(0).cpu().numpy().astype(np.float32)
            elif isinstance(vision_outputs, tuple) and len(vision_outputs) > 0:
                features = vision_outputs[0]
                logger.debug(f"Using tuple output[0]: {features.shape}")
            else:
                features = vision_outputs
                logger.debug(f"Using raw output: type={type(features)}")

            # Mean pool over spatial/patch dimension
            # Shape: (1, num_patches, hidden_size) -> (hidden_size,)
            pooled = features.mean(dim=1).squeeze(0)
            logger.debug(f"Pooled features: {pooled.shape}")

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
        video_name = Path(video_path).stem
        logger.debug(f"[{video_name}] Opening video: {video_path}")

        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            logger.error(f"[{video_name}] Failed to open video")
            raise RuntimeError(f"Failed to open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.debug(
            f"[{video_name}] Video info: {total_frames} frames, {fps:.1f} FPS, {width}x{height}")

        # Handle empty or corrupted videos
        if total_frames <= 0:
            cap.release()
            logger.error(f"[{video_name}] Video has no frames or is corrupted")
            raise RuntimeError(
                f"Video has no frames or is corrupted: {video_path}")

        # Apply frame trimming if specified (convert 1-indexed to 0-indexed)
        if start_frame is not None and end_frame is not None:
            frame_start = max(0, start_frame - 1)
            frame_end = min(total_frames - 1, end_frame - 1)
            segment_length = frame_end - frame_start + 1
            logger.debug(
                f"[{video_name}] Using trimmed segment: frames {frame_start}-{frame_end} ({segment_length} frames)")
        else:
            frame_start = 0
            frame_end = total_frames - 1
            segment_length = total_frames
            logger.debug(
                f"[{video_name}] Using full video: {segment_length} frames")

        # Handle invalid frame ranges
        if segment_length <= 0:
            cap.release()
            logger.error(
                f"[{video_name}] Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}")
            raise RuntimeError(
                f"Invalid frame range: start={start_frame}, end={end_frame}, total={total_frames}")

        # Calculate which frames to sample (uniform sampling within segment)
        if segment_length <= num_frames:
            sample_indices = [frame_start + i for i in range(segment_length)]
            logger.debug(
                f"[{video_name}] Short video - using all {len(sample_indices)} frames")
        else:
            sample_indices = np.linspace(
                frame_start, frame_end, num_frames, dtype=int).tolist()
            logger.debug(
                f"[{video_name}] Sampling {num_frames} frames uniformly from {segment_length}")

        frame_features = []
        failed_frames = 0
        extraction_start = time.time()

        for i, frame_idx in enumerate(sample_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret or frame is None:
                # Pad with zeros if frame read fails
                frame_features.append(
                    np.zeros(self.feature_dim, dtype=np.float32))
                failed_frames += 1
                logger.debug(
                    f"[{video_name}] Frame {frame_idx} read failed, using zeros")
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Extract features
            features = self.extract_frame_features(frame_rgb)
            frame_features.append(features)

        cap.release()
        extraction_time = time.time() - extraction_start

        # Handle case where no frames were successfully read
        if len(frame_features) == 0:
            logger.error(
                f"[{video_name}] Could not read any frames from video")
            raise RuntimeError(
                f"Could not read any frames from video: {video_path}")

        # Padding if needed
        original_count = len(frame_features)
        if len(frame_features) < num_frames:
            last_frame = frame_features[-1] if frame_features else np.zeros(
                self.feature_dim, dtype=np.float32)
            while len(frame_features) < num_frames:
                frame_features.append(last_frame.copy())
            logger.debug(
                f"[{video_name}] Padded from {original_count} to {num_frames} frames")

        result = np.array(frame_features[:num_frames], dtype=np.float32)

        # Log summary for this video
        if failed_frames > 0:
            logger.warning(
                f"[{video_name}] Extracted {num_frames} features in {extraction_time:.2f}s ({failed_frames} failed frames)")
        else:
            logger.debug(
                f"[{video_name}] Extracted {num_frames} features in {extraction_time:.2f}s")

        return result

    def extract_batch_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a batch of frames (more efficient).

        Args:
            frames: List of RGB frames as numpy arrays

        Returns:
            Feature array of shape (len(frames), feature_dim)
        """
        batch_size = len(frames)
        logger.debug(f"Batch extraction: {batch_size} frames")
        batch_start = time.time()

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
            logger.debug(f"Batch pixel values shape: {pixel_values.shape}")

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

            batch_time = time.time() - batch_start
            logger.debug(
                f"Batch extraction complete: {batch_size} frames in {batch_time:.2f}s ({batch_time/batch_size*1000:.1f}ms/frame)")

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
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for log files (default: logs)')

    args = parser.parse_args()

    # Setup logging based on verbosity (logs to both console and file)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger, log_file = setup_logging(log_level, log_dir=args.log_dir)

    logger.info("=" * 70)
    logger.info("PALIGEMMA FEATURE EXTRACTOR TEST")
    logger.info("=" * 70)

    # Initialize extractor
    extractor = GemmaFeatureExtractor(
        model_name=args.model_name,
        device=args.device
    )

    # Extract features
    logger.info(f"Extracting features from: {args.video_path}")
    start_time = time.time()
    features = extractor.extract_video_features(
        args.video_path,
        num_frames=args.num_frames
    )
    total_time = time.time() - start_time

    logger.info(f"✓ Features shape: {features.shape}")
    logger.info(f"✓ Feature dimension: {features.shape[1]}")
    logger.info(
        f"✓ Feature range: [{features.min():.3f}, {features.max():.3f}]")
    logger.info(f"✓ Feature mean: {features.mean():.3f}")
    logger.info(f"✓ Feature std: {features.std():.3f}")
    logger.info(f"✓ Total extraction time: {total_time:.2f}s")

    logger.info("=" * 70)
    logger.info("TEST COMPLETE!")
    logger.info(f"Log saved to: {log_file}")
    logger.info("=" * 70)
