#!/usr/bin/env python3
"""
DATA AUGMENTATION MODULE FOR HYBRID ASL RECOGNITION

Provides video frame and landmark augmentations to reduce overfitting.
Designed to be imported by train_hybrid_asl.py and train_kaggle.ipynb.

Usage:
    from augmentations import VideoAugmentation, LandmarkAugmentation
    
    video_aug = VideoAugmentation(enabled=True)
    landmark_aug = LandmarkAugmentation(enabled=True)
    
    # Apply to frames (list of numpy arrays HxWxC)
    augmented_frames = video_aug(frames)
    
    # Apply to landmarks (numpy array TxD)
    augmented_landmarks = landmark_aug(landmarks)
"""

import numpy as np
import random
from typing import List, Optional, Tuple


class VideoAugmentation:
    """
    Video frame augmentations for ASL recognition.

    Note: Horizontal flip is DISABLED by default for ASL because
    flipping can change sign meaning (e.g., directional signs).
    """

    def __init__(
        self,
        enabled: bool = True,
        brightness_range: Tuple[float, float] = (0.7, 1.3),
        contrast_range: Tuple[float, float] = (0.7, 1.3),
        saturation_range: Tuple[float, float] = (0.8, 1.2),
        rotation_degrees: float = 15.0,
        scale_range: Tuple[float, float] = (0.85, 1.0),
        horizontal_flip: bool = False,  # Disabled for ASL
        gaussian_noise_std: float = 0.02,
        temporal_dropout_prob: float = 0.1,
    ):
        self.enabled = enabled
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.saturation_range = saturation_range
        self.rotation_degrees = rotation_degrees
        self.scale_range = scale_range
        self.horizontal_flip = horizontal_flip
        self.gaussian_noise_std = gaussian_noise_std
        self.temporal_dropout_prob = temporal_dropout_prob

    def __call__(self, frames: List[np.ndarray]) -> List[np.ndarray]:
        """
        Apply augmentations to a list of video frames.

        Args:
            frames: List of numpy arrays (H, W, C) in RGB format, values 0-255

        Returns:
            List of augmented frames
        """
        if not self.enabled or len(frames) == 0:
            return frames

        # Sample augmentation parameters ONCE per video (consistency across frames)
        brightness = random.uniform(*self.brightness_range)
        contrast = random.uniform(*self.contrast_range)
        saturation = random.uniform(*self.saturation_range)
        angle = random.uniform(-self.rotation_degrees, self.rotation_degrees)
        scale = random.uniform(*self.scale_range)
        do_flip = self.horizontal_flip and random.random() > 0.5

        augmented = []
        for i, frame in enumerate(frames):
            # Temporal dropout: randomly duplicate previous frame
            if i > 0 and random.random() < self.temporal_dropout_prob:
                augmented.append(augmented[-1].copy())
                continue

            aug_frame = frame.copy().astype(np.float32)

            # Color jitter
            aug_frame = self._adjust_brightness(aug_frame, brightness)
            aug_frame = self._adjust_contrast(aug_frame, contrast)
            aug_frame = self._adjust_saturation(aug_frame, saturation)

            # Geometric transforms
            if abs(angle) > 0.5 or scale != 1.0:
                aug_frame = self._rotate_and_scale(aug_frame, angle, scale)

            # Horizontal flip
            if do_flip:
                aug_frame = np.fliplr(aug_frame)

            # Gaussian noise
            if self.gaussian_noise_std > 0:
                noise = np.random.normal(
                    0, self.gaussian_noise_std * 255, aug_frame.shape)
                aug_frame = aug_frame + noise

            # Clip to valid range
            aug_frame = np.clip(aug_frame, 0, 255).astype(np.uint8)
            augmented.append(aug_frame)

        return augmented

    def _adjust_brightness(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust brightness by multiplying pixel values."""
        return img * factor

    def _adjust_contrast(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust contrast around the mean."""
        mean = img.mean()
        return (img - mean) * factor + mean

    def _adjust_saturation(self, img: np.ndarray, factor: float) -> np.ndarray:
        """Adjust color saturation."""
        gray = np.dot(img[..., :3], [0.299, 0.587, 0.114])
        gray = np.stack([gray] * 3, axis=-1)
        return gray + (img - gray) * factor

    def _rotate_and_scale(self, img: np.ndarray, angle: float, scale: float) -> np.ndarray:
        """Apply rotation and scaling using OpenCV."""
        try:
            import cv2
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, scale)
            rotated = cv2.warpAffine(
                img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
            return rotated
        except ImportError:
            # Fallback: no rotation if cv2 not available
            return img


class LandmarkAugmentation:
    """
    Landmark augmentations for MediaPipe hand/pose landmarks.

    Landmarks shape: (T, 162) where T=num_frames, 162=landmark dimensions
    Structure: [left_hand(63), right_hand(63), upper_body(36)]
    """

    def __init__(
        self,
        enabled: bool = True,
        noise_std: float = 0.02,
        temporal_shift_range: int = 2,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        dropout_prob: float = 0.05,
        hand_swap_prob: float = 0.0,  # Disabled: changes sign meaning
        translation_range: float = 0.05,
    ):
        self.enabled = enabled
        self.noise_std = noise_std
        self.temporal_shift_range = temporal_shift_range
        self.scale_range = scale_range
        self.dropout_prob = dropout_prob
        self.hand_swap_prob = hand_swap_prob
        self.translation_range = translation_range

    def __call__(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Apply augmentations to landmark sequence.

        Args:
            landmarks: numpy array of shape (T, 162)

        Returns:
            Augmented landmarks of same shape
        """
        if not self.enabled:
            return landmarks

        landmarks = landmarks.copy().astype(np.float32)

        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, landmarks.shape)
            landmarks = landmarks + noise

        # Temporal shift (roll frames)
        if self.temporal_shift_range > 0:
            shift = random.randint(-self.temporal_shift_range,
                                   self.temporal_shift_range)
            if shift != 0:
                landmarks = np.roll(landmarks, shift, axis=0)

        # Scale landmarks (zoom in/out effect)
        if self.scale_range != (1.0, 1.0):
            scale = random.uniform(*self.scale_range)
            # Scale around center (0.5, 0.5) for normalized landmarks
            landmarks = self._scale_landmarks(landmarks, scale)

        # Translation (shift all landmarks)
        if self.translation_range > 0:
            tx = random.uniform(-self.translation_range,
                                self.translation_range)
            ty = random.uniform(-self.translation_range,
                                self.translation_range)
            landmarks = self._translate_landmarks(landmarks, tx, ty)

        # Landmark dropout (zero out random frames)
        if self.dropout_prob > 0:
            landmarks = self._apply_dropout(landmarks)

        # Hand swap (disabled by default - changes sign meaning)
        if self.hand_swap_prob > 0 and random.random() < self.hand_swap_prob:
            landmarks = self._swap_hands(landmarks)

        return landmarks.astype(np.float32)

    def _scale_landmarks(self, landmarks: np.ndarray, scale: float) -> np.ndarray:
        """Scale landmarks around center point."""
        # Assuming landmarks are normalized [0, 1]
        center = 0.5
        scaled = (landmarks - center) * scale + center
        return scaled

    def _translate_landmarks(self, landmarks: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """Translate x and y coordinates of landmarks."""
        translated = landmarks.copy()
        # Apply translation to x coordinates (indices 0, 3, 6, ... for each landmark)
        # Apply translation to y coordinates (indices 1, 4, 7, ... for each landmark)
        for i in range(0, landmarks.shape[1], 3):  # Every 3rd index is x
            translated[:, i] += tx
        for i in range(1, landmarks.shape[1], 3):  # Every 3rd+1 index is y
            translated[:, i] += ty
        return translated

    def _apply_dropout(self, landmarks: np.ndarray) -> np.ndarray:
        """Randomly zero out some frames."""
        mask = np.random.random(landmarks.shape[0]) > self.dropout_prob
        landmarks[~mask] = 0.0
        return landmarks

    def _swap_hands(self, landmarks: np.ndarray) -> np.ndarray:
        """Swap left and right hand landmarks."""
        swapped = landmarks.copy()
        # Left hand: indices 0-62, Right hand: indices 63-125
        left_hand = landmarks[:, 0:63].copy()
        right_hand = landmarks[:, 63:126].copy()
        swapped[:, 0:63] = right_hand
        swapped[:, 63:126] = left_hand
        return swapped


class CombinedAugmentation:
    """
    Combines video and landmark augmentation with coordinated transforms.

    Ensures that spatial transforms (flip, scale) are applied consistently
    to both video frames and landmarks.
    """

    def __init__(
        self,
        video_aug: Optional[VideoAugmentation] = None,
        landmark_aug: Optional[LandmarkAugmentation] = None,
    ):
        self.video_aug = video_aug or VideoAugmentation(enabled=False)
        self.landmark_aug = landmark_aug or LandmarkAugmentation(enabled=False)

    def __call__(
        self,
        frames: List[np.ndarray],
        landmarks: np.ndarray
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Apply coordinated augmentations to frames and landmarks.

        Args:
            frames: List of video frames
            landmarks: Landmark array (T, 162)

        Returns:
            Tuple of (augmented_frames, augmented_landmarks)
        """
        aug_frames = self.video_aug(frames)
        aug_landmarks = self.landmark_aug(landmarks)
        return aug_frames, aug_landmarks


def get_augmentations(
    enabled: bool = True,
    strength: str = 'medium'
) -> Tuple[VideoAugmentation, LandmarkAugmentation]:
    """
    Factory function to create augmentation objects.

    Args:
        enabled: Whether augmentations are active
        strength: 'light', 'medium', or 'strong'

    Returns:
        Tuple of (VideoAugmentation, LandmarkAugmentation)
    """
    if not enabled:
        return (
            VideoAugmentation(enabled=False),
            LandmarkAugmentation(enabled=False)
        )

    configs = {
        'light': {
            'video': {
                'brightness_range': (0.85, 1.15),
                'contrast_range': (0.85, 1.15),
                'rotation_degrees': 8.0,
                'scale_range': (0.92, 1.0),
                'gaussian_noise_std': 0.01,
            },
            'landmark': {
                'noise_std': 0.01,
                'temporal_shift_range': 1,
                'scale_range': (0.95, 1.05),
                'translation_range': 0.02,
            }
        },
        'medium': {
            'video': {
                'brightness_range': (0.7, 1.3),
                'contrast_range': (0.7, 1.3),
                'rotation_degrees': 15.0,
                'scale_range': (0.85, 1.0),
                'gaussian_noise_std': 0.02,
            },
            'landmark': {
                'noise_std': 0.02,
                'temporal_shift_range': 2,
                'scale_range': (0.9, 1.1),
                'translation_range': 0.05,
            }
        },
        'strong': {
            'video': {
                'brightness_range': (0.6, 1.4),
                'contrast_range': (0.6, 1.4),
                'rotation_degrees': 20.0,
                'scale_range': (0.75, 1.0),
                'gaussian_noise_std': 0.03,
            },
            'landmark': {
                'noise_std': 0.03,
                'temporal_shift_range': 3,
                'scale_range': (0.85, 1.15),
                'translation_range': 0.08,
                'dropout_prob': 0.1,
            }
        }
    }

    cfg = configs.get(strength, configs['medium'])

    return (
        VideoAugmentation(enabled=True, **cfg['video']),
        LandmarkAugmentation(enabled=True, **cfg['landmark'])
    )


# For quick testing
if __name__ == '__main__':
    print("Testing augmentations...")

    # Create dummy data
    frames = [np.random.randint(
        0, 255, (224, 224, 3), dtype=np.uint8) for _ in range(16)]
    landmarks = np.random.rand(16, 162).astype(np.float32)

    # Test video augmentation
    video_aug = VideoAugmentation(enabled=True)
    aug_frames = video_aug(frames)
    print(
        f"✓ Video augmentation: {len(frames)} frames → {len(aug_frames)} frames")

    # Test landmark augmentation
    landmark_aug = LandmarkAugmentation(enabled=True)
    aug_landmarks = landmark_aug(landmarks)
    print(
        f"✓ Landmark augmentation: {landmarks.shape} → {aug_landmarks.shape}")

    # Test factory function
    v_aug, l_aug = get_augmentations(enabled=True, strength='medium')
    print(f"✓ Factory function: strength='medium'")

    # Test combined
    combined = CombinedAugmentation(v_aug, l_aug)
    aug_f, aug_l = combined(frames, landmarks)
    print(
        f"✓ Combined augmentation: frames={len(aug_f)}, landmarks={aug_l.shape}")

    print("\nAll tests passed!")
