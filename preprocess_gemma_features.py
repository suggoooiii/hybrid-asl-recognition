#!/usr/bin/env python3
"""
PRE-EXTRACT GEMMA FEATURES FOR ALL VIDEOS

Run this ONCE before training to extract and save Gemma features as .npy files.
This speeds up training 10-50x since Gemma inference only happens once.

Usage:
    python preprocess_gemma_features.py \
        --data_dir data/wlasl \
        --output_dir data/wlasl/gemma_features \
        --subset nslt_100.json

Options:
    --model_name: Gemma model to use (default: google/gemma-3-4b-it)
    --device: cuda, cpu, or mps
    --batch_size: Process multiple frames at once (faster on GPU)
    --skip_existing: Skip videos that already have extracted features
"""

import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import traceback
from datetime import datetime

from gemma_feature_extractor import GemmaFeatureExtractor, setup_logging


def load_wlasl_videos(data_dir, subset_file='nslt_100.json'):
    """
    Load WLASL video list from subset file.

    Returns:
        video_paths: List of video paths
        video_ids: List of video IDs
        frame_info: List of (start_frame, end_frame) tuples
    """
    data_dir = Path(data_dir)
    videos_dir = data_dir / 'videos'

    # Load missing videos list
    missing_videos = set()
    missing_path = data_dir / 'missing.txt'
    if missing_path.exists():
        with open(missing_path, 'r') as f:
            missing_videos = {line.strip() for line in f if line.strip()}

    # Load subset config
    subset_path = data_dir / subset_file
    if not subset_path.exists():
        raise FileNotFoundError(f"Subset file not found: {subset_path}")

    with open(subset_path, 'r') as f:
        subset_data = json.load(f)

    video_paths = []
    video_ids = []
    frame_info = []
    skipped = 0

    for video_id, info in subset_data.items():
        # Skip missing videos
        if video_id in missing_videos:
            skipped += 1
            continue

        video_path = videos_dir / f"{video_id}.mp4"

        if not video_path.exists():
            skipped += 1
            continue

        # Validate annotation format
        if info is None or 'action' not in info or info['action'] is None:
            print(f"  Warning: Invalid annotation for {video_id}, skipping")
            skipped += 1
            continue

        action = info['action']
        if not isinstance(action, list) or len(action) < 3:
            print(
                f"  Warning: Invalid action format for {video_id}: {action}, skipping")
            skipped += 1
            continue

        # Extract frame info: [class_idx, start_frame, end_frame]
        start_frame = action[1]
        end_frame = action[2]

        video_paths.append(str(video_path))
        video_ids.append(video_id)
        frame_info.append((start_frame, end_frame))

    print(f"Found {len(video_paths)} videos")
    if skipped > 0:
        print(f"Skipped {skipped} missing videos")

    return video_paths, video_ids, frame_info


def main():
    parser = argparse.ArgumentParser(
        description='Pre-extract Gemma features from videos')

    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to WLASL data directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for Gemma features (.npy files)')
    parser.add_argument('--subset', type=str, default='nslt_100.json',
                        help='Subset file (nslt_100.json, nslt_300.json, etc.)')
    parser.add_argument('--model_name', type=str, default='google/paligemma-3b-pt-224',
                        help='PaliGemma model name from HuggingFace')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to run on')
    parser.add_argument('--num_frames', type=int, default=16,
                        help='Number of frames to extract per video')
    parser.add_argument('--skip_existing', action='store_true',
                        help='Skip videos with existing features')
    parser.add_argument('--use_flash_attention', action='store_true',
                        help='Enable Flash Attention 2 for faster inference (requires GPU)')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for log files (default: logs)')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose/debug logging')

    args = parser.parse_args()

    # Setup logging (both console and file)
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logger, log_file = setup_logging(
        level=log_level,
        log_file=f"gemma_preprocess_{args.subset.replace('.json', '')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",
        log_dir=args.log_dir
    )

    logger.info("=" * 70)
    logger.info("GEMMA FEATURE PRE-EXTRACTION")
    logger.info("=" * 70)
    logger.info(f"Log file: {log_file}")

    # ─────────────────────────────────────────────────────────────────
    # 1. Load video list
    # ─────────────────────────────────────────────────────────────────
    logger.info("Step 1: Loading video list...")
    logger.info(f"  Data dir: {args.data_dir}")
    logger.info(f"  Subset: {args.subset}")

    video_paths, video_ids, frame_info = load_wlasl_videos(
        args.data_dir,
        subset_file=args.subset
    )

    # ─────────────────────────────────────────────────────────────────
    # 2. Initialize Gemma feature extractor
    # ─────────────────────────────────────────────────────────────────
    logger.info("Step 2: Initializing Gemma feature extractor...")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Device: {args.device}")

    extractor = GemmaFeatureExtractor(
        model_name=args.model_name,
        device=args.device,
        use_flash_attention=args.use_flash_attention
    )

    # ─────────────────────────────────────────────────────────────────
    # 3. Create output directory
    # ─────────────────────────────────────────────────────────────────
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Step 3: Output directory: {output_dir}")

    # ─────────────────────────────────────────────────────────────────
    # 4. Extract features
    # ─────────────────────────────────────────────────────────────────
    logger.info("Step 4: Extracting features...")
    logger.info(f"  Total videos: {len(video_paths)}")
    logger.info(f"  Frames per video: {args.num_frames}")

    success_count = 0
    skip_count = 0
    error_count = 0

    for video_path, video_id, (start_frame, end_frame) in tqdm(
        zip(video_paths, video_ids, frame_info),
        total=len(video_paths),
        desc="Extracting features"
    ):
        output_path = output_dir / f"{video_id}_gemma.npy"

        # Skip if already exists
        if args.skip_existing and output_path.exists():
            skip_count += 1
            continue

        try:
            # Extract features with frame trimming
            features = extractor.extract_video_features(
                video_path,
                num_frames=args.num_frames,
                start_frame=start_frame,
                end_frame=end_frame
            )

            # Save features
            np.save(output_path, features)
            success_count += 1

        except Exception as e:
            # Log errors with full traceback for debugging
            error_msg = f"Error processing {video_id} ({type(e).__name__}): {e}"
            if error_count < 5:
                logger.error(error_msg)
                logger.debug(
                    f"Full traceback for {video_id}:\n{traceback.format_exc()}")
            else:
                logger.error(error_msg)
            error_count += 1
            continue

    logger.info("")
    logger.info("=" * 70)
    logger.info("EXTRACTION COMPLETE!")
    logger.info("=" * 70)
    logger.info(f"✓ Successfully processed: {success_count} videos")
    if skip_count > 0:
        logger.info(f"✓ Skipped (already exists): {skip_count} videos")
    if error_count > 0:
        logger.warning(f"✗ Errors: {error_count} videos")
    logger.info(f"✓ Features saved to: {output_dir}")
    logger.info(f"✓ Feature dimension: {extractor.feature_dim}")

    # Save metadata
    metadata = {
        'model_name': args.model_name,
        'num_frames': args.num_frames,
        'feature_dim': extractor.feature_dim,
        'subset': args.subset,
        'success_count': success_count,
        'error_count': error_count
    }

    metadata_path = output_dir / 'extraction_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"✓ Metadata saved to: {metadata_path}")
    logger.info(f"✓ Full log saved to: {log_file}")
    logger.info("")


if __name__ == '__main__':
    main()
