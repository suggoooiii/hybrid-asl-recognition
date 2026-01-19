# Copilot Instructions for Hybrid ASL Recognition

## Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# MediaPipe models are auto-downloaded by training scripts
# Or download manually via download_mediapipe_models() function
```

### Running the Project
```bash
# Training
python train_hybrid_asl.py --data_dir data/wlasl --subset nslt_100.json --batch_size 8

# Evaluation
python evaluate_model.py --checkpoint best_hybrid_asl_model.pth --data_dir data/wlasl

# Webcam inference (requires trained model)
python webcam_hybrid_asl.py --device cuda
```

## Project Overview

A **hybrid American Sign Language (ASL) recognition system** combining:

- **Gemma 3 Vision Encoder** (frozen) for rich visual features via pre-extraction
- **MediaPipe** for explicit hand/pose landmark features (162 dims/frame)

Research goal: Demonstrate that fusion of visual and landmark streams improves recognition accuracy, with an architecture designed for future SignGemma integration.

## Architecture

### Primary Pipeline (Gemma + MediaPipe)

Two-phase approach optimized for fast iteration and low compute requirements:

```
PHASE 1 — Pre-extraction (run once):
Video → [Gemma 3 Vision Encoder (FROZEN)] → gemma_features/{video_id}_gemma.npy
Video → [MediaPipe] → landmarks/{video_id}_landmarks.npy

PHASE 2 — Training (fast, ~2.8M params):
Gemma Features (.npy) → [Visual Projection] ─┐
                                              ├─→ Fusion → Classifier → Sign Label
Landmarks (.npy) → [Landmark Encoder] ────────┘
```

### Architecture Comparison

| Aspect             | Gemma (Primary)          | VideoMAE (Alternative)    |
| ------------------ | ------------------------ | ------------------------- |
| Visual Encoder     | Gemma 3 (4B, frozen)     | VideoMAE (86M, trainable) |
| Trainable Params   | ~2.8M                    | ~88M                      |
| GPU Memory (Train) | 2-4 GB                   | 12-16 GB                  |
| Time per Epoch     | 5-10 min                 | 30-60 min                 |
| Batch Size         | 32-64+                   | 8-16                      |
| End-to-end Tuning  | ❌ No                    | ✅ Yes                    |
| Best For           | Fast iteration, research | Final models, fine-tuning |

### Key Files

#### Primary Pipeline (Gemma)

| File                           | Purpose                                                           |
| ------------------------------ | ----------------------------------------------------------------- |
| `gemma_feature_extractor.py`   | Frozen Gemma 3 feature extraction (2048-dim per frame)            |
| `hybrid_asl_model_simple.py`   | Lightweight model: `GemmaVisualEncoder`, reuses `LandmarkEncoder` |
| `preprocess_gemma_features.py` | Batch pre-extraction script for all videos                        |
| `train_simple.py`              | Fast training with pre-extracted `.npy` features                  |
| `preprocess_landmarks.py`      | Batch MediaPipe landmark extraction                               |

#### Alternative Pipeline (VideoMAE — for end-to-end fine-tuning)

| File                  | Purpose                                                                                |
| --------------------- | -------------------------------------------------------------------------------------- |
| `hybrid_asl_model.py` | Full model: `HybridASLModel`, `VideoMAEEncoder`, `LandmarkEncoder`, `HybridASLTrainer` |
| `train_hybrid_asl.py` | End-to-end training with on-the-fly feature extraction                                 |

#### Shared

| File                   | Purpose                                  |
| ---------------------- | ---------------------------------------- |
| `extract_landmarks.py` | Standalone MediaPipe landmark extraction |
| `webcam_hybrid_asl.py` | Real-time webcam inference               |

## Feature Vector Structure (162 dims/frame)

| Component  | Indices | Size | Source                    |
| ---------- | ------- | ---- | ------------------------- |
| Left hand  | 0-62    | 63   | 21 landmarks × 3 coords   |
| Right hand | 63-125  | 63   | 21 landmarks × 3 coords   |
| Upper body | 126-161 | 36   | 12 pose landmarks (11-22) |

Pad with `[0.0] * N` when landmarks not detected.

## Critical Patterns

### MediaPipe Tasks API (NOT legacy Holistic)

```python
from mediapipe.tasks.vision import HandLandmarker, PoseLandmarker, RunningMode
```

Models auto-downloaded by `download_mediapipe_models()`.

### Gemma Feature Extraction

- Model: `google/gemma-3-4b-it` (frozen, vision encoder only)
- 16 frames sampled uniformly
- Per-frame features: 2048-dim → 256-dim (projected)
- Requires `transformers>=4.45.0` for multimodal support

```python
from gemma_feature_extractor import GemmaFeatureExtractor
extractor = GemmaFeatureExtractor(device="cuda")
features = extractor.extract_from_video(video_path, num_frames=16)  # (16, 2048)
```

### VideoMAE Processing (Alternative)

- Model: `MCG-NJU/videomae-base` (768 → 256 projected)
- 16 frames sampled uniformly
- Use `VideoMAEImageProcessor.from_pretrained()`

### Fusion Strategies

- `concat` (default): Concatenation + linear projection
- `attention`: Cross-attention between streams
- `gated`: Learnable gate for stream contribution

## Dataset (WLASL)

```
data/wlasl/
├── videos/              # {video_id}.mp4
├── WLASL_v0.3.json      # Main annotations
├── nslt_100.json        # Subset configs (100/300/1000/2000)
├── missing.txt          # ~9100 unavailable videos
├── landmarks/           # Pre-extracted MediaPipe features
│   └── {video_id}_landmarks.npy
└── gemma_features/      # Pre-extracted Gemma features
    └── {video_id}_gemma.npy
```

JSON structure: `{video_id: {"action": [class_idx, start_frame, end_frame], "subset": "train|val|test"}}`

## Commands

### Primary Pipeline (Gemma) — Recommended

```bash
# Step 1: Pre-extract Gemma features (run once, ~2-3 hours for NSLT-100)
python preprocess_gemma_features.py \
    --data_dir data/wlasl \
    --output_dir data/wlasl/gemma_features \
    --subset nslt_100.json

# Step 2: Pre-extract landmarks if not done (run once)
python preprocess_landmarks.py \
    --data_dir data/wlasl \
    --output_dir data/wlasl/landmarks \
    --subset nslt_100.json

# Step 3: Train (fast iteration)
python train_simple.py \
    --data_dir data/wlasl \
    --gemma_features_dir data/wlasl/gemma_features \
    --landmarks_dir data/wlasl/landmarks \
    --batch_size 32 \
    --fusion_type concat

# Webcam inference
python webcam_hybrid_asl.py --model_path best_simple_model.pth --device cuda
```

### Alternative Pipeline (VideoMAE) — For End-to-End Fine-Tuning

```bash
# Training (requires 12-16GB VRAM)
python train_hybrid_asl.py \
    --data_dir data/wlasl \
    --subset nslt_100.json \
    --batch_size 8 \
    --fusion_type concat

# Use --device cpu or --device mps if no CUDA GPU
```

## Common Pitfalls & Edge Cases

### GPU/Memory Issues

- **Gemma extraction**: Requires ~8GB VRAM for feature extraction (one-time)
- **Gemma training**: Only 2-4GB VRAM needed (features pre-extracted)
- **VideoMAE is memory-hungry**: Reduce `--batch_size` (try 4 or 2) if OOM
- Default device is `cuda`; use `--device cpu` or `--device mps` (Apple Silicon)

### Video Processing Failures

- **Missing videos**: ~9100 videos in `missing.txt` are unavailable; the loader skips them automatically
- **Corrupted/empty videos**: `cv2.VideoCapture` returns empty frames; padding kicks in but may hurt accuracy
- **FPS detection**: Falls back to 30 FPS if `CAP_PROP_FPS` returns 0

### MediaPipe Edge Cases

- **No hands detected**: Returns `[0.0] * 63` for that hand—common in occlusion
- **Hand chirality flip**: MediaPipe's "Left"/"Right" is from camera's perspective (mirrored in webcam)
- **Timestamp required**: Must increment monotonically for VIDEO mode; use `int(frame_idx * 1000 / fps)`

### Pre-extraction Pitfalls

- **Re-extraction required** if changing `num_frames` or frame sampling strategy
- **Disk space**: Gemma features ~500KB per video; landmarks ~50KB per video
- **Missing features**: Training scripts skip videos without both `.npy` files

### Label Mapping

- `label_mapping.json` uses **string keys** (JSON limitation): `{"0": "book", "1": "drink", ...}`
- Load with: `{int(k): v for k, v in json.load(f).items()}` if you need int keys

## Training Configuration

### Primary (Gemma Pipeline)

| Parameter       | Value             | Notes                  |
| --------------- | ----------------- | ---------------------- |
| Base LR         | 1e-4              | AdamW optimizer        |
| Scheduler       | CosineAnnealingLR | T_max=50, eta_min=1e-6 |
| Gradient clip   | max_norm=1.0      | Prevents explosion     |
| Dropout         | 0.3               | Throughout model       |
| Batch size      | 32-64             | Fits on 4GB VRAM       |
| Train/Val split | 80/20             | Random with seed=42    |

### Alternative (VideoMAE Pipeline)

| Parameter       | Value             | Notes                      |
| --------------- | ----------------- | -------------------------- |
| VideoMAE LR     | 0.1× base         | Differential learning rate |
| Scheduler       | CosineAnnealingLR | T_max=50, eta_min=1e-6     |
| Gradient clip   | max_norm=1.0      | Prevents explosion         |
| Dropout         | 0.3               | Throughout model           |
| Batch size      | 8-16              | Requires 12-16GB VRAM      |
| Train/Val split | 80/20             | Random with seed=42        |

## Output Files

### Primary (Gemma Pipeline)

- `data/wlasl/gemma_features/{video_id}_gemma.npy` — Pre-extracted visual features (16, 2048)
- `data/wlasl/landmarks/{video_id}_landmarks.npy` — Pre-extracted landmarks (N, 162)
- `best_simple_model.pth` — Model weights (saved on best val accuracy)
- `checkpoints_simple/` — Training checkpoints
- `label_mapping.json` — Class index to gloss name mapping

### Alternative (VideoMAE Pipeline)

- `best_hybrid_asl_model.pth` — Model weights (saved on best val accuracy)
- `checkpoints/` — Training checkpoints

## Research Extensions

### Ablation Studies

Compare these configurations to isolate contributions:

1. **Gemma-only**: Disable landmark branch
2. **Landmark-only**: Disable visual branch
3. **Hybrid (Gemma + Landmarks)**: Full model
4. **Hybrid (VideoMAE + Landmarks)**: Alternative pipeline for comparison

### Fusion Comparison

Benchmark fusion strategies across both pipelines:

- `concat`: Simple, fast baseline
- `attention`: Cross-attention for learned alignment
- `gated`: Adaptive stream weighting

### SignGemma Future-Proofing

The Gemma architecture is designed for **drop-in replacement** when Google releases SignGemma (trained on 10,000+ hours of ASL data). To upgrade:

1. Update model name in `gemma_feature_extractor.py`
2. Re-run `preprocess_gemma_features.py`
3. Retrain with `train_simple.py`

No architectural changes needed—same feature dimensions expected.

### Additional Research Directions

- **Cross-dataset**: Test generalization to other sign language datasets (BSL, DGS)
- **Temporal modeling**: Experiment with different `num_frames` (8, 16, 32)
- **Metrics to report**: Top-1, Top-5 accuracy; per-class accuracy for imbalanced classes
- **Feature visualization**: t-SNE/UMAP of Gemma vs VideoMAE feature spaces

## Python Coding Conventions

### General Guidelines
- **Shebang**: All executable Python scripts start with `#!/usr/bin/env python3`
- **Docstrings**: Use triple-quoted docstrings for modules, classes, and functions
- **Module docstrings**: Include usage examples and purpose at top of file
- **Type hints**: Use when it improves code clarity (see `typing` module usage in codebase)
- **Imports**: Group in order: standard library, third-party, local modules

### Naming Conventions
- **Classes**: PascalCase (e.g., `HybridASLModel`, `VideoAugmentation`)
- **Functions/methods**: snake_case (e.g., `download_mediapipe_models`, `extract_landmarks`)
- **Constants**: UPPER_SNAKE_CASE (e.g., `PEFT_AVAILABLE`, `LANDMARK_DIM`)
- **Private methods**: Prefix with underscore (e.g., `_apply_temporal_masking`)

### Code Organization
- Related functionality grouped with visual separators using `═` for major sections
- Comment blocks use `#` for explanations above code
- Keep functions focused and under 50 lines when possible
- Use descriptive variable names (e.g., `num_frames` not `n`, `landmark_features` not `lf`)

### Error Handling
- Check for missing dependencies with try/except at module level
- Provide helpful error messages with installation instructions
- Use `Path` from `pathlib` for file operations instead of `os.path`
- Gracefully handle missing/corrupted video files by skipping and logging

### MediaPipe Specific
- **Always** use the new Tasks API (`mediapipe.tasks.vision`), NOT legacy `mediapipe.solutions`
- Provide fallback values (zero padding) when landmarks not detected
- Handle timestamp calculations properly: `int(frame_idx * 1000 / fps)`
- Remember MediaPipe's hand chirality is from camera perspective (mirrored)

### PyTorch Conventions
- Use `torch.nn.Module` for all model components
- Implement `forward()` method explicitly
- Use `F.` prefix for functional operations (e.g., `F.relu`, `F.dropout`)
- Always specify `device` parameter in scripts (default to `cuda`)
- Include `.to(device)` calls for tensors and models

### Comments and Documentation
- Prefer clear code over comments when possible
- Add comments for non-obvious logic (edge cases, workarounds, research decisions)
- Document why, not what (the code shows what, comments explain why)
- Include references to papers/docs when implementing research methods

## Testing and Validation

### Current State
- No automated test suite currently exists
- Manual testing done via training runs and webcam inference
- Model validation performed during training (80/20 train/val split)

### Testing Best Practices (for future additions)
- Test landmark extraction with various video qualities
- Validate fusion strategies produce expected tensor shapes
- Test edge cases: empty frames, missing landmarks, single-hand signs
- Verify model checkpoint loading/saving
- Test data augmentation reproducibility

### Manual Validation
```bash
# Quick model sanity check
python -c "from hybrid_asl_model import HybridASLModel; model = HybridASLModel(num_classes=100); print(model)"

# Test landmark extraction
python extract_landmarks.py --video_path path/to/test.mp4 --output landmarks.npy

# Verify dataset loading
python -c "from train_hybrid_asl import load_wlasl_dataset; videos, labels = load_wlasl_dataset('data/wlasl', 'nslt_100.json'); print(f'Loaded {len(videos)} videos')"
```

## Development Workflow

### Common Tasks

**Adding a new fusion strategy:**
1. Add new option to `fusion_type` in `HybridASLModel.__init__()`
2. Implement fusion logic in `forward()` method
3. Update `train_hybrid_asl.py` argparse choices
4. Document in Architecture section above
5. Test with small batch to verify tensor shapes

**Modifying landmark extraction:**
1. Edit `MediaPipeLandmarkExtractor` class
2. Update `LANDMARK_DIM` constant if dimensions change
3. Update Feature Vector Structure table above
4. Verify with `extract_landmarks.py` script

**Debugging training issues:**
1. Reduce batch size (8 → 4 → 2) if OOM errors
2. Check GPU memory: `nvidia-smi`
3. Enable gradient clipping (already set to 1.0)
4. Verify learning rate schedule in logs
5. Check for NaN gradients: add `torch.autograd.set_detect_anomaly(True)`

**Adding new data augmentation:**
1. Implement in `augmentations.py` following existing class structure
2. Ensure augmentation preserves ASL semantic meaning
3. **Never** use horizontal flip (breaks left/right hand semantics)
4. Test augmentation preserves landmark coordinate validity

## Security Considerations

### Data Privacy
- WLASL dataset contains publicly available videos
- When adding new datasets, verify license and consent
- Never commit actual video files to repository (use `.gitignore`)

### Model Security
- Checkpoint files (`.pth`) can be large; store externally for production
- Validate input video formats before processing
- Sanitize file paths to prevent directory traversal

### Dependencies
- Keep PyTorch and transformers updated for security patches
- MediaPipe models downloaded from official Google storage
- Verify model checksums if adding new MediaPipe models

## Performance Optimization

### Training Speed
- Use mixed precision training: `torch.cuda.amp` (not yet implemented)
- Increase `num_workers` in DataLoader (default: 4)
- Consider smaller VideoMAE model: `videomae-small` instead of `videomae-base`
- Enable LoRA for faster fine-tuning with `--use_lora` flag

### Inference Speed
- Batch processing for offline evaluation
- Model quantization for deployment (not yet implemented)
- Cache landmark extraction results to avoid recomputation
- Use `--device cpu` for webcam if GPU unavailable

## Troubleshooting

### Import Errors
```bash
# Missing mediapipe
pip install mediapipe

# Missing peft (for LoRA)
pip install peft

# Missing transformers
pip install transformers>=4.30.0
```

### Video Loading Issues
- Ensure OpenCV is installed: `pip install opencv-python`
- Check video codec compatibility (H.264 recommended)
- Verify video paths are absolute or relative to script location
- Check `missing.txt` for known unavailable WLASL videos

### CUDA Errors
- Verify CUDA version matches PyTorch: `python -c "import torch; print(torch.cuda.is_available())"`
- Reduce batch size if OOM
- Clear cache: `torch.cuda.empty_cache()`
- Use CPU fallback: `--device cpu`

## Citations

```bibtex
@inproceedings{li2020wlasl,
  title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
  author={Li, Dongxu and others},
  booktitle={WACV},
  year={2020}
}

@inproceedings{tong2022videomae,
  title={VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training},
  author={Tong, Zhan and others},
  booktitle={NeurIPS},
  year={2022}
}

@article{gemma2024,
  title={Gemma: Open Models Based on Gemini Research and Technology},
  author={Google DeepMind},
  journal={arXiv preprint},
  year={2024}
}
```
