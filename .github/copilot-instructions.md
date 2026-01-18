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

- **VideoMAE** (Visual Masked Autoencoder) for deep visual features
- **MediaPipe** for explicit hand/pose landmark features (162 dims/frame)

Research goal: Demonstrate that fusion of both streams improves recognition accuracy.

## Architecture

```
Video → [VideoMAE Branch] → Visual Features (256-dim) ─┐
                                                       ├─→ Fusion → Classifier → Sign Label
Video → [MediaPipe Branch] → Landmark Features (256-dim)┘
```

### Key Files

| File                   | Purpose                                                                                                        |
| ---------------------- | -------------------------------------------------------------------------------------------------------------- |
| `hybrid_asl_model.py`  | Core: `HybridASLModel`, `MediaPipeLandmarkExtractor`, `LandmarkEncoder`, `VideoMAEEncoder`, `HybridASLTrainer` |
| `train_hybrid_asl.py`  | Training with WLASL dataset loading                                                                            |
| `webcam_hybrid_asl.py` | Real-time webcam inference                                                                                     |
| `extract_landmarks.py` | Standalone landmark extraction                                                                                 |

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

### VideoMAE Processing

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
├── WLASL_v0.3.json     # Main annotations
├── nslt_100.json       # Subset configs (100/300/1000/2000)
└── missing.txt         # ~9100 unavailable videos
```

JSON structure: `{video_id: {"action": [class_idx, start_frame, end_frame], "subset": "train|val|test"}}`

## Commands

```bash
# Training
python train_hybrid_asl.py --data_dir data/wlasl --subset nslt_100.json --batch_size 8 --fusion_type concat

# Webcam inference (requires trained model)
python webcam_hybrid_asl.py --device cuda
# Use --device cpu if no GPU
```

## Common Pitfalls & Edge Cases

### GPU/Memory Issues

- **VideoMAE is memory-hungry**: Reduce `--batch_size` (try 4 or 2) if OOM
- Default device is `cuda`; use `--device cpu` or `--device mps` (Apple Silicon) if unavailable
- VideoMAE backbone alone is ~86M params

### Video Processing Failures

- **Missing videos**: ~9100 videos in `missing.txt` are unavailable; the loader skips them automatically
- **Corrupted/empty videos**: `cv2.VideoCapture` returns empty frames; padding kicks in but may hurt accuracy
- **FPS detection**: Falls back to 30 FPS if `CAP_PROP_FPS` returns 0

### MediaPipe Edge Cases

- **No hands detected**: Returns `[0.0] * 63` for that hand—common in occlusion
- **Hand chirality flip**: MediaPipe's "Left"/"Right" is from camera's perspective (mirrored in webcam)
- **Timestamp required**: Must increment monotonically for VIDEO mode; use `int(frame_idx * 1000 / fps)`

### Label Mapping

- `label_mapping.json` uses **string keys** (JSON limitation): `{"0": "book", "1": "drink", ...}`
- Load with: `{int(k): v for k, v in json.load(f).items()}` if you need int keys

## Training Configuration

| Parameter       | Value             | Notes                      |
| --------------- | ----------------- | -------------------------- |
| VideoMAE LR     | 0.1× base         | Differential learning rate |
| Scheduler       | CosineAnnealingLR | T_max=50, eta_min=1e-6     |
| Gradient clip   | max_norm=1.0      | Prevents explosion         |
| Dropout         | 0.3               | Throughout model           |
| Train/Val split | 80/20             | Random with seed=42        |

## Output Files

- `best_hybrid_asl_model.pth` — Model weights (saved on best val accuracy)
- `label_mapping.json` — Class index to gloss name mapping

## Research Extensions (Future)

When extending for research, consider:

- **Ablation studies**: Compare VideoMAE-only vs Landmark-only vs Hybrid
- **Fusion comparison**: Benchmark `concat` vs `attention` vs `gated`
- **Cross-dataset**: Test generalization to other sign language datasets
- **Temporal modeling**: Experiment with different `num_frames` (8, 16, 32)
- **Metrics to report**: Top-1, Top-5 accuracy; per-class accuracy for imbalanced classes

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
```
