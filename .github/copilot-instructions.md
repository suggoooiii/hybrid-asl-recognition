# Copilot Instructions for Hybrid ASL Recognition

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
