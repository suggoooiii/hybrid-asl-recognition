# Copilot Instructions for Hybrid ASL Recognition

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
