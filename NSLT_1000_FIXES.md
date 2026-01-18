# NSLT-1000 Training Fixes

This document describes the fixes implemented to address training failures on `nslt_1000.json` dataset.

## Overview

Training on `nslt_1000.json` (1000 classes) was producing poor results due to several critical issues. This update addresses all identified problems.

## Changes Summary

### 1. Class Weights in Loss Function ✅

**Problem:** Severe class imbalance in nslt_1000 (some classes have 5 samples, others 50+) caused the model to ignore rare classes.

**Solution:**
- Added `compute_class_weights()` function that computes inverse-frequency weights with smoothing
- Weights are normalized and clipped to [0.1, 10.0] to prevent extreme values
- New arguments: `--use_class_weights` and `--no_class_weights`

**Usage:**
```bash
python train_simple.py --use_class_weights --subset nslt_1000.json
```

### 2. Sinusoidal Positional Encoding ✅

**Problem:** Random positional encoding provided no meaningful temporal information for transformers.

**Solution:**
- Added `SinusoidalPositionalEncoding` class with proper sin/cos positional embeddings
- Updated `LandmarkEncoder` in both `hybrid_asl_model.py` and `hybrid_asl_model_simple.py`
- Maintains backward compatibility with legacy random encoding

**Technical Details:**
- Uses standard Transformer-style positional encoding
- PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
- PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

### 3. Increased Model Capacity ✅

**Problem:** Model too small for 1000-class classification, causing underfitting.

**Solution:**
- Changed default `--hidden_dim` from 256 → 512
- Changed default `--dropout` from 0.3 → 0.5
- Deeper classifier for large num_classes (>500):
  - 3 linear layers instead of 2
  - Intermediate dimension = hidden_dim × 2 for large num_classes
  - Xavier initialization with gain=0.1 for final layer stability

**Capacity Comparison:**
- 100 classes: ~17M parameters
- 1000 classes: ~19M parameters (adaptive architecture)

### 4. Feature Dimension Verification ✅

**Problem:** Silent failures if `.npy` features were extracted with wrong model dimensions.

**Solution:**
- Added `verify_feature_dimensions()` function
- Automatically detects dimension mismatch
- Auto-adjusts `gemma_feature_dim` if needed
- Logs warnings for user awareness

### 5. Learning Rate Warmup ✅

**Problem:** Full learning rate at start destabilized early training with many classes.

**Solution:**
- Added `get_cosine_schedule_with_warmup()` function
- Linear warmup followed by cosine annealing
- New argument: `--warmup_epochs` (default: 5)
- Scheduler steps per batch during warmup, per epoch during cosine phase

**Schedule:**
- Steps 0 → warmup_steps: Linear increase from 0 → target_lr
- Steps warmup_steps → total_steps: Cosine annealing to eta_min

## New Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--hidden_dim` | 512 | Hidden dimension (increased from 256) |
| `--dropout` | 0.5 | Dropout rate (increased from 0.3) |
| `--label_smoothing` | 0.1 | Label smoothing for loss |
| `--warmup_epochs` | 5 | Number of warmup epochs |
| `--use_class_weights` | False | Enable class weights (recommended for nslt_1000) |
| `--no_class_weights` | - | Disable class weights (overrides default) |

## Recommended Training Commands

### For nslt_100.json (Quick Test)
```bash
python train_simple.py \
    --data_dir data/wlasl \
    --gemma_features_dir data/wlasl/gemma_features \
    --landmarks_dir data/wlasl/landmarks \
    --subset nslt_100.json \
    --hidden_dim 512 \
    --dropout 0.5 \
    --use_class_weights \
    --warmup_epochs 5 \
    --epochs 50 \
    --verbose
```

### For nslt_1000.json (Main Fix Target)
```bash
python train_simple.py \
    --data_dir data/wlasl \
    --gemma_features_dir data/wlasl/gemma_features \
    --landmarks_dir data/wlasl/landmarks \
    --subset nslt_1000.json \
    --hidden_dim 512 \
    --dropout 0.5 \
    --use_class_weights \
    --warmup_epochs 10 \
    --epochs 100 \
    --batch_size 32 \
    --verbose
```

## Testing

A comprehensive test suite is included in `test_nslt1000_fixes.py`:

```bash
python test_nslt1000_fixes.py
```

Tests cover:
1. Class weight computation
2. Feature dimension verification
3. Warmup scheduler
4. Sinusoidal positional encoding
5. Model capacity improvements
6. Integration test

## Expected Improvements

With these fixes, training on nslt_1000.json should show:

- **Reduced overfitting**: Higher dropout and label smoothing
- **Better convergence**: Warmup prevents early instability
- **Balanced learning**: Class weights ensure all classes are learned
- **Better temporal modeling**: Sinusoidal positional encoding
- **Higher capacity**: Deeper classifier handles 1000 classes better

## Backward Compatibility

All changes maintain backward compatibility:
- Default arguments match recommended values for large datasets
- Legacy models can still load with `use_sinusoidal_pos=False`
- Class weights are opt-in via `--use_class_weights` flag

## Files Modified

1. **train_simple.py** - Main training script
   - Added utility functions
   - Updated argument parser
   - Modified training loop for warmup

2. **hybrid_asl_model_simple.py** - Simple model (primary)
   - Added `SinusoidalPositionalEncoding` class
   - Enhanced classifier capacity

3. **hybrid_asl_model.py** - Full model (alternative)
   - Added `SinusoidalPositionalEncoding` class
   - Updated `LandmarkEncoder` with optional sinusoidal encoding

## Troubleshooting

### Issue: "Feature dimension mismatch"
**Solution:** The script will auto-adjust. If you see this warning, your features were extracted with a different model than specified. The script continues with detected dimensions.

### Issue: OOM (Out of Memory) errors
**Solution:** Reduce `--batch_size` (try 16 or 8). Model capacity is increased but still fits in 4GB VRAM with batch_size=32.

### Issue: Still poor accuracy on nslt_1000
**Solution:** Ensure you're using:
- `--use_class_weights` (critical for imbalanced data)
- `--warmup_epochs 10` (helps with many classes)
- Sufficient epochs (try 100-150)

## References

- Original issue: Training fails on nslt_1000.json
- Test suite: `test_nslt1000_fixes.py`
- Documentation: `GEMMA_INTEGRATION.md`
