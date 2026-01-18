#!/usr/bin/env python3
"""
Test script for nslt_1000.json training fixes.

Tests the new functionality added to address training failures:
1. Class weight computation
2. Feature dimension verification
3. Warmup scheduler
4. Sinusoidal positional encoding
5. Model capacity improvements

Run: python test_nslt1000_fixes.py
"""

import sys
import torch
import numpy as np
import tempfile
import os
from pathlib import Path


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 70)
    print("TEST 1: Module Imports")
    print("=" * 70)
    
    try:
        from train_simple import (
            compute_class_weights,
            verify_feature_dimensions,
            get_cosine_schedule_with_warmup
        )
        print("✓ train_simple utility functions imported successfully")
    except Exception as e:
        print(f"✗ Failed to import train_simple utilities: {e}")
        return False
    
    try:
        from hybrid_asl_model_simple import (
            SinusoidalPositionalEncoding,
            SimpleHybridASLModel
        )
        print("✓ hybrid_asl_model_simple imported successfully")
    except Exception as e:
        print(f"✗ Failed to import hybrid_asl_model_simple: {e}")
        return False
    
    try:
        from hybrid_asl_model import (
            SinusoidalPositionalEncoding as SinusoidalPosEnc2,
            LandmarkEncoder
        )
        print("✓ hybrid_asl_model imported successfully")
    except ImportError as e:
        if "mediapipe" in str(e):
            print("⚠ hybrid_asl_model import skipped (mediapipe not installed)")
        else:
            print(f"✗ Failed to import hybrid_asl_model: {e}")
            return False
    except Exception as e:
        print(f"✗ Failed to import hybrid_asl_model: {e}")
        return False
    
    print()
    return True


def test_class_weights():
    """Test class weight computation."""
    print("=" * 70)
    print("TEST 2: Class Weight Computation")
    print("=" * 70)
    
    try:
        from train_simple import compute_class_weights
        
        # Test balanced dataset
        labels = [0, 0, 1, 1, 2, 2]
        num_classes = 3
        weights = compute_class_weights(labels, num_classes)
        
        print(f"✓ Balanced dataset weights: {weights}")
        assert weights.shape[0] == num_classes, "Weight shape mismatch"
        assert torch.all(weights > 0), "All weights should be positive"
        
        # Test imbalanced dataset
        labels = [0] * 50 + [1] * 10 + [2] * 5
        weights = compute_class_weights(labels, num_classes)
        
        print(f"✓ Imbalanced dataset weights: {weights}")
        # Class 2 (5 samples) should have higher weight than class 0 (50 samples)
        assert weights[2] > weights[0], "Rare class should have higher weight"
        
        # Test with 1000 classes (nslt_1000 scenario)
        num_classes = 1000
        labels = []
        for i in range(num_classes):
            # Simulate class imbalance: some classes have 5 samples, others 50
            count = 5 if i % 2 == 0 else 50
            labels.extend([i] * count)
        
        weights = compute_class_weights(labels, num_classes)
        print(f"✓ 1000-class dataset weights computed (min: {weights.min():.2f}, max: {weights.max():.2f})")
        assert weights.shape[0] == num_classes, "Weight shape mismatch for 1000 classes"
        assert torch.all(weights >= 0.1) and torch.all(weights <= 10.0), "Weights should be clipped"
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Class weight computation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_feature_dimension_verification():
    """Test feature dimension verification."""
    print("=" * 70)
    print("TEST 3: Feature Dimension Verification")
    print("=" * 70)
    
    try:
        from train_simple import verify_feature_dimensions
        
        # Create temporary directory with dummy .npy files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy feature files
            for i in range(5):
                features = np.random.randn(16, 2048).astype(np.float32)
                np.save(os.path.join(tmpdir, f"video{i}_gemma.npy"), features)
            
            # Test correct dimension
            verified_dim = verify_feature_dimensions(tmpdir, expected_dim=2048)
            assert verified_dim == 2048, f"Expected 2048, got {verified_dim}"
            print(f"✓ Correct dimension verified: {verified_dim}")
            
            # Test dimension mismatch detection
            # Create a file with different dimension
            features_wrong = np.random.randn(16, 1152).astype(np.float32)
            np.save(os.path.join(tmpdir, f"video5_gemma.npy"), features_wrong)
            
            verified_dim = verify_feature_dimensions(tmpdir, expected_dim=2048)
            assert verified_dim == 1152, f"Should detect mismatch and return 1152, got {verified_dim}"
            print(f"✓ Dimension mismatch detected and corrected: 2048 -> {verified_dim}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Feature dimension verification failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_warmup_scheduler():
    """Test warmup scheduler."""
    print("=" * 70)
    print("TEST 4: Warmup Scheduler")
    print("=" * 70)
    
    try:
        from train_simple import get_cosine_schedule_with_warmup
        
        # Create dummy model and optimizer
        model = torch.nn.Linear(10, 10)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Create scheduler with warmup
        warmup_epochs = 5
        total_epochs = 50
        steps_per_epoch = 100
        
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            steps_per_epoch=steps_per_epoch
        )
        
        # Check warmup phase
        initial_lr = optimizer.param_groups[0]['lr']
        print(f"✓ Initial LR: {initial_lr:.6f}")
        
        # Step through warmup
        warmup_steps = warmup_epochs * steps_per_epoch
        for step in range(warmup_steps):
            scheduler.step()
        
        warmup_final_lr = optimizer.param_groups[0]['lr']
        print(f"✓ LR after warmup ({warmup_steps} steps): {warmup_final_lr:.6f}")
        assert warmup_final_lr > initial_lr * 0.9, "LR should increase during warmup"
        
        # Step through cosine annealing
        for step in range(warmup_steps, total_epochs * steps_per_epoch):
            scheduler.step()
        
        final_lr = optimizer.param_groups[0]['lr']
        print(f"✓ Final LR after cosine annealing: {final_lr:.6f}")
        assert final_lr < warmup_final_lr, "LR should decrease during cosine annealing"
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Warmup scheduler test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_sinusoidal_positional_encoding():
    """Test sinusoidal positional encoding."""
    print("=" * 70)
    print("TEST 5: Sinusoidal Positional Encoding")
    print("=" * 70)
    
    try:
        from hybrid_asl_model_simple import SinusoidalPositionalEncoding
        
        # Create positional encoding module
        d_model = 256
        max_len = 64
        pos_enc = SinusoidalPositionalEncoding(d_model=d_model, max_len=max_len, dropout=0.0)
        
        # Test forward pass
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, d_model)
        
        output = pos_enc(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == x.shape, f"Shape mismatch: {output.shape} vs {x.shape}"
        
        # Test that positional encoding is deterministic
        output2 = pos_enc(x)
        # Note: dropout makes it non-deterministic, but encoding buffer should be same
        # Check that the encoding buffer exists and has correct shape
        assert hasattr(pos_enc, 'pe'), "Positional encoding buffer should exist"
        assert pos_enc.pe.shape == (1, max_len, d_model), f"PE buffer shape mismatch: {pos_enc.pe.shape}"
        print(f"✓ Positional encoding buffer shape: {pos_enc.pe.shape}")
        
        # Test with different sequence lengths
        for test_len in [8, 16, 32]:
            x_test = torch.randn(batch_size, test_len, d_model)
            output_test = pos_enc(x_test)
            assert output_test.shape == x_test.shape, f"Shape mismatch for seq_len={test_len}"
        print(f"✓ Positional encoding works for different sequence lengths")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Sinusoidal positional encoding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_landmark_encoder_with_sinusoidal():
    """Test LandmarkEncoder with sinusoidal positional encoding."""
    print("=" * 70)
    print("TEST 6: LandmarkEncoder with Sinusoidal Encoding")
    print("=" * 70)
    
    try:
        from hybrid_asl_model import LandmarkEncoder
    except ImportError as e:
        if "mediapipe" in str(e):
            print("⚠ Test skipped (mediapipe not installed)")
            print()
            return True  # Skip test but don't fail
        else:
            raise
    
    try:
        # Create encoder with sinusoidal positional encoding
        encoder = LandmarkEncoder(
            input_dim=162,
            hidden_dim=256,
            use_sinusoidal_pos=True
        )
        
        # Test forward pass
        batch_size = 4
        seq_len = 16
        x = torch.randn(batch_size, seq_len, 162)
        
        output = encoder(x)
        print(f"✓ Input shape: {x.shape}")
        print(f"✓ Output shape: {output.shape}")
        assert output.shape == (batch_size, 256), f"Output shape mismatch: {output.shape}"
        
        # Test with legacy random positional encoding
        encoder_legacy = LandmarkEncoder(
            input_dim=162,
            hidden_dim=256,
            use_sinusoidal_pos=False
        )
        
        output_legacy = encoder_legacy(x)
        assert output_legacy.shape == (batch_size, 256), f"Legacy output shape mismatch: {output_legacy.shape}"
        print(f"✓ Legacy random positional encoding also works")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ LandmarkEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_model_capacity_improvements():
    """Test model capacity improvements for large num_classes."""
    print("=" * 70)
    print("TEST 7: Model Capacity Improvements")
    print("=" * 70)
    
    try:
        from hybrid_asl_model_simple import SimpleHybridASLModel
        
        # Test with small num_classes (100)
        model_small = SimpleHybridASLModel(
            num_classes=100,
            gemma_feature_dim=2048,
            hidden_dim=512,
            dropout=0.5
        )
        
        params_small = sum(p.numel() for p in model_small.parameters())
        print(f"✓ Model with 100 classes: {params_small:,} parameters")
        
        # Test with large num_classes (1000)
        model_large = SimpleHybridASLModel(
            num_classes=1000,
            gemma_feature_dim=2048,
            hidden_dim=512,
            dropout=0.5
        )
        
        params_large = sum(p.numel() for p in model_large.parameters())
        print(f"✓ Model with 1000 classes: {params_large:,} parameters")
        
        # Large model should have more parameters due to deeper classifier
        assert params_large > params_small, "1000-class model should have more parameters"
        
        # Test forward pass
        batch_size = 4
        num_frames = 16
        gemma_features = torch.randn(batch_size, num_frames, 2048)
        landmarks = torch.randn(batch_size, num_frames, 162)
        
        with torch.no_grad():
            logits_small = model_small(gemma_features, landmarks)
            logits_large = model_large(gemma_features, landmarks)
        
        print(f"✓ Small model output: {logits_small.shape}")
        print(f"✓ Large model output: {logits_large.shape}")
        
        assert logits_small.shape == (batch_size, 100), "Small model output shape mismatch"
        assert logits_large.shape == (batch_size, 1000), "Large model output shape mismatch"
        
        # Check that classifier has proper depth for large model
        classifier_layers = len([m for m in model_large.classifier if isinstance(m, torch.nn.Linear)])
        print(f"✓ Large model classifier has {classifier_layers} linear layers")
        assert classifier_layers == 3, "Large model should have 3 linear layers in classifier"
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Model capacity test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test integration of all components."""
    print("=" * 70)
    print("TEST 8: Integration Test")
    print("=" * 70)
    
    try:
        from hybrid_asl_model_simple import SimpleHybridASLModel
        from train_simple import compute_class_weights
        import torch.nn as nn
        
        # Simulate training setup
        num_classes = 100
        labels = []
        for i in range(num_classes):
            count = 5 if i % 2 == 0 else 15
            labels.extend([i] * count)
        
        # Compute class weights
        class_weights = compute_class_weights(labels, num_classes)
        print(f"✓ Class weights computed for {num_classes} classes")
        
        # Create loss with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print(f"✓ Loss function created with class weights and label smoothing")
        
        # Create model
        model = SimpleHybridASLModel(
            num_classes=num_classes,
            gemma_feature_dim=2048,
            hidden_dim=512,
            dropout=0.5
        )
        print(f"✓ Model created with improved capacity")
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create warmup scheduler
        from train_simple import get_cosine_schedule_with_warmup
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_epochs=5,
            total_epochs=50,
            steps_per_epoch=100
        )
        print(f"✓ Warmup scheduler created")
        
        # Test training step
        batch_size = 4
        num_frames = 16
        gemma_features = torch.randn(batch_size, num_frames, 2048)
        landmarks = torch.randn(batch_size, num_frames, 162)
        batch_labels = torch.randint(0, num_classes, (batch_size,))
        
        model.train()
        optimizer.zero_grad()
        logits = model(gemma_features, landmarks)
        loss = criterion(logits, batch_labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        print(f"✓ Training step completed successfully")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 15 + "NSLT_1000 FIXES TEST SUITE" + " " * 27 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Class Weight Computation", test_class_weights()))
    results.append(("Feature Dimension Verification", test_feature_dimension_verification()))
    results.append(("Warmup Scheduler", test_warmup_scheduler()))
    results.append(("Sinusoidal Positional Encoding", test_sinusoidal_positional_encoding()))
    results.append(("LandmarkEncoder with Sinusoidal", test_landmark_encoder_with_sinusoidal()))
    results.append(("Model Capacity Improvements", test_model_capacity_improvements()))
    results.append(("Integration Test", test_integration()))
    
    # Summary
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 15 + "ALL TESTS PASSED! 🎉" + " " * 32 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        return 0
    else:
        print()
        print("╔" + "=" * 68 + "╗")
        print("║" + " " * 15 + "SOME TESTS FAILED" + " " * 34 + "║")
        print("╚" + "=" * 68 + "╝")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
