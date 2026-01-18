#!/usr/bin/env python3
"""
QUICK TEST SCRIPT FOR GEMMA INTEGRATION

Tests the core functionality of the Gemma integration without requiring
actual data or model downloads.

Run: python test_gemma_integration.py
"""

import sys
import torch
import numpy as np

def test_imports():
    """Test that all modules can be imported."""
    print("="*70)
    print("TEST 1: Module Imports")
    print("="*70)
    
    try:
        from gemma_feature_extractor import GemmaFeatureExtractor
        print("✓ gemma_feature_extractor imported successfully")
    except Exception as e:
        print(f"✗ Failed to import gemma_feature_extractor: {e}")
        return False
    
    try:
        from hybrid_asl_model_simple import (
            HybridASLModelSimple, 
            VisualProjection, 
            PreextractedDataset
        )
        print("✓ hybrid_asl_model_simple imported successfully")
    except Exception as e:
        print(f"✗ Failed to import hybrid_asl_model_simple: {e}")
        return False
    
    print()
    return True


def test_model_architecture():
    """Test that the model can be created and runs."""
    print("="*70)
    print("TEST 2: Model Architecture")
    print("="*70)
    
    try:
        from hybrid_asl_model_simple import HybridASLModelSimple
        
        # Create model
        model = HybridASLModelSimple(
            num_classes=100,
            gemma_feature_dim=2048,
            hidden_dim=256,
            fusion_type='concat'
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"✓ Model created successfully")
        print(f"✓ Total parameters: {total_params:,}")
        print(f"✓ Trainable parameters: {trainable_params:,}")
        
        # Verify parameter count is reasonable (~2-3M)
        if trainable_params < 2_000_000 or trainable_params > 4_000_000:
            print(f"⚠ Warning: Parameter count seems unusual: {trainable_params:,}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_forward_pass():
    """Test that the model forward pass works."""
    print("="*70)
    print("TEST 3: Forward Pass")
    print("="*70)
    
    try:
        from hybrid_asl_model_simple import HybridASLModelSimple
        
        # Create model
        model = HybridASLModelSimple(
            num_classes=100,
            gemma_feature_dim=2048,
            hidden_dim=256,
            fusion_type='concat'
        )
        model.eval()
        
        # Create dummy inputs
        batch_size = 4
        num_frames = 16
        gemma_dim = 2048
        
        gemma_features = torch.randn(batch_size, num_frames, gemma_dim)
        landmarks = torch.randn(batch_size, num_frames, 162)
        
        # Forward pass
        with torch.no_grad():
            logits = model(gemma_features, landmarks)
        
        print(f"✓ Forward pass successful")
        print(f"✓ Input shapes:")
        print(f"    Gemma features: {gemma_features.shape}")
        print(f"    Landmarks: {landmarks.shape}")
        print(f"✓ Output logits shape: {logits.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, 100)
        if logits.shape != expected_shape:
            print(f"✗ Unexpected output shape: {logits.shape} (expected {expected_shape})")
            return False
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_fusion_types():
    """Test all fusion strategies."""
    print("="*70)
    print("TEST 4: Fusion Strategies")
    print("="*70)
    
    try:
        from hybrid_asl_model_simple import HybridASLModelSimple
        
        fusion_types = ['concat', 'attention', 'gated']
        
        for fusion_type in fusion_types:
            model = HybridASLModelSimple(
                num_classes=100,
                gemma_feature_dim=2048,
                hidden_dim=256,
                fusion_type=fusion_type
            )
            
            # Test forward pass
            gemma_features = torch.randn(2, 16, 2048)
            landmarks = torch.randn(2, 16, 162)
            
            with torch.no_grad():
                logits = model(gemma_features, landmarks)
            
            print(f"✓ Fusion type '{fusion_type}' works correctly")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Fusion strategy test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_padding():
    """Test the dataset padding helper function."""
    print("="*70)
    print("TEST 5: Dataset Padding")
    print("="*70)
    
    try:
        from hybrid_asl_model_simple import PreextractedDataset
        
        # Test padding logic manually
        # Create a mock dataset instance just to access the helper method
        class MockDataset(PreextractedDataset):
            def __init__(self):
                self.num_frames = 16
        
        dataset = MockDataset()
        
        # Test padding (shorter than num_frames)
        short_features = np.random.randn(8, 2048).astype(np.float32)
        padded = dataset._pad_or_truncate(short_features)
        assert padded.shape[0] == 16, f"Expected 16 frames, got {padded.shape[0]}"
        print(f"✓ Padding works: {short_features.shape} → {padded.shape}")
        
        # Test truncation (longer than num_frames)
        long_features = np.random.randn(32, 2048).astype(np.float32)
        truncated = dataset._pad_or_truncate(long_features)
        assert truncated.shape[0] == 16, f"Expected 16 frames, got {truncated.shape[0]}"
        print(f"✓ Truncation works: {long_features.shape} → {truncated.shape}")
        
        # Test exact match
        exact_features = np.random.randn(16, 2048).astype(np.float32)
        result = dataset._pad_or_truncate(exact_features)
        assert result.shape[0] == 16, f"Expected 16 frames, got {result.shape[0]}"
        print(f"✓ Exact length works: {exact_features.shape} → {result.shape}")
        
        print()
        return True
        
    except Exception as e:
        print(f"✗ Dataset padding test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "GEMMA INTEGRATION TEST SUITE" + " "*25 + "║")
    print("╚" + "="*68 + "╝")
    print()
    
    results = []
    
    # Run tests
    results.append(("Module Imports", test_imports()))
    results.append(("Model Architecture", test_model_architecture()))
    results.append(("Forward Pass", test_forward_pass()))
    results.append(("Fusion Strategies", test_all_fusion_types()))
    results.append(("Dataset Padding", test_dataset_padding()))
    
    # Summary
    print("="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "ALL TESTS PASSED! 🎉" + " "*32 + "║")
        print("╚" + "="*68 + "╝")
        print()
        return 0
    else:
        print()
        print("╔" + "="*68 + "╗")
        print("║" + " "*15 + "SOME TESTS FAILED" + " "*34 + "║")
        print("╚" + "="*68 + "╝")
        print()
        return 1


if __name__ == '__main__':
    sys.exit(main())
