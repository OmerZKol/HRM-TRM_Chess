#!/usr/bin/env python3
"""
Test script for attention policy mapping integration with HRM model.
"""

import torch
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

def test_attention_policy_integration():
    """Test the attention policy integration with HRM model."""
    
    print("Testing Attention Policy Integration...")
    
    try:
        # Import the model
        from model.ChessNNet import ChessNNet
        
        # Create model with attention policy enabled
        print("Creating HRM model with attention policy...")
        model = ChessNNet(board_size=(8, 8), action_size=1858, batch_size=2)
        
        # Create sample input
        batch_size = 2
        input_tensor = torch.randn(batch_size, 112, 8, 8)  # Chess board format
        
        print(f"Input tensor shape: {input_tensor.shape}")
        
        # Forward pass
        print("Running forward pass...")
        model.eval()
        with torch.no_grad():
            policy_log_probs, value_tanh, moves_left, q_info = model(input_tensor)
        
        print(f"Policy logits shape: {policy_log_probs.shape}")
        print(f"Value shape: {value_tanh.shape}")  
        print(f"Moves left shape: {moves_left.shape}")
        print(f"Q info keys: {list(q_info.keys())}")
        
        # Check if attention weights are available
        if 'attention_weights' in q_info:
            attention_weights = q_info['attention_weights']
            print(f"Attention weights shape: {attention_weights.shape}")
            print("✅ Attention policy integration successful!")
        else:
            print("⚠️  Attention weights not found in outputs")
            
        # Verify output shapes
        assert policy_log_probs.shape == (batch_size, 1858), f"Expected policy shape {(batch_size, 1858)}, got {policy_log_probs.shape}"
        assert value_tanh.shape == (batch_size, 3), f"Expected value shape {(batch_size, 3)}, got {value_tanh.shape}"
        
        print("✅ All output shapes correct!")
        print("✅ Attention Policy Integration Test PASSED!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_mapping_matrix():
    """Test just the attention policy mapping matrix."""
    
    print("\nTesting Attention Policy Mapping Matrix...")
    
    try:
        import numpy as np
        from model.attention_policy_map import make_attention_policy_map
        
        mapping = make_attention_policy_map()
        print(f"Mapping matrix shape: {mapping.shape}")
        print(f"Expected shape: (4288, 1858)")
        print(f"Non-zero elements: {np.count_nonzero(mapping)}")
        print(f"Matrix dtype: {mapping.dtype}")
        
        # Check basic properties
        assert mapping.shape == (4288, 1858), f"Wrong matrix shape: {mapping.shape}"
        assert mapping.dtype == np.float32, f"Wrong dtype: {mapping.dtype}"
        assert np.count_nonzero(mapping) > 0, "Matrix is all zeros!"
        
        print("✅ Attention Policy Mapping Matrix Test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Mapping matrix test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("ATTENTION POLICY INTEGRATION TESTS")
    print("=" * 60)
    
    # Test 1: Basic mapping matrix
    matrix_test_passed = test_attention_mapping_matrix()
    
    # Test 2: Full model integration 
    integration_test_passed = test_attention_policy_integration()
    
    print("\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Mapping Matrix Test: {'✅ PASSED' if matrix_test_passed else '❌ FAILED'}")
    print(f"Model Integration Test: {'✅ PASSED' if integration_test_passed else '❌ FAILED'}")
    
    if matrix_test_passed and integration_test_passed:
        print("\n🎉 ALL TESTS PASSED! Attention policy is ready to use.")
        sys.exit(0)
    else:
        print("\n💥 SOME TESTS FAILED. Please check the implementation.")
        sys.exit(1)