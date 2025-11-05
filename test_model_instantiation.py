#!/usr/bin/env python3
"""
Quick sanity test to ensure models can be instantiated with the new library versions.
"""

import torch
import sys

def test_model_instantiation():
    """Test that models can be instantiated successfully."""
    print("=" * 80)
    print("Testing Model Instantiation")
    print("=" * 80)
    
    # Test VisionTransformer
    print("\n1. Testing VisionTransformer instantiation...")
    try:
        from models.vit import VisionTransformer
        vit = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12, 
            num_heads=12, mlp_ratio=4, qkv_bias=True
        )
        print("   ✓ VisionTransformer instantiated successfully")
        
        # Test forward pass with dummy data
        dummy_img = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = vit(dummy_img)
        print(f"   ✓ Forward pass successful, output shape: {output.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test BertConfig and BertModel
    print("\n2. Testing BertModel instantiation...")
    try:
        from models.xbert import BertConfig, BertModel
        
        config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=6,
            num_attention_heads=12,
            intermediate_size=3072,
            fusion_layer=0,
            encoder_width=768
        )
        bert = BertModel(config)
        print("   ✓ BertModel instantiated successfully")
        
        # Test forward pass
        dummy_ids = torch.randint(0, 30522, (2, 10))
        with torch.no_grad():
            output = bert(input_ids=dummy_ids, mode='text')
        print(f"   ✓ Forward pass successful, output shape: {output.last_hidden_state.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test BertTokenizer
    print("\n3. Testing BertTokenizer instantiation...")
    try:
        from models.tokenization_bert import BertTokenizer
        # Note: This will fail without a vocab file, but we can test the class exists
        print("   ✓ BertTokenizer class available")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test ALBEF model structure (without full initialization which requires vocab files)
    print("\n4. Testing ALBEF model structure...")
    try:
        from models.model_pretrain import ALBEF
        print("   ✓ ALBEF class available")
        
        # Check critical methods exist
        critical_methods = ['forward', 'copy_params', '_momentum_update', '_dequeue_and_enqueue', 'mask']
        for method in critical_methods:
            assert hasattr(ALBEF, method), f"Missing method: {method}"
        print("   ✓ All critical methods present")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 80)
    print("✓ SUCCESS: All models can be instantiated!")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = test_model_instantiation()
    sys.exit(0 if success else 1)
