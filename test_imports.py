#!/usr/bin/env python3
"""
Test script to verify all imports work correctly with updated dependencies.
Tests that all modules can be imported without errors.
"""

import sys
import traceback

def test_imports():
    """Test all critical imports."""
    print("=" * 80)
    print("Testing ALBEF imports with updated dependencies")
    print("=" * 80)
    
    errors = []
    
    # Test PyTorch
    print("\n1. Testing PyTorch...")
    try:
        import torch
        print(f"   ✓ PyTorch {torch.__version__} imported successfully")
        if torch.cuda.is_available():
            print(f"   ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("   ℹ CUDA not available (CPU mode)")
    except Exception as e:
        errors.append(("PyTorch", e))
        print(f"   ✗ Failed to import PyTorch: {e}")
    
    # Test transformers
    print("\n2. Testing Transformers...")
    try:
        import transformers
        print(f"   ✓ Transformers {transformers.__version__} imported successfully")
    except Exception as e:
        errors.append(("Transformers", e))
        print(f"   ✗ Failed to import Transformers: {e}")
    
    # Test timm
    print("\n3. Testing timm...")
    try:
        import timm
        print(f"   ✓ timm {timm.__version__} imported successfully")
    except Exception as e:
        errors.append(("timm", e))
        print(f"   ✗ Failed to import timm: {e}")
    
    # Test vit.py
    print("\n4. Testing models/vit.py...")
    try:
        from models.vit import VisionTransformer, interpolate_pos_embed
        print("   ✓ VisionTransformer imported successfully")
        print("   ✓ interpolate_pos_embed imported successfully")
    except Exception as e:
        errors.append(("models.vit", e))
        print(f"   ✗ Failed to import from models/vit.py:")
        traceback.print_exc()
    
    # Test xbert.py
    print("\n5. Testing models/xbert.py...")
    try:
        from models.xbert import BertConfig, BertModel, BertForMaskedLM
        print("   ✓ BertConfig imported successfully")
        print("   ✓ BertModel imported successfully")
        print("   ✓ BertForMaskedLM imported successfully")
    except Exception as e:
        errors.append(("models.xbert", e))
        print(f"   ✗ Failed to import from models/xbert.py:")
        traceback.print_exc()
    
    # Test tokenization_bert.py
    print("\n6. Testing models/tokenization_bert.py...")
    try:
        from models.tokenization_bert import BertTokenizer
        print("   ✓ BertTokenizer imported successfully")
    except Exception as e:
        errors.append(("models.tokenization_bert", e))
        print(f"   ✗ Failed to import from models/tokenization_bert.py:")
        traceback.print_exc()
    
    # Test model_pretrain.py
    print("\n7. Testing models/model_pretrain.py...")
    try:
        from models.model_pretrain import ALBEF
        print("   ✓ ALBEF model imported successfully")
    except Exception as e:
        errors.append(("models.model_pretrain", e))
        print(f"   ✗ Failed to import from models/model_pretrain.py:")
        traceback.print_exc()
    
    # Test model_retrieval.py
    print("\n8. Testing models/model_retrieval.py...")
    try:
        from models.model_retrieval import ALBEF as ALBEF_Retrieval
        print("   ✓ ALBEF Retrieval model imported successfully")
    except Exception as e:
        errors.append(("models.model_retrieval", e))
        print(f"   ✗ Failed to import from models/model_retrieval.py:")
        traceback.print_exc()
    
    # Test model_vqa.py
    print("\n9. Testing models/model_vqa.py...")
    try:
        from models.model_vqa import ALBEF as ALBEF_VQA
        print("   ✓ ALBEF VQA model imported successfully")
    except Exception as e:
        errors.append(("models.model_vqa", e))
        print(f"   ✗ Failed to import from models/model_vqa.py:")
        traceback.print_exc()
    
    # Test model_ve.py
    print("\n10. Testing models/model_ve.py...")
    try:
        from models.model_ve import ALBEF as ALBEF_VE
        print("   ✓ ALBEF VE model imported successfully")
    except Exception as e:
        errors.append(("models.model_ve", e))
        print(f"   ✗ Failed to import from models/model_ve.py:")
        traceback.print_exc()
    
    # Test model_nlvr.py
    print("\n11. Testing models/model_nlvr.py...")
    try:
        from models.model_nlvr import ALBEF as ALBEF_NLVR
        print("   ✓ ALBEF NLVR model imported successfully")
    except Exception as e:
        errors.append(("models.model_nlvr", e))
        print(f"   ✗ Failed to import from models/model_nlvr.py:")
        traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 80)
    if errors:
        print(f"FAILED: {len(errors)} import error(s) detected:")
        for module, error in errors:
            print(f"  - {module}: {type(error).__name__}")
        return False
    else:
        print("SUCCESS: All imports working correctly!")
        return True
    print("=" * 80)

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
