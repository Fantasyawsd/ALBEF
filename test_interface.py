#!/usr/bin/env python3
"""
Test to verify that all function names, class names, and parameter names
remain unchanged after the library upgrades.
"""

import sys
import inspect

def test_vit_interface():
    """Test VisionTransformer interface."""
    from models.vit import VisionTransformer, interpolate_pos_embed, Mlp, Attention, Block
    
    print("\n=== Testing VisionTransformer Interface ===")
    
    # Check VisionTransformer init parameters
    vit_params = inspect.signature(VisionTransformer.__init__).parameters
    expected_vit_params = [
        'self', 'img_size', 'patch_size', 'in_chans', 'num_classes', 'embed_dim',
        'depth', 'num_heads', 'mlp_ratio', 'qkv_bias', 'qk_scale', 
        'representation_size', 'drop_rate', 'attn_drop_rate', 'drop_path_rate', 'norm_layer'
    ]
    
    for param in expected_vit_params:
        if param not in vit_params:
            print(f"  ✗ Missing parameter: {param}")
            return False
    
    # Check VisionTransformer methods
    vit_methods = ['forward', '_init_weights', 'no_weight_decay']
    for method in vit_methods:
        if not hasattr(VisionTransformer, method):
            print(f"  ✗ Missing method: {method}")
            return False
    
    # Check Attention class
    attn_methods = ['forward', 'save_attn_gradients', 'get_attn_gradients', 
                    'save_attention_map', 'get_attention_map']
    for method in attn_methods:
        if not hasattr(Attention, method):
            print(f"  ✗ Missing Attention method: {method}")
            return False
    
    print("  ✓ All VisionTransformer components present")
    return True


def test_xbert_interface():
    """Test BERT model interface."""
    from models.xbert import (
        BertConfig, BertModel, BertForMaskedLM, BertLMHeadModel,
        BertEmbeddings, BertSelfAttention, BertAttention, BertLayer, BertEncoder
    )
    
    print("\n=== Testing BERT Model Interface ===")
    
    # Check BertModel methods
    bert_methods = ['forward', 'get_input_embeddings', 'set_input_embeddings', 
                    '_prune_heads', 'get_extended_attention_mask']
    for method in bert_methods:
        if not hasattr(BertModel, method):
            print(f"  ✗ Missing BertModel method: {method}")
            return False
    
    # Check BertForMaskedLM methods
    masked_lm_methods = ['forward', 'get_output_embeddings', 'set_output_embeddings', 
                        'prepare_inputs_for_generation']
    for method in masked_lm_methods:
        if not hasattr(BertForMaskedLM, method):
            print(f"  ✗ Missing BertForMaskedLM method: {method}")
            return False
    
    # Check BertSelfAttention methods
    self_attn_methods = ['forward', 'save_attn_gradients', 'get_attn_gradients',
                        'save_attention_map', 'get_attention_map', 'transpose_for_scores']
    for method in self_attn_methods:
        if not hasattr(BertSelfAttention, method):
            print(f"  ✗ Missing BertSelfAttention method: {method}")
            return False
    
    # Check BertLayer init parameters
    layer_init = inspect.signature(BertLayer.__init__).parameters
    if 'layer_num' not in layer_init:
        print("  ✗ Missing BertLayer layer_num parameter")
        return False
    
    # Check BertEncoder forward mode parameter
    encoder_forward = inspect.signature(BertEncoder.forward).parameters
    if 'mode' not in encoder_forward:
        print("  ✗ Missing BertEncoder mode parameter")
        return False
    
    print("  ✓ All BERT components present")
    return True


def test_albef_models():
    """Test ALBEF model interfaces."""
    from models.model_pretrain import ALBEF
    from models.model_retrieval import ALBEF as ALBEF_Retrieval
    from models.model_vqa import ALBEF as ALBEF_VQA
    
    print("\n=== Testing ALBEF Models Interface ===")
    
    # Check ALBEF pretrain init parameters
    albef_init = inspect.signature(ALBEF.__init__).parameters
    expected_albef_params = ['self', 'text_encoder', 'tokenizer', 'config', 'temp', 'init_deit']
    for param in expected_albef_params:
        if param not in albef_init:
            print(f"  ✗ Missing ALBEF init parameter: {param}")
            return False
    
    # Check ALBEF methods
    albef_methods = ['forward', 'copy_params', '_momentum_update', 
                     '_dequeue_and_enqueue', 'mask']
    for method in albef_methods:
        if not hasattr(ALBEF, method):
            print(f"  ✗ Missing ALBEF method: {method}")
            return False
    
    # Check ALBEF attributes
    albef_attrs = ['tokenizer', 'mlm_probability', 'visual_encoder', 'text_encoder',
                   'vision_proj', 'text_proj', 'temp', 'queue_size', 'momentum',
                   'itm_head', 'visual_encoder_m', 'vision_proj_m', 'text_encoder_m',
                   'text_proj_m', 'model_pairs', 'image_queue', 'text_queue']
    
    # We can't check attributes directly on the class, but we can verify the __init__ creates them
    # by checking the source code
    import ast
    source = inspect.getsource(ALBEF.__init__)
    for attr in albef_attrs:
        if f'self.{attr}' not in source:
            print(f"  ✗ Missing ALBEF attribute initialization: {attr}")
            return False
    
    # queue_ptr uses register_buffer, check separately
    if 'register_buffer("queue_ptr"' not in source:
        print("  ✗ Missing ALBEF queue_ptr buffer registration")
        return False
    
    print("  ✓ All ALBEF model components present")
    return True


def test_tokenizer():
    """Test BertTokenizer interface."""
    from models.tokenization_bert import BertTokenizer
    
    print("\n=== Testing BertTokenizer Interface ===")
    
    # Check essential tokenizer attributes
    tokenizer_attrs = ['vocab_files_names', 'pretrained_vocab_files_map']
    
    for attr in tokenizer_attrs:
        if not hasattr(BertTokenizer, attr):
            print(f"  ✗ Missing BertTokenizer attribute: {attr}")
            return False
    
    # Check tokenizer methods  
    tokenizer_methods = ['_tokenize', 'convert_tokens_to_ids', 'convert_ids_to_tokens']
    for method in tokenizer_methods:
        if not hasattr(BertTokenizer, method):
            print(f"  ✗ Missing BertTokenizer method: {method}")
            return False
    
    print("  ✓ All BertTokenizer components present")
    return True


def test_forward_signatures():
    """Test that forward method signatures are preserved."""
    from models.vit import VisionTransformer
    from models.xbert import BertModel, BertForMaskedLM
    from models.model_pretrain import ALBEF
    
    print("\n=== Testing Forward Method Signatures ===")
    
    # Check VisionTransformer forward
    vit_forward = inspect.signature(VisionTransformer.forward).parameters
    if 'x' not in vit_forward or 'register_blk' not in vit_forward:
        print("  ✗ VisionTransformer.forward signature changed")
        return False
    
    # Check BertModel forward
    bert_forward = inspect.signature(BertModel.forward).parameters
    critical_params = ['input_ids', 'attention_mask', 'encoder_embeds', 
                      'encoder_hidden_states', 'encoder_attention_mask', 'mode']
    for param in critical_params:
        if param not in bert_forward:
            print(f"  ✗ BertModel.forward missing parameter: {param}")
            return False
    
    # Check BertForMaskedLM forward
    masked_lm_forward = inspect.signature(BertForMaskedLM.forward).parameters
    mlm_params = ['input_ids', 'encoder_embeds', 'soft_labels', 'alpha', 
                  'return_logits', 'mode']
    for param in mlm_params:
        if param not in masked_lm_forward:
            print(f"  ✗ BertForMaskedLM.forward missing parameter: {param}")
            return False
    
    # Check ALBEF forward
    albef_forward = inspect.signature(ALBEF.forward).parameters
    if 'image' not in albef_forward or 'text' not in albef_forward or 'alpha' not in albef_forward:
        print("  ✗ ALBEF.forward signature changed")
        return False
    
    print("  ✓ All forward method signatures preserved")
    return True


def main():
    """Run all interface tests."""
    print("=" * 80)
    print("Testing Interface Compatibility After Library Upgrades")
    print("=" * 80)
    
    tests = [
        ("VisionTransformer", test_vit_interface),
        ("BERT Models", test_xbert_interface),
        ("ALBEF Models", test_albef_models),
        ("Tokenizer", test_tokenizer),
        ("Forward Signatures", test_forward_signatures),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ {name} test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary:")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print("=" * 80)
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ SUCCESS: All interface tests passed!")
        print("Function names, class names, and parameter names are preserved.")
        return True
    else:
        print(f"\n✗ FAILED: {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
