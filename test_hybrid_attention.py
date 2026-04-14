#!/usr/bin/env python3
"""
Test script to verify the hybrid attention implementation in VisionTransformer.
Tests the 7:1 ratio of linear:full attention blocks.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from diffnext.models.vision_transformer_linear_decouple1_kvscale1_hybrid import VisionTransformer

def test_hybrid_attention_pattern():
    """Test the hybrid attention pattern with different depths"""

    print("Testing Hybrid Attention Pattern (7:1 Linear:Full ratio)")
    print("=" * 60)

    # Test different depths
    test_depths = [8, 16, 24, 32]

    for depth in test_depths:
        print(f"\nTesting depth = {depth}")
        print("-" * 30)

        # Create model with hybrid attention
        model = VisionTransformer(
            depth=depth,
            embed_dim=384,
            num_heads=6,
            mlp_ratio=4,
            patch_size=2,
            image_size=32,
            image_dim=4,
            encoder_depth=None,
            image_model=False,
            hybrid_ratio=7
        )

        # Analyze attention pattern
        attention_types = []
        for i, block in enumerate(model.blocks):
            attn_type = type(block.attn).__name__
            attention_types.append(attn_type)
            print(f"Block {i:2d}: {attn_type}")

        # Count different attention types
        linear_count = sum(1 for attn in attention_types if 'Linear' in attn)
        full_count = sum(1 for attn in attention_types if attn == 'FullAttention')

        print("\nSummary:")
        print(f"  Linear Attention blocks: {linear_count}")
        print(f"  Full Attention blocks: {full_count}")
        print(".1f")

        # Verify pattern
        expected_full_blocks = depth // 8  # Every 8th block should be full (7 linear + 1 full)
        if depth % 8 != 0:
            expected_full_blocks += 1  # Last incomplete cycle

        if full_count == expected_full_blocks:
            print("  ✓ Pattern verification PASSED")
        else:
            print(f"  ✗ Pattern verification FAILED (expected {expected_full_blocks} full blocks)")

def test_image_model_hybrid():
    """Test hybrid attention with image_model=True"""

    print("\n\nTesting Image Model with Hybrid Attention")
    print("=" * 60)

    depth = 16
    print(f"\nTesting image model with depth = {depth}")

    model = VisionTransformer(
        depth=depth,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        encoder_depth=None,
        image_model=True,
        hybrid_ratio=7
    )

    print("\nBlock analysis:")
    for i, block in enumerate(model.blocks):
        attn_type = type(block.attn).__name__
        base_mode = "encoder" if i < depth//2 else "decoder"
        print(f"Block {i:2d}: {attn_type} ({base_mode})")

    # Count different attention types
    attention_types = [type(block.attn).__name__ for block in model.blocks]
    linear_encoder = sum(1 for i, attn in enumerate(attention_types) if 'Linear' in attn and i < depth//2)
    full_encoder = sum(1 for i, attn in enumerate(attention_types) if attn == 'FullAttention' and i < depth//2)
    linear_decoder = sum(1 for i, attn in enumerate(attention_types) if 'Linear' in attn and i >= depth//2)
    full_decoder = sum(1 for i, attn in enumerate(attention_types) if attn == 'FullAttention' and i >= depth//2)

    print("\nSummary:")
    print(f"  Encoder blocks: {depth//2} total")
    print(f"    Linear: {linear_encoder}, Full: {full_encoder}")
    print(f"  Decoder blocks: {depth//2} total")
    print(f"    Linear: {linear_decoder}, Full: {full_decoder}")

if __name__ == "__main__":
    test_hybrid_attention_pattern()
    test_image_model_hybrid()
    print("\n" + "=" * 60)
    print("Test completed!")
