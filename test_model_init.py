#!/usr/bin/env python3
"""
Test script to verify the FullAttention layer counting feature.
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from diffnext.models.vision_transformer_linear_decouple1_kvscale1_hybrid import VisionTransformer

def test_model_initialization():
    """Test model initialization with different configurations"""

    print("Testing VisionTransformer Initialization with Attention Counting")
    print("=" * 70)

    # Test case 1: Regular model
    print("\n1. Regular VisionTransformer (depth=16, hybrid_ratio=7):")
    print("-" * 50)
    model1 = VisionTransformer(
        depth=16,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        image_model=False,
        hybrid_ratio=7
    )

    # Test case 2: Image model
    print("\n2. Image VisionTransformer (depth=16, hybrid_ratio=7):")
    print("-" * 50)
    model2 = VisionTransformer(
        depth=16,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        image_model=True,
        hybrid_ratio=7
    )

    # Test case 3: Different hybrid ratio
    print("\n3. VisionTransformer with hybrid_ratio=3 (depth=12):")
    print("-" * 50)
    model3 = VisionTransformer(
        depth=12,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        image_model=False,
        hybrid_ratio=3
    )

    # Test case 4: Small model
    print("\n4. Small VisionTransformer (depth=8, hybrid_ratio=7):")
    print("-" * 50)
    model4 = VisionTransformer(
        depth=8,
        embed_dim=384,
        num_heads=6,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        image_model=False,
        hybrid_ratio=7
    )

if __name__ == "__main__":
    test_model_initialization()
    print("\n" + "=" * 70)
    print("All tests completed!")
