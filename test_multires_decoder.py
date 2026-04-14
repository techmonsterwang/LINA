#!/usr/bin/env python3
"""
Test script to verify the multi-resolution LinearAttentionDecoder.
"""

import sys
import os
import torch
sys.path.append(os.path.dirname(__file__))

from diffnext.models.vision_transformer_linear_decouple1_kvscale1_hybrid import LinearAttentionDecoder

def test_multires_decoder():
    """Test LinearAttentionDecoder with different sequence lengths"""

    print("Testing Multi-Resolution LinearAttentionDecoder")
    print("=" * 60)

    # Test different sequence lengths
    test_lengths = [64, 128, 256, 320, 512, 1024]

    decoder = LinearAttentionDecoder(dim=384, num_heads=6)

    for seq_len in test_lengths:
        print(f"\nTesting sequence length: {seq_len}")
        print("-" * 40)

        # Calculate expected split
        head_length = seq_len // 4
        tail_length = seq_len - head_length

        print(f"Head (no conv): {head_length} tokens")
        print(f"Tail (with conv): {tail_length} tokens")

        # Test if tail_length is perfect square for 2D conv
        spatial_size = int(tail_length ** 0.5)
        is_perfect_square = spatial_size * spatial_size == tail_length

        print(f"Tail length is perfect square: {is_perfect_square}")
        if is_perfect_square:
            print(f"Spatial size for convolution: {spatial_size}x{spatial_size}")

        # Create test input
        x = torch.randn(2, seq_len, 384)  # (batch_size, seq_len, dim)

        try:
            with torch.no_grad():
                output = decoder(x)
                print(f"✓ Forward pass successful! Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Forward pass failed: {e}")

def test_specific_cases():
    """Test specific resolution cases"""

    print("\n\nTesting Specific Resolution Cases")
    print("=" * 60)

    decoder = LinearAttentionDecoder(dim=384, num_heads=6)

    # Test cases with known perfect square tail lengths
    test_cases = [
        (320, "Original: 320 tokens (16x16 + 80) -> tail 240 tokens (not square)"),
        (256, "256 tokens -> tail 192 tokens (not square)"),
        (128, "128 tokens -> tail 96 tokens (not square)"),
        (64, "64 tokens -> tail 48 tokens (not square)"),
        (324, "324 tokens -> tail 243 tokens = 9x9 (perfect square!)"),
        (400, "400 tokens -> tail 300 tokens (not square)"),
        (576, "576 tokens -> tail 432 tokens = 12x12 (perfect square!)"),
    ]

    for seq_len, description in test_cases:
        print(f"\n{description}")
        print("-" * 50)

        x = torch.randn(1, seq_len, 384)
        try:
            with torch.no_grad():
                output = decoder(x)
                print(f"✓ Success! Output shape: {output.shape}")
        except Exception as e:
            print(f"✗ Failed: {e}")

if __name__ == "__main__":
    test_multires_decoder()
    test_specific_cases()
    print("\n" + "=" * 60)
    print("Testing completed!")
