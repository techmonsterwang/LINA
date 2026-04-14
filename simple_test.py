#!/usr/bin/env python3

import torch
from diffnext.models.vision_transformer_linear_decouple1_kvscale1_hybrid import LinearAttentionDecoder

def main():
    print("Testing Multi-Resolution LinearAttentionDecoder...")
    print("=" * 60)

    # Test different sequence lengths
    test_lengths = [
        64,    # tail=48 (not square)
        256,   # tail=192 (not square)
        320,   # tail=240 (not square) - original
        324,   # tail=243 (not square)
        484,   # tail=363 (not square)
        1024,  # tail=768 (not square)
        1296,  # tail=972 = 31.176^2, not perfect square
    ]

    # Let's find some lengths where tail is actually a perfect square
    # We need 3N/4 = k^2 for some integer k
    # So N = (4/3) * k^2, and N must be integer
    # This means k^2 must be divisible by 3
    perfect_k = []
    for k in range(1, 40):  # k from 1 to 39
        tail = k * k
        if tail % 3 == 0:  # k^2 divisible by 3
            N = (4 * tail) // 3
            if N <= 2048:
                perfect_k.append((k, tail, N))

    print(f"Found {len(perfect_k)} perfect square tail configurations:")
    for k, tail, N in perfect_k[:5]:  # Show first 5
        print(f"  k={k}, tail={tail}, N={N}")

    # Add perfect square cases to test_lengths
    for _, _, N in perfect_k:
        if N not in test_lengths:
            test_lengths.append(N)

    decoder = LinearAttentionDecoder(dim=384, num_heads=6, max_seq_len=2048)

    for seq_len in test_lengths:
        print(f"\nTesting sequence length: {seq_len}")

        # Calculate expected split
        head_length = seq_len // 4
        tail_length = seq_len - head_length

        print(f"  Head (no conv): {head_length} tokens")
        print(f"  Tail (with conv): {tail_length} tokens")

        # Check if tail_length is perfect square
        spatial_size = int(tail_length ** 0.5)
        is_perfect_square = spatial_size * spatial_size == tail_length
        print(f"  Tail is perfect square: {is_perfect_square}")

        if is_perfect_square:
            print(f"  Spatial size: {spatial_size}x{spatial_size}")

        try:
            x = torch.randn(2, seq_len, 384)
            output = decoder(x)
            print(f"  ✓ Success! Output shape: {output.shape}")
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\n" + "=" * 60)
    print("All tests completed!")

if __name__ == "__main__":
    main()
