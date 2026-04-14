import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis
import matplotlib.pyplot as plt
import numpy as np


class LinearAttentionDecoder(nn.Module):
    """Multihead linear attention for decoder with extra convolution branch."""

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0., kernel_function=nn.ReLU, fp32_attention=True, image_size=(32, 32), patch_size=2):
        super(LinearAttentionDecoder, self).__init__()
        print(f"LinearAttentionDecoder initialized with image_size={image_size}, patch_size={patch_size}")
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernel_function = kernel_function()
        self.fp32_attention = fp32_attention
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

        # Calculate sequence length dynamically: (image_size/patch_size)^2 * 5/4
        # Handle image_size as tuple (height, width) or int
        if isinstance(image_size, tuple):
            # For tuple (height, width), use the smaller dimension or average
            # Assuming square patches, we'll use the smaller dimension for consistency
            effective_image_size = min(image_size)
        else:
            effective_image_size = image_size
        
        self.sequence_length = int((effective_image_size // patch_size) ** 2 * 5 / 4)
        print(f"Effective image_size={effective_image_size}, calculated sequence_length={self.sequence_length}")
        
        # Adding key_scale and value_scale
        self.key_scale = nn.Parameter(torch.ones(1, num_heads, self.sequence_length, 1))  # shape: (1, 1, N, 1)
        self.value_scale = nn.Parameter(torch.ones(1, num_heads, self.sequence_length, 1))  # shape: (1, 1, N, 1)

    def forward(self, x) -> torch.Tensor:
        B, N, C = x.shape
        assert N == self.sequence_length, f"Expected sequence length N={self.sequence_length}, but got {N}"

        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        dtype = q.dtype

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()
        
        # Apply key_scale and value_scale (element-wise multiplication)
        k = k * self.key_scale  # Apply key scaling
        v = v * self.value_scale  # Apply value scaling

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
            attn_out = q @ kv * z
            attn_out = attn_out.transpose(1, 2).flatten(2)  # (B, N, C)

        attn_out = attn_out.to(dtype)
        # ==============================
        # dwc branch
        # ==============================
        # Calculate the number of tokens for the first part (non-convolutional)
        # This should be 64 for the original case where sequence_length = 320
        # For other cases, we need to calculate proportionally
        first_part_tokens = int(self.sequence_length * 64 / 320)  # 64/320 = 0.2
        conv_tokens = self.sequence_length - first_part_tokens
        
        # Calculate the spatial dimensions for convolution
        # Assuming square patches: sqrt(conv_tokens) gives us the side length
        conv_side_length = int(conv_tokens ** 0.5)
        
        x_tail = x[:, first_part_tokens:, :].reshape(B, conv_side_length, conv_side_length, -1).permute(0, 3, 1, 2)    # shape: (B, C, conv_side_length, conv_side_length)
        conv_out = self.dw_conv(x_tail)    # shape: (B, C, conv_side_length, conv_side_length)
        conv_out = conv_out.reshape(B, C, -1).permute(0, 2, 1)  # (B, conv_tokens, C)
        conv_out_padded = torch.cat([torch.zeros(B, first_part_tokens, C, device=x.device), conv_out], dim=1)
        conv_out_padded = conv_out_padded.to(dtype)

        x = attn_out + conv_out_padded

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(Attention, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

    def forward(self, x) -> torch.Tensor:
        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)
        if self.cache_kv is not None and self.cache_kv:
            if isinstance(self.cache_kv, list):
                k = self.cache_kv[0] = torch.cat([self.cache_kv[0], k], dim=2)
                v = self.cache_kv[1] = torch.cat([self.cache_kv[1], v], dim=2)
            else:
                self.cache_kv = [k, v]
        if self.flex_attn and self.flex_attn.offsets:
            return self.proj(self.flex_attn(q, k, v).transpose(1, 2).flatten(2))
        
        # Manual implementation of scaled dot-product attention for accurate FLOPs counting
        # Standard attention computation: softmax(Q @ K^T / sqrt(d_k)) @ V
        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale  # (B, H, N, N)
        if self.attn_mask is not None:
            attn = attn + self.attn_mask
        attn = attn.softmax(dim=-1)  # (B, H, N, N)
        o = attn @ v  # (B, H, N, head_dim)
        
        return self.proj(o.transpose(1, 2).flatten(2))


def calculate_flops(model, input_shape, device='cpu', verbose=False, model_name=""):
    """
    Calculate FLOPs for a model using fvcore.
    
    Args:
        model: PyTorch model
        input_shape: Tuple of (B, N, D)
        device: Device to run on
        verbose: If True, print detailed FLOPs breakdown
        model_name: Name of the model for printing
    
    Returns:
        Total FLOPs count and FlopCountAnalysis object
    """
    model.eval()
    model = model.to(device)
    
    # Create dummy input
    dummy_input = torch.randn(input_shape).to(device)
    
    # Calculate FLOPs using FlopCountAnalysis
    flop_counter = FlopCountAnalysis(model, (dummy_input,))
    total_flops = flop_counter.total()
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Detailed FLOPs Breakdown for {model_name}")
        print(f"{'='*60}")
        
        # Get breakdown by operator type
        flops_by_op = flop_counter.by_operator()
        print("\nFLOPs by Operator Type:")
        print("-" * 60)
        for op, flops in sorted(flops_by_op.items(), key=lambda x: x[1], reverse=True):
            gflops = flops / 1e9
            percentage = (flops / total_flops) * 100
            print(f"  {op:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)")
        
        # Get breakdown by module
        flops_by_module = flop_counter.by_module()
        print("\nFLOPs by Module:")
        print("-" * 60)
        for module, flops in sorted(flops_by_module.items(), key=lambda x: x[1], reverse=True):
            if flops > 0:  # Only print non-zero FLOPs
                gflops = flops / 1e9
                percentage = (flops / total_flops) * 100
                # Simplify module name for readability
                module_name = str(module).split('(')[0] if '(' in str(module) else str(module)
                if len(module_name) > 40:
                    module_name = module_name[:37] + "..."
                print(f"  {module_name:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)")
        
        print(f"\nTotal: {total_flops:.0f} FLOPs ({total_flops/1e9:.2f} GFLOPs)")
        print(f"{'='*60}\n")
    
    return total_flops, flop_counter


def test_flops():
    """Test FLOPs for both models."""
    # Configuration
    B, N, D = 1, 5120, 1536
    num_heads = 16
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"Testing with B={B}, N={N}, D={D}, num_heads={num_heads}")
    print(f"Using device: {device}")
    
    # Calculate image_size and patch_size for LinearAttentionDecoder
    # sequence_length = (image_size // patch_size) ^ 2 * 5 / 4 = 5120
    # (image_size // patch_size) ^ 2 = 4096
    # image_size // patch_size = 64
    # We choose patch_size = 2, so image_size = 128
    patch_size = 2
    image_size = 128
    
    # Initialize models
    print("\nInitializing LinearAttentionDecoder...")
    linear_decoder = LinearAttentionDecoder(
        dim=D,
        num_heads=num_heads,
        image_size=image_size,
        patch_size=patch_size,
        fp32_attention=False  # Set to False for FLOPs calculation
    )
    
    print("\nInitializing Attention...")
    attention = Attention(
        dim=D,
        num_heads=num_heads
    )
    
    # Calculate FLOPs
    print("\nCalculating FLOPs for LinearAttentionDecoder...")
    flops_linear, flop_counter_linear = calculate_flops(
        linear_decoder, (B, N, D), device, verbose=True, model_name="LinearAttentionDecoder"
    )
    
    print("Calculating FLOPs for Attention...")
    flops_attention, flop_counter_attention = calculate_flops(
        attention, (B, N, D), device, verbose=True, model_name="Attention"
    )
    
    # Convert to GFLOPs for readability
    gflops_linear = flops_linear / 1e9
    gflops_attention = flops_attention / 1e9
    
    # Print results
    print("\n" + "="*60)
    print("FLOPs Results")
    print("="*60)
    print(f"LinearAttentionDecoder: {flops_linear:.2e} FLOPs ({gflops_linear:.2f} GFLOPs)")
    print(f"Attention: {flops_attention:.2e} FLOPs ({gflops_attention:.2f} GFLOPs)")
    print("="*60)
    
    # Save to txt file
    output_dir = "/share/project/wangjiahao/LAR/evaluations/flops"
    txt_path = f"{output_dir}/flops_results.txt"
    
    with open(txt_path, 'w') as f:
        f.write("FLOPs Test Results\n")
        f.write("="*60 + "\n")
        f.write(f"Configuration:\n")
        f.write(f"  Batch size (B): {B}\n")
        f.write(f"  Sequence length (N): {N}\n")
        f.write(f"  Dimension (D): {D}\n")
        f.write(f"  Number of heads: {num_heads}\n")
        f.write(f"  Device: {device}\n")
        f.write("\n")
        f.write("Results Summary:\n")
        f.write(f"  LinearAttentionDecoder:\n")
        f.write(f"    Total FLOPs: {flops_linear:.2e}\n")
        f.write(f"    GFLOPs: {gflops_linear:.2f}\n")
        f.write(f"  Attention:\n")
        f.write(f"    Total FLOPs: {flops_attention:.2e}\n")
        f.write(f"    GFLOPs: {gflops_attention:.2f}\n")
        f.write("\n")
        f.write(f"Ratio (LinearAttentionDecoder / Attention): {gflops_linear / gflops_attention:.2f}x\n")
        
        # Write detailed breakdown for LinearAttentionDecoder
        f.write("\n" + "="*60 + "\n")
        f.write("Detailed FLOPs Breakdown for LinearAttentionDecoder\n")
        f.write("="*60 + "\n")
        
        flops_by_op_linear = flop_counter_linear.by_operator()
        f.write("\nFLOPs by Operator Type:\n")
        f.write("-" * 60 + "\n")
        for op, flops in sorted(flops_by_op_linear.items(), key=lambda x: x[1], reverse=True):
            gflops = flops / 1e9
            percentage = (flops / flops_linear) * 100
            f.write(f"  {op:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)\n")
        
        flops_by_module_linear = flop_counter_linear.by_module()
        f.write("\nFLOPs by Module:\n")
        f.write("-" * 60 + "\n")
        for module, flops in sorted(flops_by_module_linear.items(), key=lambda x: x[1], reverse=True):
            if flops > 0:
                gflops = flops / 1e9
                percentage = (flops / flops_linear) * 100
                module_name = str(module).split('(')[0] if '(' in str(module) else str(module)
                if len(module_name) > 40:
                    module_name = module_name[:37] + "..."
                f.write(f"  {module_name:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)\n")
        
        # Write detailed breakdown for Attention
        f.write("\n" + "="*60 + "\n")
        f.write("Detailed FLOPs Breakdown for Attention\n")
        f.write("="*60 + "\n")
        
        flops_by_op_attention = flop_counter_attention.by_operator()
        f.write("\nFLOPs by Operator Type:\n")
        f.write("-" * 60 + "\n")
        for op, flops in sorted(flops_by_op_attention.items(), key=lambda x: x[1], reverse=True):
            gflops = flops / 1e9
            percentage = (flops / flops_attention) * 100
            f.write(f"  {op:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)\n")
        
        flops_by_module_attention = flop_counter_attention.by_module()
        f.write("\nFLOPs by Module:\n")
        f.write("-" * 60 + "\n")
        for module, flops in sorted(flops_by_module_attention.items(), key=lambda x: x[1], reverse=True):
            if flops > 0:
                gflops = flops / 1e9
                percentage = (flops / flops_attention) * 100
                module_name = str(module).split('(')[0] if '(' in str(module) else str(module)
                if len(module_name) > 40:
                    module_name = module_name[:37] + "..."
                f.write(f"  {module_name:40s}: {flops:15.0f} FLOPs ({gflops:8.2f} GFLOPs, {percentage:5.2f}%)\n")
    
    print(f"\nResults saved to: {txt_path}")
    
    # Create bar plot
    models = ['Linear Attention', 'Full Attention']
    gflops = [gflops_linear, gflops_attention]
    colors = ['#C5E1B4', '#F6E8E6']  # Linear Attention: light green, Full Attention: light pink
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(models, gflops, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    
    # Add value labels on bars
    for bar, value in zip(bars, gflops):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('GFLOPs', fontsize=12, fontweight='bold')
    ax.set_title('FLOPs Comparison', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    
    # Save as PDF, SVG, and PNG
    pdf_path = f"{output_dir}/flops_comparison.pdf"
    svg_path = f"{output_dir}/flops_comparison.svg"
    png_path = f"{output_dir}/flops_comparison.png"
    
    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, format='svg', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
    
    print(f"Plot saved to: {pdf_path}")
    print(f"Plot saved to: {svg_path}")
    print(f"Plot saved to: {png_path}")
    
    plt.close()
    
    return flops_linear, flops_attention


if __name__ == "__main__":
    test_flops()
