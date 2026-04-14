# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Vision Transformer."""

from typing import Tuple

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint as apply_ckpt

from diffnext.models.embeddings import PatchEmbed, RotaryEmbed3D
from diffnext.models.flex_attention import FlexAttentionCausal2D


class MLP(nn.Module):
    """Two layers MLP."""

    def __init__(self, dim, mlp_ratio=4):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.fc2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.activation = nn.GELU()

    def forward(self, x) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class FullAttention(nn.Module):
    """Multihead attention."""

    def __init__(self, dim, num_heads, qkv_bias=True):
        super(FullAttention, self).__init__()
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
        o = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.attn_mask)
        return self.proj(o.transpose(1, 2).flatten(2))


class LinearAttentionEncoder(nn.Module):
    """Multihead linear attention for encoder."""

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0., kernel_function=nn.ReLU, fp32_attention=True):
        super(LinearAttentionEncoder, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernel_function = kernel_function()
        self.fp32_attention = fp32_attention
        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

    def forward(self, x) -> torch.Tensor:
        B, N, C = x.shape
        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)
        q = self.kernel_function(q) + 1e-6   # (B, H, N, C/H)
        k = self.kernel_function(k) + 1e-6   # (B, H, N, C/H)
        
        dtype = q.dtype

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)   # (B, H, N, 1) (32, 6, 256, 1)
            kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))   # (B, H, C/H, N) @ (B, H, N, C/H) = (B, H, C/H, C/H): (32, 6, 256, 256) 
            x = q @ kv * z   # (B, H, N, C/H) @ (B, H, C/H, C/H) * (B, H, N, 1) = (B, H, N, C/H): (32, 6, 256, 64) 
            x = x.transpose(1, 2).flatten(2)    # (B, N, C): (32, 256, 384)  
            
        x = x.to(dtype)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LinearAttentionDecoder(nn.Module):
    """Multihead linear attention for decoder with extra convolution branch."""

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0., kernel_function=nn.ReLU, fp32_attention=True, max_seq_len=1024):
        super(LinearAttentionDecoder, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.kernel_function = kernel_function()
        self.fp32_attention = fp32_attention
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)
        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

        # Initialize key_scale and value_scale with maximum expected sequence length
        # These will be sliced to the actual sequence length during forward pass
        self.key_scale = nn.Parameter(torch.ones(1, num_heads, max_seq_len, 1))
        self.value_scale = nn.Parameter(torch.ones(1, num_heads, max_seq_len, 1))

    def forward(self, x) -> torch.Tensor:
        B, N, C = x.shape

        # Check if sequence length exceeds maximum expected length
        if N > self.key_scale.size(-2):
            raise ValueError(f"Sequence length {N} exceeds maximum expected length {self.key_scale.size(-2)}. "
                           f"Please increase max_seq_len in LinearAttentionDecoder.__init__")

        # Calculate split points: first 1/5 for identity, last 4/5 for convolution
        head_length = N // 5  # First 1/5 of sequence
        tail_length = N - head_length  # Last 4/5 of sequence

        qkv_shape = [-1, x.size(1), 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(dim=0)
        q, k = (self.pe_func(q), self.pe_func(k)) if self.pe_func else (q, k)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        dtype = q.dtype

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # Apply key_scale and value_scale (slice to actual sequence length)
        key_scale = self.key_scale[:, :, :N, :]  # Slice to actual sequence length
        value_scale = self.value_scale[:, :, :N, :]  # Slice to actual sequence length
        k = k * key_scale  # Apply key scaling
        v = v * value_scale  # Apply value scaling

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            z = 1 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
            attn_out = q @ kv * z
            attn_out = attn_out.transpose(1, 2).flatten(2)  # (B, N, C)

        attn_out = attn_out.to(dtype)
        # ==============================
        # dwc branch - apply convolution to last 4/5 of sequence
        # ==============================
        # For multi-resolution support, we need to handle the spatial dimensions dynamically
        # We assume the tail part can be reshaped to spatial dimensions for convolution
        # This requires that tail_length is a perfect square for 2D convolution

        # Try to infer spatial dimensions from tail_length
        # For simplicity, we assume square spatial dimensions when possible
        spatial_size = int(tail_length ** 0.5)
        if spatial_size * spatial_size == tail_length:
            # Perfect square - can reshape to 2D for convolution
            x_tail = x[:, head_length:, :].reshape(B, spatial_size, spatial_size, -1).permute(0, 3, 1, 2)  # (B, C, H, W)
            conv_out = self.dw_conv(x_tail)  # (B, C, H, W)
            conv_out = conv_out.reshape(B, C, -1).permute(0, 2, 1)  # (B, tail_length, C)

            # Create padded output: first 1/5 zeros + last 4/5 convolution output
            conv_out_padded = torch.cat([torch.zeros(B, head_length, C, device=x.device), conv_out], dim=1)
        else:
            # Not a perfect square - fall back to no convolution for tail part
            print(f"Warning: tail_length={tail_length} is not a perfect square, skipping convolution for tail part")
            conv_out_padded = torch.zeros_like(attn_out)

        conv_out_padded = conv_out_padded.to(dtype)

        x = attn_out + conv_out_padded

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, mode='encoder'):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        assert mode in ['encoder', 'decoder', 'full']
        if mode == 'encoder':
            self.attn = LinearAttentionEncoder(dim, num_heads, qkv_bias=qkv_bias)
        elif mode == 'decoder':
            self.attn = LinearAttentionDecoder(dim, num_heads, qkv_bias=qkv_bias)
        elif mode == 'full':
            self.attn = FullAttention(dim, num_heads, qkv_bias=qkv_bias)
        # self.attn = Attention(dim, num_heads, qkv_bias=qkv_bias)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio)
        self.attn_checkpointing, self.mlp_checkpointing = False, False

    def forward_attn(self, x) -> torch.Tensor:
        return self.norm1(self.attn(x))

    def forward_mlp(self, x) -> torch.Tensor:
        return self.norm2(self.mlp(x))

    def forward_ckpt(self, x, name) -> torch.Tensor:
        if getattr(self, f"{name}_checkpointing", False) and x.requires_grad:
            return apply_ckpt(getattr(self, f"forward_{name}"), x, use_reentrant=False)
        return getattr(self, f"forward_{name}")(x)

    def forward(self, x, pe_func: callable = None) -> torch.Tensor:
        self.attn.pe_func = pe_func
        x = self.forward_ckpt(x, "attn").add_(x)
        return self.forward_ckpt(x, "mlp").add_(x)


class VisionTransformer(nn.Module):
    """Vision transformer."""

    def __init__(
        self,
        depth,
        embed_dim,
        num_heads,
        mlp_ratio=4,
        patch_size=2,
        image_size=32,
        image_dim=4,
        encoder_depth=None,
        image_model=False,
        hybrid_ratio=7,  # 7:1 ratio for linear:full attention
    ):
        super(VisionTransformer, self).__init__()
        self.embed_dim, self.image_size, self.image_dim = embed_dim, image_size, image_dim
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.pos_embed, self.rope = nn.Identity(), RotaryEmbed3D(embed_dim // num_heads)
        # Create blocks with hybrid attention pattern (7:1 linear:full ratio)
        def get_block_mode(i, total_depth, hybrid_ratio, is_image_model=False):
            """Determine the attention mode for block i"""
            if is_image_model:
                half = total_depth // 2 if encoder_depth is None else encoder_depth
                base_mode = 'encoder' if i < half else 'decoder'
            else:
                base_mode = 'encoder'

            # Apply hybrid ratio pattern
            cycle_length = hybrid_ratio + 1  # 7 linear + 1 full = 8
            position_in_cycle = i % cycle_length

            # Use full attention at the end of each cycle
            if position_in_cycle == hybrid_ratio:
                return 'full'
            else:
                return base_mode

        if image_model:
            half = depth // 2 if encoder_depth is None else encoder_depth
            self.blocks = nn.ModuleList(
                Block(embed_dim, num_heads, mlp_ratio, mode=get_block_mode(i, depth, hybrid_ratio, image_model))
                for i in range(depth)
            )
        else:
            self.blocks = nn.ModuleList(
                Block(embed_dim, num_heads, mlp_ratio, mode=get_block_mode(i, depth, hybrid_ratio, False))
                for i in range(depth)
            )
        self.norm, self.mixer = nn.LayerNorm(embed_dim), nn.Identity()
        self.encoder_depth = len(self.blocks) // 2 if encoder_depth is None else encoder_depth
        self.flex_attn = FlexAttentionCausal2D()
        [setattr(blk.attn, "flex_attn", self.flex_attn) for blk in self.blocks]

        # Count and print FullAttention layers
        full_attention_count = sum(1 for blk in self.blocks if type(blk.attn).__name__ == 'FullAttention')
        linear_attention_count = len(self.blocks) - full_attention_count
        print(f"VisionTransformer initialized with {len(self.blocks)} blocks:")
        print(f"  - Linear Attention blocks: {linear_attention_count}")
        print(f"  - Full Attention blocks: {full_attention_count}")
        print(".1f")

    def prepare_pe(self, c=None, ids=None, pos=None) -> Tuple[callable, callable]:
        pad = 0 if c is None else c.size(1)
        pe1 = pe2 = self.rope.get_func(pos, pad)
        pe1 = self.rope.get_func(pos, pad, ids.expand(-1, -1, 3)) if ids is not None else pe1
        return pe1, pe2

    def forward(self, x, c=None, prev_ids=None, pos=None) -> torch.Tensor:
        x, prev_ids = x if isinstance(x, (tuple, list)) else (x, prev_ids)
        prev_ids = prev_ids if self.encoder_depth else None
        x = x_masked = self.pos_embed(self.patch_embed(x))
        pe1, pe2 = self.prepare_pe(c, prev_ids, pos) if pos is not None else [None] * 2
        if prev_ids is not None:  # Split mask from x.
            prev_ids = prev_ids.expand(-1, -1, x.size(-1))
            x = x.gather(1, prev_ids)
        x = x if c is None else torch.cat([c, x], dim=1)
        for blk in self.blocks[: self.encoder_depth]:
            x = blk(x, pe1)
        if prev_ids is not None and c is not None:  # Split c from x.
            c, x = x.split((c.size(1), x.size(1) - c.size(1)), dim=1)
        if prev_ids is not None:  # Merge mask with x.
            x = x_masked.to(dtype=x.dtype).scatter(1, prev_ids, x)
            x = x if c is None else torch.cat([c, x], dim=1)
        for blk in self.blocks[self.encoder_depth :]:
            x = blk(x, pe2)
        return self.norm(x if c is None else x[:, c.size(1) :])
