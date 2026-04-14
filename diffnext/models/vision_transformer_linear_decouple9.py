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


class LinearAttentionEncoder(nn.Module):
    """Multi-head linear attention encoder + depth-wise 1 D Conv on the first 64 tokens."""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 proj_drop: float = 0.,
                 kernel_function=nn.ReLU,
                 fp32_attention: bool = True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kernel_function = kernel_function()
        self.fp32_attention  = fp32_attention

        # --- depth-wise 1 D conv: acts on first 64 tokens only -----------------
        self.dw_conv1d = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # optional hooks
        self.attn_mask = self.cache_kv = self.pe_func = self.flex_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, 320, C]
        B, N, C = x.shape

        # ---------------- linear attention branch -----------------------------
        qkv_shape = [-1, N, 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape).permute(2, 0, 3, 1, 4).unbind(0)  # [B,H,N,D]

        if self.pe_func:
            q, k = self.pe_func(q), self.pe_func(k)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        orig_dtype = q.dtype
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        with torch.cuda.amp.autocast(enabled=not self.fp32_attention):
            z  = 1. / (q @ k.mean(-2, keepdim=True).transpose(-2, -1) + 1e-6)   # [B,H,N,1]
            kv = (k.transpose(-2, -1) * (N**-0.5)) @ (v * (N**-0.5))            # [B,H,D,D]
            attn_out = (q @ kv * z).transpose(1, 2).flatten(2)                  # [B,N,C]

        attn_out = attn_out.to(orig_dtype)

        conv_len = 64                       
        tail_len = N - conv_len             

        # ---------------- 1 D depth-wise conv branch (first 64 tokens) ---------
        x_head = x[:, :conv_len, :]         # (B, 64, C)
        conv_in  = x_head.transpose(1, 2)   # → (B, C, 64)  
        conv_out = self.dw_conv1d(conv_in)  # (B, C, 64)
        conv_out = conv_out.transpose(1, 2) # → (B, 64, C)

        # pad zeros for the remaining 256 positions to align sequence length
        padding = torch.zeros(B, tail_len, C, dtype=x.dtype, device=x.device)   # (B, N-64, C)

        conv_out_padded = torch.cat([conv_out, padding], dim=1)   # (B, N, C)

        # ---------------- fuse & project --------------------------------------
        x = attn_out + conv_out_padded
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttentionDecoder(nn.Module):
    """Multi-head linear attention decoder
       - 前 64 token : depth-wise 1-D Conv
       - 后 256 token: depth-wise 2-D Conv (16×16)"""
    def __init__(self,
                 dim: int,
                 num_heads: int,
                 qkv_bias: bool = True,
                 proj_drop: float = 0.,
                 kernel_function=nn.ReLU,
                 fp32_attention: bool = True):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.qkv  = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.kernel_function = kernel_function()
        self.fp32_attention  = fp32_attention

        # --- local-conv branches ------------------------------------------------
        self.conv_len  = 64
        self.dw_conv1d = nn.Conv1d(dim, dim, kernel_size=5, padding=2, groups=dim)
        # NEW: depth-wise 2-D conv for the remaining 256 tokens (16×16)
        self.dw_conv2d = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)   # NEW

        # optional hooks
        self.attn_mask = self.cache_kv = self.pe_func = self.flex_attn = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: [B, 320, C]
        B, N, C = x.shape
        assert N == 320, f"Expected sequence length 320 but got {N}"
        tail_len = N - self.conv_len                              # ==256

        # ---------------- linear attention branch -----------------------------
        qkv_shape = [B, N, 3, self.num_heads, self.head_dim]
        q, k, v = self.qkv(x).view(qkv_shape)\
                             .permute(2, 0, 3, 1, 4).unbind(0)    # [B,H,N,D]

        if self.pe_func:
            q, k = self.pe_func(q), self.pe_func(k)

        q = self.kernel_function(q) + 1e-6
        k = self.kernel_function(k) + 1e-6

        orig_dtype = q.dtype
        if self.fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        with torch.cuda.amp.autocast(enabled=not self.fp32_attention):
            z  = 1. / (q @ k.mean(-2, keepdim=True).transpose(-2, -1) + 1e-6)     # [B,H,N,1]
            kv = (k.transpose(-2, -1) * (N**-0.5)) @ (v * (N**-0.5))              # [B,H,D,D]
            attn_out = (q @ kv * z).transpose(1, 2).flatten(2)                    # [B,N,C]

        attn_out = attn_out.to(orig_dtype)

        # ---------------- local-conv branches ---------------------------------
        # 1) 前 64 token : depth-wise 1-D
        x_head   = x[:, :self.conv_len, :]                     # (B, 64, C)
        conv1_in = x_head.transpose(1, 2)                      # (B, C, 64)
        conv1_out = self.dw_conv1d(conv1_in).transpose(1, 2)   # (B, 64, C)

        # 2) 后 256 token: depth-wise 2-D
        x_tail = x[:, self.conv_len:, :]                       # (B, 256, C)
        H = W = int(math.sqrt(tail_len))
        assert H * W == tail_len, "tail_len 必须是完全平方数以便 2-D 卷积"
        conv2_in  = x_tail.reshape(B, H, W, C).permute(0, 3, 1, 2)   # (B, C, 16, 16)
        conv2_out = self.dw_conv2d(conv2_in)                         # (B, C, 16, 16)
        conv2_out = conv2_out.permute(0, 2, 3, 1).reshape(B, tail_len, C)  # (B, 256, C)

        # 拼接两段局部卷积结果
        conv_out_full = torch.cat([conv1_out, conv2_out], dim=1)      # (B, N, C)

        # ---------------- fuse & project --------------------------------------
        x = attn_out + conv_out_full
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    """Transformer block."""

    def __init__(self, dim, num_heads, mlp_ratio=4, qkv_bias=True, mode='encoder'):
        super(Block, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        assert mode in ['encoder', 'decoder']
        if mode == 'encoder':
            self.attn = LinearAttentionEncoder(dim, num_heads, qkv_bias=qkv_bias)
        elif mode == 'decoder':
            self.attn = LinearAttentionDecoder(dim, num_heads, qkv_bias=qkv_bias)
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
    ):
        super(VisionTransformer, self).__init__()
        self.embed_dim, self.image_size, self.image_dim = embed_dim, image_size, image_dim
        self.patch_embed = PatchEmbed(image_dim, embed_dim, patch_size)
        self.pos_embed, self.rope = nn.Identity(), RotaryEmbed3D(embed_dim // num_heads)
        if image_model:
            half = depth // 2 if encoder_depth is None else encoder_depth
            self.blocks = nn.ModuleList(
                Block(embed_dim, num_heads, mlp_ratio, mode='encoder' if i < half else 'decoder')
                for i in range(depth)
            )
        else:
            self.blocks = nn.ModuleList(Block(embed_dim, num_heads, mlp_ratio, mode='encoder') for _ in range(depth))
        self.norm, self.mixer = nn.LayerNorm(embed_dim), nn.Identity()
        self.encoder_depth = len(self.blocks) // 2 if encoder_depth is None else encoder_depth
        self.flex_attn = FlexAttentionCausal2D()
        [setattr(blk.attn, "flex_attn", self.flex_attn) for blk in self.blocks]

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
