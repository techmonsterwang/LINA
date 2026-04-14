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
import torch.nn.functional as F
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
    """Multihead linear attention for encoder."""

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0., kernel_func='relu', fp32_attention=True):
        super(LinearAttentionEncoder, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fp32_attention = fp32_attention
        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(dim, dim * 5, kernel_size=1, groups=num_heads)
        )
        
        assert kernel_func in ['identity', 'relu', 'leakyrelu', 'exp']
        if kernel_func == 'identity':
            self.phi = None
        elif kernel_func == 'relu':
            self.phi = nn.ReLU()
        elif kernel_func == 'leakyrelu':
            self.phi = nn.LeakyReLU()
        elif kernel_func == 'exp':
            self.phi = exp_kernel
        else:
            self.phi = None

        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

    def forward(self, x) -> torch.Tensor:
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        if self.phi is not None:
            q = self.phi(q)
            k = self.phi(k)
        
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        dtype = q.dtype

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):

            # 1D conv kernel: (B * C, 1, K), reshape to match conv1d expected format
            res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(b * c, 1, 5)  # e.g., 3 = kernel_size
            res_weight = res_weight.float()

            # The self.scale / n = head_dim ** -0.5 / n is a scale factor used in InLine attention.
            # This factor can be equivalently achieved by scaling \phi(Q) = \phi(Q) * self.scale / n
            # Therefore, we omit it in eq. 5 of the paper for simplicity.
            kv = (k.transpose(-2, -1) * (self.scale / n) ** 0.5) @ (v * (self.scale / n) ** 0.5)
            x = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

            x = x.transpose(1, 2).reshape(b, n, c)
            v = v.transpose(1, 2).reshape(b, n, c).permute(0, 2, 1).reshape(1, b * c, n)
            residual = F.conv1d(v, res_weight, bias=None, padding=2, groups=b * c)  # padding = (kernel_size // 2)
            x = x + residual.reshape(b, c, n).permute(0, 2, 1)

        x = x.to(dtype)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LinearAttentionDecoder(nn.Module):
    """Multihead linear attention for decoder."""

    def __init__(self, dim, num_heads, qkv_bias=True, proj_drop=0., kernel_func='relu', fp32_attention=True):
        super(LinearAttentionDecoder, self).__init__()
        self.num_heads, self.head_dim = num_heads, dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.scale = self.head_dim ** -0.5
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.fp32_attention = fp32_attention
        self.residual = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=1, groups=num_heads),
            nn.GELU(),
            nn.Conv1d(dim, dim * 9, kernel_size=1, groups=num_heads)
        )
        
        assert kernel_func in ['identity', 'relu', 'leakyrelu', 'exp']
        if kernel_func == 'identity':
            self.phi = None
        elif kernel_func == 'relu':
            self.phi = nn.ReLU()
        elif kernel_func == 'leakyrelu':
            self.phi = nn.LeakyReLU()
        elif kernel_func == 'exp':
            self.phi = exp_kernel
        else:
            self.phi = None

        self.attn_mask, self.cache_kv, self.pe_func, self.flex_attn = None, None, None, None

    def forward(self, x) -> torch.Tensor:
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)
        kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        if self.phi is not None:
            q = self.phi(q)
            k = self.phi(k)
        
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)

        dtype = q.dtype

        use_fp32_attention = getattr(self, 'fp32_attention', False)     # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):

            # 2D conv kernel: (B * C, 1, 3, 3), reshape to match conv2d expected format
            res_weight = self.residual(x.mean(dim=1).unsqueeze(dim=-1)).reshape(b * c, 1, 3, 3)  # e.g., 3 = kernel_size, torch.bfloat16

            # The self.scale / n = head_dim ** -0.5 / n is a scale factor used in InLine attention.
            # This factor can be equivalently achieved by scaling \phi(Q) = \phi(Q) * self.scale / n
            # Therefore, we omit it in eq. 5 of the paper for simplicity.
            kv = (k.transpose(-2, -1) * (self.scale / n) ** 0.5) @ (v * (self.scale / n) ** 0.5)  # [b, h, d, d]
            attn_out = q @ kv + (1 - q @ k.mean(dim=2, keepdim=True).transpose(-2, -1) * self.scale) * v.mean(dim=2, keepdim=True)

            attn_out = attn_out.transpose(1, 2).reshape(b, n, c)

        attn_out = attn_out.to(dtype)

        # ==============================
        # dwc branch
        # ==============================
        x_tail = x[:, 64:, :].reshape(b, 16, 16, -1).permute(0, 3, 1, 2).reshape(1, b * c, 16, 16)    # shape: (1, B*C, 16, 16), torch.bfloat16
        conv_out = F.conv2d(x_tail, res_weight, None, padding=(1, 1), groups=b * c)  # shape: (1, B*C, 16, 16), torch.bfloat16
        conv_out = conv_out.reshape(b, c, -1).permute(0, 2, 1)  # (B, 256, C)
        conv_out_padded = torch.cat([torch.zeros(b, 64, c, device=x.device), conv_out], dim=1)
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
