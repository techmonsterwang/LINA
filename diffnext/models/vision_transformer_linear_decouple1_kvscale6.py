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
    """Multi-head linear-attention decoder with an extra depth-wise-conv branch.

    Key/Value scaling is:
        scale = exp(log(sigmoid(WX)) / norm_const)
    where `norm_const` is a *fixed* positive constant (default = 16),
    making the gate ∈ (0, 1] and head-wise (each head gets its own scale).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool = True,
        proj_drop: float = 0.0,
        kernel_function=nn.ReLU,
        fp32_attention: bool = True,
        norm_const: float = 16.0,       # <── fixed denominator for the gate
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads   = num_heads
        self.head_dim    = dim // num_heads
        self.norm_const  = float(norm_const)

        # Q K V projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        # output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # depth-wise convolution branch
        self.dw_conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim)

        # positive kernel to linearise attention
        self.kernel_fn = kernel_function()

        # use float32 inside attention to avoid NaNs (optional)
        self.fp32_attention = fp32_attention

        # data-dependent head-wise gates for K / V scaling
        self.key_gate   = nn.Linear(dim, num_heads, bias=True)   # (B, N, num_heads)
        self.value_gate = nn.Linear(dim, num_heads, bias=True)   # (B, N, num_heads)

        # placeholders for optional features
        self.attn_mask = None
        self.cache_kv  = None
        self.pe_func   = None
        self.flex_attn = None

    # ------------------------------------------------------------------ #
    # gating helper
    # ------------------------------------------------------------------ #
    def _make_gate(self, tensor: torch.Tensor, gate_layer: nn.Linear) -> torch.Tensor:
        """
        Compute a gate ∈ (0, 1] for each (token, head):

            gate = exp( log(sigmoid(W·tensor)) / norm_const )

        Args:
            tensor: (B, N, dim)
        Returns:
            scale:  (B, N, num_heads)
        """
        logits = gate_layer(tensor)                 # (B, N, num_heads)
        sigma  = torch.sigmoid(logits)              # (0, 1)
        scale  = torch.exp(torch.log(sigma) / self.norm_const)
        return scale

    # ------------------------------------------------------------------ #
    # forward
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, C)       — N **must** be 320
        Returns:
            (B, N, C)
        """
        B, N, C = x.shape
        assert N == 320, f"Expected sequence length N = 320, got {N}"

        # -------------------------------------------------------------- #
        # build Q, K, V
        # -------------------------------------------------------------- #
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)                 # (3, B, num_heads, N, head_dim)
        )
        q, k, v = qkv.unbind(dim=0)

        if self.pe_func is not None:                # optional positional encoding
            q, k = self.pe_func(q), self.pe_func(k)

        q = self.kernel_fn(q) + 1e-6               # positive kernel trick
        k = self.kernel_fn(k) + 1e-6

        # -------------------------------------------------------------- #
        # optional fp32 for numerical stability
        # -------------------------------------------------------------- #
        dtype              = q.dtype
        use_fp32_attention = bool(self.fp32_attention)
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # -------------------------------------------------------------- #
        # head-wise, data-dependent gates based on X (not K/V)
        # -------------------------------------------------------------- #
        key_scale = self._make_gate(x, self.key_gate)      # (B, N, num_heads)
        value_scale = self._make_gate(x, self.value_gate)    # (B, N, num_heads)

        # Adjust key_scale and value_scale shapes: (B, num_heads, N, 1)
        key_scale = key_scale.permute(0, 2, 1).unsqueeze(-1)  # (B, num_heads, N, 1)
        value_scale = value_scale.permute(0, 2, 1).unsqueeze(-1)  # (B, num_heads, N, 1)

        # Apply scaling to K and V
        k = k * key_scale                      # (B, num_heads, N, head_dim)
        v = v * value_scale                    # (B, num_heads, N, head_dim)

        # -------------------------------------------------------------- #
        # linear attention
        # -------------------------------------------------------------- #
        with torch.cuda.amp.autocast(enabled=not use_fp32_attention):
            z   = 1.0 / (q @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
            kv  = (k.transpose(-2, -1) * (N ** -0.5)) @ (v * (N ** -0.5))
            out = q @ kv * z                                # (B, num_heads, N, head_dim)
            out = out.transpose(1, 2).flatten(2)            # (B, N, C)

        out = out.to(dtype)

        # -------------------------------------------------------------- #
        # depth-wise-conv branch on tail tokens (indices 64-319)
        # -------------------------------------------------------------- #
        x_tail = (
            x[:, 64:, :]                                    # (B, 256, C)
            .reshape(B, 16, 16, C)
            .permute(0, 3, 1, 2)                            # (B, C, 16, 16)
        )
        conv_out = self.dw_conv(x_tail)                     # (B, C, 16, 16)
        conv_out = (
            conv_out.reshape(B, C, -1)                      # (B, C, 256)
                    .permute(0, 2, 1)                       # (B, 256, C)
        )
        conv_out_padded = torch.cat(
            [torch.zeros(B, 64, C, device=x.device, dtype=dtype), conv_out], dim=1
        )                                                   # (B, N, C)

        # -------------------------------------------------------------- #
        # combine, project, dropout
        # -------------------------------------------------------------- #
        x = out + conv_out_padded
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
