# ------------------------------------------------------------------------
# Copyright (c) 2023-present, BAAI. All Rights Reserved.
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
"""Convert LDM VAE to diffusers model."""

import collections
import torch


if __name__ == "__main__":
    sd = torch.load("./kl16.ckpt", map_location="cpu")["model"]
    new_state_dict = collections.OrderedDict()
    # fmt: off
    for k, v in sd.items():
        new_k = k
        for arch in ("down", "up"):
            for depth in range(5):
                new_depth = depth if arch == "down" else 4 - depth
                new_k = new_k.replace(f".{arch}.{depth}.block.", f".{arch}_blocks.{new_depth}.resnets.")
                new_k = new_k.replace(f".{arch}.{depth}.downsample.", f".{arch}_blocks.{new_depth}.downsamplers.0.")
                new_k = new_k.replace(f".{arch}.{depth}.upsample.", f".{arch}_blocks.{new_depth}.upsamplers.0.")
        # mid
        new_k = new_k.replace(".mid.attn_1.", ".mid_block.attentions.0.")
        new_k = new_k.replace(".mid.block_1.", ".mid_block.resnets.0.")
        new_k = new_k.replace(".mid.block_2.", ".mid_block.resnets.1.")
        # shortcut
        new_k = new_k.replace(".nin_shortcut.", ".conv_shortcut.")
        new_k = new_k.replace(".norm_out.", ".conv_norm_out.")
        # attn
        new_k = new_k.replace("down.4.attn", "down_blocks.4.attentions")
        if "attentions" in new_k:
            new_k = new_k.replace(".norm", ".group_norm")
            new_k = new_k.replace(".q", ".query").replace(".k", ".key")
            new_k = new_k.replace(".v", ".value").replace(".proj_out", ".proj_attn")
            v = v.flatten(1) if v.dim() > 2 else v
        new_state_dict[new_k] = v
    num_params = 0
    for k, v in new_state_dict.items():
        num_params += v.numel()
        print(k, v.shape, v.dtype, num_params / 1e6)
    torch.save(new_state_dict, "diffusion_pytorch_model.bin")
