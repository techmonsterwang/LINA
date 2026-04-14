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
"""NOVA C2I initialzation."""

import os
import torch
from torch import nn
import sys

sys.path.append(os.getcwd())

from diffnext import engine
from diffnext.config import cfg
from diffnext.models.normalization import AdaLayerNormZero
from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def init_scratch(pipe):
    model = pipe.transformer.image_decoder
    nn.init.zeros_(model.patch_embed.proj.bias)
    for m in model.time_cond_embed.timestep_proj.modules():
        nn.init.normal_(m.weight, std=0.02) if isinstance(m, nn.Linear) else None
    for m in filter(lambda m: isinstance(m, AdaLayerNormZero), model.modules()):
        nn.init.zeros_(m.proj.weight), nn.init.zeros_(m.proj.bias)
    nn.init.zeros_(model.head.weight), nn.init.zeros_(model.head.bias)


if __name__ == "__main__":
    # fmt: off
    cfg_file = "/share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_256px_test.yml"
    # fmt: on

    # Prepare save path.
    cfg.merge_from_file(cfg_file)
    weights = cfg.MODEL.WEIGHTS
    out = os.path.join(weights, "transformer")
    os.makedirs(out, exist_ok=True)
    torch.save({}, os.path.join(out, "diffusion_pytorch_model.bin"))

    # Build pipeline.
    engine.manual_seed(1337)  # Fix initialization seed.
    pipe_conf = {"transformer": cfg.MODEL.CONFIG}
    pipe_path = get_pipeline_path(weights, {**cfg.PIPELINE.MODULES}, pipe_conf)
    pipe = build_pipeline(pipe_path, "nova", precison="float16", low_cpu_mem_usage=False)

    # Run initialzation and save.
    init_scratch(pipe)
    pipe.transformer.save_pretrained(out, safe_serialization=False)
