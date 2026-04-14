# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################
"""NOVA C2I training pipeline."""

from typing import Dict

from diffusers.pipelines.pipeline_utils import DiffusionPipeline
import torch

from diffnext import engine
from diffnext.pipelines.builder import PIPELINES, build_diffusion_scheduler
from diffnext.pipelines.nova.pipeline_utils import PipelineMixin


@PIPELINES.register("nova_train_c2i")
class NOVATrainC2IPipeline(DiffusionPipeline, PipelineMixin):
    """Pipeline for training NOVA C2I models."""

    _optional_components = ["transformer", "scheduler", "vae"]

    def __init__(self, transformer=None, scheduler=None, vae=None, trust_remote_code=True):
        super(NOVATrainC2IPipeline, self).__init__()
        self.vae = self.register_module(vae, "vae")
        self.transformer = self.register_module(transformer, "transformer")
        self.scheduler = self.register_module(scheduler, "scheduler")
        self.transformer.noise_scheduler = build_diffusion_scheduler(self.scheduler)
        self.transformer.sample_scheduler, self.guidance_scale = self.scheduler, 5.0

    @property
    def model(self) -> torch.nn.Module:
        """Return the trainable model."""
        return self.transformer

    def configure_model(self, loss_repeat=4, checkpointing=0, config=None) -> torch.nn.Module:
        """Configure the trainable model."""
        self.model.loss_repeat = config.TRAIN.LOSS_REPEAT if config else loss_repeat
        ckpt_lvl = config.TRAIN.CHECKPOINTING if config else checkpointing
        [setattr(blk, "mlp_checkpointing", ckpt_lvl) for blk in self.model.video_encoder.blocks]
        [setattr(blk, "mlp_checkpointing", ckpt_lvl > 1) for blk in self.model.image_encoder.blocks]
        [setattr(blk, "mlp_checkpointing", ckpt_lvl > 2) for blk in self.model.image_decoder.blocks]
        engine.freeze_module(self.model.label_embed.norm)  # We always use frozen LN.
        engine.freeze_module(self.model.video_pos_embed)  # Freeze this module during C2I.
        engine.freeze_module(self.model.video_encoder.patch_embed)  # Freeze this module during C2I.
        self.model.pipeline_preprocess = self.preprocess
        return self.model.train()

    def prepare_latents(self, inputs: Dict):
        """Prepare the video latents."""
        if "images" in inputs:
            raise NotImplementedError
        elif "moments" in inputs:
            x = torch.as_tensor(inputs.pop("moments"), device=self.device).to(dtype=self.dtype)
            inputs["x"] = self.vae.scale_(self.vae.latent_dist(x).sample())

    def encode_prompt(self, inputs: Dict):
        """Encode class prompts."""
        prompts = torch.as_tensor(inputs.pop("prompt"), device=self.device)
        inputs["c"] = [self.transformer.label_embed(prompts)]

    def preprocess(self, inputs: Dict) -> Dict:
        """Define the pipeline preprocess at every call."""
        if not self.model.training:
            raise RuntimeError("Excepted a trainable model.")
        self.prepare_latents(inputs)
        self.encode_prompt(inputs)
