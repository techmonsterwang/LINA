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

try:
    from safetensors.torch import load_file
    HAS_SAFETENSORS = True
except ImportError:
    HAS_SAFETENSORS = False

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


def load_from_nova(pipe, pretrained_path):
    """
    Load weights from pretrained NOVA model, excluding attention weights.

    Args:
        pipe: Pipeline containing the model to initialize
        pretrained_path: Path to the pretrained NOVA model directory
    """
    model = pipe.transformer

    # Load pretrained state dict
    pretrained_weights_path = os.path.join(pretrained_path, "diffusion_pytorch_model.safetensors")
    pretrained_weights_path_bin = os.path.join(pretrained_path, "diffusion_pytorch_model.bin")

    if os.path.exists(pretrained_weights_path) and HAS_SAFETENSORS:
        print("Loading pretrained NOVA weights from safetensors...")
        try:
            pretrained_state_dict = load_file(pretrained_weights_path)
        except Exception as e:
            print(f"Error loading safetensors weights: {e}")
            return
    elif os.path.exists(pretrained_weights_path_bin):
        print("Loading pretrained NOVA weights from bin file...")
        try:
            pretrained_state_dict = torch.load(pretrained_weights_path_bin, map_location='cpu')
        except Exception as e:
            print(f"Error loading bin weights: {e}")
            return
    else:
        print(f"Pretrained weights not found at {pretrained_weights_path} or {pretrained_weights_path_bin}")
        return

    # Get current model state dict
    current_state_dict = model.state_dict()

    # Track loaded and skipped weights
    loaded_keys = []
    skipped_keys = []
    not_found_keys = []

    # Load matching weights (excluding attention weights)
    for key in current_state_dict.keys():
        # Skip attention-related weights
        if 'attn' in key.lower():
            skipped_keys.append(key)
            continue

        # Check if key exists in pretrained weights
        if key in pretrained_state_dict:
            pretrained_tensor = pretrained_state_dict[key]
            current_tensor = current_state_dict[key]

            # Check if shapes match
            if pretrained_tensor.shape == current_tensor.shape:
                # Load the weight
                current_state_dict[key].copy_(pretrained_tensor)
                loaded_keys.append(key)
                print(f"Loaded: {key} - Shape: {pretrained_tensor.shape}")
            else:
                skipped_keys.append(f"{key} (shape mismatch: {current_tensor.shape} vs {pretrained_tensor.shape})")
        else:
            not_found_keys.append(key)

    # Print summary
    print("\n" + "="*50)
    print("WEIGHT LOADING SUMMARY")
    print("="*50)
    print(f"Successfully loaded weights: {len(loaded_keys)}")
    print(f"Skipped weights (attention or shape mismatch): {len(skipped_keys)}")
    print(f"Weights not found in pretrained model: {len(not_found_keys)}")
    print("\nLoaded weights:")
    for key in loaded_keys:
        print(f"  ✓ {key}")
    print("\nSkipped weights:")
    for key in skipped_keys:
        print(f"  ✗ {key}")
    print("\nWeights not found:")
    for key in not_found_keys:
        print(f"  ? {key}")

    # Save results to txt file
    output_dir = "load_file"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "finetune_nova.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("WEIGHT LOADING SUMMARY\n")
        f.write("="*60 + "\n")
        f.write(f"Successfully loaded weights: {len(loaded_keys)}\n")
        f.write(f"Skipped weights (attention or shape mismatch): {len(skipped_keys)}\n")
        f.write(f"Weights not found in pretrained model: {len(not_found_keys)}\n")
        f.write("\n")

        f.write("LOADED WEIGHTS:\n")
        f.write("-"*30 + "\n")
        for key in loaded_keys:
            f.write(f"✓ {key}\n")

        f.write("\n")
        f.write("SKIPPED WEIGHTS:\n")
        f.write("-"*30 + "\n")
        for key in skipped_keys:
            f.write(f"✗ {key}\n")

        f.write("\n")
        f.write("WEIGHTS NOT FOUND:\n")
        f.write("-"*30 + "\n")
        for key in not_found_keys:
            f.write(f"? {key}\n")

    print(f"\nResults saved to: {output_file}")


def test_load_from_nova():
    """Test function to verify load_from_nova functionality."""
    # Create a mock pipeline and model for testing
    from diffnext.models.vision_transformer_linear_decouple1_kvscale1 import VisionTransformer

    # Create a simple test model
    test_model = VisionTransformer(
        depth=2,
        embed_dim=384,
        num_heads=6,
        patch_size=2,
        image_size=32,
        image_dim=4,
        image_model=True
    )

    # Create a mock pipe object
    class MockPipe:
        def __init__(self, model):
            self.transformer = type('obj', (object,), {'image_decoder': model})()

    mock_pipe = MockPipe(test_model)

    # Test the load_from_nova function
    test_pretrained_path = "/share/project/wangjiahao/LAR/pretrained/huggingface_auto_download/models--BAAI--nova-d48w1536-sdxl1024/snapshots/76baa2f6a3be9c487045153219a7b663d7e26553/transformer"
    load_from_nova(mock_pipe, test_pretrained_path)
    print("Test completed successfully!")


if __name__ == "__main__":
    # fmt: off
    cfg_file = "/share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_256px_finetune_nova.yml"
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
    pretrained_path = "/share/project/wangjiahao/LAR/pretrained/huggingface_auto_download/models--BAAI--nova-d48w1536-sdxl1024/snapshots/76baa2f6a3be9c487045153219a7b663d7e26553/transformer"
    load_from_nova(pipe, pretrained_path)
    pipe.transformer.save_pretrained(out, safe_serialization=False)
