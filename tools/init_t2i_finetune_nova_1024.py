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
    Load weights from pretrained model, excluding key_scale and value_scale weights.

    Args:
        pipe: Pipeline containing the model to initialize
        pretrained_path: Path to the pretrained model directory
    """
    model = pipe.transformer

    # Load pretrained state dict
    pretrained_weights_path = os.path.join(pretrained_path, "diffusion_pytorch_model.safetensors")
    pretrained_weights_path_bin = os.path.join(pretrained_path, "diffusion_pytorch_model.bin")

    if os.path.exists(pretrained_weights_path) and HAS_SAFETENSORS:
        print("Loading pretrained weights from safetensors...")
        try:
            pretrained_state_dict = load_file(pretrained_weights_path)
        except Exception as e:
            print(f"Error loading safetensors weights: {e}")
            return
    elif os.path.exists(pretrained_weights_path_bin):
        print("Loading pretrained weights from bin file...")
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

    # Check key_scale and value_scale shapes before loading
    print("\n" + "="*60)
    print("KEY_SCALE AND VALUE_SCALE SHAPE COMPARISON")
    print("="*60)
    
    key_scale_keys = []
    value_scale_keys = []
    
    # Find all key_scale and value_scale keys in current model
    for key in current_state_dict.keys():
        if 'key_scale' in key.lower():
            key_scale_keys.append(key)
        elif 'value_scale' in key.lower():
            value_scale_keys.append(key)
    
    # Compare shapes for key_scale weights
    if key_scale_keys:
        print(f"\nFound {len(key_scale_keys)} key_scale weights in current model:")
        for key in key_scale_keys:
            current_shape = current_state_dict[key].shape
            if key in pretrained_state_dict:
                pretrained_shape = pretrained_state_dict[key].shape
                print(f"  {key}:")
                print(f"    Current model shape:  {current_shape}")
                print(f"    Pretrained shape:     {pretrained_shape}")
                print(f"    Shapes match:         {current_shape == pretrained_shape}")
            else:
                print(f"  {key}:")
                print(f"    Current model shape:  {current_shape}")
                print(f"    Pretrained shape:     NOT FOUND")
    else:
        print("\nNo key_scale weights found in current model.")
    
    # Compare shapes for value_scale weights
    if value_scale_keys:
        print(f"\nFound {len(value_scale_keys)} value_scale weights in current model:")
        for key in value_scale_keys:
            current_shape = current_state_dict[key].shape
            if key in pretrained_state_dict:
                pretrained_shape = pretrained_state_dict[key].shape
                print(f"  {key}:")
                print(f"    Current model shape:  {current_shape}")
                print(f"    Pretrained shape:     {pretrained_shape}")
                print(f"    Shapes match:         {current_shape == pretrained_shape}")
            else:
                print(f"  {key}:")
                print(f"    Current model shape:  {current_shape}")
                print(f"    Pretrained shape:     NOT FOUND")
    else:
        print("\nNo value_scale weights found in current model.")
    
    print("\n" + "="*60)
    print("STARTING WEIGHT LOADING (EXCLUDING KEY_SCALE AND VALUE_SCALE)")
    print("="*60)

    # Track loaded and skipped weights
    loaded_keys = []
    skipped_keys = []
    not_found_keys = []

    # Load matching weights (excluding key_scale and value_scale weights)
    for key in current_state_dict.keys():
        # Skip key_scale and value_scale weights
        if 'key_scale' in key.lower() or 'value_scale' in key.lower():
            skipped_keys.append(f"{key} (excluded: key_scale/value_scale)")
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
    output_file = os.path.join(output_dir, "finetune_nova_1024.txt")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("="*60 + "\n")
        f.write("KEY_SCALE AND VALUE_SCALE SHAPE COMPARISON\n")
        f.write("="*60 + "\n")
        
        # Write key_scale shape comparison
        if key_scale_keys:
            f.write(f"\nFound {len(key_scale_keys)} key_scale weights in current model:\n")
            for key in key_scale_keys:
                current_shape = current_state_dict[key].shape
                if key in pretrained_state_dict:
                    pretrained_shape = pretrained_state_dict[key].shape
                    f.write(f"  {key}:\n")
                    f.write(f"    Current model shape:  {current_shape}\n")
                    f.write(f"    Pretrained shape:     {pretrained_shape}\n")
                    f.write(f"    Shapes match:         {current_shape == pretrained_shape}\n")
                else:
                    f.write(f"  {key}:\n")
                    f.write(f"    Current model shape:  {current_shape}\n")
                    f.write(f"    Pretrained shape:     NOT FOUND\n")
        else:
            f.write("\nNo key_scale weights found in current model.\n")
        
        # Write value_scale shape comparison
        if value_scale_keys:
            f.write(f"\nFound {len(value_scale_keys)} value_scale weights in current model:\n")
            for key in value_scale_keys:
                current_shape = current_state_dict[key].shape
                if key in pretrained_state_dict:
                    pretrained_shape = pretrained_state_dict[key].shape
                    f.write(f"  {key}:\n")
                    f.write(f"    Current model shape:  {current_shape}\n")
                    f.write(f"    Pretrained shape:     {pretrained_shape}\n")
                    f.write(f"    Shapes match:         {current_shape == pretrained_shape}\n")
                else:
                    f.write(f"  {key}:\n")
                    f.write(f"    Current model shape:  {current_shape}\n")
                    f.write(f"    Pretrained shape:     NOT FOUND\n")
        else:
            f.write("\nNo value_scale weights found in current model.\n")
        
        f.write("\n" + "="*60 + "\n")
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


if __name__ == "__main__":
    # fmt: off
    cfg_file = "/share/project/wangjiahao/LAR/diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova.yml"
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
    pretrained_path = "/share/project/wangjiahao/LAR/exp_dir/t2i/linear_decouple1_kvscale1_w1536_6node_finetune_nova_512px/ema_checkpoints/sdxl28m_nova_d48w1536_256px_iter_600000/transformer"
    load_from_nova(pipe, pretrained_path)
    pipe.transformer.save_pretrained(out, safe_serialization=False)
