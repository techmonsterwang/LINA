#!/usr/bin/env python3
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
"""Compare GPU memory usage for multiple DiffNext models."""

import argparse
import os
import sys
from typing import Dict, List, Optional

sys.path.append(os.getcwd())

import torch
import torch.distributed as dist

from diffnext.config import cfg
from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def parse_args():
    parser = argparse.ArgumentParser(description="Compare GPU memory usage for diffnext models")
    parser.add_argument(
        "--cfg",
        type=str,
        required=True,
        help="Comma-separated list of config files (one per model)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Comma-separated list of checkpoints (one per model)",
    )
    parser.add_argument("--prompt", type=str, required=True, help="Prompt embedding .pth file")
    parser.add_argument("--num_pred_steps", type=int, default=128, help="Number of inference steps")
    parser.add_argument("--num_diff_steps", type=int, default=25, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="Guidance scale")
    parser.add_argument("--prompt_size", type=int, default=1, help="Number of unique prompts per batch")
    parser.add_argument("--sample_size", type=int, default=1, help="Number of samples per prompt")
    parser.add_argument("--vae_batch_size", type=int, default=16, help="VAE batch size")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of repeated measurements")
    parser.add_argument("--distributed", action="store_true", help="Enable distributed mode")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="gpumem_comparison.txt",
        help="Output filename (saved under evaluations/gpumem)",
    )
    return parser.parse_args()


def prepare_batch_prompts(prompts: List[torch.Tensor], prompt_size: int, sample_size: int) -> List[torch.Tensor]:
    if not prompts:
        raise ValueError("Prompt list is empty. Please provide a valid prompt file.")
    num_available = len(prompts)
    num_unique = min(prompt_size, num_available)
    selected = prompts[:num_unique]
    batch_prompts: List[torch.Tensor] = sum([[p] * sample_size for p in selected], [])
    return batch_prompts


def measure_gpu_memory(
    pipe,
    batch_prompts: List[torch.Tensor],
    device: torch.device,
    img_args: Dict,
    gen_args: Dict,
    num_runs: int,
) -> Dict[str, List[float]]:
    allocated_stats, reserved_stats = [], []

    for run in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        with torch.no_grad():
            outputs = pipe(prompt_embeds=batch_prompts, **img_args, **gen_args)
            # Ensure tensors are materialized before measuring
            _ = outputs["frames"][:, 0] if "frames" in outputs else outputs["images"]

        torch.cuda.synchronize(device)

        allocated_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        reserved_mb = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
        allocated_stats.append(allocated_mb)
        reserved_stats.append(reserved_mb)

        del outputs

    torch.cuda.empty_cache()

    return {"allocated_mb": allocated_stats, "reserved_mb": reserved_stats}


def summarize_memory(measurements: Dict[str, List[float]]) -> Dict[str, float]:
    def stats(values: List[float]) -> Dict[str, float]:
        avg = sum(values) / len(values)
        min_val = min(values)
        max_val = max(values)
        variance = sum((v - avg) ** 2 for v in values) / len(values)
        std = variance ** 0.5
        return {"avg": avg, "min": min_val, "max": max_val, "std": std}

    return {
        "allocated": stats(measurements["allocated_mb"]),
        "reserved": stats(measurements["reserved_mb"]),
    }


def run_gpu_mem_test(cfg_path: str, ckpt_path: str, args, model_name: str, prompt_dict: Dict):
    cfg.merge_from_file(cfg_path)

    num_pred_steps, num_diff_steps = args.num_pred_steps, args.num_diff_steps
    gen_args = {"num_inference_steps": num_pred_steps, "num_diffusion_steps": num_diff_steps}
    img_args = {"guidance_scale": args.guidance_scale, "output_type": "np", "vae_batch_size": args.vae_batch_size}

    rank, world_size = 0, 1
    if args.distributed:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    torch.manual_seed(1337)
    generator = torch.Generator(device).manual_seed(1337)
    gen_args.update({"generator": generator, "disable_progress_bar": True})

    is_root = device.index == 0
    if not is_root:
        return None

    pipe_path = get_pipeline_path(ckpt_path, {**cfg.PIPELINE.MODULES, "text_encoder": ""})
    pipe = build_pipeline(pipe_path, "nova", precison="bfloat16").to(device=device)

    prompts = prompt_dict["prompts"]
    batch_prompts = prepare_batch_prompts(prompts, args.prompt_size, args.sample_size)

    # Warm-up
    print(f"Warming up model {model_name}...")
    with torch.no_grad():
        _ = pipe(prompt_embeds=[batch_prompts[0]], **img_args, **gen_args)
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()

    print(f"Measuring GPU memory for {model_name}...")
    measurements = measure_gpu_memory(pipe, batch_prompts, device, img_args, gen_args, args.num_runs)
    summary = summarize_memory(measurements)

    return {
        "model_name": model_name,
        "allocated_runs": measurements["allocated_mb"],
        "reserved_runs": measurements["reserved_mb"],
        "allocated_summary": summary["allocated"],
        "reserved_summary": summary["reserved"],
    }


def format_summary_line(label: str, stats: Dict[str, float]) -> str:
    return (
        f"{label}: avg={stats['avg']:.2f} MB, "
        f"min={stats['min']:.2f} MB, max={stats['max']:.2f} MB, std={stats['std']:.2f} MB"
    )


def save_results(results: List[Dict], args) -> str:
    out_dir = "/share/project/wangjiahao/LAR/evaluations/gpumem"
    os.makedirs(out_dir, exist_ok=True)
    result_path = os.path.join(out_dir, args.output_filename)

    with open(result_path, "w") as f:
        f.write("GPU Memory Usage Comparison - Multiple Models\n")
        f.write("============================================\n\n")
        f.write(f"Number of inference steps: {args.num_pred_steps}\n")
        f.write(f"Number of diffusion steps: {args.num_diff_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Prompt size: {args.prompt_size}\n")
        f.write(f"Sample size: {args.sample_size}\n")
        f.write(f"VAE batch size: {args.vae_batch_size}\n")
        f.write(f"Runs per model: {args.num_runs}\n\n")

        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write("-" * (len(result["model_name"]) + 7) + "\n")

            f.write("Peak allocated memory per run (MB):\n")
            for i, value in enumerate(result["allocated_runs"]):
                f.write(f"  Run {i + 1}: {value:.2f} MB\n")
            f.write("Peak reserved memory per run (MB):\n")
            for i, value in enumerate(result["reserved_runs"]):
                f.write(f"  Run {i + 1}: {value:.2f} MB\n")

            f.write("\nSummary:\n")
            f.write(f"  {format_summary_line('Allocated', result['allocated_summary'])}\n")
            f.write(f"  {format_summary_line('Reserved ', result['reserved_summary'])}\n\n")

        f.write("Comparison Summary:\n")
        f.write("===================\n")
        for result in results:
            f.write(f"{result['model_name']}:\n")
            f.write(f"  {format_summary_line('Allocated', result['allocated_summary'])}\n")
            f.write(f"  {format_summary_line('Reserved ', result['reserved_summary'])}\n\n")

    return result_path


def main():
    args = parse_args()
    prompt_dict = torch.load(args.prompt, weights_only=False)

    cfg_list = args.cfg.split(",") if args.cfg else []
    ckpt_list = args.ckpt.split(",") if args.ckpt else []
    if len(cfg_list) != len(ckpt_list):
        raise ValueError("Number of cfg files must match number of ckpt files")

    model_names = [f"Model_{i + 1}_{os.path.basename(path)}" for i, path in enumerate(ckpt_list)]

    results = []
    for cfg_path, ckpt_path, model_name in zip(cfg_list, ckpt_list, model_names):
        result = run_gpu_mem_test(cfg_path, ckpt_path, args, model_name, prompt_dict)
        if result:
            results.append(result)

    if not results:
        print("No results to save (possibly running on non-root ranks).")
        return

    result_path = save_results(results, args)
    print(f"GPU memory comparison complete. Results saved to {result_path}")

    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  {format_summary_line('Allocated', result['allocated_summary'])}")
        print(f"  {format_summary_line('Reserved ', result['reserved_summary'])}")


if __name__ == "__main__":
    main()


