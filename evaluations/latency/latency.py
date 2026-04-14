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
"""Test latency for image generation."""

import argparse
import os
import json
import sys
import time
sys.path.append(os.getcwd())

import torch
import torch.distributed as dist
import PIL
import tqdm

from diffnext.config import cfg
from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Test latency for diffnext model")
    parser.add_argument("--cfg", default=None, help="config file (can be comma-separated for multiple models)")
    parser.add_argument("--metadata", type=str, help="JSONL metadata")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file (can be comma-separated for multiple models)")
    parser.add_argument("--prompt", type=str, default="", help="prompt pth file")
    parser.add_argument("--num_pred_steps", type=int, default=128, help="inference steps")
    parser.add_argument("--num_diff_steps", type=int, default=25, help="diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="guidance scale")
    parser.add_argument("--prompt_size", type=int, default=16, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=4, help="sample size for each prompt")
    parser.add_argument("--vae_batch_size", type=int, default=16, help="vae batch size")
    parser.add_argument("--distributed", action="store_true", help="distributed mode?")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size for latency test")
    parser.add_argument("--num_test_batches", type=int, default=10, help="number of test batches per run")
    parser.add_argument("--num_runs", type=int, default=3, help="number of runs to average latency over")
    parser.add_argument("--output_filename", type=str, default="latency_results.txt", help="output filename for latency results")
    return parser.parse_args()


def test_latency(pipe, prompt_embeds, batch_size, num_test_batches, num_runs, img_args, gen_args):
    """Test latency by generating images in batches and measuring time."""
    latencies = []

    # Create batch of prompts
    batch_prompts = [prompt_embeds] * batch_size

    for run in range(num_runs):
        run_latencies = []
        print(f"Run {run + 1}/{num_runs}")

        for i in tqdm.tqdm(range(num_test_batches)):
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
            start_time = time.time()

            # Generate batch of images
            outputs = pipe(prompt_embeds=batch_prompts, **img_args, **gen_args)
            img = outputs["frames"][:, 0] if "frames" in outputs else outputs["images"]

            torch.cuda.synchronize()  # Ensure generation is complete
            end_time = time.time()

            # Calculate latency per image (total time divided by batch size)
            total_time = end_time - start_time
            latency_per_image = total_time / batch_size
            run_latencies.append(latency_per_image)

        avg_run_latency = sum(run_latencies) / len(run_latencies)
        latencies.append(avg_run_latency)
        print(".4f")

    return latencies


def run_latency_test(cfg_path, ckpt_path, args, model_name, prompt_dict):
    """Run latency test for a single model."""
    # Load config for this model
    cfg.merge_from_file(cfg_path)

    num_pred_steps, num_diff_steps = args.num_pred_steps, args.num_diff_steps
    gen_args = {"num_inference_steps": num_pred_steps, "num_diffusion_steps": num_diff_steps}
    img_args = {"guidance_scale": args.guidance_scale, "output_type": "np"}
    img_args["vae_batch_size"] = args.vae_batch_size

    rank, world_size = 0, 1
    if args.distributed:
        dist.init_process_group(backend="nccl") if not dist.is_initialized() else None
        rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device), torch.manual_seed(1337)
    generator = torch.Generator(device).manual_seed(1337)
    gen_args.update({"generator": generator, "disable_progress_bar": True})
    is_root = device.index == 0

    # Only run latency test on root process
    if not is_root:
        return None

    pipe_path = get_pipeline_path(ckpt_path, {**cfg.PIPELINE.MODULES, "text_encoder": ""})
    pipe = build_pipeline(pipe_path, "nova", precison="bfloat16").to(device=device)

    prompts = prompt_dict["prompts"]
    # Use only the first prompt for latency testing
    test_prompt = prompts[0]

    # Warm up the model
    print(f"Warming up model: {model_name}...")
    _ = pipe(prompt_embeds=[test_prompt], **img_args, **gen_args)

    print(f"Starting latency test for {model_name}...")
    latencies = test_latency(pipe, test_prompt, args.batch_size, args.num_test_batches, args.num_runs, img_args, gen_args)

    # Calculate statistics
    avg_latency = sum(latencies) / len(latencies)
    min_latency = min(latencies)
    max_latency = max(latencies)

    return {
        'model_name': model_name,
        'avg_latency': avg_latency,
        'min_latency': min_latency,
        'max_latency': max_latency,
        'latencies': latencies
    }

if __name__ == "__main__":
    args = parse_args()

    prompt_dict = torch.load(args.prompt, weights_only=False)

    # Parse multiple models
    cfg_list = args.cfg.split(',') if args.cfg else []
    ckpt_list = args.ckpt.split(',') if args.ckpt else []

    if len(cfg_list) != len(ckpt_list):
        raise ValueError("Number of cfg files must match number of ckpt files")

    model_names = []
    for i, ckpt_path in enumerate(ckpt_list):
        model_names.append(f"Model_{i+1}_{os.path.basename(ckpt_path)}")

    # Run tests for all models
    results = []
    for cfg_path, ckpt_path, model_name in zip(cfg_list, ckpt_list, model_names):
        result = run_latency_test(cfg_path, ckpt_path, args, model_name, prompt_dict)
        if result:
            results.append(result)

    # Save results to txt file
    latency_dir = "/share/project/wangjiahao/LAR/evaluations/latency"
    os.makedirs(latency_dir, exist_ok=True)

    result_file = os.path.join(latency_dir, args.output_filename)

    with open(result_file, "w") as f:
        f.write("Latency Test Results - Multiple Models\n")
        f.write("======================================\n\n")
        f.write(f"Number of prediction steps: {args.num_pred_steps}\n")
        f.write(f"Number of diffusion steps: {args.num_diff_steps}\n")
        f.write(f"Guidance scale: {args.guidance_scale}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write(f"Test batches per run: {args.num_test_batches}\n")
        f.write(f"Number of runs: {args.num_runs}\n\n")

        for result in results:
            f.write(f"Model: {result['model_name']}\n")
            f.write("-" * (len(result['model_name']) + 7) + "\n")
            f.write("Latencies per run (seconds):\n")
            for i, latency in enumerate(result['latencies']):
                f.write(f"  Run {i+1}: {latency:.4f}s\n")
            f.write(f"\nAverage latency: {result['avg_latency']:.4f}s\n")
            f.write(f"Min latency: {result['min_latency']:.4f}s\n")
            f.write(f"Max latency: {result['max_latency']:.4f}s\n")
            std_dev = (sum((x - result['avg_latency'])**2 for x in result['latencies']) / len(result['latencies']))**0.5
            f.write(f"Std deviation: {std_dev:.2f}s\n\n")

        # Summary comparison
        f.write("Summary Comparison:\n")
        f.write("==================\n")
        for result in results:
            throughput = 1.0 / result['avg_latency']
            f.write(f"{result['model_name']}: {result['avg_latency']:.4f}s per image, {throughput:.2f} images/second\n")

    print(f"Latency test completed for all models. Results saved to {result_file}")

    # Print summary to console
    for result in results:
        print(f"\n{result['model_name']}:")
        print(f"  Average latency per image: {result['avg_latency']:.4f} seconds")
        throughput = 1.0 / result['avg_latency']
        print(f"  Throughput: {throughput:.2f} images/second")

