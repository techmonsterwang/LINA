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
"""Sample FID-50K images."""

import argparse
import os

import numpy as np
import PIL.Image
import torch
import torch.distributed as dist
import PIL
import tqdm

from diffnext.config import cfg
from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Generate FID-50k samples")
    parser.add_argument("--cfg", default=None, help="config file")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--num_pred_steps", type=int, default=64, help="inference steps")
    parser.add_argument("--num_diff_steps", type=int, default=25, help="diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=3.5, help="number of patches")
    parser.add_argument("--annealing_strategy", type=str, default="linear", help="diffusion steps annealing strategy")
    parser.add_argument("--prompt_size", type=int, default=25, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=50, help="sample size for each prompt")
    parser.add_argument("--vae_batch_size", type=int, default=25, help="vae batch size")
    parser.add_argument("--distributed", action="store_true", help="distrbuted mode?")
    parser.add_argument("--outdir", type=str, default="", help="write to")
    return parser.parse_args()


def create_npz_from_sample_folder(sample_dir, num=50000):
    """Builds a single .npz file from a folder of .png samples."""
    samples = []
    np.random.seed(1337)
    order = np.random.permutation(num)
    # order = np.arange(num)
    for i in tqdm.tqdm(order, desc="Building .npz file from samples"):
        sample_pil = PIL.Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    npz_path = f"{sample_dir}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


if __name__ == "__main__":
    args = parse_args()

    cfg.merge_from_file(args.cfg)

    num_pred_steps, num_diff_steps = args.num_pred_steps, args.num_diff_steps
    annealing_strategy = args.annealing_strategy
    gen_args = {"num_inference_steps": num_pred_steps, "num_diffusion_steps": num_diff_steps, "annealing_strategy": annealing_strategy}
    img_args = {"guidance_scale": args.guidance_scale, "output_type": "np"}
    img_args["min_guidance_scale"], img_args["vae_batch_size"] = 1, args.vae_batch_size

    rank, world_size = 0, 1
    if args.distributed:
        dist.init_process_group(backend="nccl") if not dist.is_initialized() else None
        rank, world_size = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device), torch.manual_seed(1337)
    generator = torch.Generator(device).manual_seed(1337)
    gen_args.update({"generator": generator, "disable_progress_bar": True})
    is_root = device.index == 0

    pipe_path = get_pipeline_path(args.ckpt, {**cfg.PIPELINE.MODULES})
    pipe = build_pipeline(pipe_path, "nova_c2i_disa", precison="bfloat16").to(device=device)

    prompts = list(range(1000))
    os.makedirs(args.outdir, exist_ok=True) if is_root else None

    grids, prompt_inds = (args.prompt_size, args.sample_size), []
    rank_prompt_inds = list(range(len(prompts)))[slice(rank, None, world_size)]
    for i, idx in enumerate(tqdm.tqdm(rank_prompt_inds, disable=not is_root)):
        prompt_inds.append(idx)
        if len(prompt_inds) != grids[0] and i != len(rank_prompt_inds) - 1:
            continue
        batch_prompts = sum([[prompts[i]] * grids[1] for i in prompt_inds], [])
        outputs = pipe(prompt=batch_prompts, **img_args, **gen_args)
        batch_img = outputs["frames"][:, 0] if "frames" in outputs else outputs["images"]
        for i, idx in enumerate(prompt_inds):
            for j in range(grids[1]):
                img_name = "{}.png".format(str(idx * args.sample_size + j).zfill(6))
                pil_img = PIL.Image.fromarray(batch_img[i * grids[1] + j])
                if batch_img.shape[-2] != 256:
                    pil_img = pil_img.resize((256, 256), resample=PIL.Image.Resampling.BILINEAR)
                pil_img.save(os.path.join(args.outdir, img_name))
        prompt_inds = []
    dist.barrier() if world_size > 1 else None
    create_npz_from_sample_folder(args.outdir) if rank == 0 else None
    print(f"Rank {rank}, done.")
    dist.destroy_process_group() if world_size > 1 else None
