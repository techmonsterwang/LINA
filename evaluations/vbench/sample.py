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
"""Sample VBench videos."""

import argparse
import collections
import os

import imageio
import tqdm
import torch

from diffnext.pipelines.builder import build_pipeline, get_pipeline_path


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="sample vbench videos")
    parser.add_argument("--ckpt", type=str, default=None, help="checkpoint file")
    parser.add_argument("--prompt", type=str, default="", help="prompt pth file")
    parser.add_argument("--max_latent_length", type=int, default=9, help="max latent length")
    parser.add_argument("--num_pred_steps", type=int, default=128, help="inference steps")
    parser.add_argument("--num_diff_steps", type=int, default=25, help="diffusion steps")
    parser.add_argument("--guidance_scale", type=float, default=7, help="guidance scale")
    parser.add_argument("--flow", default=5, type=float, help="motion flow")
    parser.add_argument("--prompt_size", type=int, default=8, help="prompt size for each batch")
    parser.add_argument("--sample_size", type=int, default=5, help="sample size for each prompt")
    parser.add_argument("--vae_batch_size", type=int, default=1, help="vae batch size")
    parser.add_argument("--outdir", type=str, default="", help="write to")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    prompt_dict = torch.load(args.prompt, weights_only=False)

    num_pred_steps, num_diff_steps = args.num_pred_steps, args.num_diff_steps
    gen_args = {"num_inference_steps": num_pred_steps, "num_diffusion_steps": num_diff_steps}
    vid_args = {"guidance_scale": args.guidance_scale, "output_type": "np"}
    vid_args["vae_batch_size"] = args.vae_batch_size
    vid_args["max_latent_length"] = args.max_latent_length

    rank, world_size = 0, 1
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device), torch.manual_seed(1337)
    generator = torch.Generator(device).manual_seed(1337)
    is_root = rank == 0
    gen_args.update({"generator": generator, "disable_progress_bar": True})
    pipe_path = get_pipeline_path(args.ckpt, {"text_encoder": ""})
    pipe = build_pipeline(pipe_path, "nova", precison="float16").to(device=device)

    grids, prompt_inds = (args.prompt_size, args.sample_size), []
    prompts, tags, texts = prompt_dict["prompts"], prompt_dict["tags"], prompt_dict["texts"]
    rank_prompt_inds = list(range(len(prompts)))[slice(rank, None, world_size)]

    for i, idx in enumerate(tqdm.tqdm(rank_prompt_inds, disable=True)):
        prompt_inds.append(idx)
        if len(prompt_inds) != grids[0] and i != len(rank_prompt_inds) - 1:
            continue
        batch_names = sum(
            [[os.path.join(args.outdir, tags[i], texts[i])] * grids[1] for i in prompt_inds], []
        )
        batch_prompts = sum([[prompts[i]] * grids[1] for i in prompt_inds], [])
        outputs = pipe(prompt_embeds=batch_prompts, motion_flow=args.flow, **vid_args, **gen_args)
        batch_frames = outputs["frames"]
        name_cnt = collections.defaultdict(int)
        for j, frames in enumerate(batch_frames):
            name = batch_names[j].replace(".mp4", "-{}.mp4".format(name_cnt[batch_names[j]]))
            name_cnt[batch_names[j]] += 1
            with imageio.get_writer(name, fps=12, ffmpeg_log_level="error") as writer:
                [writer.append_data(frames[k]) for k in range(frames.shape[0])]
        prompt_inds = []
