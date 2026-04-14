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
"""NOVA T2V application."""

import argparse
import os

import gradio as gr
import numpy as np
import PIL.Image
import torch

from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_video

# Fix tokenizer fork issue.
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# Switch to the allocator optimized for dynamic shape.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Serve NOVA T2V application")
    parser.add_argument("--model", default="", help="model path")
    parser.add_argument("--device", type=int, default=0, help="device index")
    parser.add_argument("--precision", default="float16", help="compute precision")
    return parser.parse_args()


def crop_image(image, target_h, target_w):
    """Center crop image to target size."""
    h, w = image.height, image.width
    aspect_ratio_target, aspect_ratio = target_w / target_h, w / h
    if aspect_ratio > aspect_ratio_target:
        new_w = int(h * aspect_ratio_target)
        x_start = (w - new_w) // 2
        image = image.crop((x_start, 0, x_start + new_w, h))
    else:
        new_h = int(w / aspect_ratio_target)
        y_start = (h - new_h) // 2
        image = image.crop((0, y_start, w, y_start + new_h))
    return np.array(image.resize((target_w, target_h), PIL.Image.Resampling.BILINEAR))


def generate_video(
    prompt,
    negative_prompt,
    image_prompt,
    motion_flow,
    preset,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    num_diffusion_steps,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate a video."""
    args = locals()
    preset = [p for p in video_presets if p["label"] == preset][0]
    args["max_latent_length"] = preset["#latents"]
    args["image"] = crop_image(image_prompt, preset["h"], preset["w"]) if image_prompt else None
    seed = np.random.randint(2147483647) if randomize_seed else seed
    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    frames = pipe(generator=generator, **args).frames[0]
    return export_to_video(frames, fps=12), seed


title = "Autoregressive Video Generation without Vector Quantization"
abbr = "<strong>NO</strong>n-quantized <strong>V</strong>ideo <strong>A</strong>utoregressive"
header = (
    "<div align='center'>"
    "<h2>Autoregressive Video Generation without Vector Quantization</h2>"
    "<h3><a href='https://arxiv.org/abs/2412.14169' target='_blank' rel='noopener'>[paper]</a>"
    "<a href='https://github.com/baaivision/NOVA' target='_blank' rel='noopener'>[code]</a></h3>"
    "</div>"
)
header2 = f"<div align='center'><h3>🎞️ A {abbr} model for continuous visual generation</h3></div>"

video_presets = [
    {"label": "33x768x480", "w": 768, "h": 480, "#latents": 9},
    {"label": "17x768x480", "w": 768, "h": 480, "#latents": 5},
    {"label": "1x768x480", "w": 768, "h": 480, "#latents": 1},
]


prompts = [
    "Niagara falls with colorful paint instead of water.",
    "Many spotted jellyfish pulsating under water. Their bodies are transparent and glowing in deep ocean.",  # noqa
    "An intense close-up of a soldier’s face, covered in dirt and sweat, his eyes filled with determination as he surveys the battlefield.",  # noqa
    "a close-up shot of a woman standing in a dimly lit room. she is wearing a traditional chinese outfit, which includes a red and gold dress with intricate designs and a matching headpiece. the woman has her hair styled in an updo, adorned with a gold accessory. her makeup is done in a way that accentuates her features, with red lipstick and dark eyeshadow. she is looking directly at the camera with a neutral expression. the room has a rustic feel, with wooden beams and a stone wall visible in the background. the lighting in the room is soft and warm, creating a contrast with the woman's vibrant attire. there are no texts or other objects in the video. the style of the video is a portrait, focusing on the woman and her attire.",  # noqa
    "The camera slowly rotates around a massive stack of vintage televisions that are placed within a large New York museum gallery. Each of the televisions is showing a different program. There are 1950s sci-fi movies with their distinctive visuals, horror movies with their creepy scenes, news broadcasts with moving images and words, static on some screens, and a 1970s sitcom with its characteristic look. The televisions are of various sizes and designs, some with rounded edges and others with more angular shapes. The gallery is well-lit, with light falling on the stack of televisions and highlighting the different programs being shown. There are no people visible in the immediate vicinity, only the stack of televisions and the surrounding gallery space.",  # noqa
]
motion_flows = [5, 5, 5, 5, 5]
videos = ["", "", "", "", ""]
examples = [list(x) for x in zip(prompts, motion_flows)]


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device)
    model_args = {"torch_dtype": getattr(torch, args.precision.lower()), "trust_remote_code": True}
    pipe = NOVAPipeline.from_pretrained(args.model, **model_args).to(device)

    # Application.
    app = gr.Blocks(theme="origin").__enter__()
    container = gr.Column(elem_id="col-container").__enter__()
    _, main_row = gr.Markdown(header), gr.Row().__enter__()

    # Input.
    input_col = gr.Column().__enter__()
    prompt = gr.Text(
        label="Prompt",
        placeholder="Describe the video you want to generate",
        value="Niagara falls with colorful paint instead of water.",
        lines=5,
    )
    negative_prompt = gr.Text(
        label="Negative Prompt",
        placeholder="Describe what you don't want in the video",
        value="",
        lines=1,
    )
    image_prompt = gr.Image(label="Image Prompt (Optional) ", type="pil")
    # fmt: off
    adv_opt = gr.Accordion("Advanced Options", open=False).__enter__()
    seed = gr.Slider(label="Seed", maximum=2147483647, step=1, value=0)
    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
    guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=10.0, step=0.1, value=7.0)
    with gr.Row():
        num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=128, step=1, value=128)  # noqa
        num_diffusion_steps = gr.Slider(label="Diffusion steps", minimum=1, maximum=100, step=1, value=100)  # noqa
    adv_opt.__exit__()
    generate = gr.Button("Generate Video", variant="primary", size="lg")
    input_col.__exit__()

    # Results.
    result_col, _ = gr.Column().__enter__(), gr.Markdown(header2)
    preset = gr.Dropdown([p["label"] for p in video_presets], label="Video Preset", value=video_presets[0]["label"])  # noqa
    motion_flow = gr.Slider(label="Motion Flow", minimum=1, maximum=10, step=1, value=5)
    result = gr.Video(label="Result", show_label=False, autoplay=True)
    result_col.__exit__(), main_row.__exit__()
    # fmt: on

    # Examples.
    with gr.Row():
        gr.Examples(examples=examples, inputs=[prompt, motion_flow])

    # Events.
    container.__exit__()
    gr.on(
        triggers=[generate.click, prompt.submit, negative_prompt.submit],
        fn=generate_video,
        inputs=[
            prompt,
            negative_prompt,
            image_prompt,
            motion_flow,
            preset,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
            num_diffusion_steps,
        ],
        outputs=[result, seed],
    )
    app.__exit__(), app.launch(share=False)
