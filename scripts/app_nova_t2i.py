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
"""NOVA T2I application."""

import argparse
import os

import gradio as gr
import numpy as np
import torch

from diffnext.pipelines import NOVAPipeline
from diffnext.utils import export_to_image

# Switch to the allocator optimized for dynamic shape.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser(description="Serve NOVA T2I application")
    parser.add_argument("--model", default="", help="model path")
    parser.add_argument("--device", type=int, default=0, help="device index")
    parser.add_argument("--precision", default="float16", help="compute precision")
    return parser.parse_args()


def generate_image4(
    prompt,
    negative_prompt,
    seed,
    randomize_seed,
    guidance_scale,
    num_inference_steps,
    num_diffusion_steps,
    progress=gr.Progress(track_tqdm=True),
):
    """Generate 4 images."""
    args = locals()
    seed = np.random.randint(2147483647) if randomize_seed else seed
    device = getattr(pipe, "_offload_device", pipe.device)
    generator = torch.Generator(device=device).manual_seed(seed)
    images = pipe(generator=generator, num_images_per_prompt=4, **args).images
    return [export_to_image(image, quality=95) for image in images] + [seed]


css = """#col-container {margin: 0 auto; max-width: 1366px}"""
title = "Autoregressive Video Generation without Vector Quantization"
abbr = "<strong>NO</strong>n-quantized <strong>V</strong>ideo <strong>A</strong>utoregressive"
header = (
    "<div align='center'>"
    "<h2>Autoregressive Video Generation without Vector Quantization</h2>"
    "<h3><a href='https://arxiv.org/abs/2412.14169' target='_blank' rel='noopener'>[paper]</a>"
    "<a href='https://github.com/baaivision/NOVA' target='_blank' rel='noopener'>[code]</a></h3>"
    "</div>"
)
header2 = f"<div align='center'><h3>🖼️ A {abbr} model for continuous visual generation</h3></div>"

examples = [
    "a selfie of an old man with a white beard.",
    "a woman with long hair next to a luminescent bird.",
    "a digital artwork of a cat styled in a whimsical fashion. The overall vibe is quirky and artistic.",  # noqa
    "a shiba inu wearing a beret and black turtleneck.",
    "a beautiful afghan women by red hair and green eyes.",
    "beautiful fireworks in the sky with red, white and blue.",
    "A dragon perched majestically on a craggy, smoke-wreathed mountain.",
    "A photo of llama wearing sunglasses standing on the deck of a spaceship with the Earth in the background.",  # noqa
    "Two pandas in fluffy slippers and bathrobes, lazily munching on bamboo.",
]


if __name__ == "__main__":
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", args.device)
    model_args = {"torch_dtype": getattr(torch, args.precision.lower()), "trust_remote_code": True}
    pipe = NOVAPipeline.from_pretrained(args.model, **model_args).to(device)

    # Main Application.
    app = gr.Blocks(css=css, theme="origin").__enter__()
    container = gr.Column(elem_id="col-container").__enter__()
    _, main_row = gr.Markdown(header), gr.Row().__enter__()

    # Input.
    input_col = gr.Column().__enter__()
    prompt = gr.Text(
        label="Prompt",
        placeholder="Describe the video you want to generate",
        value="a shiba inu wearing a beret and black turtleneck.",
        lines=5,
    )
    negative_prompt = gr.Text(
        label="Negative Prompt",
        placeholder="Describe what you don't want in the image",
        value="low quality, deformed, distorted, disfigured, fused fingers, bad anatomy, weird hand",  # noqa
        lines=5,
    )
    # fmt: off
    adv_opt = gr.Accordion("Advanced Options", open=True).__enter__()
    seed = gr.Slider(label="Seed", maximum=2147483647, step=1, value=0)
    randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
    guidance_scale = gr.Slider(label="Guidance scale", minimum=1, maximum=10, step=0.1, value=5)
    with gr.Row():
        num_inference_steps = gr.Slider(label="Inference steps", minimum=1, maximum=128, step=1, value=64) # noqa
        num_diffusion_steps = gr.Slider(label="Diffusion steps", minimum=1, maximum=50, step=1, value=25)  # noqa
    adv_opt.__exit__()
    generate = gr.Button("Generate Image", variant="primary", size="lg")
    input_col.__exit__()
    # fmt: on

    # Results.
    result_col, _ = gr.Column().__enter__(), gr.Markdown(header2)
    with gr.Row():
        result1 = gr.Image(label="Result1", show_label=False)
        result2 = gr.Image(label="Result2", show_label=False)
    with gr.Row():
        result3 = gr.Image(label="Result3", show_label=False)
        result4 = gr.Image(label="Result4", show_label=False)
    result_col.__exit__(), main_row.__exit__()

    # Examples.
    with gr.Row():
        gr.Examples(examples=examples, inputs=[prompt])

    # Events.
    container.__exit__()
    gr.on(
        triggers=[generate.click, prompt.submit, negative_prompt.submit],
        fn=generate_image4,
        inputs=[
            prompt,
            negative_prompt,
            seed,
            randomize_seed,
            guidance_scale,
            num_inference_steps,
            num_diffusion_steps,
        ],
        outputs=[result1, result2, result3, result4, seed],
    )
    app.__exit__(), app.launch(share=False)
