<div align="center">

<h1>LINA: Linear Autoregressive Image Generative Models with Continuous Tokens</h1>

<p align="center">
<a href="https://arxiv.org/abs/2601.22630"><img src="https://img.shields.io/badge/ArXiv-2601.22630-%23840707.svg" alt="ArXiv"></a>
<a href="https://huggingface.co/spaces/techmonsterwang/LINA-t2i-d48w1536-sdxl1024"><img src="https://img.shields.io/badge/🤗 Demo-T2I-%26840707.svg" alt="T2IDemo"></a>
<!-- <a href="https://huggingface.co/spaces/BAAI/nova-d48w1024-osp480"><img src="https://img.shields.io/badge/🤗 Demo-T2V-%26840707.svg" alt="T2VDemo"></a>
<a href="http://bitterdhg.github.io/NOVA_page"><img src="https://img.shields.io/badge/Webpage-NOVA-%237CB4F7.svg" alt="Webpage"></a> -->
</p>

[Jiahao Wang](https://scholar.google.com/citations?user=QjVR3UUAAAAJ&hl=zh-CN)<sup>1</sup>, [Ting Pan](https://scholar.google.com/citations?&user=qQv6YbsAAAAJ)<sup>2,4</sup>, [Haoge Deng](https://scholar.google.com/citations?user=S2sbvjgAAAAJ&hl=zh-CN&oi=ao)<sup>4</sup>, [Dongchen Han](https://scholar.google.com/citations?user=wv3U3tkAAAAJ&hl=zh-CN)<sup>3</sup>, [Taiqiang Wu](https://scholar.google.com/citations?user=mCtvn50AAAAJ&hl=zh-CN)<sup>1</sup><br>
[Xinlong Wang](https://scholar.google.com/citations?user=DPz0DjYAAAAJ&hl=zh-CN)<sup>4</sup>, [Ping Luo](https://scholar.google.com/citations?user=aXdjxb4AAAAJ&hl=en)<sup>1†</sup><br>

[HKU](https://www.hku.hk/)<sup>1</sup>, [ICT-CAS](http://english.ict.cas.cn)<sup>2</sup>, [THU](https://www.tsinghua.edu.cn/en/)<sup>3</sup>, [BAAI](https://www.baai.ac.cn/english.html)<sup>4</sup><br>
<sup>†</sup> Corresponding Author
<br><br><image src="assets/generated_images_lina.png"/>
<br><br><image src="assets/picture_lina.png"/>
</div>

We propose **LINA** (**LI**near **N**on-Quantized **A**utoregressive Model), a simple and clear autoregressive text-to-image baseline based purely on linear attention. We provide two key insights. First, for better *scaling behavior with model parameters*, we recommend using division-based normalization in linear attention, together with depthwise convolution to enhance local modeling. Second, we introduce a *KV gate* mechanism, which brings gating into bidirectional linear attention. **LINA** is thoroughly validated on both class-to-image and text-to-image generation.


## 🚀News
- ```[Apr 2026]``` Released [Sample Guide](./docs/sample.md).
- ```[Apr 2026]``` Released [Evaluation Guide](./docs/evaluation.md).
- ```[Apr 2026]``` Released [Training Guide](./docs/training.md).
<!-- - ```[Jan 2025]``` Accepted by ICLR 2025 ([OpenReview Page](https://openreview.net/forum?id=JE9tCwe3lp)). -->
- ```[Apr 2026]``` Released 🤗 Pretrained Weights on Hugging Face [<a href="https://huggingface.co/spaces/techmonsterwang/LINA-t2i-d48w1536-sdxl1024"><b>T2I (1024px)</b></a>, <a href="https://huggingface.co/spaces/techmonsterwang/LINA-t2i-d48w1536-sdxl512"><b>T2I (512px)</b></a>, <a href="https://huggingface.co/spaces/techmonsterwang/LINA-c2i-d48w1536-marvae"><b>C2I</b></a>]
- ```[Apr 2026]``` Released [weights](#model-zoo), and [Quick Start](#2-quick-start) guide.
- ```[Jan 2026]``` Released [paper](https://arxiv.org/abs/2601.22630), [weights](#model-zoo), and [Quick Start](#2-quick-start) guide.
- ```[Oct 2025]``` Released 🐻 [URSA](https://github.com/baaivision/URSA), a new video generation model developed by several LINA authors together with other collaborators.

## ✨Hightlights

- 🔥 **Systematic Exploration of Linear Attention for Generative Models**: Practical insights into linear attention, including normalization paradigms, architectural design, and gating mechanisms.
- 🔥 **Highly Competetive Performance**: A clear baseline with competitive t2i/c2i results.
- 🔥 **Linear Complexity**: Generative autoregressive transformer using pure linear attention.

## 🗄️Model Zoo
<a id="model-zoo"></a>
> See detailed description in [Model Zoo](./docs/model_zoo.md)

### Text to Image
<a id="text-to-image-weight"></a>

| Model       | Parameters | Resolution |   Weight                                                               | GenEval | 
|:-----------:|:----------:|:----------:|:---------------------------------------------------------------------:|:--------:|
| LINA-1.4B   | 1.4B       | 512x512    | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-t2i-d48w1536-sdxl512)          | 0.74   | 
| LINA-1.5B   | 1.5B       | 1024x1024    | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-t2i-d48w1536-sdxl1024)          | 0.72   |


### Class to Image
<a id="class-to-image-weight"></a>

| Model       | Parameters  | Resolution | Data | Weight                                                                | FID |
|:-----------:|:-----------:|:----------:|:----:|-----------------------------------------------------------------------|:------:|
| LINA-1.4B   | 1.4B        | 256x256 | ImageNet  | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-c2i-d48w1536-marvae)        |  2.18  |

## 📖Table of Contents
- [1. Installation](#1-installation)
  - [1.1 From Source](#from-source)
  - [1.2 From Git](#from-git)
- [2. Quick Start (Text to Image)](#2-quick-start)
  <!-- - [2.1 Text to Image](#text-to-image-quickstart) -->
- [3. Train](#3-train)
- [4. Evaluation](#4-evaluation)
- [5. Sample](#5-sample)

## 1. Installation
### 1.1 From Source

<a id="from-source"></a>
Clone this repository to local disk and install:

```bash
pip install diffusers transformers accelerate imageio[ffmpeg]
git clone https://github.com/techmonsterwang/LINA.git
cd LINA && pip install .
```

### 1.2 From Git
<a id="from-git"></a>

You can also install from the remote repository **if you have set your Github SSH key**: 

```bash
pip install diffusers transformers accelerate imageio[ffmpeg]
pip install git+ssh://git@github.com/techmonsterwang/LINA.git
```

## 2. Quick Start (Text to Image)
<!-- ### 2.1 Text to Image
<a id="text-to-image-quickstart"></a> -->

```python
import torch
from diffnext.pipelines import NOVAPipeline

model_id = "techmonsterwang/LINA-t2i-d48w1536-sdxl1024"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "a shiba inu wearing a beret and black turtleneck."
image = pipe(prompt).images[0]
    
image.save("shiba_inu.jpg")
```


## 3. Train
- See [Training Guide](./docs/training.md)

## 4. Evaluation
- See [Evaluation Guide](./docs/evaluation.md)

## 5. Sample
- See [Sample Guide](./docs/sample.md)

## 📋Todo List
- [X] [Model zoo](#model-zoo)
- [X] [Quick Start](#2-quick-start)
- [X] [Training guide](#3-train)
- [X] [Evaluation guide](#4-evaluation)
- [X] [Sample guide](#5-sample)


## Citation
If you find this repository useful, please consider giving a star ⭐ and citation 🦖:
```
@article{wang2026lina,
  title={LINA: Linear Autoregressive Image Generative Models with Continuous Tokens},
  author={Wang, Jiahao and Pan, Ting and Deng, Haoge and Han, Dongchen and Wu, Taiqiang and Wang, Xinlong and Luo, Ping},
  journal={arXiv preprint arXiv:2601.22630},
  year={2026}
}
```

## Acknowledgement

We thank the repositories: [NOVA](https://github.com/baaivision/nova), [Flatten-Transformer](https://github.com/LeapLabTHU/Flatten-Transformer), [InLine](https://github.com/LeapLabTHU/InLine), [MAE](https://github.com/facebookresearch/mae), [MAR](https://github.com/LTH14/mar), [MaskGIT](https://github.com/google-research/maskgit), [DiT](https://github.com/facebookresearch/DiT), [FLUX](https://github.com/black-forest-labs/flux) and [CodeWithGPU](https://github.com/seetacloud/codewithgpu).
## License
Code and models are licensed under [Apache License 2.0](LICENSE).
