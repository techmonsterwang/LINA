# 1. text 2 img
| Model       | Parameters | Resolution | Data |  Weight                                                               | GenEval | DPGBench |
|:-----------:|:----------:|:----------:|:----:|:---------------------------------------------------------------------:|:--------:|:-------:|
| NOVA-0.6B   | 0.6B       | 512x512    | 16M  | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-sd512)          | 0.75   |   81.76   |
| NOVA-0.3B   | 0.3B       | 1024x1024  | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w768-sdxl1024)        | 0.67   |   80.60   |
| NOVA-0.6B   | 0.6B       | 1024x1024  | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-sdxl1024)       | 0.69   |   82.25   |
| NOVA-1.4B   | 1.4B       | 1024x1024  | 600M | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1536-sdxl1024)       | 0.71   |   83.01   |


# 2. text 2 video
| Model       | Parameters  | Resolution | Data | Weight                                                                | VBench |
|:-----------:|:-----------:|:----------:|:----:|-----------------------------------------------------------------------|:------:|
| NOVA-0.6B   | 0.6B        | 33x768x480 | 20M  | [🤗 HF link](https://huggingface.co/BAAI/nova-d48w1024-osp480)        |  80.12  |