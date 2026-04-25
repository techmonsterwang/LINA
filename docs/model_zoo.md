# 1. text 2 image
| Model       | Parameters | Resolution |   Weight                                                               | GenEval | 
|:-----------:|:----------:|:----------:|:---------------------------------------------------------------------:|:--------:|
| LINA-1.4B   | 1.4B       | 512x512    | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-t2i-d48w1536-sdxl512)          | 0.74   | 
| LINA-1.5B   | 1.5B       | 1024x1024    | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-t2i-d48w1536-sdxl1024)          | 0.72   |


# 2. class 2 image
| Model       | Parameters  | Resolution | Data | Weight                                                                | FID |
|:-----------:|:-----------:|:----------:|:----:|-----------------------------------------------------------------------|:------:|
| LINA-1.4B   | 1.4B        | 256x256 | ImageNet  | [🤗 HF link](https://huggingface.co/techmonsterwang/LINA-c2i-d48w1536-marvae)        |  2.18  |