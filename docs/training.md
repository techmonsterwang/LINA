# Training Guide
This guide provides simple snippets to train diffnext models.

# 1. Build VAE cache
To optimize training workflow, we preprocess images or videos into VAE latents.

## Requirements:
```bash
pip install protobuf==3.20.3 codewithgpu decord
```

## Build T2I cache
Following snippet can be used to cache image latents:

```python
import os, codewithgpu, torch, PIL.Image, numpy as np
from diffnext.models.autoencoders.autoencoder_kl import AutoencoderKL

device, dtype = torch.device("cuda"), torch.float16
vae = AutoencoderKL.from_pretrained("/path/to/nova-d48w1024-sdxl1024/vae")
vae = vae.to(device=device, dtype=dtype).eval()

features = {"moments": "bytes", "caption": "string", "text": "string", "shape": ["int64"]}
_, writer = os.makedirs("./img_dataset"), codewithgpu.RecordWriter("./img_dataset", features)

img = PIL.Image.open("./assets/sample_image.jpg")
x = torch.as_tensor(np.array(img)[None, ...].transpose(0, 3, 1, 2)).to(device).to(dtype)
with torch.no_grad():
    x = vae.encode(x.sub(127.5).div(127.5)).latent_dist.parameters.cpu().numpy()[0]
example = {"caption": "long caption", "text": "short text"}
writer.write({"shape": x.shape, "moments": x.tobytes(), **example}), writer.close()
```


# 2. Train models

## Train C2I model
Following snippet provides simple C2I training arguments.

```bash
python tools/init.py
```

```bash
rank=$1
zero2_bf16='/path/to/LINA/configs/deepspeed/zero2_bf16.json'

# multi-gpu debug, success
python -u scripts/train_torchrun.py \
--deepspeed ${zero2_bf16} --tensorboard \
--rank ${rank} --world-size 4 --num-gpus 8 \
--exp-dir "exp_dir/linear_decouple1_kvscale1_w1536" \
--master-ip job-884532f3-b984-46d4-a5ba-b81c057676b8-master-0 \
--cfg /path/to/LINA/diffnext/config/imagenet/linear/linear_decouple1_kvscale1_mar1k_nova_d48w1536_256px.yml \
2>&1 | tee /path/to/LINA/logs/train/run_4node_${rank}_mar_imagenet_linear_decouple1_kvscale1_w1536.log

```




## Train T2I model
Following snippet provides simple T2I training arguments.

Download NOVA pretrained weights for initialization:

```bash
python download_nova_pretrained_and_save.py
```

Training 256px T2I model:

```bash
python tools/init_t2i_finetune_nova.py
```

```bash

```




