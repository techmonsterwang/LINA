# Evaluations

## GenEval

### 1. Generate prompt embeddings
```python
import json, torch
from transformers import CodeGenTokenizerFast
from diffnext.models.text_encoders.phi import PhiEncoderModel

model_path = "/path/to/lina-t2i-d48w1536-sdxl1024"
device, dtype = torch.device("cuda", 0), torch.float16

tokenizer = CodeGenTokenizerFast.from_pretrained(model_path + "/tokenizer")
model = PhiEncoderModel.from_pretrained(model_path + "/text_encoder", torch_dtype=dtype)
model = model.eval().to(device=device)

coll_embeds = [[], []]
for data in json.load(open("./evaluations/geneval/prompts.json")):
    for i, prompt in enumerate((data["prompt"], data["dense_prompt"])):
        input_ids = tokenizer(prompt, max_length=256, truncation=True).input_ids
        input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.int64)
        with torch.no_grad():
            coll_embeds[i].append(model(input_ids.unsqueeze_(0)).last_hidden_state[0].cpu())
torch.save({"prompts": coll_embeds[0]}, "./evaluations/geneval/prompts.pth")
torch.save({"prompts": coll_embeds[1]}, "./evaluations/geneval/prompts_rewrite.pth")
```

### 2. Sample prompt images
```bash
python ./evaluations/geneval/sample.py \
--metadata ./evaluations/geneval/metadata.jsonl \
--prompt evaluations/geneval/prompts_rewrite.pth \
--cfg ./diffnext/config/t2i/sdxl28m_nova_d48w1536_1024px_finetune_nova_v3.yml \
--ckpt /path/to/lina-t2i-d48w1536-sdxl1024 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 4 \
--outdir ./evaluations/geneval/lina-t2i-d48w1536-sdxl1024-cfg7-rewrite
```

### 3. Evaluation
<IMAGE_FOLDER>=./evaluations/geneval/lina-t2i-d48w1536-sdxl1024-cfg7-rewrite

Please refer [GenEval](https://github.com/djghosh13/geneval?tab=readme-ov-file#evaluation) evaluation guide.


## FID

### 1. Sample images
```bash
export PYTHONPATH=/path/to/LINA
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=4
export PET_NNODES=1
export TRITON_PTXAS_PATH=$(pip show torch | grep -oP 'Location:\s*\K.*' | xargs)/torch/bin/ptxas
BASELINE_ROOT=/path/to/LINA/evaluations/fid50k


torchrun --nproc_per_node 4 sample_bf16.py \
  --cfg ./diffnext/config/imagenet/linear/linear_decouple1_kvscale1_mar1k_nova_d48w1536_256px.yml \
  --ckpt /path/to/LINA-c2i-d48w1536-marvae \
  --num_pred_steps 64 --guidance_scale 2.4 --prompt_size 20 --distributed \
  --outdir ${BASELINE_ROOT}/sample_image/linear_decouple1_kvscale1/vae_mar_1k_w1536_256px/1200k_256px_ar64_cfg2.4 \
  2>&1 | tee /path/to/LINA/evaluations/fid50k/logs/linear_decouple1_kvscale1/sample_w1536/1200k_256px_ar64_cfg2.4.log


```


### 2. Evaluation
```bash
export TF_CPP_MIN_LOG_LEVEL=2
REF=/path/to/VIRTUAL_imagenet256_labeled.npz
SAMPLE=/path/to/LINA/evaluations/fid50k/sample_image/linear_decouple1_kvscale1/vae_mar_1k_w1536_256px/1200k_256px_ar64_cfg2.4.npz

python -u eval.py --ref_batch ${REF} --sample_batch ${SAMPLE} \
2>&1 | tee /path/to/LINA/evaluations/fid50k/logs/linear_decouple1_kvscale1/eval_w1536/$(basename ${SAMPLE%.*}).log


```


Please refer [ADM's TensorFlow evaluation suite](https://github.com/openai/guided-diffusion/tree/main/evaluations) for VIRTUAL_imagenet256_labeled.npz.


