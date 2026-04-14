# Evaluations

## GenEval

### 1. Generate prompt embeddings
```python
import json, torch
from transformers import CodeGenTokenizerFast
from diffnext.models.text_encoders.phi import PhiEncoderModel

model_path = "/path/to/nova-d48w1024-sdxl1024"
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
--prompt ./evaluations/geneval/prompts.pth \
--ckpt /path/to/nova-d48w1024-sdxl1024 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 16 --sample_size 4 \
--outdir ./evaluations/geneval/nova-d48w1024-sdxl1024-cfg7
```

### 3. Evaluation
<IMAGE_FOLDER>=./evaluations/geneval/nova-d48w1024-sdxl1024-cfg7

Please refer [GenEval](https://github.com/djghosh13/geneval?tab=readme-ov-file#evaluation) evaluation guide.

## VBench

### 1. Generate prompt embeddings
```python
import json, torch
from transformers import CodeGenTokenizerFast
from diffnext.models.text_encoders.phi import PhiEncoderModel

model_path = "/path/to/nova-d48w1024-osp480"
device, dtype = torch.device("cuda", 0), torch.float16

tokenizer = CodeGenTokenizerFast.from_pretrained(model_path + "/tokenizer")
model = PhiEncoderModel.from_pretrained(model_path + "/text_encoder", torch_dtype=dtype)
model = model.eval().to(device=device)

coll_embeds, tags, texts = [[], []], [], []
for data in json.load(open("./evaluations/vbench/prompts.json")):
    for i, prompt in enumerate((data["prompt"], data["dense_prompt"])):
        input_ids = tokenizer(prompt, max_length=256, truncation=True).input_ids
        input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.int64)
        with torch.no_grad():
            coll_embeds[i].append(model(input_ids.unsqueeze_(0)).last_hidden_state[0].cpu())
        tags.append(data["tag"]), texts.append(data["prompt"])
torch.save({"prompts": coll_embeds[0], "tags": tags, "texts": texts}, "./evaluations/vbench/prompts.pth")
torch.save({"prompts": coll_embeds[1], "tags": tags, "texts": texts}, "./evaluations/vbench/prompts_rewrite.pth")
```

### 2. Sample prompt videos
```bash
python ./evaluations/vbench/sample.py \
--prompt ./evaluations/vbench/prompts.pth \
--ckpt /path/to/nova-d48w1024-osp480 \
--num_pred_steps 128 --guidance_scale 7 --prompt_size 8 --sample_size 5 --max_latent_length 9 --flow 5 \
--outdir ./evaluations/vbench/nova-d48w1024-osp480-cfg7-flow5
```

### 3. Evaluation
<VIDEO_FOLDER>=./evaluations/vbench/nova-d48w1024-osp480-cfg7-flow5

Please refer [VBench](https://github.com/Vchitect/VBench?tab=readme-ov-file#evaluation-on-the-standard-prompt-suite-of-vbench) evaluation guide.
