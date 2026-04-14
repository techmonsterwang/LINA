import json, torch
import os
import sys
sys.path.append(os.getcwd())

from transformers import CodeGenTokenizerFast
from diffnext.models.text_encoders.phi import PhiEncoderModel

model_path = "/share/project/panting/mm_ckpt/nova_shared"
device, dtype = torch.device("cuda", 0), torch.float16

tokenizer = CodeGenTokenizerFast.from_pretrained(model_path + "/tokenizer/phi2")
model = PhiEncoderModel.from_pretrained(model_path + "/text_encoder/phi2", torch_dtype=dtype)
model = model.eval().to(device=device)

coll_embeds = [[], []]
for data in json.load(open("/share/project/wangjiahao/LAR/evaluations/geneval/prompts.json")):
    for i, prompt in enumerate((data["prompt"], data["dense_prompt"])):
        input_ids = tokenizer(prompt, max_length=256, truncation=True).input_ids
        input_ids = torch.as_tensor(input_ids, device=device, dtype=torch.int64)
        with torch.no_grad():
            coll_embeds[i].append(model(input_ids.unsqueeze_(0)).last_hidden_state[0].cpu())
torch.save({"prompts": coll_embeds[0]}, "/share/project/wangjiahao/LAR/evaluations/geneval/prompts.pth")
torch.save({"prompts": coll_embeds[1]}, "/share/project/wangjiahao/LAR/evaluations/geneval/prompts_rewrite.pth")