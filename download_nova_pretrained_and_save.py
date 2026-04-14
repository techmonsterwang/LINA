import torch
from diffnext.pipelines import NOVAPipeline

model_id = "BAAI/nova-d48w1536-sdxl1024"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "a shiba inu wearing a beret and black turtleneck."
image = pipe(prompt).images[0]
    
image.save("shiba_inu.jpg")

save_path = "/pretrained/nova-d48w1536-sdxl1024"
pipe.save_pretrained(save_path)
