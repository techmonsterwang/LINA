import torch
from diffnext.pipelines import NOVAPipeline

model_id = "BAAI/nova-d48w1536-sdxl1024"
model_args = {"torch_dtype": torch.float16, "trust_remote_code": True}
pipe = NOVAPipeline.from_pretrained(model_id, **model_args)
pipe = pipe.to("cuda")

prompt = "a girl named shuaizhen."
image = pipe(prompt).images[0]
    
image.save("images_sample/shuaizhen.jpg")


prompt = "cs swot."
image = pipe(prompt).images[0]
    
image.save("images_sample/cs_swot.jpg")

prompt = "what is a sana dog."
image = pipe(prompt).images[0]
    
image.save("images_sample/sana_dog.jpg")