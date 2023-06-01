# !pip install transformers accelerate

from diffusers.utils import load_image
import numpy as np
import torch
from diffusers import UniPCMultistepScheduler
from src.stable_diffusion.reference import StableDiffusionReferencePipeline


pipe = StableDiffusionReferencePipeline.from_pretrained(
    "andite/anything-v4.0", safety_checker=None, torch_dtype=torch.float16
).to("cuda:0")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

input_image = load_image("demos/png/any_girl.png")


image = pipe(
    ref_image=input_image,
    prompt="1girl, masterpiece, best quality",
    negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    num_inference_steps=20,
    reference_attn=True,
    reference_adain=False,
    style_fidelity=0.5,  # you can set style_fidelity=1.0
).images[0]

image.save("demos/png/reference_only.png")
