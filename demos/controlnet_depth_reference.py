# !pip install transformers accelerate
from diffusers import ControlNetModel, EulerAncestralDiscreteScheduler
from src.stable_diffusion.controlnet_reference import (
    StableDiffusionControlNetReferencePipeline,
)
from diffusers.utils import load_image
import numpy as np
import torch
from controlnet_aux import MidasDetector


init_image = load_image("demos/png/table_depth.jpg")
init_image = init_image.resize((512, 512))

generator = torch.Generator(device="cpu").manual_seed(4)

midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = midas(init_image)

control_image = depth_image

ref_image = load_image("demos/png/travertine.png").resize((512, 512))

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
    "SG161222/Realistic_Vision_V2.0",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    ref_image=ref_image,
    prompt="A photo of travertine table",
    num_inference_steps=20,
    generator=generator,
    image=control_image,
    reference_adain=False,
    style_fidelity=0.5,
).images[0]

image.save("demos/png/reference_depth.png")
