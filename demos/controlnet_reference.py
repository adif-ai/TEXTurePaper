import cv2
import torch
import numpy as np
from PIL import Image
from diffusers import (
    ControlNetModel,
)
from diffusers import UniPCMultistepScheduler
from diffusers.utils import load_image
from src.stable_diffusion.controlnet_reference import (
    StableDiffusionControlNetReferencePipeline,
)

input_image = load_image("demos/png/any_girl.png")

# get canny image
image = cv2.Canny(np.array(input_image), 100, 200)
image = image[:, :, None]
image = np.concatenate([image, image, image], axis=2)
canny_image = Image.fromarray(image)

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16
)
pipe = StableDiffusionControlNetReferencePipeline.from_pretrained(
    "andite/anything-v4.0",
    controlnet=controlnet,
    safety_checker=None,
    torch_dtype=torch.float16,
).to("cuda:0")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

result_img = pipe(
    ref_image=input_image,
    prompt="1girl, masterpiece, best quality, 4k",
    negative_prompt="lowres, bad anatomy, bad hands, text, error, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, jpeg artifacts, signature, watermark, username, blurry",
    image=canny_image,
    num_inference_steps=20,
    reference_attn=True,
    reference_adain=False,
).images[0]

result_img.save("demos/png/controlnet_reference.png")
