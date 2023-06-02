# !pip install transformers accelerate
from diffusers import (
    ControlNetModel,
    DDIMScheduler,
)
from src.stable_diffusion.controlnet_reference_inpainting import (
    StableDiffusionControlNetReferenceInpaintPipeline,
)
from diffusers.utils import load_image
import numpy as np
import torch
from controlnet_aux import MidasDetector


init_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy.png"
)
init_image = init_image.resize((512, 512))

generator = torch.Generator(device="cpu").manual_seed(1)

mask_image = load_image(
    "https://huggingface.co/datasets/diffusers/test-arrays/resolve/main/stable_diffusion_inpaint/boy_mask.png"
)
mask_image = mask_image.resize((512, 512))


def make_inpaint_condition(image, image_mask):
    image = np.array(image.convert("RGB")).astype(np.float32) / 255.0
    image_mask = np.array(image_mask.convert("L")).astype(np.float32) / 255.0

    assert (
        image.shape[0:1] == image_mask.shape[0:1]
    ), "image and image_mask must have the same image size"
    image[image_mask > 0.5] = -1.0  # set as masked pixel
    image = np.expand_dims(image, 0).transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return image


inpaint_image = make_inpaint_condition(init_image, mask_image)

midas = MidasDetector.from_pretrained("lllyasviel/Annotators")
depth_image = midas(init_image)

control_image = [inpaint_image, depth_image]

ref_image = load_image("demos/png/iron man 2.png").resize((512, 512))


# control net의 순서는 영향 없음
controlnet = [
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11p_sd15_inpaint", torch_dtype=torch.float16
    ),
    ControlNetModel.from_pretrained(
        "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16
    ),
]

pipe = StableDiffusionControlNetReferenceInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None,
)

# pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

from diffusers import EulerAncestralDiscreteScheduler

pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()

# generate image
image = pipe(
    ref_image=ref_image,
    prompt="a man",
    num_inference_steps=20,
    generator=generator,
    image=init_image,
    mask_image=mask_image,
    control_image=control_image,
).images[0]

image.save("demos/png/inpainting_reference_depth.png")
