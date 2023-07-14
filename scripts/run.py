import pyrallis
import os
from loguru import logger
import traceback
import datetime
import yaml


def main(
    obj_path,  # mtl_path
    save_name=str(datetime.datetime.now()).replace(" ", "-"),
    save_root="results",
    log_images=False,
    text="",
    negative_text="",
    reference_image_path=None,
    reference_image_repeat=1,
    diffusion_name="v1-5-pruned-emaonly.safetensors",
    upscale=False,
    image_resolution=512,
    texture_resolution=2048,
    seed=0,
):
    try:
        if not os.path.exists(os.path.join(save_root, save_name, "mesh", "albedo.png")):
            config_dict = {
                "log": {
                    "exp_root": save_root,
                    "exp_name": save_name,
                    "log_images": log_images,
                },
                "guide": {
                    "diffusion_name": diffusion_name,
                    "text": text,
                    "shape_path": obj_path,
                    "negative_text": negative_text,
                    "reference_image_path": reference_image_path,
                    "reference_image_repeat": reference_image_repeat,
                    "upscale": upscale,
                    "image_resolution": image_resolution,
                    "texture_resolution": texture_resolution,
                },
                "optim": {"seed": seed},
            }
            config_path = os.path.join(save_root, save_name, "run.yaml")
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                yaml.dump(config_dict, f)

            os.system(f"python -m scripts.run_texture --config_path={config_path}")
            os.remove(config_path)
        else:
            logger.info(
                f'already exists: {os.path.join(save_root, save_name, "mesh", "albedo.png")}'
            )

    except Exception as e:
        logger.info(traceback.format_exc())


if __name__ == "__main__":
    main(obj_path="shapes/ai/sofa_001.obj", text="leather sofa, brown")
