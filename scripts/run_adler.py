import yaml
import pyrallis
from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure
import os
from glob import glob
import traceback
from loguru import logger

adler_assets = glob("./shapes/adler/*.obj")
exp_root = "experiments"
concepts = [
    "modern, black and white",
    "modern, iron, black",
    "Nordic, beige, wooden",
    "Nordic, white, stainless steel",
    "vintage, dark, grey, brick",
    "classic, antique, brown, wood",
    "unique, vivid color, pink, fluorescent color",
]
sd_models = [
    "stabilityai/stable-diffusion-2-depth",
    "Linaqruf/anything-v3.0",
    "nitrosocke/Arcane-Diffusion",
    "SG161222/Realistic_Vision_V2.0",
]

for sd_model in sd_models:
    for concept in concepts:
        exp_path = f"{exp_root}/{os.path.basename(sd_model)}/{concept}/"
        for asset in adler_assets:
            try:
                prompt = f"{os.path.basename(asset).split('_')[0].replace('.obj', '')}, {'{}'} view, {concept}"
                if os.path.basename(sd_model) == "Arcane-Diffusion":
                    prompt += ", arcane style, league of legends"
                config_dict = {
                    "log": {
                        "exp_root": exp_path,
                        "exp_name": f"{os.path.basename(asset)}",
                    },
                    "guide": {
                        "text": prompt,
                        "diffusion_name": sd_model,
                        "append_direction": True,
                        "shape_path": asset,
                    },
                    "optim": {"seed": 3},
                }

                if not os.path.exists(
                    f"{exp_path}/{os.path.basename(asset)}/results/step_00010_rgb.mp4"
                ):
                    config_path = "configs/text_guided/tmp.yaml"
                    with open("configs/text_guided/tmp.yaml", "w") as f:
                        yaml.dump(config_dict, f)

                    os.system(
                        "python -m scripts.run_texture --config_path=configs/text_guided/tmp.yaml"
                    )
                else:
                    logger.info(
                        f"already exists: {exp_path}/{os.path.basename(asset)}"
                    )

            except Exception as e:
                logger.info(traceback.format_exc())
