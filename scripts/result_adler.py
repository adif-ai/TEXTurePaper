import yaml
import pyrallis
from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure
import os
from glob import glob
import traceback
from loguru import logger
import shutil

adler_assets = glob("./shapes/adler/*.obj")
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
        for asset in adler_assets:
            dir_name = os.path.join(
                f"experiments/{os.path.basename(sd_model)}/{concept}/",
                f"{os.path.basename(asset)}",
            )
            save_dir_name = os.path.join(
                f"results/{os.path.basename(sd_model)}/{concept}/",
                f"{os.path.basename(asset)}",
            )

            os.makedirs(save_dir_name, exist_ok=True)

            shutil.copytree(
                os.path.join(dir_name, "mesh"), os.path.join(save_dir_name, "mesh")
            )
            shutil.copy2(
                os.path.join(dir_name, "results", "step_00010_rgb.mp4"),
                os.path.join(save_dir_name, "result.mp4"),
            )
