import yaml
import pyrallis
from src.configs.train_config import TrainConfig
from src.training.trainer import TEXTure
import os
from glob import glob
import traceback
from loguru import logger

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
            try:
                config_dict = {
                    "log": {
                        "exp_root": f"experiments/{os.path.basename(sd_model)}/{concept}/",
                        "exp_name": f"{os.path.basename(asset)}",
                    },
                    "guide": {
                        "text": f"{os.path.basename(asset).split('_')[0].replace('.obj', '')}, {'{}'} view, {concept}",
                        "diffusion_name": sd_model,
                        "append_direction": True,
                        "shape_path": asset,
                    },
                    "optim": {"seed": 3},
                }

                config_path = "configs/text_guided/tmp.yaml"
                with open("configs/text_guided/tmp.yaml", "w") as f:
                    yaml.dump(config_dict, f)

                cfg = pyrallis.parse(config_class=TrainConfig, config_path=config_path)

                trainer = TEXTure(cfg)
                trainer.paint()
            except Exception as e:
                logger.info(traceback.format_exc())
