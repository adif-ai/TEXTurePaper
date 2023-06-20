import yaml
import os
from glob import glob
import traceback
from loguru import logger


assets = glob("./shapes/ai/*.obj")
exp_root = "experiments"
ref_images = glob("references/*.png")
sd_models = [
    "v1-5-pruned-emaonly.safetensors",
]
default_config_dict = {
    "log": {},
    "guide": {
        "use_refine": True,
        "use_dilation": False,
        "use_checkerboard": False,
        "append_direction": True,
        "inpainting_fill": 1,
    },
    "optim": {"seed": 3, "steps": 20},
}


# TEXTure with ref image + txt
for sd_model in sd_models:
    for ref_image in ref_images:
        concept = os.path.basename(ref_image).replace(".png", "")
        exp_path = f"{exp_root}/{os.path.basename(sd_model)}/{concept}_reference/"
        for asset in assets:
            try:
                prompt = f"{os.path.basename(asset).split('_')[0].replace('.obj', '')}, {'{}'} view, {concept}"

                config_dict = default_config_dict.copy()
                config_dict["log"].update(
                    {"exp_root": exp_path, "exp_name": f"{os.path.basename(asset)}"}
                )
                config_dict["guide"].update(
                    {
                        "text": prompt,
                        "diffusion_name": sd_model,
                        "shape_path": asset,
                        "reference_image_path": ref_image,
                    }
                )

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
                    logger.info(f"already exists: {exp_path}/{os.path.basename(asset)}")

            except Exception as e:
                logger.info(traceback.format_exc())

# TEXTure with txt
concepts = [os.path.basename(i).replace(".png", "") for i in ref_images] + [
    "bubble",
    "Mars",
    "mantle",
    "fish scales",
]
for sd_model in sd_models:
    for concept in concepts:
        exp_path = f"{exp_root}/{os.path.basename(sd_model)}/{concept}/"
        for asset in assets:
            try:
                prompt = f"{os.path.basename(asset).split('_')[0].replace('.obj', '')}, {'{}'} view, {concept}"
                config_dict = default_config_dict.copy()
                config_dict["log"].update(
                    {"exp_root": exp_path, "exp_name": f"{os.path.basename(asset)}"}
                )
                config_dict["guide"].update(
                    {
                        "text": prompt,
                        "diffusion_name": sd_model,
                        "shape_path": asset,
                    }
                )

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
                    logger.info(f"already exists: {exp_path}/{os.path.basename(asset)}")

            except Exception as e:
                logger.info(traceback.format_exc())

# # combine parts
# asset_parts ={"sofa_001.obj": ["leg", "small_cushion", "cushion"],
#               "Chair_002.obj": ["frame", "cushion"],
#               "Table_002.obj": ["leg", "top"]
#               }

# additional_concepts = ["bubble", "Mars", "mantle", "fish scales"]

# reference = True
# print()
# for asset in assets:
#     all_concepts = []
#     concepts = [os.path.basename(i).replace("*.png", "") for i in ref_images]
#     concepts += additional_concepts
#     for sd_model in sd_models:
#         for concept in concepts:
#             # txt + reference image
#             exp_path = f"{exp_root}/{os.path.basename(sd_model)}/{concept}_reference/"

#             # txt
#             exp_path = f"{exp_root}/{os.path.basename(sd_model)}/{concept}/"
