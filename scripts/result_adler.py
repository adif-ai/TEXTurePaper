import os
from glob import glob
import shutil


assets = glob("./shapes/ai/*.obj")
exp_root = "experiments/230622"
ref_images = glob("references/*.png")
sd_models = [
    "v1-5-pruned-emaonly.safetensors",
]
concepts = [os.path.basename(i).replace(".png", "") for i in ref_images] + [
    "bubble",
    "Mars",
    "mantle",
    "fish scales",
]


for sd_model in sd_models:
    for concept in concepts:
        for asset in assets:
            for reference in [True, False]:
                if not reference:
                    dir_name = os.path.join(
                        f"{exp_root}/{os.path.basename(sd_model)}/{concept}/",
                        f"{os.path.basename(asset)}",
                    )

                    save_dir_name = os.path.join(
                        f"results/txt/{concept}/",
                        f"{os.path.basename(asset)}",
                    )

                    if os.path.exists(
                        os.path.join(dir_name, "results", "step_00013_rgb.mp4")
                    ):
                        os.makedirs(save_dir_name, exist_ok=True)

                        shutil.copytree(
                            os.path.join(dir_name, "mesh"),
                            os.path.join(save_dir_name, "mesh"),
                        )
                        shutil.copy2(
                            os.path.join(dir_name, "results", "step_00013_rgb.mp4"),
                            os.path.join(save_dir_name, "result.mp4"),
                        )
                else:
                    dir_name = os.path.join(
                        f"{exp_root}/{os.path.basename(sd_model)}/{concept}_reference/",
                        f"{os.path.basename(asset)}",
                    )

                    save_dir_name = os.path.join(
                        f"results/reference/{concept}/",
                        f"{os.path.basename(asset)}",
                    )
                    if os.path.exists(
                        os.path.join(dir_name, "results", "step_00013_rgb.mp4")
                    ):
                        os.makedirs(save_dir_name, exist_ok=True)

                        shutil.copytree(
                            os.path.join(dir_name, "mesh"),
                            os.path.join(save_dir_name, "mesh"),
                        )
                        shutil.copy2(
                            os.path.join(dir_name, "results", "step_00013_rgb.mp4"),
                            os.path.join(save_dir_name, "result.mp4"),
                        )
