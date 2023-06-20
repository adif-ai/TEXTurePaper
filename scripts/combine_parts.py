from typing import Any, Dict, Union, List
import shutil
import os
from diffusers.utils import load_image
import PIL
import numpy as np
from itertools import product
from collections import defaultdict


def combine_parts(
    obj_path: str,
    mtl_path: str,
    png_paths: List[str],
    parts: List[str],
    parts_mask_paths: List[str],
    concepts: List[str],
    output_dir: str,
):
    os.makedirs(output_dir, exist_ok=True)

    uv_images = [np.array(load_image(i)).astype(np.float32) / 255.0 for i in png_paths]

    assert len(parts) - 1 == len(parts_mask_paths), "마지막 part는 남는 mask 영역"

    parts_images = []
    for i in parts_mask_paths:
        if i is not None:
            parts_images.append(np.array(load_image(i)).astype(np.float32) / 255.0)
    rest_part = np.zeros_like(parts_images[0])
    for i in parts_images:
        rest_part += i
    rest_part[rest_part >= 1] = 1
    rest_part -= 1
    rest_part *= -1
    parts_images.append(rest_part)

    all_combinations = defaultdict(list)
    for i, part in enumerate(parts):
        for j, concept in enumerate(concepts):
            all_combinations[part].append([i, j])

    for combination in product(*[all_combinations[i] for i in parts]):
        image = None
        dir_name = []
        for i, j in combination:
            part_mask = parts_images[i]
            concept_image = uv_images[j]
            if image is None:
                image = np.zeros_like(concept_image)
            image += part_mask * concept_image
            dir_name.append(f"{parts[i]}-{concepts[j]}")

        dir_name = "_".join(dir_name)
        save_dir = os.path.join(output_dir, dir_name)
        os.makedirs(save_dir, exist_ok=True)

        PIL.Image.fromarray((image * 255).astype(np.uint8)).save(
            os.path.join(save_dir, os.path.basename(png_paths[0]))
        )
        shutil.copy2(obj_path, save_dir)
        shutil.copy2(mtl_path, save_dir)


if __name__ == "__main__":
    combine_parts(
        obj_path="experiments/0621/gold_table_inpaint1_white_room_ref/mesh/mesh.obj",
        mtl_path="experiments/0621/gold_table_inpaint1_white_room_ref/mesh/mesh.mtl",
        png_paths=[
            "experiments/0621/gold_table_inpaint1_white_room_ref/mesh/albedo.png",
            "experiments/0621/spanish_tile_table_ref/mesh/albedo.png",
        ],
        parts=["leg", "top"],
        parts_mask_paths=["Table_002_leg.png"],
        concepts=["gold", "spanish tile"],
        output_dir="table_parts",
    )
