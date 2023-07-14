import pyrallis
import os
from loguru import logger
import traceback
import datetime
import yaml
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from scripts.combine_parts import combine_parts


def main(
    obj_path: str,  # mtl_path
    text: str,
    save_name: str = None,
    save_root: str = "results",
    negative_text: str = "",
    reference_image_path: str = None,
    reference_image_repeat: int = 1,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    if save_name is None:
        save_name = f"{text}"
    try:
        if not os.path.exists(os.path.join(save_root, save_name, "mesh", "albedo.png")):
            assert os.path.exists(
                obj_path.replace(".obj", ".mtl")
            ), "mtl file이 없습니다. UV map 고정을 위해, mtl 파일을 obj 파일과 같은 폴더 안에 위치시켜주세요."
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


def single_objects_multi_textures(
    obj_path: str,
    text_list: List[str],
    save_name_list: List[str] = None,
    save_root: str = "results",
    negative_text_list: List[str] = None,
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    uv_map_path_list = list()
    name_list = list()
    for i, text in enumerate(text_list):
        kwargs = {
            "obj_path": obj_path,
            "text": text,
            "save_root": save_root,
            "diffusion_name": diffusion_name,
            "upscale": upscale,
            "image_resolution": image_resolution,
            "texture_resolution": texture_resolution,
            "seed": seed,
            "log_images": log_images,
        }
        if save_name_list is not None:
            kwargs["save_name"] = save_name_list[i]
        else:
            kwargs["save_name"] = text
        if negative_text_list is not None:
            kwargs["negative_text"] = negative_text_list[i]
        if reference_image_path_list is not None:
            kwargs["reference_image_path"] = reference_image_path_list[i]
        if reference_image_repeat_list is not None:
            kwargs["reference_image_repeat"] = reference_image_repeat_list[i]

        main(**kwargs)

        uv_map_path_list.append(
            os.path.join(save_root, kwargs["save_name"], "mesh", "albedo.png")
        )
        name_list.append(kwargs["save_name"])

    return uv_map_path_list, name_list


def single_objects_multi_textures_parts_combination(
    obj_path: str,
    text_list: List[str],
    parts: List[str],
    parts_mask_paths: List[str],
    save_name_list: List[str] = None,
    save_root: str = "results",
    negative_text_list: List[str] = None,
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    uv_map_path_list, name_list = single_objects_multi_textures(
        obj_path,
        text_list,
        save_name_list,
        save_root,
        negative_text_list,
        reference_image_path_list,
        reference_image_repeat_list,
        diffusion_name,
        upscale,
        image_resolution,
        texture_resolution,
        seed,
        log_images,
    )

    combine_parts(
        obj_path=obj_path,
        mtl_path=obj_path.replace(".obj", ".mtl"),
        png_paths=uv_map_path_list,
        parts=parts,
        parts_mask_paths=parts_mask_paths,
        concepts=name_list,
        output_dir=os.path.join(save_root, "parts"),
    )


def multi_objects_multi_textures(
    obj_path_list: List[str],
    text_list: List[str],
    save_name_list: List[str] = None,
    save_root: str = "results",
    negative_text_list: List[str] = None,
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    for i, obj_path in enumerate(obj_path_list):
        obj_save_root = os.path.join(save_root, os.path.basename(obj_path))
        single_objects_multi_textures(
            obj_path=obj_path,
            text_list=text_list,
            save_name_list=save_name_list,
            save_root=obj_save_root,
            negative_text_list=negative_text_list,
            reference_image_path_list=reference_image_path_list,
            reference_image_repeat_list=reference_image_repeat_list,
            diffusion_name=diffusion_name,
            upscale=upscale,
            image_resolution=image_resolution,
            texture_resolution=texture_resolution,
            seed=seed,
            log_images=log_images,
        )


if __name__ == "__main__":
    main(obj_path="shapes/ai/sofa_001.obj", text="leather sofa, brown")
