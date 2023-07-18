import os
from loguru import logger
import traceback
import yaml
from typing import List
from scripts.combine_parts import combine_parts


def single_object_single_texture(
    obj_path: str,
    text: str,
    negative_text: str = "",
    save_name: str = None,
    save_root: str = "results",
    reference_image_path: str = None,
    reference_image_repeat: int = 1,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    """
    3D object 1개에 대한 텍스처를 생성하는 함수입니다.

    Args:
        obj_path (str): .obj 파일의 경로입니다. UV map 정보 보존을 위해, .obj 파일과 확장자만 다른 .mtl 파일이 .obj 파일과 동일한 폴더에 존재해야 합니다.
        text (str): 생성을 원하는 텍스처를 묘사하는 텍스트 프롬프트를 입력합니다.
        negative_text (str, optional): 생성 결과에 반영되지 않았으면 하는 텍스처를 묘사하여 입력합니다. Defaults to "".
        save_name (str, optional): 출력 결과가 저장되는 폴더 이름입니다. 따로 지정하지 않으면, text와 같은 값으로 설정됩니다. Defaults to None.
        save_root (str, optional): 출력 결과가 저장되는 경로입니다. Defaults to "results".
        reference_image_path (str, optional): 텍스처 생성 시 참고할 이미지의 경로입니다. Defaults to None.
        reference_image_repeat (int, optional): 레퍼런스 이미지를 가로, 세로로 반복하여 사용합니다. 스케일이 자잘한 패턴을 원하면 값을 올려서 사용하면 됩니다. Defaults to 1.
        diffusion_name (str, optional): 사용되는 stable diffusion 모델의 파일 이름을 입력하면 됩니다. 추가로 커뮤니티 모델을 이용하고 싶은 경우, "../stable-diffusion-webui/models/Stable-diffusion" 경로에 파일을 추가하면 됩니다. Defaults to "v1-5-pruned-emaonly.safetensors".

        지원되는 diffusion_name의 리스트입니다.

        "v1-5-pruned-emaonly.safetensors" : 기본 stable diffusion 모델입니다. 따로 지정하지 않을 경우, 디폴트 모델로 설정됩니다.
        "Realistic_Vision_V2.0-fp16-no-ema.safetensors" : 실사 느낌의 이미지를 생성하는 모델입니다.
        "anything-v3-fp16-pruned.safetensors" : 애니메이션 느낌의 이미지를 생성하는 모델입니다.
        "arcane-diffusion-v3.ckpt" : 넷플릭스 애니메이션 아케인 느낌의 이미지를 생성하는 모델입니다.

        upscale (bool, optional): 생성된 UV map 이미지를 4배 업스케일 합니다. 사용할 경우, 텍스처에서 약간의 노이즈 제거가 되고 UV map 파일이 고화질로 변환됩니다. Defaults to False.
        image_resolution (int, optional): stable diffusion에서 생성하는 이미지의 사이즈 입니다. 기본은 512 X 512 이며, 1024 X 1024 도 가능합니다. 텍스처 생성에 512는 5분, 1024는 20분 정도 소요됩니다. 그 이상으로 image resolution을 올릴 경우, GPU memory 이슈로 실행되지 않을 수 있습니다. Defaults to 512.
        texture_resolution (int, optional): UV map의 이미지의 resolution입니다. 기본은 2048 X 2048 입니다. Defaults to 2048.
        seed (int, optional): stable diffusion의 이미지 생성 결과의 랜덤성을 결정하는 seed 입니다. 값을 바꿀 경우, 약간 다른 텍스처가 생성됩니다. Defaults to 0.
        log_images (bool, optional): 텍스처 생성 과정의 중간 결과물의 저장 여부를 결정합니다. 기본은 저장하지 않음 입니다. Defaults to False.

    결과:
        save_root
            - save_name
                - mesh
                    - albedo.png : UV map 파일입니다.
                    - mesh.mtl : 3D 모델의 mtl 파일입니다.
                    - mesh.obj : 3D 모델의 obj 파일입니다.
                - result.mp4 : 3D 모델의 텍스처 생성 결과를 동영상 파일로 저장한 파일입니다.
                - config.yaml : 텍스처 생성에 사용된 configuration 파일입니다.
                - log.txt : 텍스처 생성 과정에서 발생한 log를 기록한 파일입니다.

    예시:
        single_object_single_texture(
        obj_path="shapes/ai/sofa_001.obj", text="leather sofa, brown"
        )

    """

    if save_name is None:
        save_name = text
    try:
        if not os.path.exists(os.path.join(save_root, save_name, "mesh", "albedo.png")):
            assert os.path.exists(
                obj_path.replace(".obj", ".mtl")
            ), f'{obj_path.replace(".obj", ".mtl")} mtl file이 없습니다. UV map 고정을 위해, mtl 파일을 obj 파일과 같은 폴더 안에 위치시켜주세요.'
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

            os.system(f"python -m scripts.run_texture --config_path='{config_path}'")
            os.remove(config_path)
        else:
            logger.info(
                f'already exists: {os.path.join(save_root, save_name, "mesh", "albedo.png")}'
            )

    except Exception as e:
        logger.info(traceback.format_exc())


def single_object_multi_textures(
    obj_path: str,
    text_list: List[str],
    negative_text_list: List[str] = None,
    save_name_list: List[str] = None,
    save_root: str = "results",
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    """
    3D object 1개에 대해 여러가지 텍스처를 순차적으로 생성하는 함수입니다.


    Args:
        obj_path (str): .obj 파일의 경로입니다. UV map 정보 보존을 위해, .obj 파일과 확장자만 다른 .mtl 파일이 .obj 파일과 동일한 폴더에 존재해야 합니다.
        text_list (List[str]): 생성을 원하는 텍스처를 묘사하는 텍스트 프롬프트를 리스트로 입력합니다. 리스트의 요소 개수만큼 텍스처를 생성합니다.
        negative_text_list (List[str], optional): 생성 결과에 반영되지 않았으면 하는 텍스처를 묘사하여 리스트로 입력합니다. Defaults to None.
        save_name_list (List[str], optional): 출력 결과가 저장되는 폴더 이름의 리스트입니다. 따로 지정하지 않으면, text_list와 같은 값으로 설정됩니다. Defaults to None.
        save_root (str, optional): 출력 결과가 저장되는 경로입니다. Defaults to "results".
        reference_image_path_list (List[str], optional): 텍스처 생성 시 참고할 이미지의 경로의 리스트입니다. text_list와 동일한 길이의 list로 입력해야 합니다. reference image가 없는 텍스처의 경우, list의 요소로 None을 입력하면 됩니다. Defaults to None.
        reference_image_repeat_list (List[int], optional): 레퍼런스 이미지를 반복하는 횟수의 리스트입니다. 레퍼런스 이미지를 가로, 세로로 반복하여 사용합니다. 스케일이 자잘한 패턴을 원하면 값을 올려서 사용하면 됩니다. Defaults to None.
        diffusion_name (str, optional): 사용되는 stable diffusion 모델의 파일 이름을 입력하면 됩니다. 추가로 커뮤니티 모델을 이용하고 싶은 경우, "../stable-diffusion-webui/models/Stable-diffusion" 경로에 파일을 추가하면 됩니다. Defaults to "v1-5-pruned-emaonly.safetensors".

        지원되는 diffusion_name의 리스트입니다.

        "v1-5-pruned-emaonly.safetensors" : 기본 stable diffusion 모델입니다. 따로 지정하지 않을 경우, 디폴트 모델로 설정됩니다.
        "Realistic_Vision_V2.0-fp16-no-ema.safetensors" : 실사 느낌의 이미지를 생성하는 모델입니다.
        "anything-v3-fp16-pruned.safetensors" : 애니메이션 느낌의 이미지를 생성하는 모델입니다.
        "arcane-diffusion-v3.ckpt" : 넷플릭스 애니메이션 아케인 느낌의 이미지를 생성하는 모델입니다.

        upscale (bool, optional): 생성된 UV map 이미지를 4배 업스케일 합니다. 사용할 경우, 텍스처에서 약간의 노이즈 제거가 되고 UV map 파일이 고화질로 변환됩니다. Defaults to False.
        image_resolution (int, optional): stable diffusion에서 생성하는 이미지의 사이즈 입니다. 기본은 512 X 512 이며, 1024 X 1024 도 가능합니다. 텍스처 생성에 512는 5분, 1024는 20분 정도 소요됩니다. 그 이상으로 image resolution을 올릴 경우, GPU memory 이슈로 실행되지 않을 수 있습니다. Defaults to 512.
        texture_resolution (int, optional): UV map의 이미지의 resolution입니다. 기본은 2048 X 2048 입니다. Defaults to 2048.
        seed (int, optional): stable diffusion의 이미지 생성 결과의 랜덤성을 결정하는 seed 입니다. 값을 바꿀 경우, 약간 다른 텍스처가 생성됩니다. Defaults to 0.
        log_images (bool, optional): 텍스처 생성 과정의 중간 결과물의 저장 여부를 결정합니다. 기본은 저장하지 않음 입니다. Defaults to False.

    Returns:
        uv_map_path_list: UV map 파일들의 경로 리스트입니다. object의 파츠 조합에 사용됩니다.
        name_list: 각 UV map을 표현하는 이름, save_name의 리스트입니다. object의 파츠 조합에 사용됩니다.

    예시:
        # reference image 없음
        single_object_multi_textures(
        obj_path="shapes/ai/sofa_001.obj",
        text_list=["leather sofa, brown", "pink sofa, flower"],
        save_name_list=["brown leather sofa", "핑크 꽃 소파"],
        save_root="demo",
        upscale=True,
        )

        # reference image 이용
        single_object_multi_textures(
        obj_path="shapes/ai/sofa_001.obj",
        text_list=["crocodile leather, sofa", "gold sofa"],
        save_name_list=["crocodile leather sofa", "골드 소파"],
        reference_image_path_list=[
            "references/crocodile leather.png",
            "references/gold.png",
        ],
        reference_image_repeat_list=[3, 1],
        save_root="demo",
        diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
        upscale=True,
        )
    """
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

        single_object_single_texture(**kwargs)

        uv_map_path_list.append(
            os.path.join(save_root, kwargs["save_name"], "mesh", "albedo.png")
        )
        name_list.append(kwargs["save_name"])

    return uv_map_path_list, name_list


def single_object_multi_textures_parts_combination(
    obj_path: str,
    text_list: List[str],
    parts: List[str],
    parts_mask_paths: List[str],
    negative_text_list: List[str] = None,
    save_name_list: List[str] = None,
    save_root: str = "results",
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    """
    3D object 1개에 대해 여러가지 텍스처를 순차적으로 생성하고, 파츠를 조합하여 UV map을 생성하는 함수입니다.

    Args:
        obj_path (str): .obj 파일의 경로입니다. UV map 정보 보존을 위해, .obj 파일과 확장자만 다른 .mtl 파일이 .obj 파일과 동일한 폴더에 존재해야 합니다.
        text_list (List[str]): 생성을 원하는 텍스처를 묘사하는 텍스트 프롬프트를 리스트로 입력합니다. 리스트의 요소 개수만큼 텍스처를 생성합니다.
        parts (List[str]): 파츠의 이름을 리스트로 입력합니다. 3d object 파일이 저장되는 폴더 이름에 활용됩니다.
        parts_mask_paths (List[str]): 파츠 별로 표시된 마스크 이미지의 경로를 입력합니다. 마지막 파츠는 남은 영역으로 처리되므로, 파츠의 총 개수보다 1개 적은 리스트를 입력해야 됩니다. 마스크 이미지는 파츠에 해당되는 영역을 흰색, 나머지를 검은색으로 표시한 UV map과 동일한 정사각형의 이미지이어야 합니다.
        negative_text_list (List[str], optional): 생성 결과에 반영되지 않았으면 하는 텍스처를 묘사하여 리스트로 입력합니다. Defaults to None.
        save_name_list (List[str], optional): 출력 결과가 저장되는 폴더 이름의 리스트입니다. 따로 지정하지 않으면, text_list와 같은 값으로 설정됩니다. Defaults to None.
        save_root (str, optional): 출력 결과가 저장되는 경로입니다. Defaults to "results".
        reference_image_path_list (List[str], optional): 텍스처 생성 시 참고할 이미지의 경로의 리스트입니다. text_list와 동일한 길이의 list로 입력해야 합니다. reference image가 없는 텍스처의 경우, list의 요소로 None을 입력하면 됩니다. Defaults to None.
        reference_image_repeat_list (List[int], optional): 레퍼런스 이미지를 반복하는 횟수의 리스트입니다. 레퍼런스 이미지를 가로, 세로로 반복하여 사용합니다. 스케일이 자잘한 패턴을 원하면 값을 올려서 사용하면 됩니다. Defaults to None.
        diffusion_name (str, optional): 사용되는 stable diffusion 모델의 파일 이름을 입력하면 됩니다. 추가로 커뮤니티 모델을 이용하고 싶은 경우, "../stable-diffusion-webui/models/Stable-diffusion" 경로에 파일을 추가하면 됩니다. Defaults to "v1-5-pruned-emaonly.safetensors".

        지원되는 diffusion_name의 리스트입니다.

        "v1-5-pruned-emaonly.safetensors" : 기본 stable diffusion 모델입니다. 따로 지정하지 않을 경우, 디폴트 모델로 설정됩니다.
        "Realistic_Vision_V2.0-fp16-no-ema.safetensors" : 실사 느낌의 이미지를 생성하는 모델입니다.
        "anything-v3-fp16-pruned.safetensors" : 애니메이션 느낌의 이미지를 생성하는 모델입니다.
        "arcane-diffusion-v3.ckpt" : 넷플릭스 애니메이션 아케인 느낌의 이미지를 생성하는 모델입니다.

        upscale (bool, optional): 생성된 UV map 이미지를 4배 업스케일 합니다. 사용할 경우, 텍스처에서 약간의 노이즈 제거가 되고 UV map 파일이 고화질로 변환됩니다. Defaults to False.
        image_resolution (int, optional): stable diffusion에서 생성하는 이미지의 사이즈 입니다. 기본은 512 X 512 이며, 1024 X 1024 도 가능합니다. 텍스처 생성에 512는 5분, 1024는 20분 정도 소요됩니다. 그 이상으로 image resolution을 올릴 경우, GPU memory 이슈로 실행되지 않을 수 있습니다. Defaults to 512.
        texture_resolution (int, optional): UV map의 이미지의 resolution입니다. 기본은 2048 X 2048 입니다. Defaults to 2048.
        seed (int, optional): stable diffusion의 이미지 생성 결과의 랜덤성을 결정하는 seed 입니다. 값을 바꿀 경우, 약간 다른 텍스처가 생성됩니다. Defaults to 0.
        log_images (bool, optional): 텍스처 생성 과정의 중간 결과물의 저장 여부를 결정합니다. 기본은 저장하지 않음 입니다. Defaults to False.

    예시:
        single_object_multi_textures_parts_combination(
            obj_path="shapes/ai/sofa_001.obj",
            text_list=["crocodile leather, sofa", "gold sofa"],
            parts=["leg", "pillow", "seat"],
            parts_mask_paths=[
                "shapes/ai/sofa_001_leg.png",
                "shapes/ai/sofa_001_pillow.png",
            ],
            save_name_list=["crocodile leather sofa", "골드 소파"],
            reference_image_path_list=[
                "references/crocodile leather.png",
                "references/gold.png",
            ],
            reference_image_repeat_list=[3, 1],
            save_root="demo",
            diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
            upscale=True,
        )
    """
    uv_map_path_list, name_list = single_object_multi_textures(
        obj_path=obj_path,
        text_list=text_list,
        save_name_list=save_name_list,
        save_root=save_root,
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
    obj_text_list: List[str],
    text_list: List[str],
    negative_text_list: List[str] = None,
    save_name_list: List[str] = None,
    save_root: str = "results",
    reference_image_path_list: List[str] = None,
    reference_image_repeat_list: List[int] = None,
    diffusion_name: str = "v1-5-pruned-emaonly.safetensors",
    upscale: bool = False,
    image_resolution: int = 512,
    texture_resolution: int = 2048,
    seed: int = 0,
    log_images: bool = False,
):
    """
    여러 개의 3d object 파일에 대해, 여러가지 텍스처를 동일하게 각각 적용하여 생성하는 함수입니다.

    Args:
        obj_path_list (List[str]): .obj 파일의 경로의 리스트입니다. UV map 정보 보존을 위해, .obj 파일과 확장자만 다른 .mtl 파일이 .obj 파일과 동일한 폴더에 존재해야 합니다.
        obj_text_list (List[str]): 각 object의 종류를 텍스트의 리스트로 입력합니다.
        text_list (List[str]): 생성을 원하는 텍스처를 묘사하는 텍스트 프롬프트를 리스트로 입력합니다. 리스트의 요소 개수만큼 텍스처를 생성합니다.
        negative_text_list (List[str], optional): 생성 결과에 반영되지 않았으면 하는 텍스처를 묘사하여 리스트로 입력합니다. Defaults to None.
        save_name_list (List[str], optional): 출력 결과가 저장되는 폴더 이름의 리스트입니다. 따로 지정하지 않으면, text_list와 같은 값으로 설정됩니다. Defaults to None.
        save_root (str, optional): 출력 결과가 저장되는 경로입니다. Defaults to "results".
        reference_image_path_list (List[str], optional): 텍스처 생성 시 참고할 이미지의 경로의 리스트입니다. text_list와 동일한 길이의 list로 입력해야 합니다. reference image가 없는 텍스처의 경우, list의 요소로 None을 입력하면 됩니다. Defaults to None.
        reference_image_repeat_list (List[int], optional): 레퍼런스 이미지를 반복하는 횟수의 리스트입니다. 레퍼런스 이미지를 가로, 세로로 반복하여 사용합니다. 스케일이 자잘한 패턴을 원하면 값을 올려서 사용하면 됩니다. Defaults to None.
        diffusion_name (str, optional): 사용되는 stable diffusion 모델의 파일 이름을 입력하면 됩니다. 추가로 커뮤니티 모델을 이용하고 싶은 경우, "../stable-diffusion-webui/models/Stable-diffusion" 경로에 파일을 추가하면 됩니다. Defaults to "v1-5-pruned-emaonly.safetensors".

        지원되는 diffusion_name의 리스트입니다.

        "v1-5-pruned-emaonly.safetensors" : 기본 stable diffusion 모델입니다. 따로 지정하지 않을 경우, 디폴트 모델로 설정됩니다.
        "Realistic_Vision_V2.0-fp16-no-ema.safetensors" : 실사 느낌의 이미지를 생성하는 모델입니다.
        "anything-v3-fp16-pruned.safetensors" : 애니메이션 느낌의 이미지를 생성하는 모델입니다.
        "arcane-diffusion-v3.ckpt" : 넷플릭스 애니메이션 아케인 느낌의 이미지를 생성하는 모델입니다.

        upscale (bool, optional): 생성된 UV map 이미지를 4배 업스케일 합니다. 사용할 경우, 텍스처에서 약간의 노이즈 제거가 되고 UV map 파일이 고화질로 변환됩니다. Defaults to False.
        image_resolution (int, optional): stable diffusion에서 생성하는 이미지의 사이즈 입니다. 기본은 512 X 512 이며, 1024 X 1024 도 가능합니다. 텍스처 생성에 512는 5분, 1024는 20분 정도 소요됩니다. 그 이상으로 image resolution을 올릴 경우, GPU memory 이슈로 실행되지 않을 수 있습니다. Defaults to 512.
        texture_resolution (int, optional): UV map의 이미지의 resolution입니다. 기본은 2048 X 2048 입니다. Defaults to 2048.
        seed (int, optional): stable diffusion의 이미지 생성 결과의 랜덤성을 결정하는 seed 입니다. 값을 바꿀 경우, 약간 다른 텍스처가 생성됩니다. Defaults to 0.
        log_images (bool, optional): 텍스처 생성 과정의 중간 결과물의 저장 여부를 결정합니다. 기본은 저장하지 않음 입니다. Defaults to False.

    예시:
        multi_objects_multi_textures(
        obj_path_list=["shapes/ai/sofa_001.obj", "shapes/ai/Chair_002.obj"],
        obj_text_list=["sofa", "chair"],
        text_list=["crocodile leather", "gold"],
        save_name_list=["crocodile leather", "골드"],
        reference_image_path_list=[
            "references/crocodile leather.png",
            "references/gold.png",
        ],
        reference_image_repeat_list=[3, 1],
        save_root="demo_multi",
        diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
        upscale=True,
    )
    """
    for i, obj_path in enumerate(obj_path_list):
        obj_save_root = os.path.join(save_root, os.path.basename(obj_path))
        tmp_text_list = text_list.copy()
        for j in range(len(text_list)):
            tmp_text_list[j] = obj_text_list[i] + ", " + tmp_text_list[j]

        single_object_multi_textures(
            obj_path=obj_path,
            text_list=tmp_text_list,
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
    # 예시 코드
    single_object_single_texture(
        obj_path="shapes/ai/sofa_001.obj", text="leather sofa, brown"
    )

    single_object_multi_textures(
        obj_path="shapes/ai/sofa_001.obj",
        text_list=["leather sofa, brown", "pink sofa, flower"],
        save_name_list=["brown leather sofa", "핑크 꽃 소파"],
        save_root="results/multi_textures",
        upscale=True,
    )

    single_object_multi_textures(
        obj_path="shapes/ai/sofa_001.obj",
        text_list=["crocodile leather, sofa", "gold sofa"],
        save_name_list=["crocodile leather sofa", "골드 소파"],
        reference_image_path_list=[
            "references/crocodile leather.png",
            "references/gold.png",
        ],
        reference_image_repeat_list=[3, 1],
        save_root="results/multi_textures",
        diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
        upscale=True,
    )

    single_object_multi_textures_parts_combination(
        obj_path="shapes/ai/sofa_001.obj",
        text_list=["crocodile leather, sofa", "gold sofa"],
        save_name_list=["crocodile leather sofa", "골드 소파"],
        reference_image_path_list=[
            "references/crocodile leather.png",
            "references/gold.png",
        ],
        reference_image_repeat_list=[3, 1],
        save_root="results/multi_textures",
        diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
        upscale=True,
        parts_mask_paths=[
            "shapes/ai/sofa_001_leg.png",
            "shapes/ai/sofa_001_pillow.png",
        ],
        parts=["leg", "pillow", "seat"],
    )

    multi_objects_multi_textures(
        obj_path_list=["shapes/ai/sofa_001.obj", "shapes/ai/Chair_002.obj"],
        obj_text_list=["sofa", "chair"],
        text_list=["crocodile leather", "gold"],
        save_name_list=["crocodile leather", "골드"],
        reference_image_path_list=[
            "references/crocodile leather.png",
            "references/gold.png",
        ],
        reference_image_repeat_list=[3, 1],
        save_root="results/multi_objects_multi_textures",
        diffusion_name="Realistic_Vision_V2.0-fp16-no-ema.safetensors",
        upscale=True,
    )
