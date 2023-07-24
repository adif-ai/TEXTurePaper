import gradio as gr
import os
import datetime
import PIL
import shutil
import re
import os
from loguru import logger
import traceback
import yaml
import torch


# copy from scripts.execution_functions and modified
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
    fast_mode: bool = False,
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
        fast_mode (bool, optional): 이미지 생성 각도를 위, 아래, 옆 3가지만 생성하여 빠르게 결과를 확인하는 용도. Defaults to False.

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
        if fast_mode:
            config_dict["render"] = {
                "n_views": 4,
                "views_before": [[180, 1]],
                "views_after": [[180, 179]],
            }
        config_path = os.path.join(save_root, save_name, "run.yaml")
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f)

        os.system(
            f"python -m scripts.run_texture --config_path='{config_path}' >> {os.path.join(save_root, 'app.log')}"
        )
        os.remove(config_path)
    else:
        logger.info(
            f'already exists: {os.path.join(save_root, save_name, "mesh", "albedo.png")}'
        )


def process(
    obj_path,
    mtl_file,
    text,
    negative_text,
    reference_image,
    reference_image_repeat,
    diffusion_name,
    upscale,
    image_resolution,
    texture_resolution,
    seed,
    fast_mode,
):
    try:
        if reference_image is not None:
            # save reference image
            reference_image_path = os.path.join(os.path.dirname(obj_path), "ref.png")
            PIL.Image.fromarray(reference_image).save(reference_image_path)
        else:
            reference_image_path = None

        # save mtl file
        mtl_path = os.path.join(
            os.path.dirname(obj_path),
            os.path.basename(obj_path).replace(".obj", ".mtl"),
        )
        shutil.copy2(mtl_file.name, mtl_path)

        # save path
        save_name = str(datetime.datetime.now()).replace(" ", "-")
        save_root = os.path.join(os.path.dirname(__file__), "app_outputs")
        save_path = os.path.join(save_root, save_name)

        single_object_single_texture(
            obj_path=obj_path,
            text=text,
            negative_text=negative_text,
            save_name=save_name,
            save_root=save_root,
            reference_image_path=reference_image_path,
            reference_image_repeat=reference_image_repeat,
            diffusion_name=diffusion_name,
            upscale=upscale,
            image_resolution=image_resolution,
            texture_resolution=texture_resolution,
            seed=seed,
            log_images=False,
            fast_mode=fast_mode,
        )

        # output path
        mp4_path = os.path.join(save_path, "result.mp4")
        mesh_zip_path = os.path.join(save_path, "mesh.zip")
        shutil.make_archive(
            os.path.join(save_path, "mesh"),
            "zip",
            os.path.join(save_path, "mesh"),
        )
        return mp4_path, mesh_zip_path
    except Exception as e:
        logger.info(traceback.format_exc())
        if obj_path is None:
            raise gr.Error("OBJ 파일을 추가해주세요.")
        elif mtl_file is None:
            raise gr.Error("MTL 파일을 추가해주세요.")
        else:
            raise gr.Error(
                f"오류가 발생하여, 프로세스가 정상적으로 완료되지 않았습니다. {traceback.format_exc()}"
            )


# initialize text file
os.makedirs(os.path.join(os.path.dirname(__file__), "app_outputs"), exist_ok=True)
log_path = os.path.join(os.path.dirname(__file__), "app_outputs", "app.log")
with open(log_path, "w") as f:
    f.write("app start\n")


def LastNlines(fname=log_path, N=10):
    assert N >= 0
    pos = N + 1
    lines = []
    with open(fname) as f:
        while len(lines) <= N:
            try:
                f.seek(-pos, 2)
            except IOError:
                f.seek(0)
                break
            finally:
                lines = list(f)
            pos *= 2
    return lines[-N:]


def read_logs():
    lines = LastNlines()
    lines = [re.sub("\[[0-9;]+[a-zA-Z]", "", line) for line in lines]
    return "".join(lines)


# pip install pynvml
def get_gpu_pid(return_pids=False):
    input_string = torch.cuda.list_gpu_processes()

    # 정규표현식을 이용하여 'process' 다음에 오는 숫자와 'uses' 다음에 오는 메모리를 추출하는 패턴
    pattern = r"process\s+(\d+)\s+uses\s+(\d+\.\d+)\s+MB"

    # 정규표현식 패턴과 일치하는 모든 결과를 찾기
    matches = re.findall(pattern, input_string)

    # 결과를 담을 딕셔너리 초기화
    process_gpu_usage = {}

    # 딕셔너리에 'process'와 메모리 정보를 추가
    for match in matches:
        process_number = match[0]
        gpu_usage = match[1]
        process_gpu_usage[process_number] = gpu_usage

    if return_pids:
        return list(process_gpu_usage.keys())

    # 보기 좋게 출력
    output = list()
    for process, gpu_usage in process_gpu_usage.items():
        output.append(f"Process {process}: {gpu_usage} MB GPU usage")
    output = "\n".join(output)

    if output == "":
        output = "텍스처 생성 실행 가능합니다. GPU를 사용하고 있지 않습니다."
    else:
        output = (
            "현재 GPU를 이용하고 있습니다. 현재 프로세스를 실행 중이지 않은 상태로 텍스처 생성을 요청할 경우, queue에 추가되어 이전 작업이 모두 완료되면 순차적으로 실행됩니다.\n"
            + output
        )

    return output


def kill_gpu_processes():
    pids = get_gpu_pid(return_pids=True)
    if len(pids) > 0:
        for pid in pids:
            os.system(f"kill {pid}")


sd_list = os.listdir(
    os.path.join(
        os.path.join(
            os.path.dirname(__file__),
            "../stable-diffusion-webui/models/Stable-diffusion",
        )
    )
)
sd_list.remove("Put Stable Diffusion checkpoints here.txt")
sd_list.remove("v1-5-pruned-emaonly.safetensors")
sd_list.insert(0, "v1-5-pruned-emaonly.safetensors")


with gr.Blocks(
    title="Adler 3D, Object Texture Generation",
).queue() as block:
    with gr.Row():
        gr.Markdown("# Adler 3D, Object Texture Generation")
    with gr.Row():
        with gr.Column():
            gr.Markdown("## 텍스처 생성을 위한 입력 값 설정")
            obj_path = gr.Model3D(label="OBJ 파일")
            gr.Markdown("(참고) OBJ 파일 업로드 시, 업로드 중인 상태 창이 보이더라도 텍스처 생성 가능합니다.")
            mtl_file = gr.File(label="MTL 파일")
            text = gr.Textbox(
                label="Text",
                info="텍스처에 반영될 텍스트 프롬프트를 영어로 입력하세요. object의 종류와 컬러, 재질 등을 입력해줍니다. 예시: sofa, blue, leather",
            )
            negative_text = gr.Textbox(
                label="Negative Text", info="(선택 사항) 텍스처에 반영되지 않았으면 하는 요소를 영어로 입력하세요."
            )
            diffusion_name = gr.Dropdown(
                sd_list,
                label="Stable Diffusion Model",
                value=sd_list[0],
                info="""
                (1) v1-5-pruned-emaonly.safetensors: 기본 stable diffusion 모델입니다.
                (2) anything-v3-fp16-pruned.safetensors : 애니메이션 느낌의 이미지를 생성하는 모델입니다.
                (3) arcane-diffusion-v3.ckpt: 넷플릭스 애니메이션 아케인 느낌의 이미지를 생성하는 모델입니다.
                (4) Realistic_Vision_V2.0-fp16-no-ema.safetensors: 실사 느낌의 이미지를 생성하는 모델입니다.""",
            )

            fast_mode = gr.Checkbox(
                label="Fast Mode",
                info="텍스처 이미지 생성하는 각도를 최소화하여 2배 이상 빠르게 결과를 확인할 수 있음. 텍스처가 비어있는 부분이 생길 수 있음",
            )

            image_resolution = gr.Slider(
                label="Image Resolution",
                minimum=256,
                maximum=1024,
                value=512,
                step=128,
                info="Stable diffusion 이미지 출력 결과 사이즈. 512는 최대 15분, 1024는 최대 60분 소요될 수 있음",
            )
            texture_resolution = gr.Slider(
                label="UV Map Resolution",
                minimum=1024,
                maximum=2048,
                value=2048,
                step=512,
                info="UV map 이미지 해상도",
            )
            upscale = gr.Checkbox(label="Upscale", info="UV map 해상도를 4배 고화질로 처리")
            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                value=-1,
                info="seed를 -1이 아닌 값인 경우, 출력 결과가 고정됨",
            )

            with gr.Accordion(
                "[Reference Image] 재질, 컬러, 컨셉 등을 표현하는 레퍼런스 이미지를 첨부합니다. 레퍼런스 이미지 사용 여부는 선택 사항입니다.",
                open=True,
            ):
                reference_image = gr.Image(source="upload", type="numpy")
                reference_image_repeat = gr.Slider(
                    label="Reference Image Repeat",
                    minimum=1,
                    maximum=10,
                    step=1,
                    value=1,
                    info="레퍼런스 이미지를 가로, 세로로 반복하여 사용합니다. 스케일이 자잘한 패턴을 원하면 값을 올려서 사용하면 됩니다.",
                )

        with gr.Column():
            gr.Markdown("## 텍스처 생성 결과")
            gpu_status = gr.Textbox(
                label="GPU 사용 현황", info="GPU 상황을 파악하여, 텍스처 생성 가능 여부를 출력합니다."
            )
            block.load(get_gpu_pid, None, gpu_status, every=1)
            with gr.Row():
                run_button = gr.Button(value="텍스처 생성 실행", variant="primary")
                exit_button = gr.Button(value="텍스처 생성 프로세스 취소", variant="stop")
            result_video = gr.Video(label="Result Video")
            result_mesh = gr.File(label="Result Mesh zip 파일 (obj, mtl, png)")
            logs = gr.Textbox(label="Log", info="프로세스가 진행 중임을 확인할 수 있도록 로그를 출력합니다.")
            block.load(read_logs, None, logs, every=1)

    ips = [
        obj_path,
        mtl_file,
        text,
        negative_text,
        reference_image,
        reference_image_repeat,
        diffusion_name,
        upscale,
        image_resolution,
        texture_resolution,
        seed,
        fast_mode,
    ]
    run_event = run_button.click(
        fn=process,
        inputs=ips,
        outputs=[result_video, result_mesh],
        # queue=True,
    )
    exit_button.click(fn=kill_gpu_processes, cancels=[run_event])


block.launch(
    server_name="0.0.0.0",
    favicon_path=os.path.join(os.path.dirname(__file__), "favicon.png"),
)
