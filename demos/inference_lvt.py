from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import torch

SAVE_PATH = "/home/ubuntu/jayden/TEXTurePaper/sofa.png"

pipe = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda")

lora_model_path = "/home/ubuntu/jayden/TEXTurePaper/trained_models/lvt_pattern_lora"
pipe.unet.load_attn_procs(lora_model_path)
prompt = "A photo of Louis Vuitton pattern sofa"
image = pipe(prompt, num_inference_steps=50).images[0]
image.save(SAVE_PATH)
