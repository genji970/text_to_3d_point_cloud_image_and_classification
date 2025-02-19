import torch
import logging
from diffusers import StableDiffusionPipeline

from image_generation.master.config import config

if config['cuda_on']:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

if device == 'cuda':
    logging.info("GPU.")
else:
    logging.info("CPU.")

if config['torch_to_fp16']:
    pipe = StableDiffusionPipeline.from_pretrained(config['model_name'],
                                                   torch_dtype=torch.float16,
                                                   low_cpu_mem_usage=True,
                                                   requires_safety_checker=False
                                                   ).to(device)
else:
    pipe = StableDiffusionPipeline.from_pretrained(config['model_name'],
                                                   torch_dtype=torch.float32,
                                                   low_cpu_mem_usage=True,
                                                   requires_safety_checker = False
                                                   ).to(device)

# batch size
pipe.batch_size = config['pipe_batch_size']

# resolution
pipe.width, pipe.height = config['pipe_width'] , config['pipe_height']
pipe.safety_checker = config['inappropriate_content_filter']

if config['cuda_on']:
    # CPU Offloading 활성화 (자동으로 일부 모듈을 CPU로 이동)
    pipe.text_encoder.to("cpu")  # 텍스트 인코더 CPU로 이동
    pipe.vae.to("cpu")           # VAE (디코더) CPU로 이동
    pipe.unet.to("cuda")

    # VAE slicing 및 CPU 오프로드 활성화
    pipe.enable_vae_slicing()
    pipe.enable_model_cpu_offload()
