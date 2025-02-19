import torch

from image_generation.master.config import config

if config['cuda_on']:
    # PyTorch 연산 최적화 (VRAM 단편화 방지)
    torch.backends.cuda.matmul.allow_tf32 = True

    # CUDA 메모리 초기화
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.ipc_collect()