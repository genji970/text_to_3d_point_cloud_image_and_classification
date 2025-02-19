import numpy as np
from PIL import Image
import torch

def safe_image_processing(image):
    if isinstance(image, Image.Image):  # PIL.Image면 변환하지 않음
        return image

    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Tensor → NumPy 변환

    image = np.nan_to_num(image, nan=0.0)  # NaN → 0 변환
    image = np.clip(image, 0, 1)  # 0~1 범위 제한
    image = (image * 255).round().astype("uint8")  # uint8 변환

    if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W) → (H, W, C)
        image = np.moveaxis(image, 0, -1)

    return Image.fromarray(image)  # PIL 이미지 변환
