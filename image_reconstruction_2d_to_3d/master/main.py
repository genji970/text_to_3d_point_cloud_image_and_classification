import os
import cv2
import torch
import ray
import numpy as np
import torchvision.transforms as transforms
from torchvision import models

from image_reconstruction_2d_to_3d.master.config import config
from image_save import *
from image_reconstruction_2d_to_3d.estimation.estimate import batch_process_images

if config['cuda_on']:
    device = "cuda" if torch.cuda.is_available() else "cpu"
else:
    device = "cpu"

# Ray 초기화
ray.init()

path_1 = os.path.dirname(__file__)
path_2 = os.path.dirname(path_1)
path_3 = os.path.dirname(path_2)

input_folder = os.path.join(path_3 , 'image_save')  # 입력 폴더 경로
output_folder = os.path.join(path_3 , "depth_image_save")  # 출력 폴더 경로

batch_process_images(input_folder, output_folder)