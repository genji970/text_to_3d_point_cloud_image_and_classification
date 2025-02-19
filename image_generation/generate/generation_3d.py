import os
import torch
import numpy as np

from image_generation.generate.image_process import safe_image_processing
from image_generation.master.config import config

class generate_class():

    def __init__(self , prompt : 'str' , pipe):
        self.prompt = prompt
        self.pipe = pipe

    def generate_2d_image(self):
        image = self.pipe(self.prompt, num_inference_steps=40, guidance_scale=4.0).images[0]
        image = safe_image_processing(image)

        dir_path = os.path.dirname(__file__)
        before_real_dir_path = os.path.dirname(dir_path)
        real_dir_path = os.path.dirname(before_real_dir_path)
        image_save_directory = os.path.join(real_dir_path, 'image_save')

        if not os.path.exists(real_dir_path):
            raise FileNotFoundError(f"경로가 존재하지 않습니다: {real_dir_path}")
        else:
            initial_file_name = 'generated_image_2d_0.png'
            initial_file = os.path.join(image_save_directory, initial_file_name)


            if not os.path.exists(initial_file):
                idx = 0
            else:
                idx = len([f for f in os.listdir(image_save_directory) if os.path.isfile(os.path.join(image_save_directory, f))])

        image_number = 'generated_image_2d_' + str(idx) + '.png'
        image_name = os.path.join(image_save_directory,image_number)
        image.save(image_name)
