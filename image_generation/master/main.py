from image_generation.master.info import *
from image_generation.master.torch_opt import *
from image_generation.generate.generation_3d import generate_class
from image_generation.pipeline.pipeline_construct import pipe

prompt_input = 'yellow dog sitting on chair'

generating = generate_class(prompt_input , pipe)
generating.generate_2d_image()
