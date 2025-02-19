import os
import numpy as np
from PIL import Image
import ray

from point_cloud.create_point_cloud.point_cloud_generate import process_image

path_1 = os.path.dirname(__file__)
path_2 = os.path.dirname(path_1)
path_3 = os.path.dirname(path_2)

image_dir = os.path.join(path_3 , 'image_save')
depth_dir = os.path.join(path_3 , 'depth_image_save')
point_cloud_dir = os.path.join(path_3 , 'point_cloud_image_save')

if not os.path.exists(point_cloud_dir):
    os.makedirs(point_cloud_dir)
if not os.path.exists(image_dir):
    raise FileNotFoundError(f"경로가 존재하지 않습니다: {image_dir}")
if not os.path.exists(depth_dir):
    raise FileNotFoundError(f"경로가 존재하지 않습니다: {depth_dir}")

image_files = sorted(os.listdir(image_dir))
depth_files = sorted(os.listdir(depth_dir))

ray.init()

futures = []
for image_file, depth_file in zip(image_files, depth_files):
    image_path = os.path.join(image_dir, image_file)
    depth_path = os.path.join(depth_dir, depth_file)
    output_path = os.path.join(point_cloud_dir, f"{os.path.splitext(image_file)[0]}.ply")

    futures.append(process_image.remote(image_path, depth_path, output_path))

ray.get(futures)
print("모든 포인트 클라우드 파일이 저장되었습니다.")

