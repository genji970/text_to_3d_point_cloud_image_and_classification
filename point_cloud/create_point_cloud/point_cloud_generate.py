import ray
import numpy as np
import open3d as o3d
from PIL import Image

@ray.remote
def process_image(image_path, depth_path, output_path, fx=500, fy=500):
    # 이미지와 Depth 파일 불러오기
    image = Image.open(image_path)
    depth = np.array(Image.open(depth_path)) / 255.0

    width, height = image.size
    image_np = np.array(image)
    depth_resized = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((width, height))) / 255.0

    # 좌표계 계산
    cx, cy = width // 2, height // 2
    x_indices, y_indices = np.meshgrid(np.arange(width), np.arange(height))

    # Z, X, Y 좌표 계산 (NumPy 벡터화 적용하여 속도 개선)
    Z = depth_resized * 10
    X = (x_indices - cx) * Z / fx
    Y = (y_indices - cy) * Z / fy

    # 포인트와 색상 배열 생성
    points = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
    colors = (image_np / 255.0).reshape(-1, 3)  # RGB 값을 0-1로 정규화

    # PointCloud 생성
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # .ply 파일로 저장
    o3d.io.write_point_cloud(output_path, pcd)
    return output_path