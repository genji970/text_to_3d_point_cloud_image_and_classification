import os
import torch
import torch.nn as nn

from pointnet.master.class_name import CLASS_NAMES
from pointnet.master.config import config
from pointnet.model.preprocess import load_point_cloud

if config['cuda_on']:
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# ============== 객체 탐지 ==============
def detect_objects(model, device, ply_folder):
    model.eval()
    for file_name in os.listdir(ply_folder):
        if file_name.endswith('.ply'):
            point_cloud = load_point_cloud(os.path.join(ply_folder, file_name)).to(device)
            with torch.no_grad():
                output = model(point_cloud)
                if isinstance(output, tuple):  # 튜플이면 첫 번째 요소 사용
                    output = output[0]
                pred_class_idx = torch.argmax(output).item()
                # ✅ 클래스 이름 출력
                pred_class_name = CLASS_NAMES[pred_class_idx] if pred_class_idx < len(CLASS_NAMES) else "Unknown"
                print(f'{file_name} → Predicted Class: {pred_class_name} (Index: {pred_class_idx})')