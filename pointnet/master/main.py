import torch
import os

from pointnet.master.config import config
from pointnet.model.pointnet2_cls_msg import get_model
from pointnet.model.object_detection import detect_objects

if config['cuda_on']:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device = "cpu"

# run
path_1 = os.path.dirname(__file__)
path_2 = os.path.dirname(path_1)
path_3 = os.path.dirname(path_2)

ply_folder = os.path.join(path_3 , 'point_cloud_image_save')

# 모델 초기화 (num_class는 데이터셋에 맞게 변경)
model = get_model(num_class=40).to(device)


# 사전 학습 모델 로드
checkpoint_path = os.path.join(path_3 , 'pointnet' , 'model' , 'best_model.pth')

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("✅ Pre-trained model loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Error: The checkpoint file '{checkpoint_path}' was not found.")
except KeyError as e:
    raise KeyError(f"❌ Error: Missing key in state_dict: {e}")
except RuntimeError as e:
    raise RuntimeError(f"❌ Error: State dict loading failed. Reason: {e}")
except Exception as e:
    raise Exception(f"❌ Unexpected error occurred: {e}")

detect_objects(model, device , ply_folder)