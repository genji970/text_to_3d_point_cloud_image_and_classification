import os
import cv2
import ray
import torch
import numpy as np
from transformers import DPTForDepthEstimation, DPTFeatureExtractor


@ray.remote
class DepthEstimationWorker:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").to(self.device)
        self.transform = DPTFeatureExtractor.from_pretrained("Intel/dpt-large")

    def process_image(self, image_path, output_folder):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        inputs = self.transform(images=img, return_tensors="pt").to(self.device)

        with torch.no_grad():
            depth_map = self.model(**inputs).predicted_depth.squeeze()
            depth_map = torch.nn.functional.interpolate(
                depth_map.unsqueeze(0).unsqueeze(0), size=img.shape[:2], mode="bicubic", align_corners=False
            ).squeeze()

        depth_array = depth_map.cpu().numpy()
        depth_array = (depth_array - depth_array.min()) / (depth_array.max() - depth_array.min()) * 255
        depth_array = depth_array.astype(np.uint8)

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, os.path.basename(image_path).replace(".png", "_depth.png"))
        cv2.imwrite(output_path, depth_array)
        return f"Processed {image_path} -> {output_path}"


def batch_process_images(input_folder, output_folder):
    worker = DepthEstimationWorker.remote()

    image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".png")]
    result_refs = [worker.process_image.remote(img_path, output_folder) for img_path in image_paths]
    results = ray.get(result_refs)

    for res in results:
        print(res)