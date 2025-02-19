writing prompt -> loaded diffusion model outputs 2d image -> using midas depth estimation, making depth image -> making point cloud data with (image,depth image) -> using loaded pointnet, doing object detection to ply data

## structure ##

1) image_generation , image_save.<br>
`python -m image_generation.master.main` -> made images will be saved in image_save folder

2) image_reconstruction , depth_image_save.<br>
`python -m image_reconstruction.master.main` -> made images will be saved in depth_image_sve folder

3) point_cloud , point_cloud_image_save.<br>
`python -m point_cloud.master.main` - > made images will be saved in point_cloud_image_save

4) pointnet.<br>
`python -m pointnet.master.main` -> classification by loaded pointnet model will be run to data.ply in point_cloud_image_save 


![Image](https://github.com/user-attachments/assets/b93fd8ca-1a14-417d-98f6-49b17ea9deda)

![Image](https://github.com/user-attachments/assets/9d733472-b6a4-4216-a06b-349cd058eb42)

## Reference ##
pointnet/model/best_model.pth,pointnet2_cls_msg.py,pointnet2_utils were loaded from https://github.com/yanx27/Pointnet_Pointnet2_pytorch

n_sample in pointnet/model/pointnet2_cls_msg.py were changed due to structure problem

## Next ##
1) gaussian splatting
2) change focal length,etc
3) promt using pretrained llm such as gpt2. Since realistic prompt makes better image, appropriate prompting will be good.
