import open3d as o3d
import os
import time

path_1 = os.path.dirname(__file__)
path_2 = os.path.dirname(path_1)
path_3 = os.path.dirname(path_2)
point_cloud_image_save_path = os.path.join(path_3,'point_cloud_image_save')
files = os.listdir(point_cloud_image_save_path)

if files != []:
    for file in files:
        if file.lower().endswith('.ply') and os.path.isfile(os.path.join(point_cloud_image_save_path, file)):
            ply_path = os.path.join(point_cloud_image_save_path, file)
            try:
                pcd = o3d.io.read_point_cloud(ply_path)
                point_count = len(pcd.points)
                if point_count > 0:
                    print(f"✅ {file} 파일이 정상적으로 저장되었습니다. 포인트 개수: {point_count}")
                else:
                    print(f"⚠️ {file} 파일에 포인트가 없습니다.")
            except Exception as e:
                print(f"❌ {file} 파일을 읽는 중 오류 발생: {e}")
                continue

            # ✅ 비블로킹 시각화: 자동으로 10초 후 닫힘
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name=file)
            vis.add_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            time.sleep(10)  # 10초 동안 창 유지
            vis.destroy_window()  # 창 자동으로 닫기