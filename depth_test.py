import open3d as o3d
from open3d import JVisualizer
print("Read Redwood dataset")

color_raw = o3d.io.read_image("./data/00000.jpg")
depth_raw = o3d.io.read_image("./disparity_maps/00000.png")
rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(color_raw, depth_raw)
print(rgbd_image)
pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image,o3d.camera.PinholeCameraIntrinsic(o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
# Flip it, otherwise the pointcloud will be upside down
pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
visualizer = JVisualizer()
visualizer.add_geometry(pcd)
visualizer.show()