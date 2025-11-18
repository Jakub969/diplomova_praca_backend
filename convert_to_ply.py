import open3d as o3d

points = []
colors = []

with open("C:/Users/jakub\Desktop/Jakub/diplomova_praca/Automated Tracker/04 SCENES/video_stromu1/sparse/points3D.txt", "r") as f:
    for line in f:
        if line.startswith("#"):
            continue
        vals = line.split()
        x, y, z = map(float, vals[1:4])
        r, g, b = map(float, vals[4:7])
        points.append([x, y, z])
        colors.append([r / 255.0, g / 255.0, b / 255.0])

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("points3D.ply", pcd)
