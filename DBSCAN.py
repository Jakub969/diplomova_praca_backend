import open3d as o3d
import numpy as np

INPUT_PLY = "points3D.ply"
OUTPUT_PLY = "points_tree_only_dbscan.ply"

# --------------------------------------------------
# 1. Load point cloud
# --------------------------------------------------
pcd = o3d.io.read_point_cloud(INPUT_PLY)
res = pcd.remove_non_finite_points()
if isinstance(res, tuple):
    pcd = res[0]
else:
    pcd = res


print(f"Loaded {len(pcd.points)} points")

points = np.asarray(pcd.points)

print("Y min:", points[:,1].min())
print("Y max:", points[:,1].max())

extent = pcd.get_axis_aligned_bounding_box().get_extent()
print("Extent X,Y,Z:", extent)

# --------------------------------------------------
# 2. Remove ground
# --------------------------------------------------
points = np.asarray(pcd.points)

y_min = points[:, 1].min()
y_max = points[:, 1].max()

#cela plocha
if y_min < 0:
    length_of_y = abs(y_min) + abs(y_max)
else:
    length_of_y = y_max - y_min

y_threshold = length_of_y * 0.05
print("Dolná hranica: ", y_min + y_threshold)
print("Horná hranica: ", y_max - y_threshold)

# hustota bodov spodku vs vrchu
bottom_density = np.sum(points[:,1] < y_min + y_threshold)
print("Bottom density before:", bottom_density)
top_density = np.sum(points[:,1] >= y_min + y_threshold)
print("Top density before:", top_density)

if top_density > bottom_density:
    # strom je hore nohami, otočíme
    R = pcd.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    pcd.rotate(R, center=(0, 0, 0))

y_min = points[:, 1].min()
y_max = points[:, 1].max()

#cela plocha
if y_min < 0:
    length_of_y = abs(y_min) + abs(y_max)
else:
    length_of_y = y_max - y_min

y_threshold = length_of_y * 0.365
print("Dolná hranica: ", y_min + y_threshold)
print("Horná hranica: ", y_max - y_threshold)

# hustota bodov spodku vs vrchu
bottom_density = np.sum(points[:,1] < y_min + y_threshold)
print("Bottom density after:", bottom_density)
top_density = np.sum(points[:,1] >= y_min + y_threshold)
print("Top density after:", top_density)

mask = points[:, 1] > y_min + y_threshold
pcd = pcd.select_by_index(np.where(mask)[0])

distances = pcd.compute_nearest_neighbor_distance()
avg_dist = np.mean(distances)
print("Avg NN distance:", avg_dist)

# --------------------------------------------------
# 4. Clustering (DBSCAN)
# --------------------------------------------------
eps = avg_dist * 5
labels = np.array(
    pcd.cluster_dbscan(
        eps=eps,
        min_points=50,
        print_progress=True
    )
)


max_label = labels.max()
print(f"Found {max_label + 1} clusters")

# --------------------------------------------------
# 5. Select tree cluster
# --------------------------------------------------
max_label = labels.max()

clusters = []
for i in range(max_label + 1):
    idx = np.where(labels == i)[0]
    cluster = pcd.select_by_index(idx)
    clusters.append(cluster)

# Heuristika: strom = najväčší cluster
clusters.sort(key=lambda c: len(c.points), reverse=True)
tree_pcd = clusters[0]

print(f"Tree cluster size: {len(tree_pcd.points)} points")

# --------------------------------------------------
# 6. Save result
# --------------------------------------------------
o3d.io.write_point_cloud(OUTPUT_PLY, tree_pcd)
print("Saved:", OUTPUT_PLY)
