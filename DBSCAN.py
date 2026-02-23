import open3d as o3d
import numpy as np

INPUT_PLY = "jobs/93259f1c1293404891108465e86e3566/dense/fused.ply"
OUTPUT_PLY = "points_tree_only_dbscan1.ply"

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
    points = np.asarray(pcd.points)

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

# --------------------------------------------------
# CONDITIONAL DOWNSAMPLING (len pre výpočet)
# --------------------------------------------------

MAX_POINTS = 150_000

pcd_full = pcd  # originál si uložíme
num_points = len(pcd.points)

print("Points before clustering:", num_points)

if num_points > MAX_POINTS:
    print("Using voxel downsampling for processing only...")

    # automatický voxel podľa veľkosti scény
    bbox = pcd.get_axis_aligned_bounding_box()
    scene_size = max(bbox.get_extent())

    voxel_size = scene_size / 300  # stabilné pre stromy
    print("Voxel size:", voxel_size)

    pcd_work = pcd.voxel_down_sample(voxel_size)
    points_work = np.asarray(pcd_work.points)

    sample_size = min(50000, len(points_work))
    idx = np.random.choice(len(points_work), sample_size, replace=False)
    sample_pcd = pcd_work.select_by_index(idx)

    distances = sample_pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)

    print("Avg NN distance:", avg_dist)
else:
    print("Downsampling not needed")
    pcd_work = pcd
    distances = pcd.compute_nearest_neighbor_distance()
    avg_dist = np.mean(distances)
    print("Avg NN distance:", avg_dist)
print("Points used for DBSCAN:", len(pcd_work.points))

# --------------------------------------------------
# 4. Clustering (DBSCAN)
# --------------------------------------------------

eps = avg_dist * 5

labels = np.array(
    pcd_work.cluster_dbscan(
        eps=eps,
        min_points=50,
        print_progress=True
    )
)

valid = labels >= 0
unique, counts = np.unique(labels[valid], return_counts=True)

largest_label = unique[np.argmax(counts)]
idx = np.where(labels == largest_label)[0]

tree_cluster_down = pcd_work.select_by_index(idx)

print("Largest cluster (processing cloud):", len(tree_cluster_down.points))
if num_points > MAX_POINTS:
    bbox = tree_cluster_down.get_axis_aligned_bounding_box()
    indices = bbox.get_point_indices_within_bounding_box(pcd_full.points)
    tree_pcd = pcd_full.select_by_index(indices)
else:
    tree_pcd = tree_cluster_down

print("Final tree size:", len(tree_pcd.points))
# --------------------------------------------------
# 6. Save result
# --------------------------------------------------
o3d.io.write_point_cloud(OUTPUT_PLY, tree_pcd)
print("Saved:", OUTPUT_PLY)
