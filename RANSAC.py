import open3d as o3d
import numpy as np

INPUT_PLY = "points3D.ply"
OUTPUT_PLY = "points_tree_only_ransac.ply"

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

# --------------------------------------------------
# 2. Remove ground plane (RANSAC)
# --------------------------------------------------
plane_model, inliers = pcd.segment_plane(
    distance_threshold=0.15,   # nastav podľa mierky (COLMAP!)
    ransac_n=3,
    num_iterations=1000
)

[a, b, c, d] = plane_model
print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.3f} = 0")

ground = pcd.select_by_index(inliers)
nonground = pcd.select_by_index(inliers, invert=True)

print(f"Ground removed: {len(ground.points)} points")
print(f"Remaining: {len(nonground.points)} points")

# --------------------------------------------------
# 3. Remove outliers (noise)
# --------------------------------------------------
nonground, ind = nonground.remove_radius_outlier(
    nb_points=10,
    radius=0.05
)

print(f"After outlier removal: {len(nonground.points)} points")

# --------------------------------------------------
# 4. Clustering (DBSCAN)
# --------------------------------------------------
labels = np.array(
    nonground.cluster_dbscan(
        eps=0.75,      # závisí od mierky!
        min_points=100,
        print_progress=True
    )
)

max_label = labels.max()
print(f"Found {max_label + 1} clusters")

# --------------------------------------------------
# 5. Select tree cluster
# --------------------------------------------------
clusters = []
for i in range(max_label + 1):
    idx = np.where(labels == i)[0]
    cluster = nonground.select_by_index(idx)
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
