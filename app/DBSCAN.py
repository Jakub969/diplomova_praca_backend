import open3d as o3d
import numpy as np

def filter_point_cloud(input_ply: str, output_ply: str):
    # --------------------------------------------------
    # 1. Load point cloud
    # --------------------------------------------------
    pcd = o3d.io.read_point_cloud(input_ply)
    res = pcd.remove_non_finite_points()
    pcd = res[0] if isinstance(res, tuple) else res


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

    y_threshold = length_of_y * 0.15
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

    # --------------------------------------------------
    # CONDITIONAL DOWNSAMPLING (len pre výpočet)
    # --------------------------------------------------

    MAX_POINTS = 150000

    pcd_full = pcd  # originál si uložíme
    num_points = len(pcd.points)

    print("Points before clustering:", num_points)

    if num_points > MAX_POINTS:
        print("Using voxel downsampling for processing only...")

        # automatický voxel podľa veľkosti scény
        bbox = pcd.get_axis_aligned_bounding_box()
        scene_size = max(bbox.get_extent())

        voxel_size = scene_size / 800
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
    if num_points > MAX_POINTS:
        min_points = 50
    else:
        min_points = 5

    eps = avg_dist*2

    labels = np.array(
        pcd_work.cluster_dbscan(
            eps=eps,
            min_points=min_points,
            print_progress=True
        )
    )

    valid = labels >= 0
    unique, counts = np.unique(labels[valid], return_counts=True)

    largest_label = unique[np.argmax(counts)]
    idx = np.where(labels == largest_label)[0]

    tree_cluster_down = pcd_work.select_by_index(idx)

    print("Largest cluster (processing cloud):", len(tree_cluster_down.points))
    tree_points = np.asarray(tree_cluster_down.points)

    pcd_tree = o3d.geometry.PointCloud()
    pcd_tree.points = o3d.utility.Vector3dVector(tree_points)

    tree_kdtree = o3d.geometry.KDTreeFlann(pcd_tree)

    indices_full = []

    for i, point in enumerate(pcd_full.points):
        [k, idx, _] = tree_kdtree.search_radius_vector_3d(point, eps)
        if k > 0:
            indices_full.append(i)

    tree_pcd = pcd_full.select_by_index(indices_full)

    print("Final tree size:", len(tree_pcd.points))
    bbox = tree_pcd.get_axis_aligned_bounding_box()
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    points_full = np.asarray(pcd_full.points)
    mask = (
        (points_full[:, 0] >= min_bound[0]) &
        (points_full[:, 0] <= max_bound[0]) &
        (points_full[:, 2] >= min_bound[2]) &
        (points_full[:, 2] <= max_bound[2])
    )

    indices = np.where(mask)[0]
    filtered = pcd_full.select_by_index(indices)

    filtered_np = np.asarray(filtered.points)
    bbox = filtered.get_axis_aligned_bounding_box()

    y_min = filtered_np[:, 1].min()
    y_max = filtered_np[:, 1].max()
    if y_min < 0:
        if num_points > MAX_POINTS:
            y_threshold = (y_max + abs(y_min))*0.45
        else:
            y_threshold = (y_max + abs(y_min)) * 0.17
    else:
        if num_points > MAX_POINTS:
            y_threshold = (y_max - y_min)*0.45
        else:
            y_threshold = (y_max - y_min)*0.17

    mask = filtered_np[:, 1] >= y_min + y_threshold
    filtered_np = filtered_np[mask]

    filtered_pcd = o3d.geometry.PointCloud()
    filtered_pcd.points = o3d.utility.Vector3dVector(filtered_np)

    print("Filtered size:", len(filtered_pcd.points))
    # --------------------------------------------------
    # 6. Save result
    # --------------------------------------------------
    o3d.io.write_point_cloud(output_ply, filtered_pcd)
    print("Saved:", output_ply)
