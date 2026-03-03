import open3d as o3d

def convert_ply_to_glb(input_ply, output_glb):
    pcd = o3d.io.read_point_cloud(str(input_ply))

    # Downsample (VERY important for mobile)
    pcd = pcd.voxel_down_sample(voxel_size=0.01)

    # Estimate normals (needed for meshing)
    pcd.estimate_normals()

    # Poisson surface reconstruction
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8
    )

    mesh.compute_vertex_normals()

    # Save as GLB
    o3d.io.write_triangle_mesh(str(output_glb), mesh)