import subprocess
import pycolmap
from pathlib import Path
import open3d as o3d
import numpy as np

from convert_to_glb import convert_ply_to_glb
from worker import celery
from settings import JOBS_DIR, FFMPEG_PATH
from DBSCAN import filter_point_cloud


@celery.task(name="tasks.process_video")
def process_video(job_id: str, video_path_str: str, quality):

    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "images"
    sparse_dir = job_dir / "sparse"
    dense_dir = job_dir / "dense"
    db_path = job_dir / "database.db"

    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    video_path = Path(video_path_str)

    # -------------------------
    # 1) Frame extraction
    # -------------------------
    subprocess.run([
        FFMPEG_PATH,
        "-i", str(video_path),
        "-qscale:v", "2",
        str(image_dir / "frame_%06d.jpg")
    ], check=True)

    # -------------------------
    # 2) Feature extraction (GPU)
    # -------------------------
    pycolmap.extract_features(
        database_path=db_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,
        device=pycolmap.Device.cuda
    )

    # -------------------------
    # 3) Sequential matching
    # -------------------------
    pycolmap.match_sequential(
        database_path=db_path,
        device=pycolmap.Device.cuda,
    )

    # -------------------------
    # 4) Sparse mapping
    # -------------------------
    pycolmap.incremental_mapping(
        database_path=db_path,
        image_path=image_dir,
        output_path=sparse_dir
    )

    model_path = sparse_dir / "0"

    # -------------------------
    # 5) Dense reconstruction
    # -------------------------
    if quality == "dense":
        pycolmap.undistort_images(
            output_path=dense_dir,
            input_path=model_path,
            image_path=image_dir
        )

        pycolmap.patch_match_stereo(
            workspace_path=dense_dir
        )

        pycolmap.stereo_fusion(
            output_path=dense_dir / "fused.ply",
            workspace_path=dense_dir
        )
        input_ply = dense_dir / "fused.ply"
    else:
        sparse_ply = sparse_dir / "points3D.ply"

        reconstruction = pycolmap.Reconstruction(sparse_dir)

        points = []
        colors = []

        for point3D in reconstruction.points3D.values():
            points.append(point3D.xyz)
            colors.append(point3D.color / 255.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        o3d.io.write_point_cloud(sparse_ply, pcd)
        input_ply = sparse_ply
    output_ply = job_dir / "filtered_scene.ply"
    filter_point_cloud(str(input_ply), str(output_ply))
    prediction_ply = job_dir / "prediction_output.ply"

    subprocess.run([
        "docker", "run", "--rm",
        "--gpus", "all",
        "-v", f"{job_dir}:/workspace/job",
        "pointnet2_inference:latest",
        "python", "full_inference.py",
        "/workspace/job/filtered_scene.ply",
        "/workspace/job/prediction_output.ply",
        "/workspace/best_pointnet2_model.pth"
    ], check=True)

    glb_path = job_dir / "prediction_output.glb"
    convert_ply_to_glb(input_ply, glb_path)