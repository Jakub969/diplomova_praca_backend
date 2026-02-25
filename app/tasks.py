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
def process_video(job_id: str, video_path_str: str, quality: str):
    total_progress = 0

    def update(p):
        nonlocal total_progress
        total_progress += p
        celery.update_state(
            state="PROGRESS",
            meta={"progress": total_progress}
        )

    job_dir = JOBS_DIR / job_id
    image_dir = job_dir / "images"
    sparse_dir = job_dir / "sparse"
    dense_dir = job_dir / "dense"
    db_path = job_dir / "database.db"

    image_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)
    dense_dir.mkdir(exist_ok=True)

    video_path = Path(video_path_str)

    sparse_glb = sparse_dir / "sparse_prediction.glb"
    dense_glb = dense_dir / "dense_prediction.glb"

    if not sparse_glb.exists():

        if not any(image_dir.glob("*.jpg")):
            subprocess.run([
                FFMPEG_PATH,
                "-i", str(video_path),
                "-qscale:v", "2",
                str(image_dir / "frame_%06d.jpg")
            ], check=True)
            update(5)
        if not db_path.exists():
            pycolmap.extract_features(
                database_path=db_path,
                image_path=image_dir,
                camera_mode=pycolmap.CameraMode.SINGLE,
                device=pycolmap.Device.cuda
            )
            update(20)
            pycolmap.match_sequential(
                database_path=db_path,
                device=pycolmap.Device.cuda,
            )
            update(15)
        if not (sparse_dir / "0").exists():
            pycolmap.incremental_mapping(
                database_path=db_path,
                image_path=image_dir,
                output_path=sparse_dir
            )
            update(25)
        model_path = sorted(sparse_dir.glob("*"))[0]

        sparse_ply = sparse_dir / "sparse_points.ply"

        reconstruction = pycolmap.Reconstruction(model_path)

        points = []
        colors = []

        for point3D in reconstruction.points3D.values():
            points.append(point3D.xyz)
            colors.append(point3D.color / 255.0)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.array(points))
        pcd.colors = o3d.utility.Vector3dVector(np.array(colors))
        o3d.io.write_point_cloud(sparse_ply, pcd)

        filtered_sparse = sparse_dir / "filtered_sparse.ply"
        filter_point_cloud(str(sparse_ply), str(filtered_sparse))
        update(5)

        subprocess.run([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{job_dir}:/workspace/job",
            "pointnet2_inference:latest",
            "python", "full_inference.py",
            "/workspace/job/sparse/filtered_sparse.ply",
            "/workspace/job/sparse/prediction_sparse.ply",
            "/workspace/best_pointnet2_model.pth"
        ], check=True)
        update(20)
        convert_ply_to_glb(
            sparse_dir / "prediction_sparse.ply",
            sparse_glb
        )
        update(10)
    if quality == "dense" and not dense_glb.exists():
        total_progress = 0

        model_path = sorted(sparse_dir.glob("*"))[0]

        pycolmap.undistort_images(
            output_path=dense_dir,
            input_path=model_path,
            image_path=image_dir
        )
        update(10)

        pycolmap.patch_match_stereo(
            workspace_path=dense_dir
        )
        update(40)

        pycolmap.stereo_fusion(
            output_path=dense_dir / "fused.ply",
            workspace_path=dense_dir
        )
        update(20)

        filtered_dense = dense_dir / "filtered_dense.ply"
        filter_point_cloud(
            str(dense_dir / "fused.ply"),
            str(filtered_dense)
        )
        update(10)

        subprocess.run([
            "docker", "run", "--rm",
            "--gpus", "all",
            "-v", f"{job_dir}:/workspace/job",
            "pointnet2_inference:latest",
            "python", "full_inference.py",
            "/workspace/job/dense/filtered_dense.ply",
            "/workspace/job/dense/prediction_dense.ply",
            "/workspace/best_pointnet2_model.pth"
        ], check=True)
        update(10)
        convert_ply_to_glb(
            dense_dir / "prediction_dense.ply",
            dense_glb
        )
        update(10)

    return {"status": "done"}