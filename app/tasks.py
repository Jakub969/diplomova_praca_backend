import subprocess
import pycolmap
from pathlib import Path

from worker import celery
from settings import JOBS_DIR, FFMPEG_PATH


@celery.task(name="tasks.process_video")
def process_video(job_id: str, video_path_str: str):

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

    return {
        "status": "done",
        "ply": str(dense_dir / "fused.ply")
    }
