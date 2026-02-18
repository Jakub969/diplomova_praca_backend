import subprocess
import pycolmap
from pathlib import Path

def run_colmap_video_pipeline(video_path: Path, workspace: Path, ffmpeg_path: Path):
    workspace.mkdir(parents=True, exist_ok=True)

    image_dir = workspace / "images"
    sparse_dir = workspace / "sparse"
    database_path = workspace / "database.db"

    image_dir.mkdir(exist_ok=True)
    sparse_dir.mkdir(exist_ok=True)

    # -------------------------------------------------
    # 1) Extract frames using FFmpeg
    # -------------------------------------------------
    print("Extracting frames...")
    subprocess.run([
        str(ffmpeg_path),
        "-i", str(video_path),
        "-qscale:v", "2",
        str(image_dir / "frame_%06d.jpg")
    ], check=True)

    # -------------------------------------------------
    # 2) Feature extraction
    # -------------------------------------------------
    print("Extracting features...")
    pycolmap.extract_features(
        database_path=database_path,
        image_path=image_dir,
        camera_mode=pycolmap.CameraMode.SINGLE,  # same as --ImageReader.single_camera 1
        sift_options={
            "use_gpu": True,
            "max_image_size": 4096
        }
    )

    # -------------------------------------------------
    # 3) Sequential matching (same as your .bat)
    # -------------------------------------------------
    print("Matching features (sequential)...")
    pycolmap.match_sequential(
        database_path=database_path,
        overlap=15
    )

    # -------------------------------------------------
    # 4) Sparse reconstruction
    # -------------------------------------------------
    print("Running mapper...")
    maps = pycolmap.incremental_mapping(
        database_path=database_path,
        image_path=image_dir,
        output_path=sparse_dir
    )

    print("Finished reconstruction")

    for model_id, rec in maps.items():
        print(f"Model {model_id}: "
              f"{len(rec.images)} images, "
              f"{len(rec.points3D)} points")

    return sparse_dir
