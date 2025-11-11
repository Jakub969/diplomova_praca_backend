import pycolmap
from pathlib import Path

# --- Input/output paths ---
image_dir = Path("images/tree_photos")      # Folder with your photos
workspace_dir = Path("output/tree_reconstruction")  # Output folder for COLMAP data

workspace_dir.mkdir(parents=True, exist_ok=True)

# --- Step 1: Feature extraction ---
print("Extracting features...")
pycolmap.extract_features(
    database_path=workspace_dir / "database.db",
    image_path=image_dir
)

# --- Step 2: Feature matching ---
print("Matching features...")
pycolmap.match_exhaustive(
    database_path=workspace_dir / "database.db"
)

# --- Step 3: Sparse reconstruction (Structure-from-Motion) ---
print("Running sparse reconstruction...")
maps = pycolmap.incremental_mapping(
    database_path=workspace_dir / "database.db",
    image_path=image_dir,
    output_path=workspace_dir / "sparse"
)

# You can access the reconstructed models (if multiple):
for model_id, rec in maps.items():
    print(f"Model {model_id}: {len(rec.images)} images, {len(rec.points3D)} points")

# --- Step 4: Dense reconstruction (optional, for denser point cloud) ---
print("Running dense reconstruction...")
dense_model_dir = workspace_dir / "dense"
pycolmap.stereo.run_stereo(
    workspace_dir / "sparse",
    image_dir,
    dense_model_dir
)

# --- Step 5: Convert to point cloud ---
print("Exporting point cloud...")
rec = pycolmap.Reconstruction(dense_model_dir / "dense.ply")
print(f"Exported dense point cloud: {dense_model_dir / 'dense.ply'}")
