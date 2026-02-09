import torch
import numpy as np
import open3d as o3d
from KPConv.models import KPConvNPM3D  # uprav podľa štruktúry projektu
from KPConv.datasets import NPM3DDataset  # ak existuje, alebo si priprav vlastný loader
from KPConv.config import Config  # ak máš konfiguračný súbor

# Nastavenie zariadenia
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Načítanie predtrénovaného modelu (uprav cestu k modelu)
model_path = "path_to_pretrained_npm3d_model.pth"
model = KPConvNPM3D(num_classes=... )  # nastav podľa počtu tried v NPM3D
model.load_state_dict(torch.load(model_path))
model.to(device)
model.eval()

# Načítanie point cloud
pcd = o3d.io.read_point_cloud("your_pointcloud.ply")
points = np.asarray(pcd.points, dtype=np.float32)

# Predspracovanie podľa modelu (normalizácia, prípadne voxelizácia)
# Tu si pridaj potrebné kroky podľa tréningovej pipeline

# Vytvor tensor, pridaj batch dimenziu
input_points = torch.tensor(points).float().to(device)
input_points = input_points.unsqueeze(0)  # batch size = 1

# Forward pass
with torch.no_grad():
    outputs = model(input_points)  # predpokladám, že výstup je (B, N, num_classes)
    preds = torch.argmax(outputs, dim=-1).squeeze(0).cpu().numpy()

# Výsledok - preds obsahuje labely pre každý bod
print(preds)

# Voliteľne: ulož označený point cloud
pcd.colors = o3d.utility.Vector3dVector(plt.get_cmap("tab20")(preds / preds.max())[:, :3])
o3d.io.write_point_cloud("labeled_pointcloud.ply", pcd)
