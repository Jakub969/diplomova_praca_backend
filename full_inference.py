import torch
import open3d as o3d
import numpy as np
from pointnet2_ops.pointnet2_modules import PointnetFPModule,PointnetSAModuleMSG
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PointNet2SemSegMSG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointnetSAModuleMSG(
            npoint=1024,
            radii=[0.1, 0.2, 0.4],
            nsamples=[16, 32, 128],
            mlps=[[0, 32, 32, 64],
                  [0, 64, 64, 128],
                  [0, 64, 96, 128]],
            use_xyz=True
        )

        self.sa2 = PointnetSAModuleMSG(
            npoint=256,
            radii=[0.4, 0.8],
            nsamples=[32, 64],
            mlps=[[320, 128, 128, 256],
                  [320, 128, 196, 256]],
            use_xyz=True
        )

        self.fp1 = PointnetFPModule(mlp=[512 + 320, 256, 256])
        self.fp2 = PointnetFPModule(mlp=[256, 256, 128, num_classes])

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        l1_xyz, l1_features = self.sa1(xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l1_features = self.fp1(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp2(xyz, l1_xyz, None, l1_features)
        return l0_features  # (B, num_classes, N)

def load_model(weights_path):
    model = PointNet2SemSegMSG(num_classes=2)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

def predict_pointcloud(model, ply_path, n_points=62673):
    pcd = o3d.io.read_point_cloud(ply_path)
    pts = np.asarray(pcd.points, dtype=np.float32)

    # Resampling (ako v datasete)
    min_len = len(pts)
    if min_len >= n_points:
        idx = np.random.choice(min_len, n_points, replace=False)
    else:
        idx = np.random.choice(min_len, n_points, replace=True)

    pts = pts[idx]

    # (1, 3, N)
    pts_tensor = torch.tensor(pts.T, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(pts_tensor)       # (1, num_classes, N)
        preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()

    return pts, preds

import sys

if __name__ == "__main__":
    input_ply = sys.argv[1]
    output_ply = sys.argv[2]
    weights_path = sys.argv[3]

    model = load_model(weights_path)
    pts, preds = predict_pointcloud(model, input_ply)

    pcd_out = o3d.geometry.PointCloud()
    pcd_out.points = o3d.utility.Vector3dVector(pts)

    colors = np.zeros((len(preds), 3))
    colors[preds == 0] = [1, 0, 0]
    colors[preds == 1] = [0, 1, 0]

    pcd_out.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(output_ply, pcd_out)