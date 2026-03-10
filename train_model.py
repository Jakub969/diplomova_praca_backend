import os
import ast
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d as o3d
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from pointnet2_ops.pointnet2_modules import _PointnetSAModuleBase,PointnetFPModule
from pointnet2_ops import pointnet2_utils
from typing import Optional, Tuple
import torch.nn.functional as F


# =====================
#   HYPERPARAMETRE
# =====================
BATCH_SIZE = 2
LR = 0.001
WEIGHT_DECAY = 1e-3
EPOCHS = 100
N_CLASSES = 2
N_POINTS = 62673
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.1, 0.2
SEED = 42

# =====================
#   NASTAVENIA
# =====================
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DATA_DIR = "normalized_centered_voxelized_models_0.0025"
PC_INFO = os.path.join(DATA_DIR, "point_counts.csv")
LABEL_INFO = os.path.join(DATA_DIR, "voxel_labels_0.0025.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
#   DATASET
# =====================

class BranchSegmentationDataset(Dataset):
    def __init__(self, data_dir, pc_info_path, label_path, n_points):
        self.data_dir = data_dir
        self.pc_info = pd.read_csv(pc_info_path)
        self.labels_df = pd.read_csv(label_path)
        self.n_points = n_points

        # Map tree name (without _merged_teaser.ply) to label array
        self.labels_df['labels'] = self.labels_df['labels'].apply(ast.literal_eval)
        label_dict = dict(zip(self.labels_df['tree'], self.labels_df['labels']))

        self.pc_info = self.pc_info[self.pc_info['file_name'].str.replace('_voxelized.ply', '').isin(label_dict.keys())]
        self.files = self.pc_info['file_name'].tolist()
        self.labels = [label_dict[f.replace('_voxelized.ply', '')] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label_array = np.array(self.labels[idx], dtype=np.int64)

        pcd = o3d.io.read_point_cloud(os.path.join(self.data_dir, file_name))
        pts = np.asarray(pcd.points, dtype=np.float32)

        # Match labels and points (downsample/upsample if needed)
        min_len = min(len(pts), len(label_array))
        pts, label_array = pts[:min_len], label_array[:min_len]

        if min_len > self.n_points:
            choice = np.random.choice(min_len, self.n_points, replace=False)
        else:
            choice = np.random.choice(min_len, self.n_points, replace=True)

        pts = pts[choice, :]
        label_array = label_array[choice]

        pts = torch.tensor(pts).float().T  # (3, N)
        labels = torch.tensor(label_array).long()  # (N,)
        return pts, labels

##ChannelAttention
class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=8):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):

        # x: (B,C,N,K)

        w = x.mean(dim=(2,3))     # global spatial pooling
        w = self.fc(w)            # (B,C)
        w = w.view(w.size(0), w.size(1), 1, 1)

        return x * w

##CAMA_MLP
class CAMA_MLP(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.ca1 = ChannelAttention(64)
        self.ca2 = ChannelAttention(128)
        self.ca3 = ChannelAttention(256)

        self.fuse = nn.Sequential(
            nn.Conv2d(64 + 128 + 256, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(0.3),
            nn.ReLU()
        )

    def forward(self, x):

        b1 = self.branch1(x)
        b1 = self.ca1(b1)

        b2 = self.branch2(x)
        b2 = self.ca2(b2)

        b3 = self.branch3(x)
        b3 = self.ca3(b3)

        x = torch.cat([b1, b2, b3], dim=1)

        x = self.fuse(x)

        return x

##CPF
def compute_cpf(grouped_features):

    xyz = grouped_features[:, :3]
    features = grouped_features[:, 3:] if grouped_features.shape[1] > 3 else xyz

    center_xyz = xyz[:, :, :, 0:1]
    relative_xyz = xyz - center_xyz

    return torch.cat(
        [features, relative_xyz, center_xyz.expand_as(relative_xyz)],
        dim=1
    )
# =====================
#   MODEL (PointNet++)
# =====================
class PointnetSAModuleMSG_CAMA(_PointnetSAModuleBase):

    def __init__(self, npoint, radii, nsamples, mlps, bn=True, use_xyz=True):

        super().__init__()

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        assert len(radii) == len(nsamples) == len(mlps)

        for i in range(len(radii)):

            radius = radii[i]
            nsample = nsamples[i]

            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None
                else pointnet2_utils.GroupAll(use_xyz)
            )

            mlp_spec = mlps[i].copy()

            in_channels = mlp_spec[0] + 6

            self.mlps.append(CAMA_MLP(in_channels, mlp_spec[-1]))

    def forward(
            self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        new_xyz = (
            pointnet2_utils.gather_operation(
                xyz_flipped, pointnet2_utils.furthest_point_sample(xyz, self.npoint)
            )
            .transpose(1, 2)
            .contiguous()
            if self.npoint is not None
            else None
        )

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](
                xyz, new_xyz, features
            )  # (B, C, npoint, nsample)
            new_features = compute_cpf(new_features)
            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)

            new_features = F.max_pool2d(
                new_features, kernel_size=[1, new_features.size(3)]
            )  # (B, mlp[-1], npoint, 1)
            new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)

class PointNet2SemSegMSG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointnetSAModuleMSG_CAMA(
          npoint=1024,
          radii=[0.1, 0.2, 0.4],
          nsamples=[16, 32, 64],
          mlps=[[3, 32, 32, 64],
                [3, 64, 64, 128],
                [3, 64, 96, 128]],
          use_xyz=True
        )

        self.sa2 = PointnetSAModuleMSG_CAMA(
          npoint=256,
          radii=[0.4, 0.8],
          nsamples=[32, 64],
          mlps=[[320, 128, 128, 256],
                [320, 128, 196, 256]],
          use_xyz=True
        )


        self.fp1 = PointnetFPModule(mlp=[512 + 320, 256, 256])  # correct: 832 input channels
        self.fp2 = PointnetFPModule(mlp=[256, 256, 128, num_classes])

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2).contiguous()  # (B, N, 3)
        l1_xyz, l1_features = self.sa1(xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)

        l1_features = self.fp1(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp2(xyz, l1_xyz, None, l1_features)

        return l0_features  # (B, num_classes, N)


# =====================
#   DÁTA A LOADERY
# =====================
dataset = BranchSegmentationDataset(DATA_DIR, PC_INFO, LABEL_INFO, N_POINTS)

train_len = int(TRAIN_RATIO * len(dataset))
val_len = int(VAL_RATIO * len(dataset))
test_len = len(dataset) - train_len - val_len
train_ds, val_ds, test_ds = random_split(dataset, [train_len, val_len, test_len])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# =====================
#   TRÉNING
# =====================
model = PointNet2SemSegMSG(num_classes=N_CLASSES).to(device)
# =====================
#   COMPUTE CLASS WEIGHTS FROM TRAINING DATA
# =====================
print("Computing class weights from training data...")

# Collect all labels from training dataset
all_training_labels = []
for idx in range(len(train_ds)):
    _, labels = train_ds[idx]  # Get pts, labels
    all_training_labels.extend(labels.numpy())

# Convert to numpy array
all_training_labels = np.array(all_training_labels)

class_counts = np.bincount(all_training_labels)
weights = (1.0 / class_counts) ** 2
weights = weights / weights.sum() * len(class_counts)
print("Weights: ", weights)

# Create weighted loss function
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))##criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,  # Reduce LR after 2 epochs of no improvement
    min_lr=1e-6
)

train_losses, val_losses = [], []
train_accs, val_accs = [], []
train_precisions, val_precisions = [], []
train_recalls, val_recalls = [], []
train_f1s, val_f1s = [], []
best_val_loss = float('inf')

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    all_preds, all_labels = [], []

    for pts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pts)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item()

        _, preds = torch.max(outputs, 1)  # (B, N)
        all_preds.extend(preds.cpu().numpy().ravel())
        all_labels.extend(labels.cpu().numpy().ravel())


    # metriky pre tréning
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    train_accs.append(accuracy_score(all_labels, all_preds))
    train_precisions.append(precision_score(all_labels, all_preds, average='binary', zero_division=0))
    train_recalls.append(recall_score(all_labels, all_preds, average='binary', zero_division=0))
    train_f1s.append(f1_score(all_labels, all_preds, average='binary', zero_division=0))

    # =====================
    # Validácia
    # =====================
    model.eval()
    val_loss = 0
    val_preds, val_labels_all = [], []

    with torch.no_grad():
        for pts, labels in val_loader:
            pts, labels = pts.to(device), labels.to(device)
            outputs = model(pts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            val_preds.extend(preds.cpu().numpy().flatten())
            val_labels_all.extend(labels.cpu().numpy().flatten())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    val_accs.append(accuracy_score(val_labels_all, val_preds))
    val_precisions.append(precision_score(val_labels_all, val_preds, average='binary', zero_division=0))
    val_recalls.append(recall_score(val_labels_all, val_preds, average='binary', zero_division=0))
    val_f1s.append(f1_score(val_labels_all, val_preds, average='binary', zero_division=0))

    scheduler.step(avg_val_loss)

    print(f"Epoch {epoch+1}: "
          f"Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, "
          f"Train Acc={train_accs[-1]:.3f}, Val Acc={val_accs[-1]:.3f}")
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        metrics = {
          "train_losses": train_losses,
          "val_losses": val_losses,
          "train_accs": train_accs,
          "val_accs": val_accs,
          "train_precisions": train_precisions,
          "val_precisions": val_precisions,
          "train_recalls": train_recalls,
          "val_recalls": val_recalls,
          "train_f1s": train_f1s,
          "val_f1s": val_f1s
        }
        torch.save(metrics, "training_metrics_modified_2.pt")

        torch.save(model.state_dict(),
                   "best_pointnet2_model_modified_2.pth")
        print(f"Best model saved at epoch {epoch+1} with val loss {avg_val_loss:.4f}")

# =====================
#   TESTOVANIE
# =====================
model.eval()
test_preds, test_labels_all = [], []
with torch.no_grad():
    for pts, labels in tqdm(test_loader, desc="Testing"):
        pts, labels = pts.to(device), labels.to(device)
        outputs = model(pts)
        _, preds = torch.max(outputs, 1)
        test_preds.extend(preds.cpu().numpy().flatten())
        test_labels_all.extend(labels.cpu().numpy().flatten())

test_acc = accuracy_score(test_labels_all, test_preds)
test_prec = precision_score(test_labels_all, test_preds, average='binary', zero_division=0)
test_rec = recall_score(test_labels_all, test_preds, average='binary', zero_division=0)
test_f1 = f1_score(test_labels_all, test_preds, average='binary', zero_division=0)

print(f"\n--- Výsledky na testovacej množine ---")
print(f"Accuracy: {test_acc:.4f}")
print(f"Precision: {test_prec:.4f}")
print(f"Recall:    {test_rec:.4f}")
print(f"F1 score:  {test_f1:.4f}")

# =====================
#   VIZUALIZÁCIA
# =====================
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Loss
axs[0, 0].plot(train_losses, label='Train Loss')
axs[0, 0].plot(val_losses, label='Val Loss')
axs[0, 0].set_title('Loss')
axs[0, 0].legend()
axs[0, 0].grid(True)

# Accuracy
axs[0, 1].plot(train_accs, label='Train Acc')
axs[0, 1].plot(val_accs, label='Val Acc')
axs[0, 1].set_title('Accuracy')
axs[0, 1].legend()
axs[0, 1].grid(True)

# Precision
axs[1, 0].plot(train_precisions, label='Train Precision')
axs[1, 0].plot(val_precisions, label='Val Precision')
axs[1, 0].set_title('Precision')
axs[1, 0].legend()
axs[1, 0].grid(True)

# Recall/F1
axs[1, 1].plot(train_recalls, label='Train Recall')
axs[1, 1].plot(val_recalls, label='Val Recall')
axs[1, 1].plot(train_f1s, '--', label='Train F1')
axs[1, 1].plot(val_f1s, '--', label='Val F1')
axs[1, 1].set_title('Recall / F1')
axs[1, 1].legend()
axs[1, 1].grid(True)

plt.tight_layout()
plt.savefig("metrics_plot_modified_2.png", dpi=300)
plt.show()