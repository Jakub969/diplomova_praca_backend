import os
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
from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule

# =====================
#   HYPERPARAMETRE
# =====================
BATCH_SIZE = 2
LR = 0.0025
OPTIMIZER = 'Adam'
WEIGHT_DECAY = 1e-4
LR_DECAY_FACTOR = 0.5
LR_DECAY_STEP = 20
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

DATA_DIR = r"C:\Users\jakub\Desktop\Jakub\dataset\BRANCH_v2\BRANCH_v2\BRANCH_Prediction\normalized_centered_voxelized_models_0.0025"
PC_INFO = os.path.join(DATA_DIR, "point_clouds.xlsx")
LABEL_INFO = os.path.join(DATA_DIR, "voxel_labels_0.0025.xlsx")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
#   DATASET
# =====================


class BranchDataset(Dataset):
    def __init__(self, data_dir, pc_info_path, label_path, n_points):
        self.data_dir = data_dir
        self.pc_info = pd.read_excel(pc_info_path)
        self.labels_df = pd.read_excel(label_path)
        self.n_points = n_points

        # mapovanie mena súboru na label
        label_dict = dict(zip(self.labels_df['tree'], self.labels_df['labels']))
        self.pc_info = self.pc_info[self.pc_info['file_name'].isin(label_dict.keys())]
        self.pc_info = self.pc_info[self.pc_info['point_count'] >= n_points]

        self.files = self.pc_info['file_name'].tolist()
        self.labels = [label_dict[f] for f in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label = int(self.labels[idx])

        pcd = o3d.io.read_point_cloud(os.path.join(self.data_dir, file_name))
        pts = np.asarray(pcd.points, dtype=np.float32)

        # náhodný sampling (ak viac bodov)
        if pts.shape[0] > self.n_points:
            choice = np.random.choice(pts.shape[0], self.n_points, replace=False)
            pts = pts[choice, :]
        elif pts.shape[0] < self.n_points:
            # doplnenie náhodnými duplikátmi
            choice = np.random.choice(pts.shape[0], self.n_points - pts.shape[0], replace=True)
            pts = np.concatenate([pts, pts[choice, :]], axis=0)

        pts = torch.tensor(pts).float().T  # (3, N)
        return pts, label

# =====================
#   MODEL (PointNet++)
# =====================
class PointNet2ClsMSG(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.sa1 = PointnetSAModule(npoint=1024, radii=[0.1, 0.2], nsamples=[32, 64],
                                    mlps=[[3, 32, 32, 64], [3, 64, 64, 128]])
        self.sa2 = PointnetSAModule(npoint=256, radii=[0.2, 0.4], nsamples=[32, 64],
                                    mlps=[[192, 128, 128, 256], [192, 128, 196, 256]])
        self.sa3 = PointnetSAModule(npoint=None, radii=None, nsamples=None,
                                    mlps=[[512, 256, 512, 1024]], use_xyz=False)

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, xyz):
        xyz = xyz.transpose(1, 2)  # (B, N, 3)
        l1_xyz, l1_features = self.sa1(xyz, None)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        x = torch.max(l3_features, 2)[0]
        x = self.fc(x)
        return x

# =====================
#   DÁTA A LOADERY
# =====================
dataset = BranchDataset(DATA_DIR, PC_INFO, LABEL_INFO, N_POINTS)

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
model = PointNet2ClsMSG(num_classes=N_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)

train_losses = []
val_losses = []

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0
    for pts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        pts, labels = pts.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(pts)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # validácia
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for pts, labels in val_loader:
            pts, labels = pts.to(device), labels.to(device)
            outputs = model(pts)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step()

    print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

# =====================
#   TESTOVANIE PRESNOSTI
# =====================
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for pts, labels in tqdm(test_loader, desc="Testing"):
        pts, labels = pts.to(device), labels.to(device)
        outputs = model(pts)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Presnosť na testovacej množine: {accuracy:.2f}%")

# =====================
#   ULOŽENIE MODELU
# =====================
torch.save(model.state_dict(), "pointnet2_pruned_classifier_voxel_0_0025.pth")
print("Model uložený do pointnet2_pruned_classifier.pth")

# =====================
#   VIZUALIZÁCIA STRÁT
# =====================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Priebeh stratovej funkcie')
plt.legend()
plt.grid(True)
plt.tight_layout()
# uloženie grafu
plt.savefig("loss_plot_voxel_0_0025.png", dpi=300)
plt.show()