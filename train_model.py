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

# =====================
#   HYPERPARAMETRE
# =====================
BATCH_SIZE = 2
LR = 0.0025
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
PC_INFO = os.path.join(DATA_DIR, "point_counts.csv")
LABEL_INFO = os.path.join(DATA_DIR, "voxel_labels_0.0025.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
#   DATASET
# =====================
class BranchDataset(Dataset):
    def __init__(self, data_dir, pc_info_path, label_path, n_points):
        self.data_dir = data_dir
        self.pc_info = pd.read_csv(pc_info_path)
        self.labels_df = pd.read_csv(label_path)
        self.n_points = n_points

        # upraviť názvy, aby sa zhodovali
        self.pc_info['tree_name'] = self.pc_info['file_name'].str.replace('_voxelized.ply', '', regex=False)
        self.labels_df['tree'] = self.labels_df['tree'].astype(str)

        # mapovanie mena na label
        label_dict = dict(zip(self.labels_df['tree'], self.labels_df['labels']))

        # filtrovanie len stromov, ktoré majú label aj dostatok bodov
        self.pc_info = self.pc_info[self.pc_info['tree_name'].isin(label_dict.keys())]
        self.pc_info = self.pc_info[self.pc_info['point_count'] >= n_points]

        self.files = self.pc_info['file_name'].tolist()
        self.labels = [label_dict[name.replace('_voxelized.ply', '')] for name in self.files]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        label_str = self.labels[idx]

        # labely sú v tvare "[0, 1, 0, ...]" — vyberieme 1, ak niekde existuje
        label = 1 if '1' in label_str else 0

        pcd = o3d.io.read_point_cloud(os.path.join(self.data_dir, file_name))
        pts = np.asarray(pcd.points, dtype=np.float32)

        if pts.shape[0] > self.n_points:
            choice = np.random.choice(pts.shape[0], self.n_points, replace=False)
            pts = pts[choice, :]
        elif pts.shape[0] < self.n_points:
            choice = np.random.choice(pts.shape[0], self.n_points - pts.shape[0], replace=True)
            pts = np.concatenate([pts, pts[choice, :]], axis=0)

        pts = torch.tensor(pts).float().T
        return pts, label


# =====================
#   MODEL (PointNet)
# =====================
class PointNetCls(nn.Module):
    def __init__(self, num_classes=2):
        super(PointNetCls, self).__init__()
        self.feat = nn.Sequential(
            nn.Conv1d(3, 64, 1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Conv1d(128, 1024, 1),
            nn.BatchNorm1d(1024),
            nn.ReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.feat(x)             # (B, 1024, N)
        x = torch.max(x, 2)[0]       # (B, 1024)
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
model = PointNetCls(num_classes=N_CLASSES).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_FACTOR)

train_losses, val_losses = [], []

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
correct, total = 0, 0
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
torch.save(model.state_dict(), "pointnet_pruned_classifier_voxel_0_0025.pth")
print("Model uložený do pointnet_pruned_classifier_voxel_0_0025.pth")

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
plt.savefig("loss_plot_voxel_0_0025.png", dpi=300)
plt.show()