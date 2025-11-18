import sys, os
sys.path.append(os.path.join(os.getcwd(), "models"))
import tensorflow as tf
import numpy as np
import open3d as o3d

# ------------------------------------------------------------
# Nastavenia
# ------------------------------------------------------------
CHECKPOINT_PATH = "C:/Users/jakub/Desktop/Jakub/diplomova_praca/Semantic3D/snap-37001"  # bez prípony
INPUT_PLY = "points3D.ply"
OUTPUT_PLY = "points_tree_only.ply"
TREE_CLASS_ID = 4  # uprav podľa Semantic3D labelov
NUM_POINT = 40960  # RandLA-Net používa fixné batch bloky


# ------------------------------------------------------------

class RandLANetPredictor:
    def __init__(self, ckpt_path):
        self.ckpt_path = ckpt_path
        self.sess = tf.compat.v1.Session()
        self._build_model()
        self._load_checkpoint()

    def _build_model(self):
        # Import pôvodného modelu z RandLA-Net repozitára
        from models.RandLANet import Network

        self.inputs = tf.compat.v1.placeholder(tf.float32, shape=[1, NUM_POINT, 3], name='points')
        self.net = Network({'pointclouds': self.inputs}, is_training=False)
        self.logits = self.net.logits

    def _load_checkpoint(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, self.ckpt_path)

    def infer(self, points):
        pts = points.copy()
        if pts.shape[0] < NUM_POINT:
            repeat = NUM_POINT - pts.shape[0]
            pts = np.vstack([pts, pts[:repeat]])
        else:
            pts = pts[:NUM_POINT]

        pts = pts.reshape(1, NUM_POINT, 3)
        logits = self.sess.run(self.logits, feed_dict={self.inputs: pts})
        pred = np.argmax(logits, axis=-1).flatten()
        return pred[:points.shape[0]]


# ------------------------------------------------------------
# 1. Načítať PLY pointcloud
# ------------------------------------------------------------
pcd = o3d.io.read_point_cloud(INPUT_PLY)
points = np.asarray(pcd.points)

# ------------------------------------------------------------
# 2. Inicializovať RandLA-Net model
# ------------------------------------------------------------
predictor = RandLANetPredictor(CHECKPOINT_PATH)

# ------------------------------------------------------------
# 3. Predikcia labelov
# ------------------------------------------------------------
pred_labels = predictor.infer(points)

# ------------------------------------------------------------
# 4. Vybrať body triedy TREE_CLASS_ID
# ------------------------------------------------------------
tree_points = points[pred_labels == TREE_CLASS_ID]

# ------------------------------------------------------------
# 5. Uložiť výsledok do PLY
# ------------------------------------------------------------
pcd_tree = o3d.geometry.PointCloud()
pcd_tree.points = o3d.utility.Vector3dVector(tree_points)
o3d.io.write_point_cloud(OUTPUT_PLY, pcd_tree)

print("Hotovo. Výsledok je uložený v", OUTPUT_PLY)
