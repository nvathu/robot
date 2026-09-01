# run_full_pipeline.py
import json, glob, os
import pandas as pd
import numpy as np
from geometric_projection import world_to_camera_frame

# ==== CONFIG ====
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
OUTPUT_DIR = "dataset/pointclouds"
TARGET_ROBOTS = ["S1", "S2"]
FRAME_STEP = 20          # cach nhau bao nhieu dong CSV moi xu ly 1 frame
FLOOR_GRID_SPACING = 0.1  # khoang cach giua cac diem luoi san (met)
FLOOR_MARGIN = 0.3        # mo rong them bien quanh vung robot da di qua (met)

df = pd.read_csv(POSE_CSV)
print(f"Tong so dong CSV: {len(df)}")

# ==== BUOC 1: HIGH-ACCURACY LAYER - vi tri that cua robot muc tieu, hoan toan tu dong ====
def transform_to_world(X_local, Z_local, self_pose):
    angle = np.radians(self_pose['angle'])
    c, s = np.cos(angle), np.sin(angle)
    X_world = c * X_local + s * Z_local
    Z_world = -s * X_local + c * Z_local
    x_final = X_world + self_pose['x'] / 1000.0
    z_final = Z_world + self_pose['y'] / 1000.0
    return x_final, z_final


high_acc_points = []
high_acc_colors = []

for idx in range(0, len(df), FRAME_STEP):
    row = df.iloc[idx]
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}

    for tgt in TARGET_ROBOTS:
        tx_col, ty_col = f'car_{tgt}_pose_x', f'car_{tgt}_pose_y'
        if tx_col not in row or pd.isna(row[tx_col]):
            continue
        target_pose = {'x': row[tx_col], 'y': row[ty_col]}

        forward, lateral = world_to_camera_frame(
            self_pose['x'], self_pose['y'], self_pose['angle'],
            target_pose['x'], target_pose['y']
        )
        Z = forward / 1000.0
        X = lateral / 1000.0
        if Z <= 0:
            continue  # ngoai FOV, bo qua

        x_w, z_w = transform_to_world(X, Z, self_pose)
        high_acc_points.append([x_w, 0.0, z_w])
        high_acc_colors.append([1.0, 0.0, 0.0])

high_acc_points = np.array(high_acc_points) if high_acc_points else np.zeros((0, 3))
high_acc_colors = np.array(high_acc_colors) if high_acc_colors else np.zeros((0, 3))
print(f"High-accuracy points (robot MoCap, world frame): {len(high_acc_points)}")


# ==== BUOC 2: FLOOR LAYER - mat phang san TONG HOP TU KIEN THUC DA BIET, khong can pixel/model ====
# Xac dinh vung khong gian robot da di qua, dua truc tiep vao pose CSV (khong can anh)
all_x = np.concatenate([
    df['self_pose_x'].dropna().values,
    *[df[f'car_{t}_pose_x'].dropna().values for t in TARGET_ROBOTS if f'car_{t}_pose_x' in df.columns]
]) / 1000.0
all_y = np.concatenate([
    df['self_pose_y'].dropna().values,
    *[df[f'car_{t}_pose_y'].dropna().values for t in TARGET_ROBOTS if f'car_{t}_pose_y' in df.columns]
]) / 1000.0

x_min, x_max = all_x.min() - FLOOR_MARGIN, all_x.max() + FLOOR_MARGIN
z_min, z_max = all_y.min() - FLOOR_MARGIN, all_y.max() + FLOOR_MARGIN
print(f"Vung san (world frame): X=[{x_min:.2f},{x_max:.2f}], Z=[{z_min:.2f},{z_max:.2f}]")

x_grid = np.arange(x_min, x_max, FLOOR_GRID_SPACING)
z_grid = np.arange(z_min, z_max, FLOOR_GRID_SPACING)
xx, zz = np.meshgrid(x_grid, z_grid)
floor_points = np.stack([xx.ravel(), np.zeros(xx.size), zz.ravel()], axis=1)
floor_colors = np.tile([0.7, 0.7, 0.7], (len(floor_points), 1))  # mau xam nhat cho san
print(f"Floor points (synthetic grid, Y=0): {len(floor_points)}")


# ==== BUOC 3: LUU KET QUA ====
os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "full_auto_pointcloud.npz")
np.savez(out_path,
         high_acc_points=high_acc_points,
         high_acc_colors=high_acc_colors,
         floor_points=floor_points,
         floor_colors=floor_colors)
print(f"\nDa luu: {out_path}")
print("PIPELINE HOAN TAT - khong can buoc thu cong nao.")