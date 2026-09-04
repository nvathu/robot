import cv2, glob, os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geometric_projection import get_target_pixel, world_to_camera_frame

SESSION = "2025_0813_1543-S3"
IMG_DIR = f"dataset/rgb_dino/{SESSION}/20250813"
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
PITCH_HEIGHT_PATH = "dataset/calib/pitch_height.json"
TARGET_ROBOTS = ["S1", "S2"]
MARGIN_RATIO = 0.15
ROBOT_SIZE_M = 0.15

MIN_WALL_DIST_M = 0.3
MAX_WALL_DIST_M = 2.6
BRIGHT_THRESH = 150
RUN_LENGTH = 12
DARK_FLOOR_MAX = 130

N_SAMPLE_FRAMES = 20

with open(INTRINSICS_PATH) as f:
    K = json.load(f)
with open(PITCH_HEIGHT_PATH) as f:
    PH = json.load(f)

H_CAM = PH["h_free_m"]
PITCH_DEG = PH["pitch_free_deg"]
THETA = np.radians(PITCH_DEG)

def forward_to_row(forward_m, h, theta, fy, cy):
    k = (h*np.cos(theta) - forward_m*np.sin(theta)) / (forward_m*np.cos(theta) + h*np.sin(theta))
    return cy + k*fy

v_at_max_dist = forward_to_row(MAX_WALL_DIST_M, H_CAM, THETA, K['fy'], K['cy'])
v_at_min_dist = forward_to_row(MIN_WALL_DIST_M, H_CAM, THETA, K['fy'], K['cy'])

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))

def get_ts(f):
    return int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9

img_ts = np.array([get_ts(f) for f in img_files])

def find_img(ts, max_gap=0.03):
    i = np.argmin(np.abs(img_ts - ts))
    return img_files[i] if abs(img_ts[i] - ts) <= max_gap else None

def mask_dynamic_robots(img_gray, row, self_pose, w0, h0, K):
    masked = img_gray.copy()
    for tgt in TARGET_ROBOTS:
        tx = f'car_{tgt}_pose_x'
        if tx not in row or pd.isna(row[tx]):
            continue
        target_pose = {'x': row[tx], 'y': row[f'car_{tgt}_pose_y']}
        pixel = get_target_pixel(self_pose, target_pose, K, margin_ratio=MARGIN_RATIO)
        if pixel is None:
            continue
        u, v = pixel
        forward, _ = world_to_camera_frame(self_pose['x'], self_pose['y'], self_pose['angle'],
                                           target_pose['x'], target_pose['y'])
        forward_m = forward / 1000.0
        if forward_m <= 0.05:
            continue
        r_px = (K['fx'] * ROBOT_SIZE_M) / forward_m
        cv2.rectangle(masked, (int(u - r_px), 0), (int(u + r_px), h0), 0, thickness=-1)
    return masked

def detect_floor_wall_row_v2(img_gray, v_top, v_bot):
    h0, w0 = img_gray.shape
    top = max(0, int(v_top))
    bot = min(h0, int(v_bot))
    if bot - top < RUN_LENGTH + 5:
        return np.full(w0, -1), np.zeros(w0, dtype=bool)
    v_result = np.full(w0, -1)
    valid = np.zeros(w0, dtype=bool)
    col_strip = img_gray[top:bot, :].astype(np.float32)
    n_rows = col_strip.shape[0]
    for u in range(w0):
        col = col_strip[:, u]
        for r in range(n_rows - RUN_LENGTH):
            run = col[r:r+RUN_LENGTH]
            if run.min() < BRIGHT_THRESH:
                continue
            before = col[max(0, r-6):r]
            if len(before) > 0 and before.mean() < DARK_FLOOR_MAX:
                v_result[u] = top + r
                valid[u] = True
                break
    return v_result, valid

def row_to_forward_distance(v, h_cam, theta, fy, cy):
    k = (v - cy) / fy
    denom = (np.sin(theta) + k*np.cos(theta))
    if denom <= 1e-6:
        return None
    forward = h_cam * (np.cos(theta) - k*np.sin(theta)) / denom
    return forward

valid_rows = []
for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue
    ts = row['pose_timestamp'] / 1e9
    if find_img(ts) is not None:
        valid_rows.append(idx)

sample_idx = np.linspace(0, len(valid_rows)-1, N_SAMPLE_FRAMES).astype(int)
sample_rows = [valid_rows[i] for i in sample_idx]

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, csv_idx in zip(axes.flat, sample_rows):
    row = df.iloc[csv_idx]
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h0, w0 = img_gray.shape
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}

    img_masked = mask_dynamic_robots(img_gray, row, self_pose, w0, h0, K)
    v_boundary, valid = detect_floor_wall_row_v2(img_masked, v_at_max_dist, v_at_min_dist)

    local_x, local_z = []
    local_z = []
    for u in range(w0):
        if not valid[u]:
            continue
        v = v_boundary[u]
        forward_m = row_to_forward_distance(v, H_CAM, THETA, K['fy'], K['cy'])
        if forward_m is None or not (MIN_WALL_DIST_M < forward_m < MAX_WALL_DIST_M):
            continue
        lateral_m = (u - K['cx']) * forward_m / K['fx']
        local_x.append(lateral_m)
        local_z.append(forward_m)

    ax.scatter(local_x, local_z, s=8, alpha=0.7)
    ax.scatter([0], [0], c='red', marker='^', s=60, label='camera')
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("forward (m)")
    ax.set_title(f"csv_idx={csv_idx}, {len(local_x)} valid points")
    ax.axis('equal')
    ax.legend(fontsize=7)

plt.suptitle("Single-frame floor-wall boundary shape check")
plt.tight_layout()
plt.savefig("debug_depth_and_scale/single_frame_lidar_scan.png", dpi=150)
print("Saved debug_depth_and_scale/single_frame_lidar_scan.png")
