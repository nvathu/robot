import torch, cv2, glob, os, json
import pandas as pd
import numpy as np
from model import ResNetDepth
from geometric_projection import get_target_pixel, world_to_camera_frame

SESSION = "2025_0813_1543-S3"
IMG_DIR = f"dataset/rgb_dino/{SESSION}/20250813"
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
GLOBAL_SCALE_PATH = "dataset/calib/global_scale.json"
WEIGHTS_PATH = "weights/best_model.pth"
OUTPUT_DIR = "dataset/pointclouds"
TARGET_ROBOTS = ["S1", "S2"]
MAX_TS_GAP = 0.03
NUM_FRAMES = 10000
VOXEL_SIZE = 0.04
MIN_DEPTH_M = 0.30
MAX_DEPTH_M = 1.87
MARGIN_RATIO = 0.15
ROBOT_SIZE_M = 0.15
MASK_DYNAMIC_ROBOTS = True
ROBOT_COLORS = {"S1": np.array([[1.0, 0.0, 0.0]]), "S2": np.array([[0.0, 0.0, 1.0]])}

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(INTRINSICS_PATH) as f:
    K = json.load(f)
with open(GLOBAL_SCALE_PATH) as f:
    GS = json.load(f)
GLOBAL_SCALE = GS["global_scale"]

model = ResNetDepth().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))
img_ts = np.array([int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9 for f in img_files])


def find_img(ts, max_gap=MAX_TS_GAP):
    i = np.argmin(np.abs(img_ts - ts))
    return img_files[i] if abs(img_ts[i] - ts) <= max_gap else None


@torch.no_grad()
def predict_depth(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None, None, None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (180, 180))
    tensor = torch.tensor(img_small / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pred = model(tensor).squeeze().cpu().numpy()
    return pred, img.shape[:2], img_small


def mask_dynamic_robots(depth_map, row, self_pose, w0, h0):
    masked = depth_map.copy()
    n_masked = 0
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
        r_orig = (K['fx'] * (ROBOT_SIZE_M / 2.0)) / forward_m
        u_s, v_s = u * 180 / w0, v * 180 / h0
        r_s = max(2, int(round(r_orig * 180 / w0)))
        cv2.circle(masked, (int(round(u_s)), int(round(v_s))), r_s, -1.0, thickness=-1)
        n_masked += 1
    return masked, n_masked


def unproject(depth_map, fx, fy, cx, cy, rgb_image, min_depth=MIN_DEPTH_M, max_depth=MAX_DEPTH_M):
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_map > min_depth) & (depth_map < max_depth)
    z = depth_map[valid]
    x = (u[valid] - cx) * z / fx
    y = (v[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = rgb_image[valid] / 255.0
    return points, colors


def transform_to_world(points, self_pose):
    if len(points) == 0:
        return points
    angle = np.radians(self_pose['angle'])
    c, s = np.cos(angle), np.sin(angle)
    X, Z = points[:, 0], points[:, 2]
    X_w = c * X + s * Z
    Z_w = -s * X + c * Z
    x_f = X_w + self_pose['x'] / 1000.0
    z_f = Z_w + self_pose['y'] / 1000.0
    return np.stack([x_f, points[:, 1], z_f], axis=1)


def voxel_downsample(points, colors, voxel_size=VOXEL_SIZE):
    if len(points) == 0:
        return points, colors
    keys = np.floor(points / voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx], colors[idx]


valid_rows = []
for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue
    if find_img(row['pose_timestamp'] / 1e9) is not None:
        valid_rows.append(idx)

step = max(1, len(valid_rows) // NUM_FRAMES)
selected_rows = valid_rows[::step][:NUM_FRAMES]
print(f"{len(selected_rows)} frames selected out of {len(valid_rows)} valid rows")

all_scene_points, all_scene_colors = [], []
all_robot_points, all_robot_colors = [], []
all_self_points = []
total_masked = 0

for i, csv_idx in enumerate(selected_rows):
    row = df.iloc[csv_idx]
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}
    all_self_points.append([self_pose['x'] / 1000.0, 0.0, self_pose['y'] / 1000.0])

    raw_depth, orig_shape, img_small = predict_depth(img_path)
    if raw_depth is None:
        continue
    h0, w0 = orig_shape

    if MASK_DYNAMIC_ROBOTS:
        raw_depth, n_masked = mask_dynamic_robots(raw_depth, row, self_pose, w0, h0)
        total_masked += n_masked

    depth_metric = raw_depth * GLOBAL_SCALE
    scene_points, scene_colors = unproject(depth_metric, K['fx'], K['fy'], K['cx'] * 180 / w0, K['cy'] * 180 / h0, img_small)
    scene_points_world = transform_to_world(scene_points, self_pose)

    all_scene_points.append(scene_points_world)
    all_scene_colors.append(scene_colors)

    for tgt in TARGET_ROBOTS:
        tx = f'car_{tgt}_pose_x'
        if tx not in row or pd.isna(row[tx]):
            continue
        target_pose = {'x': row[tx], 'y': row[f'car_{tgt}_pose_y']}
        pixel = get_target_pixel(self_pose, target_pose, K, margin_ratio=MARGIN_RATIO)
        if pixel is None:
            continue
        forward, lateral = world_to_camera_frame(self_pose['x'], self_pose['y'], self_pose['angle'],
                                                   target_pose['x'], target_pose['y'])
        pt_world = transform_to_world(np.array([[lateral / 1000.0, 0.0, forward / 1000.0]]), self_pose)
        all_robot_points.append(pt_world)
        all_robot_colors.append(ROBOT_COLORS[tgt])

    if (i + 1) % 25 == 0:
        print(f"{i+1}/{len(selected_rows)}")

scene_merged = np.concatenate(all_scene_points, axis=0)
colors_merged = np.concatenate(all_scene_colors, axis=0)

nan_mask = np.isnan(scene_merged).any(axis=1)
if nan_mask.any():
    scene_merged = scene_merged[~nan_mask]
    colors_merged = colors_merged[~nan_mask]

scene_merged, colors_merged = voxel_downsample(scene_merged, colors_merged, VOXEL_SIZE)
robots_merged = np.concatenate(all_robot_points, axis=0) if all_robot_points else np.zeros((0, 3))
robots_colors_merged = np.concatenate(all_robot_colors, axis=0) if all_robot_colors else np.zeros((0, 3))
self_merged = np.array(all_self_points)

print(f"final point count: {len(scene_merged)}, robots masked: {total_masked}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.savez(os.path.join(OUTPUT_DIR, "room_pointcloud_near_range.npz"),
         scene_points=scene_merged, scene_colors=colors_merged,
         robot_points=robots_merged, robot_colors=robots_colors_merged,
         self_trajectory=self_merged,
         min_depth=MIN_DEPTH_M, max_depth=MAX_DEPTH_M)