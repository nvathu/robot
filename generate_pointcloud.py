# build_room_pointcloud.py
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
MAX_DEPTH_M = 5.0
NUM_FRAMES = 10000       # so frame se ghep - cang nhieu cang day, nhung cang cham
FRAME_STEP = None       # se tinh tu dong de trai deu toan bo dataset
VOXEL_SIZE = 0.04

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

with open(INTRINSICS_PATH) as f:
    K = json.load(f)
with open(GLOBAL_SCALE_PATH) as f:
    GS = json.load(f)
GLOBAL_SCALE = GS["global_scale"]
print(f"Dung global_scale = {GLOBAL_SCALE:.4f} (tu {GS['n_samples']} mau)")

model = ResNetDepth().to(device)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
model.eval()

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))

def get_ts(f):
    return int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9
img_ts = np.array([get_ts(f) for f in img_files])

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
    tensor = torch.tensor(img_small/255.).permute(2,0,1).unsqueeze(0).float().to(device)
    pred = model(tensor).squeeze().cpu().numpy()
    return pred, img.shape[:2], img_small

def unproject(depth_map, fx, fy, cx, cy, rgb_image, max_depth=MAX_DEPTH_M):
    H, W = depth_map.shape
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    valid = (depth_map > 0) & (depth_map < max_depth)
    z = depth_map[valid]
    x = (u[valid]-cx)*z/fx
    y = (v[valid]-cy)*z/fy
    points = np.stack([x,y,z], axis=1)
    colors = rgb_image[valid]/255.0
    return points, colors

def transform_to_world(points, self_pose):
    if len(points) == 0:
        return points
    angle = np.radians(self_pose['angle'])
    c, s = np.cos(angle), np.sin(angle)
    X, Z = points[:,0], points[:,2]
    X_w = c*X + s*Z
    Z_w = -s*X + c*Z
    x_f = X_w + self_pose['x']/1000.0
    z_f = Z_w + self_pose['y']/1000.0
    return np.stack([x_f, points[:,1], z_f], axis=1)

def voxel_downsample(points, colors, voxel_size=VOXEL_SIZE):
    if len(points) == 0:
        return points, colors
    keys = np.floor(points/voxel_size).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[idx], colors[idx]


# Chon frame trai deu toan bo dataset (khong chi noi co anchor)
valid_rows = []
for idx in range(len(df)):
    ts = df.iloc[idx]['pose_timestamp'] / 1e9
    if find_img(ts) is not None:
        valid_rows.append(idx)

step = max(1, len(valid_rows) // NUM_FRAMES)
selected_rows = valid_rows[::step][:NUM_FRAMES]
print(f"Chon {len(selected_rows)} frame trai deu tu {len(valid_rows)} frame hop le")

all_scene_points = []
all_scene_colors = []
all_robot_points = []  # vi tri that cua robot (MoCap), tat ca target robots
all_robot_colors = []

for i, csv_idx in enumerate(selected_rows):
    row = df.iloc[csv_idx]
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    self_pose = {'x': row['self_pose_x'], 'y': row['self_pose_y'], 'angle': row['self_pose_angle']}

    raw_depth, orig_shape, img_small = predict_depth(img_path)
    if raw_depth is None:
        continue
    h0, w0 = orig_shape

    # Dung 1 GLOBAL SCALE co dinh cho MOI frame - khong tinh rieng nua
    depth_metric = raw_depth * GLOBAL_SCALE
    scene_points, scene_colors = unproject(
        depth_metric, K['fx'], K['fy'], K['cx']*180/w0, K['cy']*180/h0, img_small
    )
    scene_points_world = transform_to_world(scene_points, self_pose)

    all_scene_points.append(scene_points_world)
    all_scene_colors.append(scene_colors)

    # Ghi lai vi tri that cua cac robot muc tieu (MoCap), de hien thi doi chieu
    for tgt in TARGET_ROBOTS:
        tx = f'car_{tgt}_pose_x'
        if tx not in row or pd.isna(row[tx]):
            continue
        target_pose = {'x': row[tx], 'y': row[f'car_{tgt}_pose_y']}
        forward, lateral = world_to_camera_frame(self_pose['x'], self_pose['y'], self_pose['angle'],
                                                    target_pose['x'], target_pose['y'])
        if forward <= 0:
            continue
        pt_local = np.array([[lateral/1000.0, 0.0, forward/1000.0]])
        pt_world = transform_to_world(pt_local, self_pose)
        all_robot_points.append(pt_world)
        all_robot_colors.append(np.array([[1.0, 0.0, 0.0]]))

    if (i+1) % 25 == 0:
        print(f"  Da xu ly {i+1}/{len(selected_rows)} frame")

scene_merged = np.concatenate(all_scene_points, axis=0)
colors_merged = np.concatenate(all_scene_colors, axis=0)
print(f"\nTruoc downsample: {len(scene_merged)} diem")

scene_merged, colors_merged = voxel_downsample(scene_merged, colors_merged, VOXEL_SIZE)
print(f"Sau downsample: {len(scene_merged)} diem")

robots_merged = np.concatenate(all_robot_points, axis=0) if all_robot_points else np.zeros((0,3))
robots_colors_merged = np.concatenate(all_robot_colors, axis=0) if all_robot_colors else np.zeros((0,3))

os.makedirs(OUTPUT_DIR, exist_ok=True)
out_path = os.path.join(OUTPUT_DIR, "room_pointcloud.npz")
np.savez(out_path,
         scene_points=scene_merged, scene_colors=colors_merged,
         robot_points=robots_merged, robot_colors=robots_colors_merged)
print(f"Da luu: {out_path}")