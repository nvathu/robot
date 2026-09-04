import torch, cv2, glob, os, json
import pandas as pd
import numpy as np
from model import ResNetDepth
from geometric_projection import get_target_pixel, world_to_camera_frame

SESSION = "2025_0813_1543-S3"
IMG_DIR = f"dataset/rgb_dino/{SESSION}/20250813"
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
WEIGHTS_PATH = "weights/best_model.pth"
TARGET_ROBOTS = ["S1", "S2"]
MAX_TS_GAP = 0.03
MARGIN_RATIO = 0.15

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(INTRINSICS_PATH) as f:
    K = json.load(f)

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

depth_cache = {}

@torch.no_grad()
def get_raw_depth(img_path):
    if img_path in depth_cache:
        return depth_cache[img_path]
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_small = cv2.resize(img_rgb, (180, 180))
    tensor = torch.tensor(img_small / 255.).permute(2, 0, 1).unsqueeze(0).float().to(device)
    pred = model(tensor).squeeze().cpu().numpy()
    result = {'depth': pred, 'shape': img.shape[:2]}
    depth_cache[img_path] = result
    return result

scale_factors = []

for idx in range(len(df)):
    row = df.iloc[idx]
    ts = row['pose_timestamp'] / 1e9
    img_path = find_img(ts)
    if img_path is None:
        continue

    self_pose = {
        'x': row['self_pose_x'],
        'y': row['self_pose_y'],
        'angle': row['self_pose_angle']
    }

    for tgt in TARGET_ROBOTS:
        tx = f'car_{tgt}_pose_x'
        if tx not in row or pd.isna(row[tx]):
            continue

        target_pose = {
            'x': row[tx],
            'y': row[f'car_{tgt}_pose_y']
        }

        pixel = get_target_pixel(self_pose, target_pose, K, margin_ratio=MARGIN_RATIO)
        if pixel is None:
            continue

        u, v = pixel
        forward, lateral = world_to_camera_frame(
            self_pose['x'], self_pose['y'], self_pose['angle'],
            target_pose['x'], target_pose['y']
        )

        d_true = forward / 1000.0
        cached = get_raw_depth(img_path)
        if cached is None:
            continue

        h0, w0 = cached['shape']
        u_r = int(np.clip(u * 180 / w0, 0, 179))
        v_r = int(np.clip(v * 180 / h0, 0, 179))

        d_pred_raw = cached['depth'][v_r, u_r]
        if d_pred_raw > 1e-3:
            scale_factors.append(d_true / d_pred_raw)

    if idx % 4000 == 0:
        print(f"{idx}/{len(df)}, {len(scale_factors)} samples, cache {len(depth_cache)} images")

scale_factors = np.array(scale_factors)
global_scale = float(np.median(scale_factors))

print(f"\nTotal samples: {len(scale_factors)}")
print(f"Median (global scale): {global_scale:.4f}")
print(f"Mean: {scale_factors.mean():.4f}, Std: {scale_factors.std():.4f}")

os.makedirs("dataset/calib", exist_ok=True)

with open("dataset/calib/global_scale.json", "w") as f:
    json.dump({
        "global_scale": global_scale,
        "n_samples": len(scale_factors),
        "mean": float(scale_factors.mean()),
        "std": float(scale_factors.std())
    }, f, indent=2)

print("Saved dataset/calib/global_scale.json")
