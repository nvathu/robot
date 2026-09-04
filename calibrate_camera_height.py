import cv2, glob, os, json
import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from geometric_projection import world_to_camera_frame, project_to_pixel

SESSION = "2025_0813_1543-S3"
IMG_DIR = f"dataset/rgb_dino/{SESSION}/20250813"
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
TARGET_ROBOTS = ["S1", "S2"]
MAX_TS_GAP = 0.03
ROBOT_SIZE_M = 0.15
N_SAMPLES = 600
OUT_DIR = "debug_depth_and_scale/pitch_height_calib"
os.makedirs(OUT_DIR, exist_ok=True)

YELLOW_LOWER = np.array([15, 80, 80])
YELLOW_UPPER = np.array([40, 255, 255])
MIN_YELLOW_PIXELS = 15

with open(INTRINSICS_PATH) as f:
    K = json.load(f)

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))

def get_ts(f):
    return int(os.path.splitext(os.path.basename(f))[0].replace("image_", "")) / 1e9

img_ts = np.array([get_ts(f) for f in img_files])

def find_img(ts, max_gap=MAX_TS_GAP):
    i = np.argmin(np.abs(img_ts - ts))
    return img_files[i] if abs(img_ts[i] - ts) <= max_gap else None

def detect_wheel_contact_row(img_bgr, u_center, crop_half_w):
    h0, w0 = img_bgr.shape[:2]
    u0 = max(0, int(u_center - crop_half_w))
    u1 = min(w0, int(u_center + crop_half_w))
    if u1 - u0 < 10:
        return None, False

    crop = img_bgr[:, u0:u1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, YELLOW_LOWER, YELLOW_UPPER)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))

    ys, xs = np.where(mask > 0)
    if len(ys) < MIN_YELLOW_PIXELS:
        return None, False

    return int(ys.max()), True

records = []
candidates = []

for idx in range(len(df)):
    row = df.iloc[idx]
    if pd.isna(row['self_pose_x']) or pd.isna(row['self_pose_y']) or pd.isna(row['self_pose_angle']):
        continue

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

        forward, lateral = world_to_camera_frame(
            self_pose['x'], self_pose['y'], self_pose['angle'],
            target_pose['x'], target_pose['y']
        )

        forward_m = forward / 1000.0
        lateral_m = lateral / 1000.0
        if forward_m <= 0.1:
            continue

        u = K['cx'] - (lateral_m * K['fx'] / forward_m)
        margin = 0.15 * (2 * K['cx'])
        if not (margin <= u < 2*K['cx'] - margin):
            continue

        candidates.append((idx, img_path, forward_m, lateral_m, u))

rng = np.random.default_rng(42)
if len(candidates) > N_SAMPLES:
    pick = rng.choice(len(candidates), N_SAMPLES, replace=False)
    candidates = [candidates[i] for i in pick]

n_ok, n_fail = 0, 0

for csv_idx, img_path, forward_m, lateral_m, u in candidates:
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        continue

    crop_half_w = max(25, (K['fx'] * ROBOT_SIZE_M / forward_m) * 1.1)
    v_contact, ok = detect_wheel_contact_row(img_bgr, u, crop_half_w)
    if not ok:
        n_fail += 1
        continue

    n_ok += 1
    records.append({
        "forward_m": forward_m,
        "lateral_m": lateral_m,
        "v_contact": v_contact,
        "u": u,
        "img_path": img_path,
        "csv_idx": csv_idx
    })

forward_arr = np.array([r['forward_m'] for r in records])
lateral_arr = np.array([r['lateral_m'] for r in records])
v_arr = np.array([r['v_contact'] for r in records])

rng2 = np.random.default_rng(0)
idx_all = rng2.permutation(len(forward_arr))
n_train = int(0.8 * len(idx_all))
train_i, test_i = idx_all[:n_train], idx_all[n_train:]

def predict_v(h, pitch_deg, forward, lateral):
    v_pred = np.full_like(forward, np.nan)
    for i in range(len(forward)):
        res = project_to_pixel(
            forward[i]*1000, lateral[i]*1000,
            K['fx'], K['fy'], K['cx'], K['cy'],
            height_mm=h*1000, pitch_deg=pitch_deg
        )
        v_pred[i] = res[1] if res is not None else np.nan
    return v_pred

def residual_pitch0(params):
    h = params[0]
    pred = predict_v(h, 0.0, forward_arr[train_i], lateral_arr[train_i])
    return np.nan_to_num(pred - v_arr[train_i], nan=1e3)

def residual_free(params):
    h, pitch = params
    pred = predict_v(h, pitch, forward_arr[train_i], lateral_arr[train_i])
    return np.nan_to_num(pred - v_arr[train_i], nan=1e3)

res0 = least_squares(residual_pitch0, x0=[0.05], loss='soft_l1', f_scale=15.0)
h0_fit = res0.x[0]

res1 = least_squares(residual_free, x0=[0.05, 0.0], loss='soft_l1', f_scale=15.0)
h1_fit, pitch1_fit = res1.x

pred0_test = predict_v(h0_fit, 0.0, forward_arr[test_i], lateral_arr[test_i])
pred1_test = predict_v(h1_fit, pitch1_fit, forward_arr[test_i], lateral_arr[test_i])

mae0 = np.nanmean(np.abs(pred0_test - v_arr[test_i]))
mae1 = np.nanmean(np.abs(pred1_test - v_arr[test_i]))

with open("dataset/calib/pitch_height.json", "w") as f:
    json.dump({
        "h_pitch0_m": float(h0_fit),
        "mae_pitch0_px": float(mae0),
        "h_free_m": float(h1_fit),
        "pitch_free_deg": float(pitch1_fit),
        "mae_free_px": float(mae1),
        "n_samples": len(records),
        "n_detect_fail": n_fail
    }, f, indent=2)
