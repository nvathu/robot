import cv2, json, glob, os, shutil
import pandas as pd
import numpy as np
from geometric_projection import get_target_pixel

SESSION = "2025_0813_1543-S3"
IMG_DIR = f"dataset/rgb_dino/{SESSION}/20250813"
POSE_CSV = "dataset/poses/records_20250813-1535-S3.csv"
INTRINSICS_PATH = "dataset/calib/intrinsics.json"
OUTPUT_DIR = "debug_projection"
TARGET_ROBOTS = {
    "S1": (0, 0, 255),
    "S2": (255, 0, 0),
}
MAX_TS_GAP = 0.03
MARGIN_RATIO = 0.15
TEST_INDICES = [78, 156, 312, 390, 624, 702, 858, 936, 1092, 1170, 1248, 1326]

with open(INTRINSICS_PATH) as f:
    K = json.load(f)

df = pd.read_csv(POSE_CSV)
img_files = sorted(glob.glob(f"{IMG_DIR}/*.png"))
print(f"Total images in folder: {len(img_files)}")

def get_ts_from_filename(filepath):
    name = os.path.splitext(os.path.basename(filepath))[0]
    return int(name.replace("image_", "")) / 1e9

img_timestamps = np.array([get_ts_from_filename(f) for f in img_files])

def circular_interp(angles, weights):
    sin_avg = np.average(np.sin(angles), weights=weights)
    cos_avg = np.average(np.cos(angles), weights=weights)
    return np.arctan2(sin_avg, cos_avg)

def interpolate_pose(df, image_timestamp, max_gap=0.1):
    ts = df['pose_timestamp'].values / 1e9
    idx = np.searchsorted(ts, image_timestamp)
    if idx == 0 or idx >= len(df):
        return None
    t0, t1 = ts[idx-1], ts[idx]
    if (image_timestamp - t0) > max_gap or (t1 - image_timestamp) > max_gap:
        return None
    alpha = (image_timestamp - t0) / (t1 - t0 + 1e-9)
    row0, row1 = df.iloc[idx-1], df.iloc[idx]
    result = {}
    for col in df.columns:
        if col in ('pose_timestamp', 'image', 'image_timestamp'):
            continue
        if col.endswith('_angle'):
            angle0 = np.radians(row0[col]) if 'self' in col else row0[col]
            angle1 = np.radians(row1[col]) if 'self' in col else row1[col]
            result[col] = np.degrees(circular_interp(
                np.array([angle0, angle1]), weights=[1-alpha, alpha]
            ))
        else:
            result[col] = (1-alpha)*row0[col] + alpha*row1[col]
    return result

if os.path.exists(OUTPUT_DIR):
    shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

count = 0
for img_idx in TEST_INDICES:
    if img_idx >= len(img_files):
        print(f"  Skipping idx={img_idx}: exceeds available images ({len(img_files)})")
        continue

    img_path = img_files[img_idx]
    img_ts = img_timestamps[img_idx]

    interpolated = interpolate_pose(df, img_ts, max_gap=MAX_TS_GAP)
    if interpolated is None:
        print(f"  idx={img_idx}: pose interpolation failed")
        continue

    self_pose = {
        'x': interpolated['self_pose_x'],
        'y': interpolated['self_pose_y'],
        'angle': interpolated['self_pose_angle'],
    }

    img = cv2.imread(img_path)
    if img is None:
        continue

    drawn_any = False
    for tgt, color in TARGET_ROBOTS.items():
        tx_col, ty_col = f'car_{tgt}_pose_x', f'car_{tgt}_pose_y'
        if tx_col not in interpolated or pd.isna(interpolated[tx_col]) or pd.isna(interpolated[ty_col]):
            continue

        target_pose = {'x': interpolated[tx_col], 'y': interpolated[ty_col]}
        pixel = get_target_pixel(self_pose, target_pose, K, margin_ratio=MARGIN_RATIO)
        if pixel is None:
            continue

        u, v = pixel
        print(f"  idx={img_idx} {tgt}: computed (u,v) = ({u:.1f}, {v:.1f})")

        d_true = np.hypot(target_pose['x'] - self_pose['x'],
                           target_pose['y'] - self_pose['y']) / 1000.0

        cv2.circle(img, (int(u), int(v)), 10, color, 2)
        cv2.putText(img, f"{tgt} ({d_true:.2f}m)", (int(u) + 12, int(v)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        drawn_any = True

    if not drawn_any:
        print(f"  idx={img_idx}: no robot inside FOV")
        continue

    out_path = os.path.join(OUTPUT_DIR, f"check_{img_idx}.png")
    cv2.imwrite(out_path, img)
    count += 1
    print(f"  Saved {out_path}")

print(f"\nSaved {count}/{len(TEST_INDICES)} images to {OUTPUT_DIR}/")
