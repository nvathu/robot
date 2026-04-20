import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

VALID_DIR = "./valid_data"
DEPTH_DIR = "./valid_data/depth"

CLICK_FILE = "click_points.npy"  
SCALE_BIAS_FILE = "scale_bias.npy"



def load_valid_data(valid_dir):

    csv_path = None
    for f in os.listdir(valid_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(valid_dir, f)
            break

    df = pd.read_csv(csv_path)
    data = []

    for _, row in df.iterrows():

        if pd.isna(row["self_pose_x"]) or pd.isna(row["car_S3_pose_x"]):
            continue

        img_name = os.path.basename(str(row["image"]))
        depth_path = os.path.join(DEPTH_DIR, img_name)

        img_path = None
        for root, _, files in os.walk(valid_dir):
            if img_name in files:
                img_path = os.path.join(root, img_name)
                break

        if img_path is None or not os.path.exists(depth_path):
            continue

        data.append({
            "img_name": img_name,
            "depth": depth_path,
            "self": (row["self_pose_x"], row["self_pose_y"]),
            "s1": (row["car_S1_pose_x"], row["car_S1_pose_y"]),
            "s3": (row["car_S3_pose_x"], row["car_S3_pose_y"])
        })

    return data



def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def depth_at_point(depth, pt):
    x, y = pt
    return float(depth[y, x])



def compute_metrics(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)


    rmse = np.sqrt(np.mean((gt - pred) ** 2))

    abs_rel = np.mean(np.abs(gt - pred) / gt)

   
    ratio = np.maximum(gt / pred, pred / gt)
    delta = np.mean(ratio < 1.25)

    return rmse, abs_rel, delta



if __name__ == "__main__":

    print("Loading data...")
    data = load_valid_data(VALID_DIR)

    print("Loading click points...")
    click_dict = np.load(CLICK_FILE, allow_pickle=True).item()

    print("Loading scale & bias...")
    scale, bias = np.load(SCALE_BIAS_FILE)

    print("Scale:", scale)
    print("Bias :", bias)

    gt_all = []
    pred_all = []

    for item in tqdm(data):

        img_name = item["img_name"]

        if img_name not in click_dict:
            continue

        pts = click_dict[img_name]

        depth = cv2.imread(item["depth"], cv2.IMREAD_GRAYSCALE)
        if depth is None:
            continue

        z1 = depth_at_point(depth, pts[0])
        z2 = depth_at_point(depth, pts[1])

        d1 = compute_distance(item["self"], item["s1"])
        d2 = compute_distance(item["self"], item["s3"])

        
        pred1 = scale * z1 + bias
        pred2 = scale * z2 + bias

        gt_all.extend([d1, d2])
        pred_all.extend([pred1, pred2])


    rmse, abs_rel, delta = compute_metrics(gt_all, pred_all)

    print("\n========== METRIC DEPTH EVALUATION ==========")
    print(f"RMSE        : {rmse:.4f}")
    print(f"Abs Rel     : {abs_rel:.4f}")
    print(f"Delta <1.25 : {delta:.4f}")