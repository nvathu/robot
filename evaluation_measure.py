import os
import cv2
import json
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm


VALID_DIR = "./valid_data"
DEPTH_DIR = "./valid_data/depth"

SAVE_DIR = "./saved_data"

CLICK_FILE = os.path.join(SAVE_DIR, "click_points.json")
PAIR_FILE = os.path.join(SAVE_DIR, "depth_distance_pairs.csv")
SCALE_BIAS_FILE = os.path.join(SAVE_DIR, "scale_bias.npy")

RESULT_CSV = os.path.join(SAVE_DIR, "metric_depth_results.csv")

SUMMARY_TXT = os.path.join(SAVE_DIR, "metric_depth_summary.txt")

EXPORT_DIR = os.path.join(SAVE_DIR, "evaluation_images")

os.makedirs(EXPORT_DIR, exist_ok=True)


def load_valid_data(valid_dir):

    csv_path = None

    for f in os.listdir(valid_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(valid_dir, f)
            break

    if csv_path is None:
        raise Exception("No CSV file found")

    df = pd.read_csv(csv_path)

    data = []

    for _, row in df.iterrows():

        if pd.isna(row["self_pose_x"]):
            continue

        if pd.isna(row["car_S1_pose_x"]):
            continue

        if pd.isna(row["car_S3_pose_x"]):
            continue

        img_name = os.path.basename(str(row["image"]))

        depth_path = os.path.join(DEPTH_DIR, img_name)

        image_path = str(row["image"])

        if not os.path.exists(depth_path):
            continue

        data.append(
            {
                "img_name": img_name,
                "image_path": image_path,
                "depth": depth_path,
                "self": (row["self_pose_x"], row["self_pose_y"]),
                "s1": (row["car_S1_pose_x"], row["car_S1_pose_y"]),
                "s3": (row["car_S3_pose_x"], row["car_S3_pose_y"]),
                
                "car_S1_pose_x": row["car_S1_pose_x"],
                "car_S1_pose_y": row["car_S1_pose_y"],
                "car_S1_pose_angle": row["car_S1_pose_angle"],
                "self_pose_x": row["self_pose_x"],
                "self_pose_y": row["self_pose_y"],
                "self_pose_angle": row["self_pose_angle"],
                "car_S3_pose_x": row["car_S3_pose_x"],
                "car_S3_pose_y": row["car_S3_pose_y"],
                "car_S3_pose_angle": row["car_S3_pose_angle"],
            }
        )

    return data


def compute_distance(p1, p2):

    return np.linalg.norm(np.array(p1) - np.array(p2))


def depth_at_point(depth, pt):

    x, y = pt

    h, w = depth.shape

    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    return float(depth[y, x])


def compute_rmse(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)

    return np.sqrt(np.mean((gt - pred) ** 2))


def compute_abs_rel(gt, pred):

    gt = np.array(gt)
    pred = np.array(pred)

    return np.mean(np.abs(gt - pred) / (gt + 1e-8))


def compute_delta(gt, pred, threshold=1.25):

    gt = np.array(gt)
    pred = np.array(pred)

    ratio = np.maximum(gt / (pred + 1e-8), pred / (gt + 1e-8))

    return np.mean(ratio < threshold)


if __name__ == "__main__":

    print("METRIC DEPTH EVALUATION")

    data = load_valid_data(VALID_DIR)

    print("Total valid samples:", len(data))

    if not os.path.exists(CLICK_FILE):
        raise Exception("click_points.json not found")

    with open(CLICK_FILE, "r") as f:
        click_data = json.load(f)

    scale, bias = np.load(SCALE_BIAS_FILE)

    print("\nScale:", scale)
    print("Bias :", bias)

    gt_depths = []
    pred_depths = []

    detailed_rows = []

    for item in tqdm(data):

        img_name = item["img_name"]

        if img_name not in click_data:
            continue

        pts = click_data[img_name]

        pt1 = tuple(pts[0])
        pt2 = tuple(pts[1])

        depth = cv2.imread(item["depth"], cv2.IMREAD_GRAYSCALE)

        if depth is None:
            continue

        z1 = depth_at_point(depth, pt1)

        z2 = depth_at_point(depth, pt2)

        if z1 is None or z2 is None:
            continue

        d1 = compute_distance(item["self"], item["s1"])

        d2 = compute_distance(item["self"], item["s3"])

        pred1 = scale * z1 + bias

        pred2 = scale * z2 + bias

        gt_depths.extend([d1, d2])

        pred_depths.extend([pred1, pred2])

        detailed_rows.append(
            {
                "image": img_name,
                "z1": z1,
                "z2": z2,
                "gt1": d1,
                "gt2": d2,
                "pred1": pred1,
                "pred2": pred2,
                "error1": abs(pred1 - d1),
                "error2": abs(pred2 - d2),
                
                "car_S1_pose_x": item["car_S1_pose_x"],
                "car_S1_pose_y": item["car_S1_pose_y"],
                "car_S1_pose_angle": item["car_S1_pose_angle"],
                "self_pose_x": item["self_pose_x"],
                "self_pose_y": item["self_pose_y"],
                "self_pose_angle": item["self_pose_angle"],
                "car_S3_pose_x": item["car_S3_pose_x"],
                "car_S3_pose_y": item["car_S3_pose_y"],
                "car_S3_pose_angle": item["car_S3_pose_angle"],
            }
        )


        rgb_path = item["image_path"]

        if os.path.exists(rgb_path):

            shutil.copy(rgb_path, os.path.join(EXPORT_DIR, img_name))


        shutil.copy(item["depth"], os.path.join(EXPORT_DIR, "depth_" + img_name))


    rmse = compute_rmse(gt_depths, pred_depths)

    abs_rel = compute_abs_rel(gt_depths, pred_depths)

    delta125 = compute_delta(gt_depths, pred_depths, 1.25)

    delta125_2 = compute_delta(gt_depths, pred_depths, 1.25**2)

    delta125_3 = compute_delta(gt_depths, pred_depths, 1.25**3)

    print("\nFINAL RESULT")

    print(f"RMSE          : {rmse:.4f}")

    print(f"Abs Rel Error : {abs_rel:.4f}")

    print(f"Delta <1.25   : {delta125:.4f}")

    print(f"Delta <1.25²  : {delta125_2:.4f}")

    print(f"Delta <1.25³  : {delta125_3:.4f}")

    pd.DataFrame(detailed_rows).to_csv(RESULT_CSV, index=False)

    with open(SUMMARY_TXT, "w") as f:

        f.write("METRIC DEPTH EVALUATION\n\n")

        f.write(f"Scale: {scale}\n")

        f.write(f"Bias : {bias}\n\n")

        f.write(f"RMSE          : {rmse:.4f}\n")

        f.write(f"Abs Rel Error : {abs_rel:.4f}\n")

        f.write(f"Delta <1.25   : {delta125:.4f}\n")

        f.write(f"Delta <1.25²  : {delta125_2:.4f}\n")

        f.write(f"Delta <1.25³  : {delta125_3:.4f}\n")

    print("\nSaved:")
    print(RESULT_CSV)
    print(SUMMARY_TXT)
    print(EXPORT_DIR)
