import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

VALID_DIR = "./valid_data"
RGB_ROOT = "./dataset/rgb"
DEPTH_ROOT = "./dataset/depth"


def load_manual_split(valid_list, rgb_root):
    data = []
    for img_name in valid_list:
        found = False

        for session in os.listdir(rgb_root):
            session_path = os.path.join(rgb_root, session)
            if not os.path.isdir(session_path):
                continue

            for sub in os.listdir(session_path):
                sub_path = os.path.join(session_path, sub)
                if os.path.isdir(sub_path):
                    candidate = os.path.join(sub_path, img_name)
                    if os.path.exists(candidate):
                        img_path = candidate

                        
                        csv_files = [f for f in os.listdir(session_path) if f.endswith(".csv")]
                        for csv_file in csv_files:
                            df = pd.read_csv(os.path.join(session_path, csv_file))
                            row = df[df["image"].str.contains(img_name, na=False)]

                            if len(row) == 0:
                                continue

                            row = row.iloc[0]

                            if (
                                np.isnan(row["self_pose_x"])
                                or np.isnan(row["car_S3_pose_x"])
                            ):
                                continue

                            data.append({
                                "session": session,
                                "img_path": img_path,
                                "self": (row["self_pose_x"], row["self_pose_y"]),
                                "other": (row["car_S3_pose_x"], row["car_S3_pose_y"])
                            })

                            found = True
                            break

                if found:
                    break
            if found:
                break

    return data


def compute_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)


def get_depth_value(depth_path):
    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)
    if depth is None:
        return None

    h, w = depth.shape
    crop = depth[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
    return np.mean(crop)


def build_dataset(data_list):
    depths = []
    distances = []

    for item in tqdm(data_list):
        filename = os.path.basename(item["img_path"])
        session = item["session"]
        depth_path = os.path.join(DEPTH_ROOT, session, filename)

        if not os.path.exists(depth_path):
            continue

        depth_val = get_depth_value(depth_path)
        if depth_val is None:
            continue

        dist = compute_distance(item["self"], item["other"])
        depths.append(depth_val)
        distances.append(dist)

    return np.array(depths), np.array(distances)


def fit_scale_bias(depths, distances):
    A = np.vstack([depths, np.ones(len(depths))]).T
    scale, bias = np.linalg.lstsq(A, distances, rcond=None)[0]
    return scale, bias


if __name__ == "__main__":
    valid_images = sorted(os.listdir(VALID_DIR))
    np.random.seed(42)
    np.random.shuffle(valid_images)

    print("Total valid images:", len(valid_images))

    split_idx = int(len(valid_images) * 0.8 )
    train_imgs = valid_images[:split_idx]
    test_imgs = valid_images[split_idx:]

    train_data = load_manual_split(train_imgs, RGB_ROOT)
    test_data = load_manual_split(test_imgs, RGB_ROOT)

    print("Train samples:", len(train_data))
    print("Test samples :", len(test_data))

    train_depths, train_distances = build_dataset(train_data)
    test_depths, test_distances = build_dataset(test_data)

    scale, bias = fit_scale_bias(train_depths, train_distances)
    print("Scale:", scale)
    print("Bias :", bias)

    pred_test = scale * test_depths + bias
    r2 = r2_score(test_distances, pred_test)
    print("R2 score:", r2)

    plt.scatter(train_depths, train_distances, label="Train")
    plt.scatter(test_depths, test_distances, label="Test")

    x_line = np.linspace(min(train_depths), max(train_depths), 100)
    y_line = scale * x_line + bias
    plt.plot(x_line, y_line, color="red", label="Fitted line")

    plt.xlabel("Depth (MiDaS)")
    plt.ylabel("Real Distance")
    plt.legend()
    plt.title("Depth → Distance")
    plt.savefig("result.png")
    plt.show()

    np.save("scale_bias.npy", np.array([scale, bias]))
    print("\nSaved to scale_bias.npy")
