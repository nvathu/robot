import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm



RGB_ROOT = "./dataset/rgb"
DEPTH_ROOT = "./dataset/depth"



def load_all_valid_data(rgb_root):

    all_data = []

    for session in os.listdir(rgb_root):

        session_path = os.path.join(rgb_root, session)

        if not os.path.isdir(session_path):
            continue


        csv_files = [f for f in os.listdir(session_path) if f.endswith(".csv")]

        for csv_file in csv_files:

            csv_path = os.path.join(session_path, csv_file)

            try:
                df = pd.read_csv(csv_path)
            except:
                continue

            for _, row in df.iterrows():


                if (
                    np.isnan(row.get("self_pose_x", np.nan)) or
                    np.isnan(row.get("self_pose_y", np.nan)) or
                    np.isnan(row.get("car_S3_pose_x", np.nan)) or
                    np.isnan(row.get("car_S3_pose_y", np.nan))
                ):
                    continue

                image_path = row.get("image", None)

                if not isinstance(image_path, str):
                    continue

                filename = os.path.basename(image_path)


                img_found = None

                for sub in os.listdir(session_path):
                    sub_path = os.path.join(session_path, sub)

                    if os.path.isdir(sub_path):
                        candidate = os.path.join(sub_path, filename)

                        if os.path.exists(candidate):
                            img_found = candidate
                            break

                if img_found is None:
                    continue

                all_data.append({
                    "session": session,
                    "img_path": img_found,
                    "self": (row["self_pose_x"], row["self_pose_y"]),
                    "other": (row["car_S3_pose_x"], row["car_S3_pose_y"])
                })

    return all_data



def compute_distance(p1, p2):

    return np.sqrt(
        (p1[0] - p2[0])**2 +
        (p1[1] - p2[1])**2
    )



def get_depth_value(depth_path):

    depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

    if depth is None:
        return None

    h, w = depth.shape


    return depth[h // 2, w // 2]




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

    # y = ax + b
    A = np.vstack([depths, np.ones(len(depths))]).T

    scale, bias = np.linalg.lstsq(A, distances, rcond=None)[0]

    return scale, bias




if __name__ == "__main__":

    print("Loading data...")
    data = load_all_valid_data(RGB_ROOT)

    print("Total valid samples:", len(data))

    if len(data) == 0:
        print("No valid data")
        exit()

    print("Building dataset...")
    depths, distances = build_dataset(data)

    print("Total usable pairs:", len(depths))

    if len(depths) == 0:
        print("No depth-distance")
        exit()

    print("Fitting scale & bias...")
    scale, bias = fit_scale_bias(depths, distances)

    print("\n========== RESULT ==========")
    print("Scale:", scale)
    print("Bias :", bias)


    np.save("scale_bias.npy", np.array([scale, bias]))

    print("\nSaved to scale_bias.npy")