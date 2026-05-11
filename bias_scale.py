import os
import cv2
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

VALID_DIR = "./valid_data"
DEPTH_DIR = "./valid_data/depth"

DEBUG_DIR = "./debug_click"
SAVE_DIR = "./saved_data"

os.makedirs(DEBUG_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

CLICK_SAVE = os.path.join(SAVE_DIR, "click_points.json")
PAIR_SAVE = os.path.join(SAVE_DIR, "depth_distance_pairs.csv")
SCALE_BIAS_SAVE = os.path.join(SAVE_DIR, "scale_bias.npy")


def load_valid_data(valid_dir):

    csv_path = None

    for f in os.listdir(valid_dir):
        if f.endswith(".csv"):
            csv_path = os.path.join(valid_dir, f)
            break

    if csv_path is None:
        raise Exception("No CSV found in valid_data")

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

        img_path = None

        for root, _, files in os.walk(valid_dir):

            if img_name in files:
                img_path = os.path.join(root, img_name)
                break

        if img_path is None:
            continue

        if not os.path.exists(depth_path):
            continue

        data.append({
            "img_name": img_name,
            "img": img_path,
            "depth": depth_path,

            "self": (
                row["self_pose_x"],
                row["self_pose_y"]
            ),

            "s1": (
                row["car_S1_pose_x"],
                row["car_S1_pose_y"]
            ),

            "s3": (
                row["car_S3_pose_x"],
                row["car_S3_pose_y"]
            )
        })

    return data



click_points = []


def mouse_callback(event, x, y, flags, param):

    global click_points

    if event == cv2.EVENT_LBUTTONDOWN:
        click_points.append((x, y))


def get_2_clicks(img):

    global click_points

    click_points = []

    cv2.namedWindow("Click Robot 1 and Robot 2")
    cv2.setMouseCallback(
        "Click Robot 1 and Robot 2",
        mouse_callback
    )

    while True:

        vis = img.copy()

        for i, p in enumerate(click_points):

            cv2.circle(vis, p, 6, (0, 255, 0), -1)

            cv2.putText(
                vis,
                f"{i+1}",
                (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imshow("Click Robot 1 and Robot 2", vis)

        key = cv2.waitKey(1)

        # ENTER or SPACE
        if key == 13 or key == 32:
            break

        # ESC
        if key == 27:
            click_points = []
            break

    cv2.destroyAllWindows()

    if len(click_points) < 2:
        return None

    return click_points[:2]


def depth_at_point(depth, pt):

    x, y = pt

    h, w = depth.shape

    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    return float(depth[y, x])


def compute_distance(p1, p2):

    return np.linalg.norm(
        np.array(p1) - np.array(p2)
    )


def solve_scale_bias(z1, z2, d1, d2):

    A = np.array([
        [z1, 1],
        [z2, 1]
    ])

    b = np.array([
        d1,
        d2
    ])

    try:

        scale, bias = np.linalg.solve(A, b)

        return scale, bias

    except:

        return None, None


def load_old_clicks():

    if not os.path.exists(CLICK_SAVE):
        return {}

    with open(CLICK_SAVE, "r") as f:
        return json.load(f)


def save_clicks(click_data):

    with open(CLICK_SAVE, "w") as f:
        json.dump(click_data, f, indent=4)



def build_dataset(data):

    scales = []
    biases = []

    pair_rows = []

    click_data = load_old_clicks()

    for item in tqdm(data):

        img_name = item["img_name"]

        
        img = cv2.imread(item["img"])

        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(
            item["depth"],
            cv2.IMREAD_GRAYSCALE
        )

        if depth is None:
            continue

        
        if img_name in click_data:

            pts = click_data[img_name]

            pts = [
                tuple(pts[0]),
                tuple(pts[1])
            ]

            print(f"\nUsing saved clicks: {img_name}")

        else:

            print(f"\nClick robots: {img_name}")

            pts = get_2_clicks(img)

            if pts is None:
                continue

            click_data[img_name] = pts

            save_clicks(click_data)

       
        z1 = depth_at_point(depth, pts[0])
        z2 = depth_at_point(depth, pts[1])

        if z1 is None or z2 is None:
            continue

      
        d1 = compute_distance(
            item["self"],
            item["s1"]
        )

        d2 = compute_distance(
            item["self"],
            item["s3"]
        )

       
        scale, bias = solve_scale_bias(
            z1, z2,
            d1, d2
        )

        if scale is None:
            continue

       
        print("\nImage:", img_name)

        print("\nDepth:")
        print("Robot1:", z1)
        print("Robot2:", z2)

        print("\nDistance:")
        print("Robot1:", d1)
        print("Robot2:", d2)

        print("\nSolved:")
        print("Scale:", scale)
        print("Bias :", bias)

        
        scales.append(scale)
        biases.append(bias)

        pair_rows.append({
            "image": img_name,

            "z1": z1,
            "z2": z2,

            "d1": d1,
            "d2": d2,

            "scale": scale,
            "bias": bias,

            "x1": pts[0][0],
            "y1": pts[0][1],

            "x2": pts[1][0],
            "y2": pts[1][1]
        })

        
        vis = img.copy()

        for i, p in enumerate(pts):

            cv2.circle(
                vis,
                p,
                7,
                (0, 255, 0),
                -1
            )

            cv2.putText(
                vis,
                f"R{i+1}",
                (p[0] + 5, p[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

        cv2.imwrite(
            os.path.join(DEBUG_DIR, img_name),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

   
    df = pd.DataFrame(pair_rows)

    df.to_csv(
        PAIR_SAVE,
        index=False
    )

    print("\nSaved pair CSV:")
    print(PAIR_SAVE)

    return np.array(scales), np.array(biases)

if __name__ == "__main__":

    print("Loading data...")

    data = load_valid_data(VALID_DIR)

    np.random.shuffle(data)

    split = int(len(data) * 0.8)

    train = data[:split]
    test = data[split:]

    print("\nTrain:", len(train))
    print("Test :", len(test))

    print("\n TRAIN ")

    train_s, train_b = build_dataset(train)

    print("\n TEST")

    test_s, test_b = build_dataset(test)

   
    mean_scale = np.mean(train_s)
    mean_bias = np.mean(train_b)

    np.save(
        SCALE_BIAS_SAVE,
        np.array([
            mean_scale,
            mean_bias
        ])
    )

   
    print("\nFINAL RESULT")

    print("Mean Scale:", mean_scale)
    print("Mean Bias :", mean_bias)

    print("\nSaved:")
    print(SCALE_BIAS_SAVE)

  
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)

    plt.hist(train_s, bins=20)

    plt.title("Scale Distribution")

    plt.subplot(1, 2, 2)

    plt.hist(train_b, bins=20)

    plt.title("Bias Distribution")

    plt.tight_layout()

    plt.savefig("scale_bias_dist.png")

    plt.show()

    print("\nSaved scale_bias_dist.png")