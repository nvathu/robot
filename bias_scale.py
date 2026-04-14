import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from tqdm import tqdm

VALID_DIR = "./valid_data"
DEPTH_DIR = "./valid_data/depth"
DEBUG_DIR = "./debug_click"

os.makedirs(DEBUG_DIR, exist_ok=True)


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
            "img": img_path,
            "depth": depth_path,
            "self": (row["self_pose_x"], row["self_pose_y"]),
            "s1": (row["car_S1_pose_x"], row["car_S1_pose_y"]),
            "s3": (row["car_S3_pose_x"], row["car_S3_pose_y"])
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

    vis = img.copy()

    cv2.imshow("Click Robot 1 & Robot 2", vis)
    cv2.setMouseCallback("Click Robot 1 & Robot 2", mouse_callback)

    while True:
        cv2.imshow("Click Robot 1 & Robot 2", vis)
        key = cv2.waitKey(1)

        # ENTER or SPACE to confirm
        if key == 13 or key == 32:
            break

        # ESC skip
        if key == 27:
            click_points = []
            break

       
        tmp = img.copy()
        for i, p in enumerate(click_points):
            cv2.circle(tmp, p, 5, (0, 255, 0), -1)
            cv2.putText(tmp, str(i), p,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        vis = tmp

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
    return np.linalg.norm(np.array(p1) - np.array(p2))



def build_dataset(data):

    depths = []
    distances = []

    for item in tqdm(data):

        img = cv2.imread(item["img"])
        if img is None:
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(item["depth"], cv2.IMREAD_GRAYSCALE)
        if depth is None:
            continue

        pts = get_2_clicks(img)

        if pts is None:
            continue

        d1 = depth_at_point(depth, pts[0])
        d2 = depth_at_point(depth, pts[1])

        if d1 is None or d2 is None:
            continue

        
        depths.append(d1)
        depths.append(d2)

        dist1 = compute_distance(item["self"], item["s1"])
        dist2 = compute_distance(item["self"], item["s3"])

        distances.append(dist1)
        distances.append(dist2)

        
        vis = img.copy()
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 255, 0), -1)

        cv2.imwrite(
            os.path.join(DEBUG_DIR, os.path.basename(item["img"])),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

    return np.array(depths), np.array(distances)



def fit(depths, distances):

    A = np.vstack([depths, np.ones(len(depths))]).T
    scale, bias = np.linalg.lstsq(A, distances, rcond=None)[0]
    return scale, bias


if __name__ == "__main__":

    print("Loading data...")
    data = load_valid_data(VALID_DIR)

    np.random.shuffle(data)

    split = int(len(data) * 0.8)

    train = data[:split]
    test = data[split:]

    print("Train:", len(train))
    print("Test :", len(test))

    print("\n--- TRAIN PHASE (click manual) ---")
    train_d, train_y = build_dataset(train)

    print("\n--- TEST PHASE (click manual) ---")
    test_d, test_y = build_dataset(test)

    print("Train samples:", len(train_d))
    print("Test samples :", len(test_d))

    scale, bias = fit(train_d, train_y)

    print("\n")
    print("Scale:", scale)
    print("Bias :", bias)

    pred = scale * test_d + bias
    r2 = r2_score(test_y, pred)

    print("R2 score:", r2)

   
    plt.scatter(train_d, train_y, label="train")
    plt.scatter(test_d, test_y, label="test")

    x = np.linspace(min(train_d), max(train_d), 100)
    y = scale * x + bias

    plt.plot(x, y, color="red")
    plt.legend()
    plt.title("Depth → Distance")

    plt.savefig("result.png")
    plt.show()

    np.save("scale_bias.npy", np.array([scale, bias]))

    print("\nSaved scale_bias.npy")
