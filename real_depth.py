import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

VALID_DIR = "./valid_data"
DEPTH_DIR = "./valid_data/depth"
DEBUG_DIR = "./debug_vis"

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

        img_path = None
        depth_path = os.path.join(DEPTH_DIR, img_name)

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



def detect_robots(img):

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    edges = cv2.Canny(blur, 50, 150)

    kernel = np.ones((5,5), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=2)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H, W = img.shape[:2]
    candidates = []

    for c in contours:
        area = cv2.contourArea(c)

        if area < 1500 or area > 50000:
            continue

        x, y, w, h = cv2.boundingRect(c)

        aspect = w / (h + 1e-6)

        if 0.3 < aspect < 3.5:
            cx = x + w//2
            cy = y + h//2

           
            score = area * (cy / H)

            candidates.append((score, (x,y,w,h)))

    candidates = sorted(candidates, key=lambda x: x[0], reverse=True)

    boxes = [b for _, b in candidates[:2]]

    return boxes



def depth_from_bbox(depth, bbox):

    x, y, w, h = bbox

    roi = depth[y:y+h, x:x+w]

    if roi.size == 0:
        return None

    return np.median(roi)



def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def debug_visual(img, boxes, depth_map, save_path):

    vis = img.copy()

    for i, (x,y,w,h) in enumerate(boxes):

        d = depth_from_bbox(depth_map, (x,y,w,h))

        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)

        label = f"B{i} d={d:.1f}" if d else "None"

        cv2.putText(vis, label, (x, y-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    cv2.imwrite(save_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


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

        boxes = detect_robots(img)

        if len(boxes) != 2:
            continue

        d1 = depth_from_bbox(depth, boxes[0])
        d2 = depth_from_bbox(depth, boxes[1])

        if d1 is None or d2 is None:
            continue

        
        dist_s1 = compute_distance(item["self"], item["s1"])
        dist_s3 = compute_distance(item["self"], item["s3"])

        depths.append(d1)
        distances.append(dist_s1)

        depths.append(d2)
        distances.append(dist_s3)

        debug_visual(
            img,
            boxes,
            depth,
            os.path.join(DEBUG_DIR, os.path.basename(item["img"]))
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

    train_d, train_y = build_dataset(train)
    test_d, test_y = build_dataset(test)

    print("Train usable:", len(train_d))
    print("Test usable :", len(test_d))

    scale, bias = fit(train_d, train_y)

    print("\n===== RESULT =====")
    print("Scale:", scale)
    print("Bias :", bias)

    pred = scale * test_d + bias
    r2 = r2_score(test_y, pred)

    print("R2:", r2)

    plt.scatter(train_d, train_y, s=5, label="train")
    plt.scatter(test_d, test_y, s=5, label="test")

    x = np.linspace(min(train_d), max(train_d), 100)
    y = scale * x + bias

    plt.plot(x, y, color="red")
    plt.legend()
    plt.title("Depth → Distance (FIXED 2 ROBOTS)")
    plt.savefig("result.png")
    plt.show()

    np.save("scale_bias.npy", np.array([scale, bias]))

    print("Saved OK")