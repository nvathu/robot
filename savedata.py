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
CLICK_SAVE = "click_points.npy"

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
            "name": img_name,
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


def get_clicks(img, img_name, click_db):

    global click_points

    if img_name in click_db:
        return click_db[img_name]

    click_points = []
    vis = img.copy()

    cv2.imshow("Click 2 robots", vis)
    cv2.setMouseCallback("Click 2 robots", mouse_callback)

    while True:
        tmp = img.copy()

        for i, p in enumerate(click_points):
            cv2.circle(tmp, p, 5, (0, 255, 0), -1)
            cv2.putText(tmp, str(i), p,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2)

        cv2.imshow("Click 2 robots", tmp)

        key = cv2.waitKey(1)

        if key == 13 or key == 32:  # enter / space
            break
        if key == 27:  # ESC skip
            click_points = []
            break

    cv2.destroyAllWindows()

    if len(click_points) < 2:
        return None

    pts = click_points[:2]

    click_db[img_name] = pts

    return pts


def depth_at_point(depth, pt):
    x, y = pt
    h, w = depth.shape

    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    return float(depth[y, x])


def compute_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))


def build_dataset(data, click_db):

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

        pts = get_clicks(img, item["name"], click_db)

        if pts is None:
            continue

        d1 = depth_at_point(depth, pts[0])
        d2 = depth_at_point(depth, pts[1])

        if d1 is None or d2 is None:
            continue

        dist1 = compute_distance(item["self"], item["s1"])
        dist2 = compute_distance(item["self"], item["s3"])

       
        print("\nImage:", item["name"])
        print("Depth:", d1, d2)
        print("Distance:", dist1, dist2)

        depths += [d1, d2]
        distances += [dist1, dist2]

    
        vis = img.copy()
        for p in pts:
            cv2.circle(vis, p, 5, (0, 255, 0), -1)

        cv2.imwrite(
            os.path.join(DEBUG_DIR, item["name"]),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

    return np.array(depths), np.array(distances)



def fit(depths, distances):
    A = np.vstack([depths, np.ones(len(depths))]).T
    return np.linalg.lstsq(A, distances, rcond=None)[0]


if __name__ == "__main__":

    print("Loading data...")
    data = load_valid_data(VALID_DIR)

    np.random.shuffle(data)

    split = int(len(data) * 0.8)
    train = data[:split]
    test = data[split:]

    print("Train:", len(train))
    print("Test :", len(test))

    
    if os.path.exists(CLICK_SAVE):
        click_db = np.load(CLICK_SAVE, allow_pickle=True).item()
        print("Loaded click DB:", len(click_db))
    else:
        click_db = {}

   
    print("\n--- TRAIN ---")
    train_d, train_y = build_dataset(train, click_db)

    print("\n--- TEST ---")
    test_d, test_y = build_dataset(test, click_db)

    np.save(CLICK_SAVE, click_db)

    print("Saved click database")

    scale, bias = fit(train_d, train_y)

    print("\nRESULT ")
    print("Scale:", scale)
    print("Bias :", bias)

    pred = scale * test_d + bias
    r2 = r2_score(test_y, pred)

    print("R2:", r2)

  
    np.save("train_depth.npy", train_d)
    np.save("train_dist.npy", train_y)
    np.save("test_depth.npy", test_d)
    np.save("test_dist.npy", test_y)

    pd.DataFrame({
        "depth": train_d,
        "distance": train_y
    }).to_csv("train.csv", index=False)

    pd.DataFrame({
        "depth": test_d,
        "distance": test_y
    }).to_csv("test.csv", index=False)

    plt.scatter(train_d, train_y, label="train")
    plt.scatter(test_d, test_y, label="test")

    x = np.linspace(min(train_d), max(train_d), 100)
    y = scale * x + bias

    plt.plot(x, y, color="red")

    plt.legend()
    plt.title("Manual Depth → Distance")

    plt.savefig("result.png")
    plt.show()


    np.save("scale_bias.npy", np.array([scale, bias]))

    print("\nSaved EVERYTHING successfully")