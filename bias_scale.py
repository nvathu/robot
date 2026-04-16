import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        tmp = img.copy()

        for i, p in enumerate(click_points):
            cv2.circle(tmp, p, 5, (0, 255, 0), -1)

        cv2.imshow("Click Robot 1 & Robot 2", tmp)

        key = cv2.waitKey(1)

        if key == 13 or key == 32:
            break
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
    return np.linalg.norm(np.array(p1) - np.array(p2))



def solve_scale_bias(z1, z2, d1, d2):

    

    A = np.array([
        [z1, 1],
        [z2, 1]
    ])

    b = np.array([d1, d2])

    try:
        s, bias = np.linalg.solve(A, b)
        return s, bias
    except:
        return None, None


def build_dataset(data):

    scales = []
    biases = []

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

        z1 = depth_at_point(depth, pts[0])
        z2 = depth_at_point(depth, pts[1])

        if z1 is None or z2 is None:
            continue

        d1 = compute_distance(item["self"], item["s1"])
        d2 = compute_distance(item["self"], item["s3"])

        print("\nImage:", item["img"])
        print("Depth:", z1, z2)
        print("Distance:", d1, d2)

        scale, bias = solve_scale_bias(z1, z2, d1, d2)

        if scale is None:
            continue

        print("Scale:", scale, "Bias:", bias)

        scales.append(scale)
        biases.append(bias)

       
        vis = img.copy()
        for p in pts:
            cv2.circle(vis, p, 5, (0, 255, 0), -1)

        cv2.imwrite(
            os.path.join(DEBUG_DIR, os.path.basename(item["img"])),
            cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
        )

    return np.array(scales), np.array(biases)



if __name__ == "__main__":

    print("Loading data...")
    data = load_valid_data(VALID_DIR)

    np.random.shuffle(data)

    split = int(len(data) * 0.8)

    train = data[:split]
    test = data[split:]

    print("Train:", len(train))
    print("Test :", len(test))

    print("\n TRAIN ")
    train_s, train_b = build_dataset(train)

    print("\n TEST ")
    test_s, test_b = build_dataset(test)

 
    np.save("scale_bias.npy", np.array([
        np.mean(train_s),
        np.mean(train_b)
    ]))

   
    print("Mean Scale:", np.mean(train_s))
    print("Mean Bias :", np.mean(train_b))


    plt.figure()

    plt.subplot(1,2,1)
    plt.hist(train_s, bins=20)
    plt.title("Scale distribution")

    plt.subplot(1,2,2)
    plt.hist(train_b, bins=20)
    plt.title("Bias distribution")

    plt.savefig("scale_bias_dist.png")
    plt.show()

    print("\nSaved scale_bias.npy + plot")