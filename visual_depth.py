import os
import numpy as np
import cv2
from tqdm import tqdm

INPUT_DIR = "./depth_real"
OUTPUT_DIR = "./depth_vis"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def npy_to_image(npy_path, save_path):

    depth = np.load(npy_path)

    depth = np.nan_to_num(depth)

    d_min = depth.min()
    d_max = depth.max()

    if d_max - d_min < 1e-6:
        return

    depth_norm = (depth - d_min) / (d_max - d_min)
    depth_uint8 = (depth_norm * 255).astype(np.uint8)

    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_INFERNO)

    cv2.imwrite(save_path, depth_color)


for session in os.listdir(INPUT_DIR):

    session_path = os.path.join(INPUT_DIR, session)

    if not os.path.isdir(session_path):
        continue

    save_session = os.path.join(OUTPUT_DIR, session)
    os.makedirs(save_session, exist_ok=True)

    files = [f for f in os.listdir(session_path) if f.endswith(".npy")]

    for f in tqdm(files, desc=session):

        npy_path = os.path.join(session_path, f)

        save_path = os.path.join(save_session, f.replace(".npy", ".png"))

        npy_to_image(npy_path, save_path)

print("\nDone visualize depth!")