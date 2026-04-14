import os
import cv2
import numpy as np
from tqdm import tqdm

DEPTH_INPUT = "./dataset/depth"      
OUTPUT_DIR = "./depth_real"  

os.makedirs(OUTPUT_DIR, exist_ok=True)


scale, bias = np.load("scale_bias.npy")

print("Loaded scale:", scale)
print("Loaded bias :", bias)


for session in os.listdir(DEPTH_INPUT):

    session_path = os.path.join(DEPTH_INPUT, session)

    if not os.path.isdir(session_path):
        continue

    save_session = os.path.join(OUTPUT_DIR, session)
    os.makedirs(save_session, exist_ok=True)

    for f in tqdm(os.listdir(session_path), desc=session):

        if not f.endswith(".png"):
            continue

        depth_path = os.path.join(session_path, f)

        depth = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)

        if depth is None:
            continue

        depth = depth.astype(np.float32)

      
        real_depth = scale * depth + bias

      
        real_depth = np.clip(real_depth, 0, None)

        np.save(
            os.path.join(save_session, f.replace(".png", ".npy")),
            real_depth
        )

        vis = real_depth.copy()
        vis = vis / (vis.max() + 1e-6) * 255
        vis = vis.astype(np.uint8)

        cv2.imwrite(
            os.path.join(save_session, f),
            vis
        )

print("\nDone converting to REAL DEPTH")
