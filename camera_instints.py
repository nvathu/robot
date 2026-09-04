import json, os
import numpy as np

W, H = 640, 480
FOV_deg = 70.0
fx = (W / 2) / np.tan(np.radians(FOV_deg) / 2)

intrinsics = {
    "fx": float(fx),
    "fy": float(fx),
    "cx": float(W / 2),
    "cy": float(H / 2),
    "source": "computed from FOV=70deg, resolution=640x480"
}

os.makedirs("dataset/calib", exist_ok=True)
with open("dataset/calib/intrinsics.json", "w") as fp:
    json.dump(intrinsics, fp, indent=2)

print(intrinsics)
