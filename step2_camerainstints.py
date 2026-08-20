import numpy as np
import json, os
W, H = 640, 480
FOV_deg = 70.0
FOV_rad = np.radians(FOV_deg)

fx = (W / 2) / np.tan(FOV_rad / 2)
cx, cy = W / 2.0, H / 2.0

fy = fx  

print(f"fx = fy = {fx:.2f}")
print(f"cx = {cx}, cy = {cy}")

intrinsics = {
    "fx": float(fx),
    "fy": float(fx),
    "cx": float(W / 2),
    "cy": float(H / 2),
    "source": "computed from published FOV=70deg, resolution=640x480 (Osoyoo Pi Robot Car spec)"
}

os.makedirs("dataset/calib", exist_ok=True)
with open("dataset/calib/intrinsics.json", "w") as fp:
    json.dump(intrinsics, fp, indent=2)

print(intrinsics)