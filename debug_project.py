# check_ground_plane_validity.py
import json
import numpy as np
from ground_plane_projection import pixel_to_ground_point

with open("dataset/calib/ground_plane_params.json") as f:
    GP = json.load(f)
with open("dataset/calib/intrinsics.json") as f:
    K = json.load(f)

print(f"camera_height = {GP['camera_height_m']:.4f} m")
print(f"pitch = {GP['pitch_rad']:.4f} rad ({np.degrees(GP['pitch_rad']):.2f} deg)")

# Test thu voi vai pixel dai dien trong anh 180x180
test_pixels = [(90, 90), (90, 150), (90, 170), (0, 170), (179, 170)]
for u, v in test_pixels:
    result = pixel_to_ground_point(u, v, K['fx'], K['fy'], K['cx'], K['cy'],
                                     GP['camera_height_m'], GP['pitch_rad'])
    print(f"pixel ({u},{v}) -> {result}")