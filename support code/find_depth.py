import os
import shutil

valid_img_dir = "../valid_data"
depth_root = "../dataset/depth"
valid_depth_dir = "../valid_data/depth"

os.makedirs(valid_depth_dir, exist_ok=True)

for img_name in os.listdir(valid_img_dir):
    # depth file có cùng tên
    depth_name = img_name

    found = False
    for root, dirs, files in os.walk(depth_root):
        if depth_name in files:
            src = os.path.join(root, depth_name)
            dst = os.path.join(valid_depth_dir, depth_name)
            shutil.copy(src, dst)
            print("Copied:", depth_name)
            found = True
            break

    if not found:
        print("⚠️ Không tìm thấy depth cho:", img_name)
