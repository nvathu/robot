import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

dataset = "./dataset/rgb"
OUTPUT_NPY = "./dataset/depth_npy"
OUTPUT_VIS = "./dataset/depth"

os.makedirs(OUTPUT_NPY, exist_ok=True)
os.makedirs(OUTPUT_VIS, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type).to(device).eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform


def collect_images(root):
    image_paths = []
    for session in os.listdir(root):
        session_path = os.path.join(root, session)

        if not os.path.isdir(session_path):
            continue
        for sub in os.listdir(session_path):
            sub_path = os.path.join(session_path,sub)
            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith(".png"):
                        image_paths.append((session, os.path.join(sub_path, f)))

    return image_paths


image_list = collect_images(dataset)
print("Total images:", len(image_list))

BATCH_SIZE = 16 

for i in tqdm(range(0, len(image_list), BATCH_SIZE)):

    batch_items = image_list[i:i + BATCH_SIZE]

    imgs = []
    metas = []

    for session, img_path in batch_items:

        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W = img_rgb.shape[:2]
        inp = transform(img_rgb) 
        imgs.append(inp)
        metas.append((session, img_path))

    batch_tensor = torch.cat(imgs, dim=0).to(device)

    with torch.no_grad():
        pred = midas(batch_tensor)

    for j, (session, img_path) in enumerate(metas):

        depth = pred[j]

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(0).unsqueeze(0),
            size=(H, W),
            mode="bicubic",
            align_corners=False
        ).squeeze()

        depth = depth.cpu().numpy().astype(np.float32)

        save_dir = os.path.join(OUTPUT_NPY, session)
        os.makedirs(save_dir, exist_ok=True)

        filename = os.path.basename(img_path)
        filename_no_ext = os.path.splitext(filename)[0]
        np.save(os.path.join(save_dir,filename_no_ext + ".npy"),depth)
        
        vis = depth.copy()
        vis = vis - vis.min()
        vis = vis / (vis.max() + 1e-8)
        vis = (vis * 255).astype(np.uint8)
        vis = cv2.applyColorMap(vis,cv2.COLORMAP_INFERNO)
        save_dir_vis = os.path.join(OUTPUT_VIS,session)
        os.makedirs(save_dir_vis, exist_ok=True)
        cv2.imwrite(os.path.join(save_dir_vis, filename),vis)
