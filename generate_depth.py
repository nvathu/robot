import torch
import cv2
import os
import numpy as np
from tqdm import tqdm

dataset = "./dataset/rgb"
output = "./dataset/depth"

os.makedirs(output, exist_ok=True)

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

for session, img_path in tqdm(image_list):

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = transform(img_rgb).to(device)

    with torch.no_grad():
        pred = midas(input_tensor)

    pred = pred.squeeze().cpu().numpy()

    pred = cv2.normalize(pred, None, 0, 255, cv2.NORM_MINMAX)
    pred = pred.astype(np.uint8)

    save_dir = os.path.join(output, session)
    os.makedirs(save_dir, exist_ok=True)

    filename = os.path.basename(img_path)
    cv2.imwrite(os.path.join(save_dir, filename), pred)
