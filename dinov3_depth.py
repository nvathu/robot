import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
from torchvision.transforms import v2
from PIL import Image



REPO_DIR = '/home/thu/Thu/dinov3' 
DEPTHER_CKPT = "/home/thu/Thu/dinov3/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth"
BACKBONE_CKPT = "/home/thu/Thu/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth" 

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

depther = torch.hub.load(
    REPO_DIR, 
    'dinov3_vit7b16_dd', 
    source="local", 
    weights=DEPTHER_CKPT, 
    backbone_weights=BACKBONE_CKPT,
    pretrained=False
).to(device).eval()

def make_transform(resize_size: int | list[int] = 768):
    return v2.Compose([
        v2.ToImage(),
        v2.Resize((resize_size, resize_size), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])


def collect_images(root):
    image_paths = []
    for session in os.listdir(root):
        session_path = os.path.join(root, session)

        if not os.path.isdir(session_path):
            continue

        for sub in os.listdir(session_path):
            sub_path = os.path.join(session_path, sub)

            if os.path.isdir(sub_path):
                for f in os.listdir(sub_path):
                    if f.endswith(".png"):
                        image_paths.append((session, sub, os.path.join(sub_path, f)))

    return image_paths



input_dir = "./dataset/rgb"
output_dir = "./dataset/dinov3_depth"
IMG_SIZE = 518
BATCH_SIZE = 3  
transform = make_transform(IMG_SIZE)

image_list = collect_images(input_dir)


for i in tqdm(range(0, len(image_list), BATCH_SIZE)):
    batch_items = image_list[i:i + BATCH_SIZE]
    tensors = []
    metas = []

    for session, sub, img_path in batch_items:
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        tensors.append(transform(img))
        metas.append((session, sub, img_path, (orig_h, orig_w)))

    batch_img = torch.stack(tensors).to(device)

    with torch.inference_mode():
        with torch.autocast('cuda', dtype=torch.bfloat16):
            
            depths = depther(batch_img)

   
    preds_np = depths.squeeze(1).float().cpu().numpy()

    for j, (session, sub, img_path, (oh, ow)) in enumerate(metas):
        depth = preds_np[j]
        
        
        d_min, d_max = depth.min(), depth.max()
        depth_norm = (depth - d_min) / (d_max - d_min + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)


        depth_final = cv2.resize(depth_uint8, (ow, oh), interpolation=cv2.INTER_CUBIC)

        save_path = os.path.join(output_dir, session, sub)
        os.makedirs(save_path, exist_ok=True)
        cv2.imwrite(os.path.join(save_path, os.path.basename(img_path)), depth_final)