import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

class DepthDataset(Dataset):

    def __init__(self, rgb_root, depth_root):

        self.samples = []

        self.rgb_root = rgb_root
        self.depth_root = depth_root

        for session in os.listdir(rgb_root):

            session_path = os.path.join(
                rgb_root,
                session
            )

            if not os.path.isdir(session_path):
                continue

            for sub in os.listdir(session_path):

                sub_path = os.path.join(
                    session_path,
                    sub
                )

                if not os.path.isdir(sub_path):
                    continue

                for f in os.listdir(sub_path):

                    if not f.endswith(".png"):
                        continue

                    rgb_path = os.path.join(
                        sub_path,
                        f
                    )

                    
                    npy_name = os.path.splitext(f)[0] + ".npy"

                    depth_path = os.path.join(
                        depth_root,
                        session,
                        npy_name
                    )

                    if os.path.exists(depth_path):

                        self.samples.append(
                            (rgb_path, depth_path)
                        )

        print("Dataset size:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        rgb_path, depth_path = self.samples[idx]
        img = cv2.imread(rgb_path)
        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )
        img = cv2.resize(
            img,
            (180, 180),
            interpolation=cv2.INTER_AREA
        )

        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).permute(2, 0, 1)
        depth = np.load(depth_path).astype(np.float32)

        depth = cv2.resize(
            depth,
            (180, 180),
            interpolation=cv2.INTER_LINEAR
        )
        depth = np.log(depth + 1.0)

        depth = depth / depth.max()

        depth = torch.tensor(depth).unsqueeze(0)

        return img.float(), depth.float()
