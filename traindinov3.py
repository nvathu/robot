import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F  

from tqdm import tqdm
from datetime import datetime

from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.tensorboard import SummaryWriter

from pytorch_msssim import SSIM

from dataset import DepthDataset
from model import ResNetDepth


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs("./outputs", exist_ok=True)
os.makedirs("./weights", exist_ok=True)

run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"./runs/{run_name}")

full_dataset = DepthDataset("./dataset/rgb_dino", "./dataset/dino_depth")
total_size = len(full_dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

torch.manual_seed(42)
train_set, val_set, test_set = random_split(
    full_dataset, [train_size, val_size, test_size],
    generator=torch.Generator().manual_seed(42)
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=8)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=8)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=8)


model = ResNetDepth().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)


class ScaleInvariantLoss(nn.Module):
    def __init__(self, lam=0.5):
        super().__init__()
        self.lam = lam

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.000001)
        target = torch.clamp(target, 0.000001)

        d = torch.log(pred) - torch.log(target)
        mse = torch.mean(d**2)
        correction = self.lam * (torch.mean(d) ** 2)

        return mse - correction


class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]

        gt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        gt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        loss_x = torch.mean(torch.abs(pred_dx - gt_dx))
        loss_y = torch.mean(torch.abs(pred_dy - gt_dy))

        return loss_x + loss_y


ssi_loss_fn = ScaleInvariantLoss(lam=0.0)
grad_loss_fn = GradientLoss()
ssim_loss_fn = SSIM(data_range=1.0, size_average=True, channel=1)
smooth_l1_loss_fn = nn.SmoothL1Loss(beta=0.1)

def compute_loss(pred, target):
    if pred.shape[2:] != target.shape[2:]:
        pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=True)
    pred = F.elu(pred) + 1.0 + 1e-3
    target = target + 1e-31
 

    l_ssi = ssi_loss_fn(pred, target)
    l_grad = grad_loss_fn(pred, target)
    def normalize_for_ssim(x):
        b = x.shape[0]
        x_flat = x.view(b, -1)
        x_min = x_flat.min(dim=1)[0].view(b, 1, 1, 1)
        x_max = x_flat.max(dim=1)[0].view(b, 1, 1, 1)
        return (x - x_min) / (x_max - x_min + 1e-6)
    l_ssim = 1.0 - ssim_loss_fn(normalize_for_ssim(pred), normalize_for_ssim(target))
    l_l1 = smooth_l1_loss_fn(pred, target)

    total = 0.3 * l_ssi + 1.0 * l_grad + 0.5 * l_ssim + 3.0 * l_l1
    return total, l_ssi, l_grad, l_ssim


def to_inferno(tensor):

    img = tensor.detach().cpu().squeeze().numpy()
    
    
    if len(img.shape) == 3:
        img = img[0]

    img_min = img.min()
    img_max = img.max()
    
    if (img_max - img_min) > 1e-6:
        img = (img - img_min) / (img_max - img_min)
    else:
        img = np.zeros_like(img)
    img = (img * 255).astype(np.uint8)
    img = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    return img


def log_images(writer, imgs, preds, targets, epoch):
    N = min(10, imgs.shape[0])

    if preds.shape[2:] != targets.shape[2:]:
        preds = F.interpolate(preds, size=targets.shape[2:], mode="bilinear", align_corners=True)

    for i in range(N):
        writer.add_image(f"Input/{i}", imgs[i].cpu(), epoch)
        writer.add_image(f"Prediction/{i}", to_inferno(preds[i]), epoch)
        writer.add_image(f"GT/{i}", to_inferno(targets[i]), epoch)


num_epochs = 100
train_losses = []
val_losses = []
best_val_loss = 999999

start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_ssi = 0
    train_grad = 0
    train_ssim = 0

    for img, depth in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        img = img.to(device)
        depth = depth.to(device)

        pred = model(img)
        loss, l_ssi, l_grad, l_ssim = compute_loss(pred, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_ssi += l_ssi.item()
        train_grad += l_grad.item()
        train_ssim += l_ssim.item()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, depth in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            img = img.to(device)
            depth = depth.to(device)

            pred = model(img)
            loss, _, _, _ = compute_loss(pred, depth)
            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "./weights/best_model.pth")

    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)
    

    with torch.no_grad():
        sample_imgs, sample_depths = next(iter(val_loader))
        sample_imgs = sample_imgs.to(device)
        sample_preds = model(sample_imgs)

    log_images(writer, sample_imgs, sample_preds, sample_depths, epoch)
    print(f"Epoch {epoch}: Train={train_loss:.4f} Val={val_loss:.4f}")

end_time = time.time()
print(f"\nTraining Time: {(end_time-start_time)/60:.2f} min")


model.load_state_dict(torch.load("./weights/best_model.pth", map_location=device))
model.eval()
test_loss = 0

with torch.no_grad():
    for img, depth in test_loader:
        img = img.to(device)
        depth = depth.to(device)

        pred = model(img)
        loss, _, _, _ = compute_loss(pred, depth)
        test_loss += loss.item()

test_loss /= len(test_loader)
print(f"\nFinal Test Loss: {test_loss:.4f}")

plt.figure()
plt.plot(train_losses, label="Train")
plt.plot(val_losses, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.savefig("./outputs/loss_curve.png")
plt.close()

print("\nTraining complete.")
