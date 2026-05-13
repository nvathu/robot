import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import DepthDataset
from model import ResNetDepth
from tqdm import tqdm
import time
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import os
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./outputs", exist_ok=True)
run_name = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"./runs/{run_name}")



dataset = DepthDataset("./dataset/rgb", "./dataset/depth")

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True,num_workers=8)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False,num_workers=8)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False,num_workers=8)

model = ResNetDepth().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# loss_fn = nn.L1Loss()
loss_fn = nn.MSELoss()
# loss_fn = nn.CrossEntropyLoss()

def to_inferno(tensor):
    
    img = tensor.detach().cpu().numpy()[0]

    img = (img - img.min()) / (img.max() - img.min() + 0.000001)
    img = (img * 255).astype(np.uint8)

    img_color = cv2.applyColorMap(img, cv2.COLORMAP_INFERNO)
    img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

    img_color = torch.tensor(img_color).permute(2, 0, 1).float() / 255.0

    return img_color

def log_images(writer, imgs, preds, targets, epoch):

    N = min(20, imgs.shape[0])

    for i in range(N):

        writer.add_image(
            f"Input/{i}",
            imgs[i].cpu(),
            epoch
        )

        writer.add_image(
            f"Prediction/{i}",
            to_inferno(preds[i]),
            epoch
        )

        writer.add_image(
            f"GT/{i}",
            to_inferno(targets[i]),
            epoch
        )

num_epochs = 100

train_losses = []
val_losses = []


start_time = time.time()
for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for img, depth in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):

        img = img.to(device)
        depth = depth.to(device)

        pred = model(img)
        loss = loss_fn(pred, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()


    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    val_loss = 0

    with torch.no_grad():
        for img, depth in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            img = img.to(device)
            depth = depth.to(device)

            pred = model(img)
            loss = loss_fn(pred, depth)

            val_loss += loss.item()

    val_loss /= len(val_loader)
    val_losses.append(val_loss)


    writer.add_scalar("Loss/train", train_loss, epoch)
    writer.add_scalar("Loss/val", val_loss, epoch)



    with torch.no_grad():
        sample_imgs, sample_depths = next(iter(val_loader))

        sample_imgs = sample_imgs.to(device)
        sample_preds = model(sample_imgs)

    log_images(
        writer,
        sample_imgs,
        sample_preds,
        sample_depths,
        epoch
    )

    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
end_time = time.time()
print(f"Total training time: {end_time - start_time:.2f} seconds")



model.eval()
test_loss = 0

with torch.no_grad():
    for img, depth in test_loader:
        img = img.to(device)
        depth = depth.to(device)

        pred = model(img)
        loss = loss_fn(pred, depth)

        test_loss += loss.item()

test_loss /= len(test_loader)

print(f"Final Test Loss: {test_loss:.4f}")


plt.figure()
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")

plt.legend()
plt.grid()

plt.savefig("./outputs/loss_curve.png")
plt.close()

print("Training complete. Loss curve saved.")
