import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torch.utils.tensorboard import SummaryWriter

from dataset import DepthDataset
from model import ResNetDepth

import matplotlib.pyplot as plt
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("./outputs", exist_ok=True)
writer = SummaryWriter("./runs")

dataset = DepthDataset("./dataset/rgb", "./dataset/depth")

total_size = len(dataset)

train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_set, val_set, test_set = random_split(
    dataset, [train_size, val_size, test_size]
)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

model = ResNetDepth().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.L1Loss()

def log_images(writer, img, pred, target, epoch):

    writer.add_image("Input", img, epoch)
    writer.add_image("Prediction", pred, epoch)
    writer.add_image("GroundTruth", target, epoch)

def visualize(img, pred, target, epoch):
    img = img.permute(1,2,0).cpu().numpy()
    pred = pred.detach().cpu().numpy()[0]
    target = target.detach().cpu().numpy()[0]

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input")

    plt.subplot(1,3,2)
    plt.imshow(pred, cmap="inferno")
    plt.title("Prediction")

    plt.subplot(1,3,3)
    plt.imshow(target, cmap="inferno")
    plt.title("GT")

    plt.tight_layout()
    plt.savefig(f"./outputs/epoch_{epoch}.png")
    plt.close()

num_epochs = 5

train_losses = []
val_losses = []

fixed_img = None
fixed_depth = None

for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    for img, depth in train_loader:

        img = img.to(device)
        depth = depth.to(device)

        if fixed_img is None:
            fixed_img = img[0].unsqueeze(0).clone()
            fixed_depth = depth[0].unsqueeze(0).clone()

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
        for img, depth in val_loader:
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
        fixed_pred = model(fixed_img.to(device))
    
    log_images(
        writer,
        fixed_img[0].cpu(),
        fixed_pred[0].cpu(),
        fixed_depth[0].cpu(),
        epoch
    )
    visualize(
        fixed_img[0].cpu(),
        fixed_pred[0].cpu(),
        fixed_depth[0].cpu(),
        epoch
    )
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")



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
