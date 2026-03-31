import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import DepthDataset
from model import ResNetDepth

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DepthDataset("./dataset/rgb", "./dataset/depth")

loader = DataLoader(dataset, batch_size=8, shuffle=True)

model = ResNetDepth().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = nn.L1Loss()

writer = SummaryWriter("./runs")

step = 0

def visualize(img, pred, target, step):
    img = img.permute(1,2,0).cpu().numpy()
    pred = pred.detach().cpu().numpy()[0]
    target = target.detach().cpu().numpy()[0]

    plt.figure(figsize=(10,3))

    plt.subplot(1,3,1)
    plt.imshow(img)
    plt.title("Input")

    plt.subplot(1,3,2)
    plt.imshow(pred, cmap="inferno")
    plt.title("Pred")

    plt.subplot(1,3,3)
    plt.imshow(target, cmap="inferno")
    plt.title("GT")

    plt.savefig(f"./outputs/{step}.png")
    plt.close()


for epoch in range(3):

    for img, depth in loader:

        img = img.to(device)
        depth = depth.to(device)

        pred = model(img)

        loss = loss_fn(pred, depth)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        writer.add_scalar("loss", loss.item(), step)

        if step % 200 == 0:
            visualize(img[0], pred[0], depth[0], step)

        step += 1

print("Training done")
