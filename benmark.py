import torch
import time
import numpy as np
import cv2

from model import ResNetDepth


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE = (180, 180)
NUM_RUNS = 10


print("Loading models...")


resnet = ResNetDepth().to(DEVICE)
resnet.eval()


midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large").to(DEVICE)
midas.eval()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
midas_transform = midas_transforms.dpt_transform



dummy_img = np.random.randint(0, 255, (IMG_SIZE[0], IMG_SIZE[1], 3), dtype=np.uint8)


resnet_input = torch.tensor(dummy_img / 255.).permute(2,0,1).unsqueeze(0).float().to(DEVICE)


midas_input = midas_transform(dummy_img).to(DEVICE)



def benchmark_model(model, input_tensor, is_midas=False):

    
    for _ in range(10):
        with torch.no_grad():
            if is_midas:
                _ = model(input_tensor)
            else:
                _ = model(input_tensor)

    torch.cuda.synchronize() if DEVICE.type == "cuda" else None

    start = time.time()

    for _ in range(NUM_RUNS):
        with torch.no_grad():
            if is_midas:
                pred = model(input_tensor)
                pred = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=IMG_SIZE,
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)
            else:
                _ = model(input_tensor)

    torch.cuda.synchronize() if DEVICE.type == "cuda" else None

    end = time.time()

    total_time = end - start
    avg_time = total_time / NUM_RUNS
    fps = 1.0 / avg_time

    return avg_time, fps



print("\nRunning benchmark...\n")

resnet_time, resnet_fps = benchmark_model(resnet, resnet_input)
midas_time, midas_fps = benchmark_model(midas, midas_input, is_midas=True)




print(f"Device: {DEVICE}")
print()

print("ResNetDepth:")
print(f"  Avg time : {resnet_time*1000:.2f} ms")
print(f"  FPS      : {resnet_fps:.2f}")

print()

print("MiDaS (DPT_Large):")
print(f"  Avg time : {midas_time*1000:.2f} ms")
print(f"  FPS      : {midas_fps:.2f}")

print()

speedup = midas_time / resnet_time
print(f"Speedup (ResNet vs MiDaS): {speedup:.2f}x faster")