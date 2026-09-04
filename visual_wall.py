import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

WALL_HEIGHT_M = 2.0
N_VERTICAL_LEVELS = 15

data = np.load("dataset/pointclouds/wwall_outline_geometric.npz")
wall_points_2d = data['wall_points']

y_levels = np.linspace(0, WALL_HEIGHT_M, N_VERTICAL_LEVELS)
wall_points_3d = []
for y in y_levels:
    layer = np.column_stack([
        wall_points_2d[:, 0],
        np.full(len(wall_points_2d), y),
        wall_points_2d[:, 1],
    ])
    wall_points_3d.append(layer)
wall_points_3d = np.concatenate(wall_points_3d, axis=0)

print(f"Original 2D points: {len(wall_points_2d)}")
print(f"Extruded 3D points: {len(wall_points_3d)}")

np.savez("dataset/pointclouds/wall_outline_3d_extruded.npz", wall_points_3d=wall_points_3d)
print("Saved dataset/pointclouds/wall_outline_3d_extruded.npz")

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(
    wall_points_3d[:, 0],
    wall_points_3d[:, 2],
    wall_points_3d[:, 1],
    s=1.5,
    alpha=0.3,
    c=wall_points_3d[:, 1],
    cmap='Blues'
)
ax1.set_xlabel("X (m)")
ax1.set_ylabel("Z (m)")
ax1.set_zlabel("Y (m)")
ax1.set_title(f"3D Wall Surface (extruded to {WALL_HEIGHT_M}m)")
ax1.view_init(elev=20, azim=-50)
ax1.set_box_aspect([1, 1, 0.5])

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(wall_points_2d[:, 0], wall_points_2d[:, 1], s=3, alpha=0.4)
ax2.set_xlabel("X (m)")
ax2.set_ylabel("Z (m)")
ax2.set_aspect('equal')
ax2.set_title("Original 2D outline")

plt.tight_layout()
plt.savefig("wall_3d_visualization.png", dpi=150)
print("Saved wall_3d_visualization.png")
