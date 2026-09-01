# visualize_room.py
import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset/pointclouds/room_pointcloud.npz")
points = data['scene_points']
colors = np.clip(data['scene_colors'] * 2.2 + 0.1, 0, 1)
robots = data['robot_points']

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(points[:,0], points[:,2], points[:,1], c=colors, s=1, alpha=0.5)
ax1.scatter(robots[:,0], robots[:,2], robots[:,1], c='red', s=3, alpha=0.6)
ax1.view_init(elev=25, azim=-60)
ax1.set_xlabel('X (m)'); ax1.set_ylabel('Z (m)'); ax1.set_zlabel('Y (m)')
ax1.set_title('3D Room Point Cloud (goc nghieng)')
ax1.set_box_aspect([1,1,0.3])

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(points[:,0], points[:,2], c=colors, s=1, alpha=0.5)
ax2.scatter(robots[:,0], robots[:,2], c='red', s=5, alpha=0.7)
ax2.set_xlabel('X (m)'); ax2.set_ylabel('Z (m)')
ax2.set_aspect('equal')
ax2.set_title('Top-down view')

plt.tight_layout()
plt.savefig("room_pointcloud_result.png", dpi=150)
print("Da luu room_pointcloud_result.png")