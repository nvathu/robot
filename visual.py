import numpy as np
import matplotlib.pyplot as plt

data = np.load("dataset/pointclouds/room_pointcloud_near_range.npz")
points = data['scene_points']
colors = np.clip(data['scene_colors'] * 2.2 + 0.1, 0, 1)
robots = data['robot_points']
robot_colors = data['robot_colors']
self_traj = data['self_trajectory']

fig = plt.figure(figsize=(16, 7))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax1.scatter(points[:,0], points[:,2], points[:,1], c=colors, s=1, alpha=0.5)
ax1.scatter(robots[:,0], robots[:,2], robots[:,1], c=robot_colors, s=4, alpha=0.7)
ax1.plot(self_traj[:,0], self_traj[:,2], self_traj[:,1], c='lime', linewidth=0.8, alpha=0.8)
ax1.view_init(elev=25, azim=-60)
ax1.set_xlabel('X (m)')
ax1.set_ylabel('Z (m)')
ax1.set_zlabel('Y (m)')
ax1.set_title('3D Room Point Cloud (angled view)')
ax1.set_box_aspect([1,1,0.3])

ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(points[:,0], points[:,2], c=colors, s=1, alpha=0.5)
ax2.scatter(robots[:,0], robots[:,2], c=robot_colors, s=6, alpha=0.7)
ax2.plot(self_traj[:,0], self_traj[:,2], c='lime', linewidth=1.0, alpha=0.9, label='self trajectory')
ax2.set_xlabel('X (m)')
ax2.set_ylabel('Z (m)')
ax2.set_aspect('equal')
ax2.set_title('Top-down view')
ax2.legend(loc='upper right', fontsize=8)

plt.tight_layout()
plt.savefig("room_pointcloud_result.png", dpi=150)
print("Saved room_pointcloud_result.png")
