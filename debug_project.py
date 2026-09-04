import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

data = np.load("dataset/pointclouds/wall_outline_geometric.npz")
points = data['wall_points']

RADIUS = 0.05
tree = cKDTree(points)
counts = np.array(tree.query_ball_point(points, r=RADIUS, return_length=True))
thresh = np.percentile(counts, 90)
hd = points[counts >= thresh]
print(f"High-density points: {len(hd)}")

centroid = hd.mean(axis=0)
centered = hd - centroid
_, _, Vt = np.linalg.svd(centered, full_matrices=False)
line_residuals = centered @ Vt[1]
line_rmse = np.sqrt(np.mean(line_residuals**2))

def circle_residuals(params, pts):
    cx, cy, r = params
    d = np.sqrt((pts[:,0]-cx)**2 + (pts[:,1]-cy)**2)
    return d - r

x0 = [hd[:,0].mean(), hd[:,1].mean(), 1.5]
res = least_squares(circle_residuals, x0, args=(hd,))
cx, cy, r = res.x
circle_res = circle_residuals(res.x, hd)
circle_rmse = np.sqrt(np.mean(circle_res**2))

print("\n=== FIT COMPARISON ===")
print(f"Line RMSE = {line_rmse*100:.2f} cm")
print(f"Circle RMSE = {circle_rmse*100:.2f} cm  (center=({cx:.2f},{cy:.2f}), r={r:.2f}m)")

if circle_rmse < line_rmse * 0.6:
    print("\n-> Circle fits significantly better; likely artifact rather than real geometry.")
else:
    print("\n-> No strong difference; more data or methods needed.")

fig, ax = plt.subplots(figsize=(7,7))
ax.scatter(points[:,0], points[:,1], s=3, alpha=0.1, color='gray')
ax.scatter(hd[:,0], hd[:,1], s=8, color='red', label='high density')

theta = np.linspace(0, 2*np.pi, 200)
ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), 'g--', label=f'circle fit (r={r:.2f}m)')

line_dir = Vt[0]
proj = centered @ line_dir
line_pts = np.array([centroid + line_dir*proj.min(), centroid + line_dir*proj.max()])
ax.plot(line_pts[:,0], line_pts[:,1], 'b-', label='line fit')

ax.set_aspect('equal')
ax.legend()
ax.set_xlabel("X(m)")
ax.set_ylabel("Z(m)")
ax.set_title(f"Line RMSE={line_rmse*100:.1f}cm vs Circle RMSE={circle_rmse*100:.1f}cm")
plt.tight_layout()
plt.savefig("debug_depth_and_scale/line_vs_circle.png", dpi=150)
print("Saved debug_depth_and_scale/line_vs_circle.png")
