import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

# Define the points
points_A = np.array([[1 + i, i - 1] for i in range(6)])
points_B = np.array([[-i, i] for i in range(6)])
points_C = np.array([[0, i] for i in range(6)])
points = np.vstack([points_A, points_B, points_C])

# Create the Voronoi diagram
vor = Voronoi(points)

# Count the number of half-lines (unbounded edges)
num_half_lines = sum(1 for region in vor.regions if len(region) > 0 and -1 in region)

# Plot the Voronoi diagram
fig, ax = plt.subplots(figsize=(10, 10))
voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1.5)

# Plot points for visibility
ax.scatter(points[:, 0], points[:, 1], color='blue', s=80, label='Points')

# Set zoom-out limits for half-lines
x_min, x_max = points[:, 0].min() - 10, points[:, 0].max() + 10
y_min, y_max = points[:, 1].min() - 10, points[:, 1].max() + 10
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)

# Title and annotation
ax.set_title('Voronoi Diagram with Half-Lines')
ax.text(0.95, 0.05, f"Number of half-lines: {num_half_lines}", transform=ax.transAxes,
        fontsize=12, ha='right', bbox=dict(facecolor='white', alpha=0.6, edgecolor='gray'))

# Show the plot
plt.show()
