import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull, Voronoi, voronoi_plot_2d
import random

def generate_A7_A8(initial_points, max_attempts=10000, range_min=-10, range_max=25):
    """
    Generates two random points A7 and A8 such that the convex hull
    of all 8 points has exactly 4 vertices.
    """
    for attempt in range(max_attempts):
        A7 = np.array([random.uniform(range_min, range_max), random.uniform(range_min, range_max)])
        A8 = np.array([random.uniform(range_min, range_max), random.uniform(range_min, range_max)])
        all_points = np.vstack([initial_points, A7, A8])
        hull = ConvexHull(all_points)
        if len(hull.vertices) == 4:
            return A7, A8
    raise ValueError(f"Could not find suitable A7 and A8 within {max_attempts} attempts.")

def plot_voronoi(all_points, hull):
    """
    Plots the Voronoi diagram, convex hull, and points with requested aesthetics.
    """
    vor = Voronoi(all_points)
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # White background and no grid
    ax.set_facecolor('white')
    
    # Plot Voronoi diagram (gray edges)
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='gray', line_width=1.5, line_alpha=0.9, point_size=2)
    
    # Plot points
    ax.scatter(all_points[:6, 0], all_points[:6, 1], color='blue', label='A1-A6', zorder=2)
    ax.scatter(all_points[6, 0], all_points[6, 1], color='red', s=80, label='A7', marker='s', zorder=3)
    ax.scatter(all_points[7, 0], all_points[7, 1], color='green', s=80, label='A8', marker='s', zorder=3)
    
    # Convex hull
    for simplex in hull.simplices:
        ax.plot(all_points[simplex, 0], all_points[simplex, 1], 'k-', lw=2, zorder=1)
    ax.scatter(all_points[hull.vertices, 0], all_points[hull.vertices, 1], color='orange', s=100, zorder=4, label='Convex Hull Points')
    
    # Increase padding for more zoom-out
    padding = 50  # Increased for even more zoom-out
    x_min, x_max = all_points[:, 0].min() - padding, all_points[:, 0].max() + padding
    y_min, y_max = all_points[:, 1].min() - padding, all_points[:, 1].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    
    ax.set_title('Voronoi Diagram with Convex Hull', fontsize=16)
    ax.legend()
    ax.set_aspect('equal')
    plt.show()

def main():
    initial_points = np.array([
        [5, 1],    # A1
        [7, -1],   # A2
        [9, -1],   # A3
        [7, 3],    # A4
        [11, 1],   # A5
        [9, 3]     # A6
    ])
    
    try:
        A7, A8 = generate_A7_A8(initial_points)
        all_points = np.vstack([initial_points, A7, A8])
        hull = ConvexHull(all_points)
        plot_voronoi(all_points, hull)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
