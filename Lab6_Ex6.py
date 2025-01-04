import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay
from matplotlib.animation import FuncAnimation

def get_points(lambda_val):
    """
    Get the set of points {A, B, C, D, E, M} with M = (1, λ).
    """
    points = np.array([
        [1, -1],   # A
        [-1, 1],   # B
        [2, -1],   # C
        [1, 1],    # D
        [0, 2],    # E
        [1, lambda_val]  # M
    ])
    return points

def delaunay_triangulation(points):
    """
    Perform Delaunay triangulation and return the simplices (triangles).
    """
    return Delaunay(points)

def calculate_num_edges_and_hull(triangulation):
    """
    Calculate the number of edges and the number of points on the convex hull.
    """
    n = len(triangulation.points)  # Number of points
    hull = triangulation.convex_hull  # Convex hull simplices
    h = len(np.unique(hull))  # Number of unique points on the convex hull (h)
    num_edges = 3 * n - h - 3  # Calculate number of edges using e = 3n - h - 3
    return num_edges, h

def update(frame):
    """
    Update function for the animation.
    """
    lambda_val = lambda_values[frame]
    points = get_points(lambda_val)
    delaunay = delaunay_triangulation(points)
    num_edges, h = calculate_num_edges_and_hull(delaunay)
    
    ax.clear()
    ax.set_facecolor("white")
    ax.set_title(f"λ = {lambda_val:.2f}, Edges: {num_edges}", fontsize=16)
    
    # Zoomed-out view to ensure the moving points remain visible
    ax.set_xlim(-2, 3)
    ax.set_ylim(-3, 10)

    # Plot Delaunay triangulation
    ax.triplot(points[:, 0], points[:, 1], delaunay.simplices, color='blue', linewidth=1.5)
    
    # Plot points
    for (x, y) in points:
        ax.scatter(x, y, color='red', s=80)
        ax.text(x + 0.1, y + 0.1, f"({x:.1f}, {y:.1f})", fontsize=10, ha='left', color='black')

# Set up the animation
fig, ax = plt.subplots(figsize=(8, 8))
lambda_values = np.linspace(-2, 8, 200)  # Expanded range of lambda values for M = (1, λ)
ani = FuncAnimation(fig, update, frames=len(lambda_values), interval=50)  # Faster animation (50ms interval)

plt.show()
