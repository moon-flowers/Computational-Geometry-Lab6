import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay, Voronoi, voronoi_plot_2d

def construct_points_example_1():
    """
    Construct the first set of points that ensures exactly 5 internal triangles.
    """
    points = np.array([
        [0, 0],    # P1 (central point)
        [2, -1],   # P2
        [-2, -1],  # P3
        [-1, 3],   # P4
        [1, 3],    # P5
        [0, 5]     # P6
    ])
    return points

def construct_points_example_2():
    """
    Construct the second set of points that ensures exactly 5 internal triangles.
    """
    points = np.array([
        [1, 0],    # Q1 (central point)
        [3, 1],    # Q2
        [-1, 1],   # Q3
        [0, 4],    # Q4
        [2, 4],    # Q5
        [-2, 4]    # Q6
    ])
    return points

def delaunay_triangulation(points):
    """
    Perform Delaunay triangulation on a set of points.
    """
    tri = Delaunay(points)
    return tri

def plot_voronoi_delaunay(points, delaunay, voronoi, title):
    """
    Plot Delaunay triangulation, Voronoi diagram, and specify the number of edges and half-lines.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot Voronoi diagram
    voronoi_plot_2d(voronoi, ax=ax, show_vertices=False, line_colors='gray', line_width=1.5)
    
    # Plot Delaunay triangulation
    ax.triplot(points[:, 0], points[:, 1], delaunay.simplices, color='blue', linewidth=1.5, label="Delaunay Triangulation")
    
    # Plot points
    ax.scatter(points[:, 0], points[:, 1], color='red', s=100, zorder=3, label="Points")

    # Count edges and half-lines
    num_points = len(points)
    num_half_lines = sum(1 for region in voronoi.regions if len(region) > 0 and -1 in region)
    num_edges = 3 * num_points - num_half_lines - 3  # Relation for number of edges

    # Annotate the graph with details
    ax.set_title(f"{title}\nEdges in Triangulation: {num_edges}, Half-line edges: {num_half_lines}")

    # Zoom out to show half-lines
    padding = 10
    x_min, x_max = points[:, 0].min() - padding, points[:, 0].max() + padding
    y_min, y_max = points[:, 1].min() - padding, points[:, 1].max() + padding
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ax.legend()
    plt.show()

def main():
    # First example
    points1 = construct_points_example_1()
    delaunay1 = delaunay_triangulation(points1)
    voronoi1 = Voronoi(points1)
    plot_voronoi_delaunay(points1, delaunay1, voronoi1, "Example 1: Delaunay Triangulation and Voronoi Diagram")

    # Second example
    points2 = construct_points_example_2()
    delaunay2 = delaunay_triangulation(points2)
    voronoi2 = Voronoi(points2)
    plot_voronoi_delaunay(points2, delaunay2, voronoi2, "Example 2: Delaunay Triangulation and Voronoi Diagram")

main()
