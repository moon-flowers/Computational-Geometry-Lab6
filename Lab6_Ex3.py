import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import combinations

class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1

def kruskal_mst(points):
    n = len(points)
    edges = []

    # Generate all pairwise distances as edges (i, j, weight)
    for i, j in combinations(range(n), 2):
        distance = np.linalg.norm(points[i] - points[j])
        edges.append((distance, i, j))
    
    # Sort edges by distance
    edges.sort()

    # Kruskal's MST algorithm
    dsu = DisjointSet(n)
    mst_edges = []
    mst_length = 0

    for distance, u, v in edges:
        if dsu.find(u) != dsu.find(v):
            dsu.union(u, v)
            mst_edges.append((u, v))
            mst_length += distance
        if len(mst_edges) == n - 1:
            break

    return mst_edges, mst_length

def get_points(lambda_val):
    return np.array([
        [1, 6],    # A
        [1, 1],    # B
        [-4, 7],   # C
        [6, 7],    # D
        [1, -1],   # E
        [5, 3],    # F
        [-2, 3],   # P
        [lambda_val - 2, 3]  # Q
    ])

# Animation function
def update(frame):
    lambda_val = lambda_values[frame]
    points = get_points(lambda_val)
    mst_edges, mst_length = kruskal_mst(points)
    
    ax.clear()
    ax.set_facecolor("white")
    ax.set_title(f"MST for λ = {lambda_val:.2f}, Length = {mst_length:.2f}", fontsize=16)
    ax.set_xlim(-10, 10)
    ax.set_ylim(-5, 10)

    # Plot the points
    for i, (x, y) in enumerate(points):
        ax.scatter(x, y, color='blue', s=80)
        ax.text(x + 0.2, y + 0.2, f"({x:.1f}, {y:.1f})", fontsize=10, ha='left', color='black')

    ax.scatter(points[-1, 0], points[-1, 1], color='red', s=100, marker='s', label='Q (varies with λ)')

    # Plot MST edges
    for u, v in mst_edges:
        ax.plot([points[u, 0], points[v, 0]], [points[u, 1], points[v, 1]], 'k-', lw=2)

    ax.legend()

# Set up figure and animation
fig, ax = plt.subplots(figsize=(8, 8))
lambda_values = np.linspace(-10, 10, 200)  # Range of lambda values
ani = FuncAnimation(fig, update, frames=len(lambda_values), interval=30)  # Faster animation (30ms interval)

plt.show()
