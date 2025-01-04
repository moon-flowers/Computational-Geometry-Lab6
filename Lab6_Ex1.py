import math
import matplotlib.pyplot as plt
from itertools import combinations

# -------------------------------------------------------------
# 1) Geometry helpers
# -------------------------------------------------------------

def circle_from_3pts(a, b, c):
    """
    Returns the circumcircle (cx, cy, r) of triangle (a,b,c),
    or None if the points are collinear.
    """
    (x1, y1), (x2, y2), (x3, y3) = a, b, c
    d = 2.0 * (x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))
    if abs(d) < 1e-12:
        return None  # collinear or numerical issue
    
    ux = ((x1**2 + y1**2)*(y2 - y3) +
          (x2**2 + y2**2)*(y3 - y1) +
          (x3**2 + y3**2)*(y1 - y2)) / d
    uy = ((x1**2 + y1**2)*(x3 - x2) +
          (x2**2 + y2**2)*(x1 - x3) +
          (x3**2 + y3**2)*(x2 - x1)) / d
    r = math.hypot(ux - x1, uy - y1)
    return (ux, uy, r)

def in_circle(pt, circle):
    """Check if point pt=(x,y) lies inside/on circle (cx,cy,r)."""
    if circle is None:
        return False
    cx, cy, r = circle
    x, y = pt
    return (x - cx)**2 + (y - cy)**2 <= r**2 + 1e-12

def triangle_edges(tri):
    """Return the 3 edges of triangle tri as frozensets of points."""
    A, B, C = tri
    return {
        frozenset([A,B]),
        frozenset([B,C]),
        frozenset([C,A])
    }

# -------------------------------------------------------------
# 2) Naive Delaunay
# -------------------------------------------------------------

def delaunay(points):
    """
    For each combination of 3 points, compute its circumcircle
    and check if no other point lies inside. If so, keep the triangle.
    Returns list of triangles as ((x1,y1),(x2,y2),(x3,y3)).
    """
    triangles = []
    for combo in combinations(points, 3):
        c = circle_from_3pts(*combo)
        if c is None:  
            # collinear or degenerate
            continue
        # check if any other point is inside
        if not any(in_circle(p, c) for p in points if p not in combo):
            triangles.append(combo)
    return triangles

# -------------------------------------------------------------
# 3) Voronoi edges
# -------------------------------------------------------------

def get_voronoi_finite_edges(triangles):
    """
    Connect circumcenters of adjacent triangles (sharing an edge).
    Returns list of line segments [(p1, p2), ...].
    """
    # compute circumcenters
    centers = []
    for tri in triangles:
        cx, cy, _ = circle_from_3pts(*tri)
        centers.append((cx, cy))
    
    # map from edge -> list of indices of triangles that share it
    edge_map = {}
    for i, tri in enumerate(triangles):
        for e in triangle_edges(tri):
            edge_map.setdefault(e, []).append(i)
    
    # if exactly 2 triangles share edge => connect their circumcenters
    vor_edges = []
    for e, tri_inds in edge_map.items():
        if len(tri_inds) == 2:
            i1, i2 = tri_inds
            p1 = centers[i1]
            p2 = centers[i2]
            vor_edges.append((p1, p2))
    return vor_edges

# -------------------------------------------------------------
# 4) "Infinite" Voronoi edges on the hull
# -------------------------------------------------------------

def get_voronoi_infinite_rays(triangles, points, bbox=None):
    """
    For each hull edge (i.e., an edge belonging to exactly 1 triangle),
    draw a ray from that triangle's circumcenter outward.
    We intersect it with a bounding box to visualize.
    """
    # bounding box to which we clip infinite edges
    # (minX, minY, maxX, maxY)
    if bbox is None:
        margin = 2
        xs, ys = zip(*points)
        minx, maxx = min(xs)-margin, max(xs)+margin
        miny, maxy = min(ys)-margin, max(ys)+margin
        bbox = (minx, miny, maxx, maxy)
    
    # compute circle for each tri
    centers = []
    for tri in triangles:
        centers.append(circle_from_3pts(*tri))  # (cx, cy, r)
    
    # build edge->list of triangles
    edge_map = {}
    for i, tri in enumerate(triangles):
        for e in triangle_edges(tri):
            edge_map.setdefault(e, []).append(i)
    
    rays = []
    for e, tri_inds in edge_map.items():
        if len(tri_inds) == 1:
            # hull edge => one triangle
            i_tri = tri_inds[0]
            cx, cy, _ = centers[i_tri]
            A, B = list(e)
            
            # midpoint
            mx = 0.5*(A[0] + B[0])
            my = 0.5*(A[1] + B[1])
            
            # direction is from circumcenter away from midpoint
            vx = mx - cx
            vy = my - cy
            # but we want "outward" direction. 
            # Actually, we want from (cx,cy) out to "far away" in that direction.
            # We'll just param in the same direction:
            #   param:  X(t) = cx + t*vx,  Y(t) = cy + t*vy, t >= 0
            
            # pick a large t, then clip to bounding box
            big_t = 1e6
            x1 = cx + vx*big_t
            y1 = cy + vy*big_t
            
            xA, yA = clip_segment_to_box(cx, cy, x1, y1, *bbox)
            rays.append(((cx, cy), (xA, yA)))
    return rays

def clip_segment_to_box(x0, y0, x1, y1, minx, miny, maxx, maxy):
    """
    Clips segment (x0,y0)->(x1,y1) to bounding box.
    Returns intersection with the box boundary from (x0,y0) outward.
    """
    dx = x1 - x0
    dy = y1 - y0
    # param eqn: X(t) = x0 + t*dx, Y(t)=y0 + t*dy
    # find smallest t that hits a boundary
    t_candidates = []
    
    if abs(dx) > 1e-12:
        t_candidates += [(minx - x0)/dx, (maxx - x0)/dx]
    if abs(dy) > 1e-12:
        t_candidates += [(miny - y0)/dy, (maxy - y0)/dy]
    
    # We want the smallest positive t
    t_candidates = [t for t in t_candidates if t >= 0]
    if not t_candidates:
        return (x0, y0)  # no valid intersection if all negative
    t_min = min(t_candidates)
    
    xx = x0 + t_min*dx
    yy = y0 + t_min*dy
    # clamp to the box in case of numerical fuzz
    xx = max(minx, min(xx, maxx))
    yy = max(miny, min(yy, maxy))
    return (xx, yy)

# -------------------------------------------------------------
# 5) Main
# -------------------------------------------------------------
if __name__ == "__main__":
    # Points
    A = (3, 5)
    B = (6, 6)
    C = (6, 4)
    D = (9, 5)
    E = (9, 7)
    points = [A, B, C, D, E]
    
    # 5.1) Delaunay
    triangles = delaunay(points)
    
    # 5.2) Voronoi finite edges
    vor_finite = get_voronoi_finite_edges(triangles)
    
    # 5.3) Voronoi infinite edges (rays) for hull edges
    # We'll pick a bounding box that roughly matches your figure
    bounding_box = (2.5, 3.5, 9.5, 7.5)
    vor_infinite = get_voronoi_infinite_rays(triangles, points, bbox=bounding_box)
    
    # ---------------------------------------------------------
    # Plot everything
    # ---------------------------------------------------------
    fig, ax = plt.subplots()
    
    # Plot points (in red) and label them
    for (x, y) in points:
        ax.plot(x, y, 'o', color='red')
    ax.text(A[0], A[1], "A", fontsize=10, ha='left', va='top')
    ax.text(B[0], B[1], "B", fontsize=10, ha='left', va='bottom')
    ax.text(C[0], C[1], "C", fontsize=10, ha='left', va='top')
    ax.text(D[0], D[1], "D", fontsize=10, ha='left', va='top')
    ax.text(E[0], E[1], "E", fontsize=10, ha='left', va='bottom')
    
    # Plot Delaunay triangles (blue)
    for tri in triangles:
        cyc = list(tri) + [tri[0]]
        ax.plot([p[0] for p in cyc], [p[1] for p in cyc], color='blue')
    
    # Plot Voronoi finite edges (dashed gray)
    for (p1, p2) in vor_finite:
        x1, y1 = p1
        x2, y2 = p2
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--')
    
    # Plot Voronoi infinite rays (dashed gray)
    for (p1, p2) in vor_infinite:
        x1, y1 = p1
        x2, y2 = p2
        ax.plot([x1, x2], [y1, y2], color='gray', linestyle='--')
    
    # Set up the axes/limits and title
    ax.set_aspect('equal')
    ax.set_xlim(bounding_box[0], bounding_box[2])
    ax.set_ylim(bounding_box[1], bounding_box[3])
    ax.set_title("Delaunay Triangulation and Voronoi Diagram")
    
    plt.show()
