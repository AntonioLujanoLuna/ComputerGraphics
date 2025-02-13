# renderer/cuda_geometry.py

from numba import cuda, float32
from .cuda_utils import dot, cross_inplace, normalize_inplace
import math

INFINITY = 1e20
EPSILON = 1e-8

@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius, t_min, t_max):
    """Ray-sphere intersection test on the GPU."""
    oc = cuda.local.array(3, dtype=float32)
    for i in range(3):
        oc[i] = ray_origin[i] - sphere_center[i]
    
    a = dot(ray_dir, ray_dir)
    half_b = dot(oc, ray_dir)
    c = dot(oc, oc) - sphere_radius * sphere_radius
    discriminant = half_b * half_b - a * c
    
    if discriminant < 0:
        return -1.0
        
    sqrtd = math.sqrt(discriminant)
    root = (-half_b - sqrtd) / a
    
    if root < t_min or root > t_max:
        root = (-half_b + sqrtd) / a
        if root < t_min or root > t_max:
            return -1.0
            
    return root

@cuda.jit(device=True)
def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, t_min, t_max, out_uv):
    """
    Ray-triangle intersection using the Möller–Trumbore algorithm.
    If an intersection occurs, stores the barycentric coordinates (u,v) in out_uv[0] and out_uv[1],
    and returns the intersection distance t. Otherwise, returns -1.0.
    """
    edge1 = cuda.local.array(3, dtype=float32)
    edge2 = cuda.local.array(3, dtype=float32)
    h = cuda.local.array(3, dtype=float32)
    s = cuda.local.array(3, dtype=float32)
    q = cuda.local.array(3, dtype=float32)

    for i in range(3):
        edge1[i] = v1[i] - v0[i]
        edge2[i] = v2[i] - v0[i]

    cross_inplace(h, ray_dir, edge2)
    a = dot(edge1, h)

    if abs(a) < EPSILON:
        return -1.0

    f = 1.0 / a
    for i in range(3):
        s[i] = ray_origin[i] - v0[i]
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:
        return -1.0

    cross_inplace(q, s, edge1)
    v = f * dot(ray_dir, q)
    if v < 0.0 or (u + v) > 1.0:
        return -1.0

    t = f * dot(edge2, q)
    if t < t_min or t > t_max:
        return -1.0

    out_uv[0] = u
    out_uv[1] = v
    return t

@cuda.jit(device=True)
def compute_triangle_normal(triangle, out_normal):
    """
    Compute the normal for a triangle.
    
    Parameters:
      triangle: 1D array of 9 floats, representing [v0x, v0y, v0z, v1x, v1y, v1z, v2x, v2y, v2z].
      out_normal: 1D array of 3 floats in which the normalized normal will be stored.
    """
    # Allocate temporary arrays for edge computations.
    edge1 = cuda.local.array(3, dtype=float32)
    edge2 = cuda.local.array(3, dtype=float32)
    
    # Compute edge1 = v1 - v0 and edge2 = v2 - v0.
    for i in range(3):
        edge1[i] = triangle[i + 3] - triangle[i]
        edge2[i] = triangle[i + 6] - triangle[i]
    
    # Compute the cross product: out_normal = edge1 x edge2.
    cross_inplace(out_normal, edge1, edge2)
    
    # Normalize the resulting normal.
    normalize_inplace(out_normal)

@cuda.jit(device=True)
def calculate_sphere_uv(hit_point, center, out_uv):
    """Calculate UV coordinates for a point on a sphere."""
    # Convert hit point to sphere-local coordinates
    local_point = cuda.local.array(3, dtype=float32)
    for i in range(3):
        local_point[i] = hit_point[i] - center[i]
    
    # Normalize the point
    length = math.sqrt(dot(local_point, local_point))
    for i in range(3):
        local_point[i] /= length
    
    # Calculate spherical coordinates
    phi = math.atan2(local_point[2], local_point[0])
    theta = math.asin(local_point[1])
    
    # Convert to UV coordinates
    out_uv[0] = 1.0 - (phi + math.pi) / (2.0 * math.pi)
    out_uv[1] = (theta + math.pi/2.0) / math.pi

@cuda.jit(device=True)
def interpolate_uv(uv0, uv1, uv2, u: float, v: float, out_uv):
    """Interpolate UV coordinates using barycentric coordinates."""
    w = 1.0 - u - v
    out_uv[0] = w * uv0[0] + u * uv1[0] + v * uv2[0]
    out_uv[1] = w * uv0[1] + u * uv1[1] + v * uv2[1]
