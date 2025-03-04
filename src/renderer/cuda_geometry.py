# renderer/cuda_geometry.py

from numba import cuda, float32, int32
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
    This version handles two-sided triangles by not requiring the ray to hit from the front.
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

    # For two-sided triangles, we allow intersection from both sides
    if abs(a) < EPSILON:  # Ray is parallel to triangle, no intersection
        return -1.0

    f = 1.0 / a
    for i in range(3):
        s[i] = ray_origin[i] - v0[i]
    
    u = f * dot(s, h)
    if u < 0.0 or u > 1.0:  # u outside triangle
        return -1.0

    cross_inplace(q, s, edge1)
    v = f * dot(ray_dir, q)
    if v < 0.0 or (u + v) > 1.0:  # v outside triangle or u+v > 1
        return -1.0

    t = f * dot(edge2, q)
    if t < t_min or t > t_max:  # Intersection too close or too far
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

@cuda.jit(device=True)
def aabb_hit(ray_origin, ray_dir, box_min, box_max, t_min, t_max):
    for i in range(3):
        invD = 1.0 / ray_dir[i]
        t0 = (box_min[i] - ray_origin[i]) * invD
        t1 = (box_max[i] - ray_origin[i]) * invD
        if invD < 0.0:
            temp = t0
            t0 = t1
            t1 = temp
        t_min = t0 if t0 > t_min else t_min
        t_max = t1 if t1 < t_max else t_max
        if t_max <= t_min:
            return False
    return True

@cuda.jit(device=True)
def gpu_bvh_traverse(ray_origin, ray_dir,
                     bbox_min, bbox_max,
                     left_indices, right_indices,
                     is_leaf, object_indices, num_nodes,
                     t_min, t_max,
                     sphere_centers, sphere_radii,
                     triangle_vertices, uv_temp):
    # Use a fixed-size local stack
    stack = cuda.local.array(64, dtype=int32)
    stack_ptr = 0
    hit_object = -1
    hit_t = t_max
    stack[stack_ptr] = 0  # start at root
    stack_ptr += 1
    
    # Pre-calculate ray direction inverse and signs for faster AABB tests
    ray_dir_inv = cuda.local.array(3, dtype=float32)
    ray_dir_sign = cuda.local.array(3, dtype=int32)
    
    for i in range(3):
        if abs(ray_dir[i]) > EPSILON:
            ray_dir_inv[i] = 1.0 / ray_dir[i]
        else:
            ray_dir_inv[i] = 1.0 / EPSILON
        ray_dir_sign[i] = 1 if ray_dir[i] < 0.0 else 0

    # Pre-check how many spheres we have for faster type checking
    sphere_count = sphere_centers.shape[0]

    while stack_ptr > 0:
        stack_ptr -= 1
        node_idx = stack[stack_ptr]
        
        # Skip if box doesn't intersect - use optimized AABB test
        # We don't call the function to avoid the overhead
        tmin = 0.0
        tmax = hit_t
        
        for i in range(3):
            box_min_val = bbox_min[node_idx][i]
            box_max_val = bbox_max[node_idx][i]
            
            if ray_dir_sign[i]:
                t1 = (box_max_val - ray_origin[i]) * ray_dir_inv[i]
                t2 = (box_min_val - ray_origin[i]) * ray_dir_inv[i]
            else:
                t1 = (box_min_val - ray_origin[i]) * ray_dir_inv[i]
                t2 = (box_max_val - ray_origin[i]) * ray_dir_inv[i]
                
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            
            if tmax <= tmin:
                # Box doesn't intersect
                continue
            
        if is_leaf[node_idx] == 1:
            # For leaf nodes, get the object index and check the specific primitive type
            obj_idx = object_indices[node_idx]
            if obj_idx >= 0:  # Valid object
                if obj_idx < sphere_count:
                    # It's a sphere
                    # Direct access to sphere data to avoid indirection
                    center = sphere_centers[obj_idx]
                    radius = sphere_radii[obj_idx]
                    
                    # Inline sphere intersection test for better performance
                    oc = cuda.local.array(3, dtype=float32)
                    for i in range(3):
                        oc[i] = ray_origin[i] - center[i]
                    
                    a = dot(ray_dir, ray_dir)
                    half_b = dot(oc, ray_dir)
                    c = dot(oc, oc) - radius * radius
                    discriminant = half_b * half_b - a * c
                    
                    if discriminant >= 0:
                        sqrtd = math.sqrt(discriminant)
                        root = (-half_b - sqrtd) / a
                        
                        if root < t_min or root > hit_t:
                            root = (-half_b + sqrtd) / a
                            
                        if root >= t_min and root < hit_t:
                            hit_t = root
                            hit_object = obj_idx
                else:
                    # It's a triangle
                    tri_idx = obj_idx - sphere_count
                    t = ray_triangle_intersect(ray_origin, ray_dir,
                          triangle_vertices[tri_idx, 0:3],
                          triangle_vertices[tri_idx, 3:6],
                          triangle_vertices[tri_idx, 6:9],
                          t_min, hit_t, uv_temp)
                    if t > 0.0 and t < hit_t:
                        hit_t = t
                        hit_object = obj_idx
        else:
            # For internal nodes, push both children onto the stack
            left_idx = left_indices[node_idx]
            right_idx = right_indices[node_idx]
            
            # Calculate distances using the midpoints of the child boxes
            left_mid = cuda.local.array(3, dtype=float32)
            right_mid = cuda.local.array(3, dtype=float32)
            
            # Faster approximation of which node is closer using dot product with ray direction
            for i in range(3):
                left_mid[i] = (bbox_min[left_idx][i] + bbox_max[left_idx][i]) * 0.5
                right_mid[i] = (bbox_min[right_idx][i] + bbox_max[right_idx][i]) * 0.5
                left_mid[i] -= ray_origin[i]
                right_mid[i] -= ray_origin[i]
            
            left_dist = dot(left_mid, ray_dir)
            right_dist = dot(right_mid, ray_dir)
            
            # Push the further node first (so closer one is processed first)
            if left_dist > right_dist:
                stack[stack_ptr] = left_idx
                stack_ptr += 1
                stack[stack_ptr] = right_idx
                stack_ptr += 1
            else:
                stack[stack_ptr] = right_idx
                stack_ptr += 1
                stack[stack_ptr] = left_idx
                stack_ptr += 1
                
    return hit_object, hit_t

