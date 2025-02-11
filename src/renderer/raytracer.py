# renderer/raytracer.py
import numpy as np
from numba import cuda, float32, int32
import math
from core.vector import Vector3
from core.ray import Ray
from materials.metal import Metal

# CUDA device constants
INFINITY = float32(1e20)
EPSILON = float32(0.001)
MAX_BOUNCES = 3

@cuda.jit(device=True)
def normalize_inplace(v):
    """Normalize a vector on the GPU in-place."""
    length_sq = v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    if length_sq > 0.0:
        length = math.sqrt(length_sq)
        v[0] /= length
        v[1] /= length
        v[2] /= length

@cuda.jit(device=True)
def dot(v1, v2):
    """Compute dot product on the GPU."""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@cuda.jit(device=True)
def cross_inplace(out, v1, v2):
    """Compute cross product on the GPU, storing result in out."""
    temp0 = v1[1] * v2[2] - v1[2] * v2[1]
    temp1 = v1[2] * v2[0] - v1[0] * v2[2]
    temp2 = v1[0] * v2[1] - v1[1] * v2[0]
    out[0] = temp0
    out[1] = temp1
    out[2] = temp2

@cuda.jit(device=True)
def reflect(out, v, n):
    """Reflect vector v about normal n, storing result in out."""
    d = dot(v, n)
    for i in range(3):
        out[i] = v[i] - 2.0 * d * n[i]
    return out

@cuda.jit(device=True)
def get_random_dir(seed, out):
    """Generate a random direction vector into the output array."""
    x = math.sin(seed * 12.9898) * 43758.5453
    y = math.sin(seed * 78.233) * 43758.5453
    z = math.sin(seed * 37.719) * 43758.5453

    out[0] = math.fmod(abs(x), 2.0) - 1.0
    out[1] = math.fmod(abs(y), 2.0) - 1.0
    out[2] = math.fmod(abs(z), 2.0) - 1.0

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
def ray_triangle_intersect(ray_origin, ray_dir, v0, v1, v2, t_min, t_max):
    """Ray-triangle intersection using Möller–Trumbore algorithm."""
    edge1 = cuda.local.array(3, dtype=float32)
    edge2 = cuda.local.array(3, dtype=float32)
    h = cuda.local.array(3, dtype=float32)
    s = cuda.local.array(3, dtype=float32)
    q = cuda.local.array(3, dtype=float32)
    
    # When v0, v1, v2 are passed in, they're already 3-element arrays
    for i in range(3):
        edge1[i] = v1[i] - v0[i]
        edge2[i] = v2[i] - v0[i]
    
    # Compute cross product of ray_dir and edge2
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
    if v < 0.0 or u + v > 1.0:
        return -1.0
        
    t = f * dot(edge2, q)
    if t < t_min or t > t_max:
        return -1.0
        
    return t

@cuda.jit(device=True)
def trace_ray(origin, direction, max_bounces,
              sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
              triangle_vertices, triangle_materials, triangle_material_types,
              out_color, seed):
    # Initialize arrays
    current_origin = cuda.local.array(3, dtype=float32)
    current_direction = cuda.local.array(3, dtype=float32)
    color = cuda.local.array(3, dtype=float32)
    attenuation = cuda.local.array(3, dtype=float32)
    
    # Copy input arrays
    for i in range(3):
        current_origin[i] = origin[i]
        current_direction[i] = direction[i]
        color[i] = 0.0
        attenuation[i] = 1.0

    # Set current ray parameters.
    current_origin = cuda.local.array(3, dtype=float32)
    current_direction = cuda.local.array(3, dtype=float32)
    for i in range(3):
        current_origin[i] = origin[i]
        current_direction[i] = direction[i]

    # Iterative loop for ray bounces.
    for bounce in range(max_bounces):
        closest_t = 1e20  # A large number for the nearest intersection.
        hit_is_sphere = True
        hit_idx = -1

        # Test intersection with spheres.
        for i in range(sphere_centers.shape[0]):
            t = ray_sphere_intersect(current_origin, current_direction,
                                     sphere_centers[i], sphere_radii[i], EPSILON, closest_t)
            if t > 0.0:
                closest_t = t
                hit_is_sphere = True
                hit_idx = i

        # Test intersection with triangles
        for i in range(triangle_vertices.shape[0]):
            # Create local arrays for vertices
            v0 = cuda.local.array(3, dtype=float32)
            v1 = cuda.local.array(3, dtype=float32)
            v2 = cuda.local.array(3, dtype=float32)
            
            # Copy vertex data
            for j in range(3):
                v0[j] = triangle_vertices[i][j]
                v1[j] = triangle_vertices[i][3 + j]
                v2[j] = triangle_vertices[i][6 + j]

        # If no hit is found, add background color and break.
        if hit_idx < 0:
            # Compute background gradient color.
            t_bg = 0.5 * (current_direction[1] + 1.0)
            for i in range(3):
                color[i] += attenuation[i] * ((1.0 - t_bg) * 1.0 + t_bg * (1.0, 0.7, 0.5)[i])
            break

        # Compute the hit point.
        hit_point = cuda.local.array(3, dtype=float32)
        for i in range(3):
            hit_point[i] = current_origin[i] + closest_t * current_direction[i]

        # Compute the normal at the hit.
        normal = cuda.local.array(3, dtype=float32)
        if hit_is_sphere:
            for i in range(3):
                normal[i] = (hit_point[i] - sphere_centers[hit_idx][i]) / sphere_radii[hit_idx]
            normalize_inplace(normal)
            # Retrieve material information for the sphere.
            material_type = sphere_material_types[hit_idx]
            mat_idx = hit_idx * 3
            material_color = cuda.local.array(3, dtype=float32)
            for i in range(3):
                material_color[i] = sphere_materials[mat_idx + i]
        else:
            # Compute the base index for this triangle.
            base = hit_idx * 9

            # Load v0, v1, v2 into local arrays.
            v0 = cuda.local.array(3, dtype=float32)
            v1 = cuda.local.array(3, dtype=float32)
            v2 = cuda.local.array(3, dtype=float32)
            for i in range(3):
                v0[i] = triangle_vertices[base + i]
                v1[i] = triangle_vertices[base + 3 + i]
                v2[i] = triangle_vertices[base + 6 + i]

            # Now compute the edge vectors.
            v1_minus_v0 = cuda.local.array(3, dtype=float32)
            v2_minus_v0 = cuda.local.array(3, dtype=float32)
            for i in range(3):
                v1_minus_v0[i] = v1[i] - v0[i]
                v2_minus_v0[i] = v2[i] - v0[i]

            # Compute the normal, etc.
            cross_inplace(normal, v1_minus_v0, v2_minus_v0)
            normalize_inplace(normal)
            material_type = triangle_material_types[hit_idx]
            mat_idx = hit_idx * 3
            material_color = cuda.local.array(3, dtype=float32)
            for i in range(3):
                material_color[i] = triangle_materials[mat_idx + i]

        # Example: handle a simple metal material.
        if material_type == 1:  # Metal
            reflected = cuda.local.array(3, dtype=float32)
            random_dir = cuda.local.array(3, dtype=float32)
            
            # Initialize reflected direction
            for i in range(3):
                reflected[i] = 0.0
            
            # Compute reflection
            reflect(reflected, current_direction, normal)
            
            # Get random direction and add fuzz
            get_random_dir(seed + bounce, random_dir)
            normalize_inplace(random_dir)
            
            # Apply fuzz
            for i in range(3):
                reflected[i] = reflected[i] + random_dir[i] * float32(0.1)
            normalize_inplace(reflected)
            
            # Update attenuation and setup new ray
            for i in range(3):
                attenuation[i] *= material_color[i]
                current_origin[i] = hit_point[i]
                current_direction[i] = reflected[i]
        else:
            # For diffuse materials, add the diffuse contribution and terminate.
            diffuse = max(0.1, dot(normal, [0.5, 0.7, -0.5]))  # Example diffuse term.
            for i in range(3):
                color[i] += attenuation[i] * material_color[i] * diffuse
            break

    # Write the final accumulated color to out_color.
    for i in range(3):
        out_color[i] = color[i]

@cuda.jit
def render_kernel(width, height,
                 camera_pos, camera_forward, camera_right, camera_up,
                 sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
                 triangle_vertices, triangle_materials, triangle_material_types,
                 out_image, frame_number):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return
    
    # Initialize arrays with explicit indices
    ray_dir = cuda.local.array(3, dtype=float32)
    ray_origin = cuda.local.array(3, dtype=float32)
    color = cuda.local.array(3, dtype=float32)
    
    # Initialize ray origin (camera position)
    for i in range(3):
        ray_origin[i] = camera_pos[i]
    
    # Compute ray direction
    u = float32(x) / float32(width - 1)
    v = float32(height - 1 - y) / float32(height - 1)
    
    for i in range(3):
        ray_dir[i] = camera_forward[i] + (u - 0.5) * camera_right[i] + (v - 0.5) * camera_up[i]
    
    # Normalize ray direction
    normalize_inplace(ray_dir)
    
    # Generate a seed for random number generation
    seed = x + y * width + frame_number * width * height
    
    # Trace the ray and get the color
    trace_ray(ray_origin, ray_dir, MAX_BOUNCES,
             sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
             triangle_vertices, triangle_materials, triangle_material_types,
             color, seed)
    
    # Write the color to the output image (converting from float [0,1] to uint8 [0,255])
    for i in range(3):
        out_image[x, y, i] = min(255, max(0, int(color[i] * 255)))

class Renderer:
    def __init__(self, width: int, height: int, max_depth: int = 3):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.frame_number = 0
        
        # CUDA configuration
        self.threadsperblock = (16, 16)
        self.blockspergrid_x = math.ceil(width / self.threadsperblock[0])
        self.blockspergrid_y = math.ceil(height / self.threadsperblock[1])
        self.blockspergrid = (self.blockspergrid_x, self.blockspergrid_y)
        
        # Initialize device arrays
        self.d_sphere_centers = None
        self.d_sphere_radii = None
        self.d_sphere_materials = None
        self.d_sphere_material_types = None
        self.d_triangle_vertices = None
        self.d_triangle_materials = None
        self.d_triangle_material_types = None
        
    def update_scene_data(self, world):
        """Convert scene data to GPU-friendly format with material types."""
        # Add debug prints
        sphere_count = len([obj for obj in world.objects if hasattr(obj, 'radius')])
        mesh_objects = [obj for obj in world.objects if hasattr(obj, 'triangles')]
        triangle_count = sum(len(obj.triangles) for obj in mesh_objects)
        
        print(f"Uploading scene data:")
        print(f"- {sphere_count} spheres")
        print(f"- {triangle_count} triangles")
        # Clean up old arrays
        self.cleanup()
        
        # Prepare sphere arrays (ensure at least one element)
        centers = np.zeros((max(1, sphere_count), 3), dtype=np.float32)
        radii = np.zeros(max(1, sphere_count), dtype=np.float32)
        sphere_materials = np.zeros(max(1, sphere_count) * 3, dtype=np.float32)
        sphere_material_types = np.zeros(max(1, sphere_count), dtype=np.int32)
        
        # Fill sphere arrays
        sphere_idx = 0
        for obj in world.objects:
            if hasattr(obj, 'radius'):
                centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
                radii[sphere_idx] = obj.radius
                
                if hasattr(obj.material, 'albedo'):
                    mat_idx = sphere_idx * 3
                    sphere_materials[mat_idx:mat_idx+3] = [
                        obj.material.albedo.x,
                        obj.material.albedo.y,
                        obj.material.albedo.z
                    ]
                    # Set material type (0 = diffuse, 1 = metal)
                    sphere_material_types[sphere_idx] = 1 if isinstance(obj.material, Metal) else 0
                sphere_idx += 1
        
        # Prepare triangle arrays (ensure at least one element)
        triangle_vertices = np.zeros((max(1, triangle_count), 9), dtype=np.float32)
        triangle_materials = np.zeros(max(1, triangle_count) * 3, dtype=np.float32)
        triangle_material_types = np.zeros(max(1, triangle_count), dtype=np.int32)
        
        # Fill triangle arrays
        triangle_idx = 0
        for mesh in mesh_objects:
            for triangle in mesh.triangles:
                # Store vertices sequentially: v0 (3), v1 (3), v2 (3)
                triangle_vertices[triangle_idx, 0:3] = [
                    triangle.v0.x, triangle.v0.y, triangle.v0.z
                ]
                triangle_vertices[triangle_idx, 3:6] = [
                    triangle.v1.x, triangle.v1.y, triangle.v1.z
                ]
                triangle_vertices[triangle_idx, 6:9] = [
                    triangle.v2.x, triangle.v2.y, triangle.v2.z
                ]
                
                # Store material color and type
                if hasattr(mesh.material, 'albedo'):
                    mat_idx = triangle_idx * 3
                    triangle_materials[mat_idx:mat_idx+3] = [
                        mesh.material.albedo.x,
                        mesh.material.albedo.y,
                        mesh.material.albedo.z
                    ]
                    triangle_material_types[triangle_idx] = 1 if isinstance(mesh.material, Metal) else 0
                triangle_idx += 1
        
        # Ensure arrays are contiguous and transfer to GPU
        self.d_sphere_centers = cuda.to_device(np.ascontiguousarray(centers))
        self.d_sphere_radii = cuda.to_device(np.ascontiguousarray(radii))
        self.d_sphere_materials = cuda.to_device(np.ascontiguousarray(sphere_materials))
        self.d_sphere_material_types = cuda.to_device(np.ascontiguousarray(sphere_material_types))
        self.d_triangle_vertices = cuda.to_device(np.ascontiguousarray(triangle_vertices))
        self.d_triangle_materials = cuda.to_device(np.ascontiguousarray(triangle_materials))
        self.d_triangle_material_types = cuda.to_device(np.ascontiguousarray(triangle_material_types))
    
    def render_frame(self, camera, world) -> np.ndarray:
        """Render a frame using CUDA acceleration."""
        # Update scene data if needed
        if self.d_sphere_centers is None:
            self.update_scene_data(world)
        
        # Prepare camera data
        camera_pos = np.ascontiguousarray(
            np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
        )
        camera_forward = np.ascontiguousarray(
            np.array([camera.forward.x, camera.forward.y, camera.forward.z], dtype=np.float32)
        )
        camera_right = np.ascontiguousarray(
            np.array([camera.right.x, camera.right.y, camera.right.z], dtype=np.float32)
        )
        camera_up = np.ascontiguousarray(
            np.array([camera.up.x, camera.up.y, camera.up.z], dtype=np.float32)
        )
        
        # Create output array
        output = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        d_output = cuda.to_device(output)
        
        # Launch kernel
        render_kernel[self.blockspergrid, self.threadsperblock](
            self.width, self.height,
            camera_pos, camera_forward, camera_right, camera_up,
            self.d_sphere_centers, self.d_sphere_radii, self.d_sphere_materials, self.d_sphere_material_types,
            self.d_triangle_vertices, self.d_triangle_materials, self.d_triangle_material_types,
            d_output, self.frame_number
        )
        
        # Increment frame number (used for random number generation)
        self.frame_number += 1
        
        # Get result
        output = d_output.copy_to_host()
        return output
    
    def cleanup(self):
        """Clean up CUDA memory."""
        try:
            if self.d_sphere_centers is not None:
                del self.d_sphere_centers
                del self.d_sphere_radii
                del self.d_sphere_materials
                del self.d_sphere_material_types
                del self.d_triangle_vertices
                del self.d_triangle_materials
                del self.d_triangle_material_types
            
            self.d_sphere_centers = None
            self.d_sphere_radii = None
            self.d_sphere_materials = None
            self.d_sphere_material_types = None
            self.d_triangle_vertices = None
            self.d_triangle_materials = None
            self.d_triangle_material_types = None
        except Exception as e:
            print(f"Warning: Error during CUDA cleanup: {str(e)}")