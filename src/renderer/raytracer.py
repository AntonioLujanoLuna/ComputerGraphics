# renderer/raytracer.py
import numpy as np
from numba import cuda, float32, int32
import math
from core.vector import Vector3
from core.ray import Ray

# CUDA device constants
INFINITY = float32(1e20)
EPSILON = float32(0.001)

@cuda.jit(device=True)
def normalize(vector):
    """Normalize a vector on the GPU."""
    length = math.sqrt(vector[0]**2 + vector[1]**2 + vector[2]**2)
    if length > 0:
        vector[0] /= length
        vector[1] /= length
        vector[2] /= length
    return vector

@cuda.jit(device=True)
def dot(v1, v2):
    """Compute dot product on the GPU."""
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2]

@cuda.jit(device=True)
def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius, t_min, t_max):
    """Ray-sphere intersection test on the GPU."""
    oc = cuda.local.array(3, dtype=float32)
    oc[0] = ray_origin[0] - sphere_center[0]
    oc[1] = ray_origin[1] - sphere_center[1]
    oc[2] = ray_origin[2] - sphere_center[2]
    
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

@cuda.jit
def render_kernel(width, height, camera_pos, camera_forward, camera_right, camera_up,
                 sphere_centers, sphere_radii, sphere_materials, out_image):
    """Main rendering kernel."""
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return
        
    # Compute ray direction
    u = x / (width - 1)
    v = (height - 1 - y) / (height - 1)
    
    ray_dir = cuda.local.array(3, dtype=float32)
    ray_dir[0] = camera_forward[0] + (u - 0.5) * camera_right[0] + (v - 0.5) * camera_up[0]
    ray_dir[1] = camera_forward[1] + (u - 0.5) * camera_right[1] + (v - 0.5) * camera_up[1]
    ray_dir[2] = camera_forward[2] + (u - 0.5) * camera_right[2] + (v - 0.5) * camera_up[2]
    normalize(ray_dir)
    
    # Initialize color
    color = cuda.local.array(3, dtype=float32)
    color[0] = color[1] = color[2] = 0.0
    
    # Ray trace
    closest_t = INFINITY
    hit_anything = False
    
    # Test intersection with all spheres
    for i in range(sphere_centers.shape[0]):
        t = ray_sphere_intersect(camera_pos, ray_dir, sphere_centers[i], sphere_radii[i], 
                               EPSILON, closest_t)
        if t > 0:
            hit_anything = True
            closest_t = t
            
            # Simple diffuse shading
            hit_point = cuda.local.array(3, dtype=float32)
            hit_point[0] = camera_pos[0] + t * ray_dir[0]
            hit_point[1] = camera_pos[1] + t * ray_dir[1]
            hit_point[2] = camera_pos[2] + t * ray_dir[2]
            
            normal = cuda.local.array(3, dtype=float32)
            normal[0] = (hit_point[0] - sphere_centers[i][0]) / sphere_radii[i]
            normal[1] = (hit_point[1] - sphere_centers[i][1]) / sphere_radii[i]
            normal[2] = (hit_point[2] - sphere_centers[i][2]) / sphere_radii[i]
            normalize(normal)
            
            # Material color
            mat_idx = i * 3
            color[0] = sphere_materials[mat_idx]
            color[1] = sphere_materials[mat_idx + 1]
            color[2] = sphere_materials[mat_idx + 2]
    
    if not hit_anything:
        # Sky color
        t = 0.5 * (ray_dir[1] + 1.0)
        color[0] = (1.0 - t) + t * 0.5
        color[1] = (1.0 - t) + t * 0.7
        color[2] = (1.0 - t) + t * 1.0
    
    # Write final color, flip y coordinate
    out_image[x, y, 0] = int(min(255, max(0, color[0] * 255)))
    out_image[x, y, 1] = int(min(255, max(0, color[1] * 255)))
    out_image[x, y, 2] = int(min(255, max(0, color[2] * 255)))

class Renderer:
    def __init__(self, width: int, height: int, max_depth: int = 3):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        
        # CUDA configuration
        self.threadsperblock = (16, 16)
        self.blockspergrid_x = math.ceil(width / self.threadsperblock[0])
        self.blockspergrid_y = math.ceil(height / self.threadsperblock[1])
        self.blockspergrid = (self.blockspergrid_x, self.blockspergrid_y)
        
        # Initialize device arrays
        self.d_sphere_centers = None
        self.d_sphere_radii = None
        self.d_sphere_materials = None
        
    def update_scene_data(self, world):
        """Convert scene data to GPU-friendly format."""
        # Count spheres in the world
        sphere_count = len([obj for obj in world.objects if hasattr(obj, 'radius')])
        
        # Prepare CPU arrays
        centers = np.zeros((sphere_count, 3), dtype=np.float32)
        radii = np.zeros(sphere_count, dtype=np.float32)
        materials = np.zeros(sphere_count * 3, dtype=np.float32)  # RGB for each sphere
        
        # Fill arrays
        sphere_idx = 0
        for obj in world.objects:
            if hasattr(obj, 'radius'):  # Is a sphere
                centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
                radii[sphere_idx] = obj.radius
                
                # Simple diffuse color from material
                if hasattr(obj.material, 'albedo'):
                    mat_idx = sphere_idx * 3
                    materials[mat_idx:mat_idx+3] = [
                        obj.material.albedo.x,
                        obj.material.albedo.y,
                        obj.material.albedo.z
                    ]
                sphere_idx += 1
        
        # Transfer to GPU
        self.d_sphere_centers = cuda.to_device(np.ascontiguousarray(centers))
        self.d_sphere_radii = cuda.to_device(np.ascontiguousarray(radii))
        self.d_sphere_materials = cuda.to_device(np.ascontiguousarray(materials))
        
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
        
        # Create output array with correct orientation
        output = np.zeros((self.width, self.height, 3), dtype=np.uint8)
        output = np.ascontiguousarray(output)  # Ensure contiguous memory
        d_output = cuda.to_device(output)
        
        # Launch kernel
        render_kernel[self.blockspergrid, self.threadsperblock](
            self.width, self.height,
            camera_pos, camera_forward, camera_right, camera_up,
            self.d_sphere_centers, self.d_sphere_radii, self.d_sphere_materials,
            d_output
        )
        
        # Get result
        output = d_output.copy_to_host()
        return output