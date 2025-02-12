# renderer/raytracer.py
import numpy as np
from numba import cuda, float32, int32
import math
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from core.vector import Vector3
from core.ray import Ray
from materials.metal import Metal

# CUDA device constants
INFINITY = float32(1e20)
EPSILON = float32(1e-7)
MAX_BOUNCES = 6

@cuda.jit(device=True)
def random_in_unit_sphere(rng_states, thread_id, out):
    """
    Fills 'out' with a random 3D point uniformly inside the unit sphere.
    'rng_states' is the device array of Xoroshiro128+ states,
    'thread_id' is the pixel/thread index.
    """
    for _ in range(10):
        rx = 2.0 * xoroshiro128p_uniform_float32(rng_states, thread_id) - 1.0
        ry = 2.0 * xoroshiro128p_uniform_float32(rng_states, thread_id) - 1.0
        rz = 2.0 * xoroshiro128p_uniform_float32(rng_states, thread_id) - 1.0

        length_sq = rx * rx + ry * ry + rz * rz
        if length_sq <= 1.0:
            out[0] = rx
            out[1] = ry
            out[2] = rz
            return

    # Fallback if we never get a point <= 1.0 (very rare):
    out[0] = 0.0
    out[1] = 0.0
    out[2] = 1.0

@cuda.jit(device=True)
def sample_environment(direction, out_color):
    """
    Sample environment map (simple gradient sky)
    Writes directly to out_color array instead of returning
    """
    # Normalize direction if needed
    t = 0.5 * (direction[1] + 1.0)
    
    # Horizon colors (warm)
    horizon_r = 0.7
    horizon_g = 0.6
    horizon_b = 0.5
    
    # Zenith colors (blue sky)
    zenith_r = 0.2
    zenith_g = 0.4
    zenith_b = 0.8
    
    # Lerp between horizon and zenith
    out_color[0] = (1.0 - t) * horizon_r + t * zenith_r
    out_color[1] = (1.0 - t) * horizon_g + t * zenith_g
    out_color[2] = (1.0 - t) * horizon_b + t * zenith_b
    
    # Add sun disk
    sun_dir = cuda.local.array(3, float32)
    sun_dir[0] = 0.5
    sun_dir[1] = 0.8
    sun_dir[2] = 0.5
    normalize_inplace(sun_dir)
    
    sun_dot = dot(direction, sun_dir)
    if sun_dot > 0.995:  # Sharp sun disk
        out_color[0] += 5.0
        out_color[1] += 4.8
        out_color[2] += 4.5

@cuda.jit(device=True)
def sample_light_contribution(
    hit_point, normal,
    sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
    rng_states, thread_id,
    out_color
):
    """
    Improved light sampling with multiple importance sampling
    """
    # Sample multiple light points for better convergence
    num_light_samples = 4
    for _ in range(num_light_samples):
        # Get a random emissive sphere
        center = cuda.local.array(3, float32)
        emission = cuda.local.array(3, float32)
        rad = cuda.local.array(1, float32)
        sample_emissive_sphere(
            sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
            rng_states, thread_id,
            center, rad, emission
        )
        
        if rad[0] < 0.0:
            return
            
        # Generate point on sphere surface using uniform sampling
        # This gives better results than random sampling
        u = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, thread_id)
        v = 1.0 - 2.0 * xoroshiro128p_uniform_float32(rng_states, thread_id)
        r = math.sqrt(1.0 - v * v)
        
        light_point = cuda.local.array(3, float32)
        light_point[0] = center[0] + rad[0] * (r * math.cos(u))
        light_point[1] = center[1] + rad[0] * v
        light_point[2] = center[2] + rad[0] * (r * math.sin(u))
        
        # Calculate direction and distance to light
        to_light = cuda.local.array(3, float32)
        for i in range(3):
            to_light[i] = light_point[i] - hit_point[i]
        
        dist = math.sqrt(dot(to_light, to_light))
        if dist < 1e-4:
            continue
            
        inv_dist = 1.0 / dist
        for i in range(3):
            to_light[i] *= inv_dist
            
        # Calculate geometric term
        cos_theta = max(0.0, dot(normal, to_light))
        if cos_theta <= 0.0:
            continue
            
        # Add contribution with proper attenuation
        scale = cos_theta / (num_light_samples * math.pi * dist * dist)
        for i in range(3):
            out_color[i] += emission[i] * scale

@cuda.jit(device=True)
def sample_emissive_sphere(
    sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
    rng_states, thread_id,
    out_center, out_radius, out_emission
):
    """
    Randomly select one emissive sphere (mat_type=2) among all spheres,
    returning its center, radius, and emission color.

    If no emissive sphere is found, set out_radius to -1 to indicate "no light."
    """
    # Count how many are emissive
    emissive_count = 0
    for s in range(sphere_centers.shape[0]):
        if sphere_material_types[s] == 2:
            emissive_count += 1

    if emissive_count == 0:
        out_radius[0] = -1.0
        return

    # Random index among emissive spheres
    pick = int(xoroshiro128p_uniform_float32(rng_states, thread_id) * emissive_count)
    if pick >= emissive_count:
        pick = emissive_count - 1

    found_index = -1
    running = 0
    for s in range(sphere_centers.shape[0]):
        if sphere_material_types[s] == 2:
            if running == pick:
                found_index = s
                break
            running += 1

    if found_index < 0:
        out_radius[0] = -1.0
        return

    # Return chosen sphere's center, radius, and emission color
    for i in range(3):
        out_center[i] = sphere_centers[found_index, i]
        out_emission[i] = sphere_materials[found_index * 3 + i]
    out_radius[0] = sphere_radii[found_index]


@cuda.jit(device=True)
def compute_direct_light(
    hit_point, normal,
    sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
    triangle_vertices, triangle_materials, triangle_material_types,
    rng_states, thread_id,
    out_color
):
    """
    Next-event estimation (NEE) for emissive spheres with proper normalization.
    This function samples a point on an emissive sphere's surface uniformly and
    accumulates its contribution into out_color.
    
    The contribution is computed as:
    
        L_direct = (L_e * G) / pdf
    
    where:
      - L_e is the emission (radiance) of the light,
      - G = cos(theta) / (dist^2) is the geometry term, and
      - pdf is the probability density of sampling that point on the light.
    
    For a uniform sampling on a sphere, the pdf is:
    
        pdf = 1 / (4 * pi * rad^2)
    
    The contribution is then averaged over the number of light samples.
    
    Args:
      hit_point: 3-element array (float32) for the ray–hit position.
      normal: 3-element array (float32) for the surface normal at hit_point.
      sphere_centers: GPU array of sphere centers.
      sphere_radii: GPU array of sphere radii.
      sphere_materials: GPU array containing emission (or albedo) for each sphere.
      sphere_material_types: GPU array of material types (e.g. 0: Lambertian, 1: Metal, 2: Emissive).
      triangle_vertices, triangle_materials, triangle_material_types:
          (Included for completeness; not used here.)
      rng_states: RNG states array for the current thread.
      thread_id: The current thread's index.
      out_color: A 3-element array (float32) into which the direct light contribution is accumulated.
    """
    num_light_samples = 4  # Use multiple samples for better convergence.
    
    for s in range(num_light_samples):
        # Sample an emissive sphere (if available).
        center = cuda.local.array(3, float32)
        emission = cuda.local.array(3, float32)
        rad = cuda.local.array(1, float32)
        sample_emissive_sphere(
            sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
            rng_states, thread_id,
            center, rad, emission
        )
        
        # If no emissive sphere was found, skip this sample.
        if rad[0] < 0.0:
            continue
        
        # Uniformly sample a point on the sphere's surface.
        # Here we use two uniformly distributed random numbers to compute spherical coordinates.
        u = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, thread_id)
        v = 1.0 - 2.0 * xoroshiro128p_uniform_float32(rng_states, thread_id)
        r_sample = math.sqrt(max(0.0, 1.0 - v * v))
        
        light_point = cuda.local.array(3, float32)
        light_point[0] = center[0] + rad[0] * (r_sample * math.cos(u))
        light_point[1] = center[1] + rad[0] * v
        light_point[2] = center[2] + rad[0] * (r_sample * math.sin(u))
        
        # Compute the vector from the hit point to the sampled point on the light.
        to_light = cuda.local.array(3, float32)
        for i in range(3):
            to_light[i] = light_point[i] - hit_point[i]
        
        dist = math.sqrt(dot(to_light, to_light))
        if dist < 1e-4:
            continue  # Skip samples that are too close.
        
        # Normalize the direction toward the light.
        inv_dist = 1.0 / dist
        for i in range(3):
            to_light[i] *= inv_dist
        
        # Calculate the cosine of the angle between the surface normal and the light direction.
        cos_theta = dot(normal, to_light)
        if cos_theta <= 0.0:
            continue  # Light is below the surface.
        
        # Geometry term: G = cos_theta / (dist^2)
        G = cos_theta / (dist * dist)
        
        # Compute the PDF for uniformly sampling a point on the sphere's surface:
        # pdf = 1 / (4 * pi * rad^2)
        pdf = 1.0 / (4.0 * math.pi * rad[0] * rad[0])
        
        # Compute the contribution from this sample:
        # Contribution = L_e * (G / pdf)
        # Average over the number of samples:
        scale = (G / pdf) / num_light_samples
        
        # Accumulate the contribution into out_color.
        for i in range(3):
            out_color[i] += emission[i] * scale

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
    
    # Improve numerical precision    
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
              out_color,
              rng_states, thread_id):
    """
    Improved path tracing with next-event estimation only on the first bounce
    to avoid double-counting bright light sources.
    """
    current_origin = cuda.local.array(3, float32)
    current_dir   = cuda.local.array(3, float32)
    attenuation   = cuda.local.array(3, float32)
    color         = cuda.local.array(3, float32)
    temp_color    = cuda.local.array(3, float32)

    # Initialize local arrays
    for i in range(3):
        current_origin[i] = origin[i]
        current_dir[i]    = direction[i]
        attenuation[i]    = 1.0
        color[i]          = 0.0
        temp_color[i]     = 0.0

    for bounce in range(max_bounces):
        # 1) Find the closest intersection
        closest_t   = INFINITY
        hit_index   = -1
        hit_sphere  = True

        # Check spheres
        for s in range(sphere_centers.shape[0]):
            t = ray_sphere_intersect(current_origin, current_dir,
                                     sphere_centers[s], sphere_radii[s],
                                     float32(1e-4), closest_t)
            if t > 0.0:
                closest_t  = t
                hit_index  = s
                hit_sphere = True

        # Check triangles
        for t_idx in range(triangle_vertices.shape[0]):
            t = ray_triangle_intersect(current_origin, current_dir,
                                       triangle_vertices[t_idx, 0:3],
                                       triangle_vertices[t_idx, 3:6],
                                       triangle_vertices[t_idx, 6:9],
                                       float32(1e-4), closest_t)
            if t > 0.0:
                closest_t  = t
                hit_index  = t_idx
                hit_sphere = False

        # 2) If no hit => sample environment (sky / sun) and break
        if hit_index < 0:
            sample_environment(current_dir, temp_color)
            for i in range(3):
                color[i] += attenuation[i] * temp_color[i]
            break

        # 3) Compute actual hit point, normal, and material
        hit_point = cuda.local.array(3, float32)
        normal    = cuda.local.array(3, float32)
        mat_color = cuda.local.array(3, float32)
        mat_type  = 0

        for i in range(3):
            hit_point[i] = current_origin[i] + closest_t * current_dir[i]

        if hit_sphere:
            # Sphere
            sphere_radius = sphere_radii[hit_index]
            center        = sphere_centers[hit_index]
            for i in range(3):
                normal[i] = (hit_point[i] - center[i]) / sphere_radius
            normalize_inplace(normal)

            mat_type = sphere_material_types[hit_index]
            base_idx = hit_index * 3
            for i in range(3):
                mat_color[i] = sphere_materials[base_idx + i]

        else:
            # Triangle
            v0 = cuda.local.array(3, float32)
            v1 = cuda.local.array(3, float32)
            v2 = cuda.local.array(3, float32)
            for i in range(3):
                v0[i] = triangle_vertices[hit_index, i]
                v1[i] = triangle_vertices[hit_index, 3 + i]
                v2[i] = triangle_vertices[hit_index, 6 + i]

            e1 = cuda.local.array(3, float32)
            e2 = cuda.local.array(3, float32)
            for i in range(3):
                e1[i] = v1[i] - v0[i]
                e2[i] = v2[i] - v0[i]
            cross_inplace(normal, e1, e2)
            normalize_inplace(normal)

            mat_type = triangle_material_types[hit_index]
            base_idx = hit_index * 3
            for i in range(3):
                mat_color[i] = triangle_materials[base_idx + i]

        # 4) If emissive material, add emission & break
        if mat_type == 2:
            for i in range(3):
                color[i] += attenuation[i] * mat_color[i]
            break

        # 5) Next-event estimation: do it ONLY on the first bounce
        #    to avoid double-counting big light sources.
        if bounce == 0:
            for i in range(3):
                temp_color[i] = 0.0

            compute_direct_light(
                hit_point, normal,
                sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
                triangle_vertices, triangle_materials, triangle_material_types,
                rng_states, thread_id,
                temp_color
            )
            for i in range(3):
                color[i] += attenuation[i] * temp_color[i]

        # 6) Scatter the ray according to material type

        if mat_type == 1:
            # Metal
            reflected = cuda.local.array(3, float32)
            reflect(reflected, current_dir, normal)

            # Add some small fuzz for demonstration
            fuzz_vec = cuda.local.array(3, float32)
            random_in_unit_sphere(rng_states, thread_id, fuzz_vec)
            fuzz_amount = 0.05
            for i in range(3):
                reflected[i] += fuzz_amount * fuzz_vec[i]
            normalize_inplace(reflected)

            # Update attenuation
            for i in range(3):
                attenuation[i] *= mat_color[i]

            # Next ray
            for i in range(3):
                current_origin[i] = hit_point[i]
                current_dir[i]    = reflected[i]

            # If reflection is inward, end
            if dot(current_dir, normal) <= 0.0:
                break

        else:
            # Lambertian / diffuse
            scatter_vec = cuda.local.array(3, float32)
            random_in_unit_sphere(rng_states, thread_id, scatter_vec)

            # direction = normal + random
            for i in range(3):
                scatter_vec[i] += normal[i]
            normalize_inplace(scatter_vec)

            # Update attenuation
            for i in range(3):
                attenuation[i] *= mat_color[i]

            # Next ray
            for i in range(3):
                current_origin[i] = hit_point[i]
                current_dir[i]    = scatter_vec[i]

        # 7) Russian roulette for bounces > 3
        if bounce >= 3:
            luminance = 0.2126 * attenuation[0] + 0.7152 * attenuation[1] + 0.0722 * attenuation[2]
            q = max(0.05, min(0.95, luminance))
            if xoroshiro128p_uniform_float32(rng_states, thread_id) > q:
                break

            inv_q = 1.0 / q
            for i in range(3):
                attenuation[i] *= inv_q

    # Write final color
    for i in range(3):
        out_color[i] = color[i]

@cuda.jit
def render_kernel(width, height,
                  camera_origin,
                  camera_lower_left,
                  camera_horizontal,
                  camera_vertical,
                  sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
                  triangle_vertices, triangle_materials, triangle_material_types,
                  out_image, frame_number,
                  rng_states):
    """
    GPU kernel that computes the linear radiance for each pixel by tracing rays.
    
    This modified version outputs raw, linear radiance values instead of converting
    the computed color to 8-bit integers. The accumulation and tone mapping are then
    performed on the host in linear space.
    
    Args:
        width (int): Image width.
        height (int): Image height.
        camera_origin (array of float32, shape [3]): The camera origin.
        camera_lower_left (array of float32, shape [3]): The lower-left corner of the viewport.
        camera_horizontal (array of float32, shape [3]): The horizontal vector for the viewport.
        camera_vertical (array of float32, shape [3]): The vertical vector for the viewport.
        sphere_centers: GPU array of sphere centers.
        sphere_radii: GPU array of sphere radii.
        sphere_materials: GPU array of sphere material colors/emissions.
        sphere_material_types: GPU array of sphere material types.
        triangle_vertices: GPU array of triangle vertex data.
        triangle_materials: GPU array of triangle material colors/emissions.
        triangle_material_types: GPU array of triangle material types.
        out_image: Output GPU array to store the linear radiance (float32) for each pixel.
        frame_number (int): Current frame number (unused here, but passed in).
        rng_states: RNG states array for random number generation.
    """
    # Get pixel coordinates from the CUDA grid.
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Compute a unique pixel index for RNG state access.
    pixel_idx = x + y * width

    # Allocate local arrays for the ray origin, direction, and color.
    ray_origin = cuda.local.array(3, dtype=float32)
    ray_dir    = cuda.local.array(3, dtype=float32)
    color      = cuda.local.array(3, dtype=float32)

    # Initialize the ray origin and color.
    for i in range(3):
        ray_origin[i] = camera_origin[i]
        color[i] = 0.0

    # Anti-aliasing: add a small jitter using random numbers.
    jitter_x = xoroshiro128p_uniform_float32(rng_states, pixel_idx) / width
    jitter_y = xoroshiro128p_uniform_float32(rng_states, pixel_idx) / height

    # Compute normalized u and v coordinates in the viewport.
    u = (float32(x) + jitter_x) / float32(width - 1)
    v = 1.0 - ((float32(y) + jitter_y) / float32(height - 1))

    # Reconstruct the ray direction from the camera parameters.
    # The direction is computed from the lower-left corner plus horizontal and vertical offsets.
    for i in range(3):
        val = camera_lower_left[i] + u * camera_horizontal[i] + v * camera_vertical[i] - camera_origin[i]
        ray_dir[i] = val

    # Normalize the computed ray direction.
    normalize_inplace(ray_dir)

    # Trace the ray through the scene. The trace_ray function accumulates the radiance.
    trace_ray(ray_origin, ray_dir, MAX_BOUNCES,
              sphere_centers, sphere_radii,
              sphere_materials, sphere_material_types,
              triangle_vertices, triangle_materials, triangle_material_types,
              color,
              rng_states, pixel_idx)

    # Write the linear radiance to the output image.
    # Do not convert to 8-bit here; conversion is done after accumulation.
    for i in range(3):
        out_image[x, y, i] = color[i]

class Renderer:
    def __init__(self, width: int, height: int, max_depth: int = 3):
        self.width = width
        self.height = height
        self.max_depth = max_depth

        self.frame_number = 0
        self.accumulation_buffer = np.zeros((width, height, 3), dtype=np.float32)
        self.samples = 0

        # CUDA config
        self.threadsperblock = (16, 16)
        self.blockspergrid_x = math.ceil(width / self.threadsperblock[0])
        self.blockspergrid_y = math.ceil(height / self.threadsperblock[1])
        self.blockspergrid   = (self.blockspergrid_x, self.blockspergrid_y)

        # Create random states (one per pixel)
        n_states = width * height
        self.rng_states = create_xoroshiro128p_states(n_states, seed=42)

        # Prepare GPU arrays (set to None initially)
        self.d_sphere_centers = None
        self.d_sphere_radii   = None
        self.d_sphere_materials = None
        self.d_sphere_material_types = None
        self.d_triangle_vertices = None
        self.d_triangle_materials = None
        self.d_triangle_material_types = None

    def reset_accumulation(self):
        """Reset the accumulation buffer when the camera moves."""
        self.accumulation_buffer.fill(0)
        self.samples = 0
        
    def update_scene_data(self, world):
        """
        Convert scene data (spheres, meshes) into GPU‐friendly arrays,
        handling Lambertian, Metal, and DiffuseLight materials properly.
        """

        # 1. Count spheres & triangles
        sphere_count = sum(1 for obj in world.objects if hasattr(obj, 'radius'))
        mesh_objects = [obj for obj in world.objects if hasattr(obj, 'triangles')]
        triangle_count = sum(len(mesh.triangles) for mesh in mesh_objects)
        
        # 2. Cleanup old GPU arrays
        self.cleanup()
        
        # 3. Prepare arrays for spheres
        centers = np.zeros((max(1, sphere_count), 3), dtype=np.float32)
        radii = np.zeros(max(1, sphere_count), dtype=np.float32)
        sphere_materials = np.zeros(max(1, sphere_count) * 3, dtype=np.float32)
        sphere_material_types = np.zeros(max(1, sphere_count), dtype=np.int32)
        
        # 4. Fill sphere arrays
        sphere_idx = 0
        for obj in world.objects:
            if hasattr(obj, 'radius'):
                # Record geometry
                centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
                radii[sphere_idx] = obj.radius

                # Prepare the base index for storing color
                mat_idx = sphere_idx * 3
                
                # Distinguish between material types
                mat = obj.material
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    # It's a DiffuseLight (emissive)
                    emission = mat.emitted(0, 0, obj.center)
                    sphere_materials[mat_idx:mat_idx+3] = [
                        emission.x, emission.y, emission.z
                    ]
                    sphere_material_types[sphere_idx] = 2  # "2" means emissive
                elif isinstance(mat, Metal):
                    # Metal
                    sphere_materials[mat_idx:mat_idx+3] = [
                        mat.albedo.x, mat.albedo.y, mat.albedo.z
                    ]
                    sphere_material_types[sphere_idx] = 1  # "1" means metal
                else:
                    # Assume Lambertian (or fallback)
                    # You might do: "elif isinstance(mat, Lambertian): ..."
                    # but “else” is fine if there are no other types
                    sphere_materials[mat_idx:mat_idx+3] = [
                        mat.albedo.x, mat.albedo.y, mat.albedo.z
                    ]
                    sphere_material_types[sphere_idx] = 0  # "0" means lambertian
                
                sphere_idx += 1
        
        # 5. Prepare arrays for triangles
        triangle_vertices = np.zeros((max(1, triangle_count), 9), dtype=np.float32)
        triangle_materials = np.zeros(max(1, triangle_count) * 3, dtype=np.float32)
        triangle_material_types = np.zeros(max(1, triangle_count), dtype=np.int32)
        
        # 6. Fill triangle arrays
        triangle_idx = 0
        for mesh in mesh_objects:
            for triangle in mesh.triangles:
                # Store the triangle’s 3 vertices
                triangle_vertices[triangle_idx, 0:3] = [triangle.v0.x, triangle.v0.y, triangle.v0.z]
                triangle_vertices[triangle_idx, 3:6] = [triangle.v1.x, triangle.v1.y, triangle.v1.z]
                triangle_vertices[triangle_idx, 6:9] = [triangle.v2.x, triangle.v2.y, triangle.v2.z]

                # Prepare the base index for color
                mat_idx = triangle_idx * 3
                mat = mesh.material
                
                # Check if emissive
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    emission = mat.emitted(0, 0, triangle.v0)  # pass any vertex or data
                    triangle_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    triangle_material_types[triangle_idx] = 2
                elif isinstance(mat, Metal):
                    triangle_materials[mat_idx:mat_idx+3] = [
                        mat.albedo.x, mat.albedo.y, mat.albedo.z
                    ]
                    triangle_material_types[triangle_idx] = 1
                else:
                    # Assume Lambertian if not metal or emissive
                    triangle_materials[mat_idx:mat_idx+3] = [
                        mat.albedo.x, mat.albedo.y, mat.albedo.z
                    ]
                    triangle_material_types[triangle_idx] = 0

                triangle_idx += 1
        
        # 7. Transfer arrays to GPU
        self.d_sphere_centers = cuda.to_device(np.ascontiguousarray(centers))
        self.d_sphere_radii   = cuda.to_device(np.ascontiguousarray(radii))
        self.d_sphere_materials = cuda.to_device(np.ascontiguousarray(sphere_materials))
        self.d_sphere_material_types = cuda.to_device(np.ascontiguousarray(sphere_material_types))
        
        self.d_triangle_vertices = cuda.to_device(np.ascontiguousarray(triangle_vertices))
        self.d_triangle_materials = cuda.to_device(np.ascontiguousarray(triangle_materials))
        self.d_triangle_material_types = cuda.to_device(np.ascontiguousarray(triangle_material_types))
    
    def render_frame(self, camera, world) -> np.ndarray:
        """
        Renders a frame with temporal accumulation for noise reduction.
        """
        if self.d_sphere_centers is None:
            self.update_scene_data(world)

        # Camera setup remains the same...
        camera_origin = np.array([
            camera.position.x,
            camera.position.y,
            camera.position.z
        ], dtype=np.float32)

        camera_lower_left = np.array([
            camera.lower_left_corner.x,
            camera.lower_left_corner.y,
            camera.lower_left_corner.z
        ], dtype=np.float32)

        camera_horizontal = np.array([
            camera.horizontal.x,
            camera.horizontal.y,
            camera.horizontal.z
        ], dtype=np.float32)

        camera_vertical = np.array([
            camera.vertical.x,
            camera.vertical.y,
            camera.vertical.z
        ], dtype=np.float32)

        # Create output array for this frame
        frame_output = np.zeros((self.width, self.height, 3), dtype=np.float32)
        d_frame_output = cuda.to_device(frame_output)

        # Render new frame
        render_kernel[self.blockspergrid, self.threadsperblock](
            self.width,
            self.height,
            camera_origin,
            camera_lower_left,
            camera_horizontal,
            camera_vertical,
            self.d_sphere_centers, self.d_sphere_radii,
            self.d_sphere_materials, self.d_sphere_material_types,
            self.d_triangle_vertices, self.d_triangle_materials, self.d_triangle_material_types,
            d_frame_output,
            self.frame_number,
            self.rng_states
        )

        # Get frame data back from GPU
        frame_output = d_frame_output.copy_to_host()

        # Accumulate into buffer
        self.accumulation_buffer += frame_output
        self.samples += 1
        
        # Average accumulated frames
        averaged = self.accumulation_buffer / self.samples

        # Tone mapping (to handle bright areas better)
        exposure = 1.0
        mapped = 1.0 - np.exp(-averaged * exposure)
        
        # Gamma correction with slightly higher gamma for better contrast
        gamma = 2.2
        mapped = np.power(mapped, 1.0 / gamma)
        
        # Convert to 8-bit RGB
        output = np.clip(mapped * 255, 0, 255).astype(np.uint8)
        
        self.frame_number += 1
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