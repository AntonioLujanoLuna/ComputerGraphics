# renderer/raytracer.py
import numpy as np
from numba import cuda, float32, int32
import math
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from core.vector import Vector3
from core.ray import Ray
from core.uv import UV
from materials.metal import Metal
from materials.dielectric import Dielectric
from materials.textures import Texture, SolidTexture, ImageTexture
from renderer.tone_mapping import reinhard_tone_mapping
from typing import List

# CUDA device constants
INFINITY = float32(1e20)
EPSILON = float32(1e-20)
MAX_BOUNCES = 16

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
def scatter_metal(incident, normal, albedo, rng_states, thread_id, out_scattered):
    """Compute metal scattering."""
    reflected = cuda.local.array(3, dtype=float32)
    reflect(incident, normal, reflected)
    
    # Add fuzz
    fuzz_vec = cuda.local.array(3, dtype=float32)
    random_in_unit_sphere(rng_states, thread_id, fuzz_vec)
    for i in range(3):
        reflected[i] += 0.1 * fuzz_vec[i]  # Using fixed fuzz of 0.1
    
    normalize_inplace(reflected)
    for i in range(3):
        out_scattered[i] = reflected[i]
    
    return dot(out_scattered, normal) > 0

@cuda.jit(device=True)
def scatter_lambertian(normal, rng_states, thread_id, out_scattered):
    """Compute Lambertian scattering."""
    scatter_vec = cuda.local.array(3, dtype=float32)
    random_in_unit_sphere(rng_states, thread_id, scatter_vec)
    for i in range(3):
        scatter_vec[i] += normal[i]
    
    normalize_inplace(scatter_vec)
    for i in range(3):
        out_scattered[i] = scatter_vec[i]
    
    return True

@cuda.jit(device=True)
def scatter_dielectric(current_dir, normal, ior, rng_states, thread_id, out_scattered):
    """
    Compute dielectric (glass) scattering.
    """
    # Normalize incoming ray direction
    unit_direction = cuda.local.array(3, float32)
    length = math.sqrt(dot(current_dir, current_dir))
    for i in range(3):
        unit_direction[i] = current_dir[i] / length
    
    # Determine if we're entering or exiting the material
    front_face = dot(unit_direction, normal) < 0
    etai_over_etat = 1.0 / ior if front_face else ior
    
    # Calculate cos_theta
    cos_theta = min(-dot(unit_direction, normal), 1.0)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    
    # Check for total internal reflection
    if etai_over_etat * sin_theta > 1.0:
        # Must reflect
        reflected = cuda.local.array(3, float32)
        reflect(unit_direction, normal, reflected)
        for i in range(3):
            out_scattered[i] = reflected[i]
        return True
    
    # Calculate reflection probability using Schlick's approximation
    reflect_prob = schlick(cos_theta, etai_over_etat)
    
    # Probabilistically choose reflection or refraction
    if xoroshiro128p_uniform_float32(rng_states, thread_id) < reflect_prob:
        reflected = cuda.local.array(3, float32)
        reflect(unit_direction, normal, reflected)
        for i in range(3):
            out_scattered[i] = reflected[i]
    else:
        refracted = cuda.local.array(3, float32)
        if refract(unit_direction, normal, etai_over_etat, refracted):
            for i in range(3):
                out_scattered[i] = refracted[i]
        else:
            # Fallback to reflection if refraction fails
            reflected = cuda.local.array(3, float32)
            reflect(unit_direction, normal, reflected)
            for i in range(3):
                out_scattered[i] = reflected[i]
    
    return True

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

@cuda.jit
def compute_variance_kernel(accum_buffer, accum_buffer_sq, samples, out_mask):
    """
    For each pixel (x,y), compute the variance per channel from the accumulated color
    and accumulated color squared, then average over channels.
    If the average variance is below a threshold (here hard-coded to 0.001), mark the pixel as converged.
    """
    x, y = cuda.grid(2)
    width = accum_buffer.shape[0]
    height = accum_buffer.shape[1]
    if x < width and y < height:
        # Compute per-channel means.
        mean_r = accum_buffer[x, y, 0] / samples
        mean_g = accum_buffer[x, y, 1] / samples
        mean_b = accum_buffer[x, y, 2] / samples
        # Compute per-channel means of the square.
        mean_sq_r = accum_buffer_sq[x, y, 0] / samples
        mean_sq_g = accum_buffer_sq[x, y, 1] / samples
        mean_sq_b = accum_buffer_sq[x, y, 2] / samples
        # Compute variance for each channel.
        var_r = mean_sq_r - mean_r * mean_r
        var_g = mean_sq_g - mean_g * mean_g
        var_b = mean_sq_b - mean_b * mean_b
        # Average variance across channels.
        avg_var = (var_r + var_g + var_b) / 3.0
        # Write mask: 1 if converged, 0 otherwise.
        if avg_var < 0.001:
            out_mask[x, y] = 1
        else:
            out_mask[x, y] = 0

@cuda.jit(device=True)
def sample_texture(texture_data, texture_width, texture_height, u: float, v: float, out_color):
    """Sample a texture at given UV coordinates."""
    # Wrap UV coordinates
    u = u % 1.0
    v = 1.0 - (v % 1.0)  # Flip V coordinate for OpenGL-style UV
    
    # Convert to pixel coordinates
    x = min(int(u * texture_width), texture_width - 1)
    y = min(int(v * texture_height), texture_height - 1)
    
    # Get pixel index
    idx = (y * texture_width + x) * 3
    
    # Copy color to output
    out_color[0] = texture_data[idx]
    out_color[1] = texture_data[idx + 1]
    out_color[2] = texture_data[idx + 2]

@cuda.jit(device=True)
def interpolate_uv(uv0, uv1, uv2, u: float, v: float, out_uv):
    """Interpolate UV coordinates using barycentric coordinates."""
    w = 1.0 - u - v
    out_uv[0] = w * uv0[0] + u * uv1[0] + v * uv2[0]
    out_uv[1] = w * uv0[1] + u * uv1[1] + v * uv2[1]

@cuda.jit(device=True)
def halton(index, base):
    """
    Compute the Halton sequence value for a given index and base.
    This is a simple low-discrepancy sequence generator.
    """
    f = 1.0
    r = 0.0
    while index > 0:
        f = f / base
        r = r + f * (index % base)
        index = index // base
    return r

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
def refract(v, n, ni_over_nt, out_refracted):
    cos_theta = min(-dot(v, n), 1.0)
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    
    # Check for total internal reflection
    if ni_over_nt * sin_theta > 1.0:
        return False
    
    # Calculate refracted ray components
    r_out_perp = cuda.local.array(3, float32)
    r_out_parallel = cuda.local.array(3, float32)
    
    # Perpendicular component: ni_over_nt * (v + cos_theta * n)
    for i in range(3):
        r_out_perp[i] = ni_over_nt * v[i] + (ni_over_nt * cos_theta - math.sqrt(1.0 - ni_over_nt * ni_over_nt * sin_theta * sin_theta)) * n[i]
    
    # Copy result to output
    for i in range(3):
        out_refracted[i] = r_out_perp[i]
    
    # Ensure the refracted ray is normalized
    normalize_inplace(out_refracted)
    return True

@cuda.jit(device=True)
def schlick(cosine, ref_idx):
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * math.pow((1 - cosine), 5)

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
def trace_ray(origin, direction, max_bounces,
              sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
              triangle_vertices, triangle_uvs, triangle_materials, triangle_material_types,
              texture_data, texture_dimensions, texture_indices,
              out_color,
              rng_states, thread_id):
    """
    Enhanced path tracing with texture support.
    
    Parameters:
      origin, direction: Ray parameters.
      max_bounces: Maximum number of bounces.
      sphere_centers, sphere_radii, sphere_materials, sphere_material_types: Sphere data.
      triangle_vertices, triangle_uvs, triangle_materials, triangle_material_types: Triangle data.
      texture_data: Flattened array of all texture data.
      texture_dimensions: Width/height for each texture.
      texture_indices: Maps materials to their textures.
      out_color: Output color (array of 3 floats).
      rng_states, thread_id: RNG state and thread index.
    """
    # Local arrays for current ray parameters and accumulation.
    current_origin = cuda.local.array(3, dtype=float32)
    current_dir = cuda.local.array(3, dtype=float32)
    attenuation = cuda.local.array(3, dtype=float32)
    color = cuda.local.array(3, dtype=float32)
    temp_color = cuda.local.array(3, dtype=float32)
    uv_coords = cuda.local.array(2, dtype=float32)  # For texture sampling

    for i in range(3):
        current_origin[i] = origin[i]
        current_dir[i] = direction[i]
        attenuation[i] = 1.0
        color[i] = 0.0
        temp_color[i] = 0.0

    # Main path-tracing loop over bounces.
    for bounce in range(max_bounces):
        closest_t = INFINITY
        hit_index = -1
        hit_sphere = True  # True if the closest hit is a sphere; False for a triangle.
        hit_u = 0.0
        hit_v = 0.0

        # --- Intersect with spheres ---
        for s in range(sphere_centers.shape[0]):
            t = ray_sphere_intersect(current_origin, current_dir,
                                     sphere_centers[s], sphere_radii[s],
                                     float32(1e-4), closest_t)
            if t > 0.0:
                closest_t = t
                hit_index = s
                hit_sphere = True
                # Calculate spherical UV coordinates (for texture mapping)
                hit_point = cuda.local.array(3, dtype=float32)
                for i in range(3):
                    hit_point[i] = current_origin[i] + t * current_dir[i]
                calculate_sphere_uv(hit_point, sphere_centers[s], uv_coords)

        # --- Intersect with triangles ---
        for t_idx in range(triangle_vertices.shape[0]):
            t = ray_triangle_intersect(current_origin, current_dir,
                                       triangle_vertices[t_idx, 0:3],
                                       triangle_vertices[t_idx, 3:6],
                                       triangle_vertices[t_idx, 6:9],
                                       float32(1e-4), closest_t)
            if t > 0.0:
                closest_t = t
                hit_index = t_idx
                hit_sphere = False
                # Interpolate triangle UV coordinates.
                interpolate_uv(
                    triangle_uvs[t_idx, 0:2],
                    triangle_uvs[t_idx, 2:4],
                    triangle_uvs[t_idx, 4:6],
                    hit_u, hit_v, uv_coords
                )

        # --- If no hit, sample the environment ---
        if hit_index < 0:
            sample_environment(current_dir, temp_color)
            for i in range(3):
                color[i] += attenuation[i] * temp_color[i]
            break

        # --- Compute hit point, surface normal, and material color ---
        hit_point = cuda.local.array(3, dtype=float32)
        normal = cuda.local.array(3, dtype=float32)
        material_color = cuda.local.array(3, dtype=float32)

        for i in range(3):
            hit_point[i] = current_origin[i] + closest_t * current_dir[i]

        if hit_sphere:
            # Compute sphere normal.
            center = sphere_centers[hit_index]
            for i in range(3):
                normal[i] = (hit_point[i] - center[i]) / sphere_radii[hit_index]
            # Flip the normal if the ray is inside the sphere.
            if dot(current_dir, normal) > 0.0:
                for i in range(3):
                    normal[i] = -normal[i]
            mat_type = sphere_material_types[hit_index]
            mat_idx = hit_index * 3
            if texture_indices[hit_index] >= 0:
                tex_idx = texture_indices[hit_index]
                sample_texture(
                    texture_data,
                    texture_dimensions[tex_idx, 0],
                    texture_dimensions[tex_idx, 1],
                    uv_coords[0], uv_coords[1],
                    material_color
                )
            else:
                for i in range(3):
                    material_color[i] = sphere_materials[mat_idx + i]
        else:
            # For triangles: compute normal and get material color.
            compute_triangle_normal(triangle_vertices[hit_index], normal)
            mat_type = triangle_material_types[hit_index]
            mat_idx = hit_index * 3
            if texture_indices[hit_index + sphere_centers.shape[0]] >= 0:
                tex_idx = texture_indices[hit_index + sphere_centers.shape[0]]
                sample_texture(
                    texture_data,
                    texture_dimensions[tex_idx, 0],
                    texture_dimensions[tex_idx, 1],
                    uv_coords[0], uv_coords[1],
                    material_color
                )
            else:
                for i in range(3):
                    material_color[i] = triangle_materials[mat_idx + i]

        # --- For dielectric materials (mat_type == 3), override material_color to white ---
        if mat_type == 3:
            ior = material_color[0]  # Extract the refractive index.
            for i in range(3):
                material_color[i] = 1.0
        # --- Handle emissive materials (mat_type == 2) ---
        if mat_type == 2:
            for i in range(3):
                color[i] += attenuation[i] * material_color[i]
            break

        # --- Next-event estimation on the first bounce ---
        if bounce == 0:
            compute_direct_light(
                hit_point, normal,
                sphere_centers, sphere_radii,
                sphere_materials, sphere_material_types,
                triangle_vertices, triangle_materials, triangle_material_types,
                rng_states, thread_id,
                temp_color
            )
            for i in range(3):
                color[i] += attenuation[i] * temp_color[i]

        # --- Scatter based on material type ---
        scattered = cuda.local.array(3, dtype=float32)
        scattered_valid = False
        if mat_type == 1:  # Metal
            scattered_valid = scatter_metal(current_dir, normal, material_color,
                                            rng_states, thread_id, scattered)
        elif mat_type == 3:  # Dielectric
            scattered_valid = scatter_dielectric(current_dir, normal, ior,
                                                 rng_states, thread_id, scattered)
        else:  # Lambertian
            scattered_valid = scatter_lambertian(normal, rng_states, thread_id, scattered)

        if not scattered_valid:
            break

        # *** NEW: Normalize the scattered ray direction ***
        normalize_inplace(scattered)

        # Update the ray and attenuation for the next bounce.
        for i in range(3):
            current_origin[i] = hit_point[i]
            current_dir[i] = scattered[i]
            attenuation[i] *= material_color[i]

        # --- Russian roulette termination after 3 bounces ---
        if bounce >= 3:
            luminance = 0.2126 * attenuation[0] + 0.7152 * attenuation[1] + 0.0722 * attenuation[2]
            q = max(0.05, min(0.95, luminance))
            if xoroshiro128p_uniform_float32(rng_states, thread_id) > q:
                break
            inv_q = 1.0 / q
            for i in range(3):
                attenuation[i] *= inv_q

    # Write the final accumulated color to the output.
    for i in range(3):
        out_color[i] = color[i]

@cuda.jit
def render_kernel(width, height,
                 camera_origin,
                 camera_lower_left,
                 camera_horizontal,
                 camera_vertical,
                 sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
                 triangle_vertices, triangle_uvs, triangle_materials, triangle_material_types,
                 texture_data, texture_dimensions, texture_indices,
                 out_image, frame_number,
                 rng_states):
    """
    GPU kernel that computes the linear radiance for each pixel.
    This version uses shared memory for camera parameters for better memory access.
    """
    # Allocate shared memory for camera parameters (size 3 each)
    shared_origin = cuda.shared.array(shape=3, dtype=float32)
    shared_lower_left = cuda.shared.array(shape=3, dtype=float32)
    shared_horizontal = cuda.shared.array(shape=3, dtype=float32)
    shared_vertical = cuda.shared.array(shape=3, dtype=float32)

    # Let thread (0,0) of each block load the camera parameters.
    if cuda.threadIdx.x == 0 and cuda.threadIdx.y == 0:
        for i in range(3):
            shared_origin[i] = camera_origin[i]
            shared_lower_left[i] = camera_lower_left[i]
            shared_horizontal[i] = camera_horizontal[i]
            shared_vertical[i] = camera_vertical[i]
    cuda.syncthreads()

    # Compute pixel coordinates.
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    pixel_idx = x + y * width

    # Local arrays for the ray origin, direction, and color.
    ray_origin = cuda.local.array(3, dtype=float32)
    ray_dir = cuda.local.array(3, dtype=float32)
    color = cuda.local.array(3, dtype=float32)
    for i in range(3):
        ray_origin[i] = shared_origin[i]
        color[i] = 0.0

    # Anti-aliasing: add a small jitter.
    jitter_x = halton(pixel_idx, 2) / width
    jitter_y = halton(pixel_idx, 3) / height
    u = (float32(x) + jitter_x) / float32(width - 1)
    v = 1.0 - ((float32(y) + jitter_y) / float32(height - 1))

    # Reconstruct the ray direction.
    for i in range(3):
        # Calculate: lower_left + u * horizontal + v * vertical - origin.
        temp = shared_lower_left[i] + u * shared_horizontal[i] + v * shared_vertical[i] - shared_origin[i]
        ray_dir[i] = temp

    # Normalize the ray direction.
    normalize_inplace(ray_dir)

    # Trace the ray (using your existing trace_ray function).
    trace_ray(ray_origin, ray_dir, MAX_BOUNCES,
            sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
            triangle_vertices, triangle_uvs, triangle_materials, triangle_material_types,
            texture_data, texture_dimensions, texture_indices,
            color,
            rng_states, pixel_idx)

    # Write the computed color to the output image.
    for i in range(3):
        out_image[x, y, i] = color[i]

class Renderer:
    def __init__(self, width: int, height: int, max_depth: int = 3):
        self.width = width
        self.height = height
        self.max_depth = max_depth

        self.frame_number = 0
        self.accumulation_buffer = np.zeros((width, height, 3), dtype=np.float32)
        self.accumulation_buffer_sq = np.zeros((width, height, 3), dtype=np.float32)  # For variance
        self.samples = 0

        # CUDA config remains as before.
        self.threadsperblock = (16, 16)
        self.blockspergrid_x = math.ceil(width / self.threadsperblock[0])
        self.blockspergrid_y = math.ceil(height / self.threadsperblock[1])
        self.blockspergrid   = (self.blockspergrid_x, self.blockspergrid_y)

        # Create random states (one per pixel)
        n_states = width * height
        self.rng_states = create_xoroshiro128p_states(n_states, seed=42)

        # GPU arrays for scene data are initialized to None.
        self.d_sphere_centers = None
        self.d_sphere_radii   = None
        self.d_sphere_materials = None
        self.d_sphere_material_types = None
        self.d_triangle_vertices = None
        self.d_triangle_materials = None
        self.d_triangle_material_types = None

        # Add UV and texture-related GPU arrays
        self.d_triangle_uvs = None
        self.d_texture_data = None
        self.d_texture_indices = None
        self.texture_dimensions = None

        # Add texture-related GPU arrays
        self.d_texture_data = None
        self.d_texture_indices = None  # Maps materials to textures
        self.texture_dimensions = None  # Stores width/height for each texture
    
    def load_textures(self, textures: List[Texture]):
        """Load textures to GPU memory."""
        # Convert textures to flat array
        texture_data = []
        texture_dimensions = []
        
        for texture in textures:
            if isinstance(texture, ImageTexture):
                # Flatten RGB data
                flat_data = texture.data.reshape(-1)
                texture_data.extend(flat_data)
                texture_dimensions.append((texture.width, texture.height))
            elif isinstance(texture, SolidTexture):
                # Create 1x1 texture for solid colors
                texture_data.extend([texture.color.x, texture.color.y, texture.color.z])
                texture_dimensions.append((1, 1))
                
        # Transfer to GPU
        self.d_texture_data = cuda.to_device(np.array(texture_data, dtype=np.float32))
        self.texture_dimensions = cuda.to_device(np.array(texture_dimensions, dtype=np.int32))

    def reset_accumulation(self):
        """Reset the accumulation buffers when the camera or scene changes."""
        self.accumulation_buffer.fill(0)
        self.accumulation_buffer_sq.fill(0)
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

        num_texture_entries = sphere_count + triangle_count
        # Create a default texture indices array (all -1 means “no texture”)
        default_texture_indices = -1 * np.ones((num_texture_entries,), dtype=np.int32)
        self.d_texture_indices = cuda.to_device(default_texture_indices)
        
        # 4. Fill sphere arrays
        sphere_idx = 0
        for obj in world.objects:
            if hasattr(obj, 'radius'):
                # Record geometry
                centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
                radii[sphere_idx] = obj.radius

                # Prepare the base index for storing material parameters.
                mat_idx = sphere_idx * 3
                
                mat = obj.material
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    # DiffuseLight (emissive)
                    emission = mat.emitted(0, 0, obj.center)
                    sphere_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    sphere_material_types[sphere_idx] = 2  # emissive
                elif isinstance(mat, Metal):
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        sphere_materials[mat_idx:mat_idx+3] = [
                            mat.texture.color.x, mat.texture.color.y, mat.texture.color.z
                        ]
                    sphere_material_types[sphere_idx] = 1  # metal
                elif isinstance(mat, Dielectric):
                    # For dielectrics, we store the refractive index in the red channel.
                    sphere_materials[mat_idx:mat_idx+3] = [mat.ref_idx, 0.0, 0.0]
                    sphere_material_types[sphere_idx] = 3  # dielectric
                else:
                    # Lambertian or other materials
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):  # SolidTexture
                            sphere_materials[mat_idx:mat_idx+3] = [
                                mat.texture.color.x, mat.texture.color.y, mat.texture.color.z
                            ]
                        else:  # Other texture types
                            # Sample center of texture for base color
                            color = mat.texture.sample(UV(0.5, 0.5))
                            sphere_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                    sphere_material_types[sphere_idx] = 0  # lambertian
                sphere_idx += 1
        
        # 5. Prepare arrays for triangles
        triangle_vertices = np.zeros((max(1, triangle_count), 9), dtype=np.float32)
        triangle_uvs = np.zeros((max(1, triangle_count), 6), dtype=np.float32)  # 6 values for 3 UV coordinates
        triangle_materials = np.zeros(max(1, triangle_count) * 3, dtype=np.float32)
        triangle_material_types = np.zeros(max(1, triangle_count), dtype=np.int32)
        
        # 6. Fill triangle arrays
        triangle_idx = 0
        for mesh in mesh_objects:
            for triangle in mesh.triangles:
                # Store vertices as before
                triangle_vertices[triangle_idx, 0:3] = [triangle.v0.x, triangle.v0.y, triangle.v0.z]
                triangle_vertices[triangle_idx, 3:6] = [triangle.v1.x, triangle.v1.y, triangle.v1.z]
                triangle_vertices[triangle_idx, 6:9] = [triangle.v2.x, triangle.v2.y, triangle.v2.z]
                
                # Store UV coordinates
                triangle_uvs[triangle_idx, 0:2] = [triangle.uv0.u, triangle.uv0.v]
                triangle_uvs[triangle_idx, 2:4] = [triangle.uv1.u, triangle.uv1.v]
                triangle_uvs[triangle_idx, 4:6] = [triangle.uv2.u, triangle.uv2.v]

                # Prepare the base index for color
                mat_idx = triangle_idx * 3
                mat = mesh.material
                
                # Check if emissive
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    emission = mat.emitted(0, 0, triangle.v0)  # pass any vertex or data
                    triangle_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    triangle_material_types[triangle_idx] = 2
                elif isinstance(mat, Metal):
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        triangle_materials[mat_idx:mat_idx+3] = [
                            mat.texture.color.x, mat.texture.color.y, mat.texture.color.z
                        ]
                    triangle_material_types[triangle_idx] = 1
                else:
                    # Lambertian or other materials
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):  # SolidTexture
                            triangle_materials[mat_idx:mat_idx+3] = [
                                mat.texture.color.x, mat.texture.color.y, mat.texture.color.z
                            ]
                        else:  # Other texture types
                            # Sample center of texture for base color
                            color = mat.texture.sample(UV(0.5, 0.5))
                            triangle_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                    triangle_material_types[triangle_idx] = 0

                triangle_idx += 1
        
        # 7. Transfer arrays to GPU
        self.d_sphere_centers = cuda.to_device(np.ascontiguousarray(centers))
        self.d_sphere_radii   = cuda.to_device(np.ascontiguousarray(radii))
        self.d_sphere_materials = cuda.to_device(np.ascontiguousarray(sphere_materials))
        self.d_sphere_material_types = cuda.to_device(np.ascontiguousarray(sphere_material_types))
        self.d_triangle_uvs = cuda.to_device(np.ascontiguousarray(triangle_uvs))
        self.d_triangle_vertices = cuda.to_device(np.ascontiguousarray(triangle_vertices))
        self.d_triangle_materials = cuda.to_device(np.ascontiguousarray(triangle_materials))
        self.d_triangle_material_types = cuda.to_device(np.ascontiguousarray(triangle_material_types))

        # Collect all unique textures
        textures = set()
        for obj in world.objects:
            if hasattr(obj, 'material') and hasattr(obj.material, 'texture'):
                textures.add(obj.material.texture)
        
        # Load textures to GPU
        self.load_textures(list(textures))

    def render_frame(self, camera, world) -> np.ndarray:
        if self.d_sphere_centers is None:
            self.update_scene_data(world)

        # Set up camera parameters
        camera_origin = np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
        camera_lower_left = np.array([camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z], dtype=np.float32)
        camera_horizontal = np.array([camera.horizontal.x, camera.horizontal.y, camera.horizontal.z], dtype=np.float32)
        camera_vertical = np.array([camera.vertical.x, camera.vertical.y, camera.vertical.z], dtype=np.float32)

        # Allocate output array
        frame_output = cuda.pinned_array((self.width, self.height, 3), dtype=np.float32)
        stream = cuda.stream()
        d_frame_output = cuda.to_device(frame_output, stream=stream)

        # Launch kernel with ALL parameters
        render_kernel[self.blockspergrid, self.threadsperblock, stream](
            self.width,
            self.height,
            camera_origin,
            camera_lower_left,
            camera_horizontal,
            camera_vertical,
            self.d_sphere_centers,
            self.d_sphere_radii,
            self.d_sphere_materials,
            self.d_sphere_material_types,
            self.d_triangle_vertices,
            self.d_triangle_uvs,
            self.d_triangle_materials,
            self.d_triangle_material_types,
            self.d_texture_data,
            self.texture_dimensions,
            self.d_texture_indices,
            d_frame_output,
            self.frame_number,
            self.rng_states
        )

        stream.synchronize()
        frame_output = d_frame_output.copy_to_host(stream=stream)
        stream.synchronize()

        self.accumulation_buffer += frame_output
        self.accumulation_buffer_sq += frame_output ** 2
        self.samples += 1

        averaged = self.accumulation_buffer / self.samples
        output = reinhard_tone_mapping(averaged, exposure=1.0, white_point=1.0, gamma=2.2)

        self.frame_number += 1
        return output

    def has_converged(self, threshold: float = 0.001) -> bool:
        """
        Computes the per‑pixel variance over the accumulation buffer.
        Returns True if the average variance is below the threshold.
        """
        if self.samples == 0:
            return False
        mean = self.accumulation_buffer / self.samples
        mean_sq = self.accumulation_buffer_sq / self.samples
        variance = mean_sq - mean ** 2
        avg_variance = np.mean(variance)
        return avg_variance < threshold
    
    def cleanup(self):
        """Clean up CUDA memory."""
        try:
            if self.d_sphere_centers is not None:
                del self.d_sphere_centers
                del self.d_sphere_radii
                del self.d_sphere_materials
                del self.d_sphere_material_types
                del self.d_triangle_vertices
                del self.d_triangle_uvs  # Add this
                del self.d_triangle_materials
                del self.d_triangle_material_types
                if self.d_texture_data is not None:
                    del self.d_texture_data
                    del self.d_texture_indices
            
            self.d_sphere_centers = None
            self.d_sphere_radii = None
            self.d_sphere_materials = None
            self.d_sphere_material_types = None
            self.d_triangle_vertices = None
            self.d_triangle_uvs = None  # Add this
            self.d_triangle_materials = None
            self.d_triangle_material_types = None
            self.d_texture_data = None
            self.d_texture_indices = None
            self.texture_dimensions = None
        except Exception as e:
            print(f"Warning: Error during CUDA cleanup: {str(e)}")