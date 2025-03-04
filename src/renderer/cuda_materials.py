# renderer/cuda_materials.py

from numba import cuda, float32
from .cuda_utils import dot, normalize_inplace, random_in_unit_sphere, cross_inplace
from numba.cuda.random import xoroshiro128p_uniform_float32
import math

INFINITY = 1e20
EPSILON = 1e-8

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
def scatter_microfacet_metal(incident, normal, albedo, roughness, rng_states, thread_id, out_scattered):
    """Compute microfacet metal scattering."""
    # Compute a half-vector via importance-sampling
    half_vector = cuda.local.array(3, dtype=float32)
    
    # Sample a direction with GGX distribution
    # Simplified implementation - can be improved with true GGX sampling
    a2 = roughness * roughness
    
    # Importance sample based on roughness
    phi = 2.0 * math.pi * xoroshiro128p_uniform_float32(rng_states, thread_id)
    cos_theta = math.sqrt((1.0 - xoroshiro128p_uniform_float32(rng_states, thread_id)) / 
                         (1.0 + (a2 - 1.0) * xoroshiro128p_uniform_float32(rng_states, thread_id)))
    sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
    
    # Compute local coords
    x = sin_theta * math.cos(phi)
    y = sin_theta * math.sin(phi)
    z = cos_theta
    
    # Transform to world space (simplistic version)
    # First create an arbitrary tangent
    tangent = cuda.local.array(3, dtype=float32)
    bitangent = cuda.local.array(3, dtype=float32)
    
    if abs(normal[0]) > 0.1:
        tangent[0] = normal[1]
        tangent[1] = -normal[0]
        tangent[2] = 0.0
    else:
        tangent[0] = 0.0
        tangent[1] = normal[2]
        tangent[2] = -normal[1]
    
    normalize_inplace(tangent)
    cross_inplace(bitangent, normal, tangent)
    
    for i in range(3):
        half_vector[i] = x * tangent[i] + y * bitangent[i] + z * normal[i]
    
    normalize_inplace(half_vector)
    
    # Reflect incident ray around half vector
    dot_i_h = dot(incident, half_vector)
    for i in range(3):
        out_scattered[i] = incident[i] - 2.0 * dot_i_h * half_vector[i]
        
    normalize_inplace(out_scattered)
    
    return dot(out_scattered, normal) > 0

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
