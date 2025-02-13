# renderer/cuda_bsdf.py

from numba import cuda, float32
import math
from .cuda_materials import scatter_dielectric, scatter_lambertian, scatter_metal
from .cuda_utils import dot
from .cuda_geometry import ray_sphere_intersect, ray_triangle_intersect


@cuda.jit(device=True)
def bsdf_sample(mat_type, incoming, normal, mat_color, ior, rng_states, thread_id, out_scatter):
    """
    Sample the BSDF based on material type.
    
    Parameters:
      mat_type: Material type identifier (0: Lambertian, 1: Metal, 3: Dielectric)
      incoming: 3-element array for the incoming ray direction.
      normal:   3-element array for the surface normal at the hit point.
      mat_color: 3-element array representing the material’s color or albedo.
      ior:       Index of refraction (for dielectrics).
      rng_states, thread_id: Random number generator state and thread index.
      out_scatter: Output 3-element array for the sampled direction.
      
    Returns:
      True if scattering occurred, False otherwise.
    """
    if mat_type == 1:
        return scatter_metal(incoming, normal, mat_color, rng_states, thread_id, out_scatter)
    elif mat_type == 3:
        return scatter_dielectric(incoming, normal, ior, rng_states, thread_id, out_scatter)
    else:
        # Default to Lambertian scattering.
        return scatter_lambertian(normal, rng_states, thread_id, out_scatter)

@cuda.jit(device=True)
def bsdf_pdf(mat_type, incoming, normal, out_dir, ior):
    """
    Compute the probability density for the BSDF sample.
    
    For a Lambertian (diffuse) surface, the PDF is max(dot(normal, out_dir), 0)/pi.
    For specular materials (Metal, Dielectric), which are delta distributions,
    we return a very small epsilon value.
    """
    cos_theta = dot(normal, out_dir)
    if mat_type == 0:  # Lambertian
        return max(cos_theta, 0.0) / math.pi
    else:
        return 1e-8  # Delta function approximation for specular reflection

@cuda.jit(device=True)
def scene_occlusion_test(hit_point, light_dir, sphere_centers, sphere_radii, triangle_vertices):
    """
    Test whether a ray cast from hit_point along light_dir is occluded by any scene geometry.
    
    Returns:
      The intersection distance if an occluder is found,
      or -1.0 if no occlusion is detected.
    """
    t_min = float32(1e-4)
    t_max = float32(1e20)
    
    # Check sphere intersections.
    for s in range(sphere_centers.shape[0]):
        t = ray_sphere_intersect(hit_point, light_dir, sphere_centers[s],
                                 sphere_radii[s], t_min, t_max)
        if t > 0.0 and t < t_max:
            return t
    # Check triangle intersections.
    uv_dummy = cuda.local.array(2, dtype=float32)
    for t in range(triangle_vertices.shape[0]):
        t_val = ray_triangle_intersect(hit_point, light_dir,
                                       triangle_vertices[t, 0:3],
                                       triangle_vertices[t, 3:6],
                                       triangle_vertices[t, 6:9],
                                       t_min, t_max, uv_dummy)
        if t_val > 0.0 and t_val < t_max:
            return t_val
    return -1.0

@cuda.jit(device=True)
def update_scatter_attenuation(mat_type, mat_color, attenuation):
    """
    Update the ray’s attenuation after a scattering event.
    
    For Lambertian or metal surfaces, the attenuation is multiplied by the material color.
    For dielectric materials (e.g. glass), we assume no attenuation in this simple model.
    """
    if mat_type == 3:
        # Dielectric: typically, the ray is not attenuated.
        return
    else:
        for i in range(3):
            attenuation[i] *= mat_color[i]
