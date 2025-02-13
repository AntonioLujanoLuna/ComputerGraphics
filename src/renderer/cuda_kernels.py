# renderer/cuda_kernels.py

from numba import cuda, float32
import math
from numba.cuda.random import xoroshiro128p_uniform_float32
from .cuda_utils import dot, normalize_inplace, halton, compute_env_pdf, mis_power_heuristic, sample_cosine_hemisphere
from .cuda_geometry import (
    ray_sphere_intersect, calculate_sphere_uv, 
    ray_triangle_intersect, compute_triangle_normal
)
from .cuda_env import eval_env_map, sample_env_importance
from .cuda_bsdf import bsdf_pdf, bsdf_sample, update_scatter_attenuation, scene_occlusion_test
from .cuda_materials import compute_direct_light

INFINITY = 1e20
EPSILON = 1e-20
MAX_BOUNCES = 16

@cuda.jit
def adaptive_render_kernel(width, height,
                           camera_origin, camera_lower_left,
                           camera_horizontal, camera_vertical,
                           sphere_centers, sphere_radii,
                           sphere_materials, sphere_material_types,
                           triangle_vertices, triangle_uvs,
                           triangle_materials, triangle_material_types,
                           texture_data, texture_dimensions, texture_indices,
                           d_accum_buffer, d_accum_buffer_sq,
                           d_sample_count, d_mask, d_frame_output,
                           frame_number, rng_states, N,
                           d_env_map, d_env_cdf, env_total, env_width, env_height):
    """
    An adaptive path tracing kernel that:
      1) Skips already converged pixels (d_mask[x,y]==1)
      2) Does multiple samples per pixel
      3) Calls trace_ray(...) for each sample
      4) Accumulates color in d_accum_buffer
      5) Writes the final average to d_frame_output

    Added parameters for environment sampling and MIS:
      d_env_map, d_env_cdf, env_total, env_width, env_height
    """
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # If this pixel is marked converged, just output the existing average color.
    if d_mask[x, y] == 1:
        count = d_sample_count[x, y]
        if count > 0:
            avg_r = d_accum_buffer[x, y, 0] / count
            avg_g = d_accum_buffer[x, y, 1] / count
            avg_b = d_accum_buffer[x, y, 2] / count
            d_frame_output[x, y, 0] = avg_r
            d_frame_output[x, y, 1] = avg_g
            d_frame_output[x, y, 2] = avg_b
        else:
            d_frame_output[x, y, 0] = 0.0
            d_frame_output[x, y, 1] = 0.0
            d_frame_output[x, y, 2] = 0.0
        return

    # Accumulate color for the sub-samples in local memory.
    local_color = cuda.local.array(3, dtype=float32)
    for i in range(3):
        local_color[i] = 0.0

    pixel_idx = x + y * width

    # Loop over sub-samples
    for s in range(N):
        seq_idx = frame_number * N + s

        # Example: use simple random or Halton-based jitter
        # Halton is available from your posted halton() function.
        jitter_x = halton(pixel_idx * 4096 + seq_idx, 2) / width
        jitter_y = halton(pixel_idx * 4096 + seq_idx, 3) / height

        u = (float32(x) + jitter_x) / float32(width - 1)
        v = 1.0 - ((float32(y) + jitter_y) / float32(height - 1))

        # Build the ray from camera parameters
        ray_origin = cuda.local.array(3, dtype=float32)
        ray_dir = cuda.local.array(3, dtype=float32)
        for i in range(3):
            ray_origin[i] = camera_origin[i]
            ray_dir[i] = (camera_lower_left[i]
                          + u * camera_horizontal[i]
                          + v * camera_vertical[i]
                          - camera_origin[i])
        normalize_inplace(ray_dir)

        # Trace the ray
        sample_col = cuda.local.array(3, dtype=float32)
        for i in range(3):
            sample_col[i] = 0.0

        trace_ray(ray_origin, ray_dir, MAX_BOUNCES,
                  sphere_centers, sphere_radii,
                  sphere_materials, sphere_material_types,
                  triangle_vertices, triangle_uvs,
                  triangle_materials, triangle_material_types,
                  texture_data, texture_dimensions, texture_indices,
                  sample_col,
                  rng_states, pixel_idx*N + s,  # pass unique seed per sample
                  d_env_map, d_env_cdf, env_total, env_width, env_height)

        # Accumulate into local_color
        for i in range(3):
            local_color[i] += sample_col[i]

    # Average over N sub-samples
    invN = 1.0 / float32(N)
    for i in range(3):
        local_color[i] *= invN

    # Atomically add to the accumulation buffers
    cuda.atomic.add(d_accum_buffer, (x, y, 0), local_color[0])
    cuda.atomic.add(d_accum_buffer, (x, y, 1), local_color[1])
    cuda.atomic.add(d_accum_buffer, (x, y, 2), local_color[2])

    cuda.atomic.add(d_accum_buffer_sq, (x, y, 0), local_color[0]*local_color[0])
    cuda.atomic.add(d_accum_buffer_sq, (x, y, 1), local_color[1]*local_color[1])
    cuda.atomic.add(d_accum_buffer_sq, (x, y, 2), local_color[2]*local_color[2])

    cuda.atomic.add(d_sample_count, (x, y), N)

    # Compute final average color = accum / sample_count
    count_after = d_sample_count[x, y]
    if count_after > 0:
        avg_r = d_accum_buffer[x, y, 0] / float32(count_after)
        avg_g = d_accum_buffer[x, y, 1] / float32(count_after)
        avg_b = d_accum_buffer[x, y, 2] / float32(count_after)
        d_frame_output[x, y, 0] = avg_r
        d_frame_output[x, y, 1] = avg_g
        d_frame_output[x, y, 2] = avg_b
    else:
        d_frame_output[x, y, 0] = 0.0
        d_frame_output[x, y, 1] = 0.0
        d_frame_output[x, y, 2] = 0.0

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

from numba import cuda, float32

@cuda.jit
def compute_variance_mask_kernel(d_accum_buffer, d_accum_buffer_sq, d_sample_count, d_mask, threshold, min_samples):
    """
    For each pixel (x, y), compute the per-channel means and variances.
    If the sample count is below 'min_samples' then mark the pixel as not converged.
    Otherwise, if the average variance is below the 'threshold', mark the pixel as converged.
    """
    x, y = cuda.grid(2)
    width = d_accum_buffer.shape[0]
    height = d_accum_buffer.shape[1]
    if x < width and y < height:
        count = d_sample_count[x, y]
        # Only mark a pixel as converged if we've taken enough samples.
        if count < min_samples:
            d_mask[x, y] = 0
            return

        mean_r = d_accum_buffer[x, y, 0] / count
        mean_g = d_accum_buffer[x, y, 1] / count
        mean_b = d_accum_buffer[x, y, 2] / count

        mean_sq_r = d_accum_buffer_sq[x, y, 0] / count
        mean_sq_g = d_accum_buffer_sq[x, y, 1] / count
        mean_sq_b = d_accum_buffer_sq[x, y, 2] / count

        var_r = mean_sq_r - mean_r * mean_r
        var_g = mean_sq_g - mean_g * mean_g
        var_b = mean_sq_b - mean_b * mean_b

        avg_var = (var_r + var_g + var_b) / 3.0

        if avg_var < threshold:
            d_mask[x, y] = 1
        else:
            d_mask[x, y] = 0

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
def trace_ray(origin, direction, max_bounces,
              sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
              triangle_vertices, triangle_uvs, triangle_materials, triangle_material_types,
              texture_data, texture_dimensions, texture_indices,
              out_color,
              rng_states, thread_id,
              d_env_map, d_env_cdf, env_total, env_width, env_height):
    """
    Path tracer that does:
      - Sphere & triangle intersection
      - Emissive materials
      - Basic next-event from sphere lights
      - *NEW*: Environment importance sampling with MIS

    Parameters:
      origin, direction: the primary ray
      max_bounces: maximum number of path bounces
      ... [existing geometry & material params] ...
      out_color: 3-float local array to accumulate result
      rng_states, thread_id: random states for the current thread
      d_env_map, d_env_cdf, env_total, env_width, env_height: environment data
    """

    # Local copies
    current_origin = cuda.local.array(3, dtype=float32)
    current_dir    = cuda.local.array(3, dtype=float32)
    attenuation    = cuda.local.array(3, dtype=float32)
    color_accum    = cuda.local.array(3, dtype=float32)

    for i in range(3):
        current_origin[i] = origin[i]
        current_dir[i]    = direction[i]
        attenuation[i]    = 1.0
        color_accum[i]    = 0.0

    # For storing barycentric coords for triangle hits
    uv_temp = cuda.local.array(2, dtype=float32)

    bounce = 0
    while bounce < max_bounces:
        bounce += 1

        # --- Find the closest intersection in the scene ---
        closest_t = float32(1e20)
        hit_idx   = -1
        hit_sphere = True

        # Intersect spheres
        s_count = sphere_centers.shape[0]
        t_count = triangle_vertices.shape[0]

        tmp_hit_point = cuda.local.array(3, dtype=float32)
        hit_normal = cuda.local.array(3, dtype=float32)

        # Track some UV for texturing
        uv_coords = cuda.local.array(2, dtype=float32)
        uv_coords[0] = 0.0
        uv_coords[1] = 0.0

        # Sphere intersection
        for s_i in range(s_count):
            tval = ray_sphere_intersect(
                current_origin, current_dir,
                sphere_centers[s_i], sphere_radii[s_i],
                float32(1e-4), closest_t
            )
            if tval > 0.0:
                closest_t = tval
                hit_idx   = s_i
                hit_sphere = True

        # Triangle intersection
        for t_i in range(t_count):
            tval = ray_triangle_intersect(
                current_origin, current_dir,
                triangle_vertices[t_i, 0:3],
                triangle_vertices[t_i, 3:6],
                triangle_vertices[t_i, 6:9],
                float32(1e-4), closest_t, uv_temp
            )
            if tval > 0.0:
                closest_t = tval
                hit_idx   = t_i
                hit_sphere = False
                uv_coords[0] = uv_temp[0]
                uv_coords[1] = uv_temp[1]

        # --- If no intersection => sample environment and terminate ---
        if hit_idx < 0:
            # Evaluate environment map in the direction current_dir
            temp_env = cuda.local.array(3, dtype=float32)
            eval_env_map(current_dir, d_env_map, env_width, env_height, temp_env)

            for i in range(3):
                color_accum[i] += attenuation[i] * temp_env[i]
            break  # no more bounces

        # --- We have a valid hit. Compute hit point and normal. ---
        for i in range(3):
            tmp_hit_point[i] = current_origin[i] + closest_t * current_dir[i]

        mat_type = 0
        mat_color = cuda.local.array(3, dtype=float32)
        for i in range(3):
            mat_color[i] = 0.0

        if hit_sphere:
            # Spheres
            # fetch sphere normal
            center = sphere_centers[hit_idx]
            radius = sphere_radii[hit_idx]

            for i in range(3):
                hit_normal[i] = (tmp_hit_point[i] - center[i]) / radius

            # Flip normal if we are inside
            if dot(current_dir, hit_normal) > 0.0:
                for i in range(3):
                    hit_normal[i] = -hit_normal[i]

            mat_type = sphere_material_types[hit_idx]
            base_idx = hit_idx * 3

            # If you want texture: check texture_indices[hit_idx]
            # or just store a base color
            for i in range(3):
                mat_color[i] = sphere_materials[base_idx + i]

            # Optionally compute sphere UV (already done if you want)
            calculate_sphere_uv(tmp_hit_point, center, uv_coords)

        else:
            # Triangles
            compute_triangle_normal(triangle_vertices[hit_idx], hit_normal)
            # Flip if needed
            if dot(current_dir, hit_normal) > 0.0:
                for i in range(3):
                    hit_normal[i] = -hit_normal[i]
            mat_type = triangle_material_types[hit_idx]
            base_idx = hit_idx * 3
            for i in range(3):
                mat_color[i] = triangle_materials[base_idx + i]

        normalize_inplace(hit_normal)

        # --- Check for Emissive material ---
        if mat_type == 2:
            # Add emissive color and terminate
            for i in range(3):
                color_accum[i] += attenuation[i] * mat_color[i]
            break

        # If it's Dielectric, we store IOR in mat_color[0]
        ior_val = mat_color[0] if (mat_type == 3) else 1.0

        # --- Next-Event Estimation from local lights (optional) ---
        # Allocate a local array to hold the direct lighting contribution.
        temp_light = cuda.local.array(3, dtype=float32)
        for i in range(3):
            temp_light[i] = 0.0

        # Call compute_direct_light; make sure the parameters match your implementation.
        compute_direct_light(tmp_hit_point, hit_normal,
                            sphere_centers, sphere_radii, sphere_materials, sphere_material_types,
                            triangle_vertices, triangle_materials, triangle_material_types,
                            rng_states, thread_id, temp_light)

        # Add the computed direct lighting contribution to the accumulated color.
        for i in range(3):
            color_accum[i] += attenuation[i] * temp_light[i]

        # --- Next-Event from Environment (MIS sample) ---
        if env_total > 1e-8:
            # 1) Sample environment direction
            env_dir = cuda.local.array(3, dtype=float32)
            env_pdf = cuda.local.array(1, dtype=float32)
            sample_env_importance(rng_states, thread_id, d_env_cdf, env_total,
                                  env_width, env_height, env_dir, env_pdf)

            cosN = dot(env_dir, hit_normal)
            if cosN > 0.0 and env_pdf[0] > 1e-8:
                # Check occlusion by shooting a shadow ray:
                shadow_dist = scene_occlusion_test(tmp_hit_point, env_dir,
                                                   sphere_centers, sphere_radii,
                                                   triangle_vertices)
                if shadow_dist < 0.0:
                    # Not occluded
                    # Evaluate environment color
                    env_L = cuda.local.array(3, dtype=float32)
                    eval_env_map(env_dir, d_env_map, env_width, env_height, env_L)

                    # Evaluate BSDF pdf for the same direction
                    pdf_bsdf = bsdf_pdf(mat_type, current_dir, hit_normal, env_dir, ior_val)

                    # MIS weight: env_pdf -> "pLight", pdf_bsdf -> "pBSDF"
                    w = mis_power_heuristic(env_pdf[0], pdf_bsdf)

                    # geometric factor for Lambertian or so is cosN
                    # So contribution = (env_L * cosN / env_pdf[0]) * w
                    for i in range(3):
                        color_accum[i] += attenuation[i] * env_L[i] * (cosN / env_pdf[0]) * w

        # --- Now do the standard BSDF sampling of the next bounce ---
        scatter_dir = cuda.local.array(3, dtype=float32)
        valid_scatter = bsdf_sample(mat_type, current_dir, hit_normal,
                                    mat_color, ior_val,
                                    rng_states, thread_id,
                                    scatter_dir)
        if not valid_scatter:
            # e.g. was fully absorbed
            break

        # Evaluate pdf for the scattered direction:
        pdf_bsdf = bsdf_pdf(mat_type, current_dir, hit_normal, scatter_dir, ior_val)

        # Evaluate environment pdf for the same scatter_dir, if environment is present
        pdf_env = 0.0
        if env_total > 1e-8:
            pdf_env = compute_env_pdf(scatter_dir, d_env_map, d_env_cdf,
                                      env_total, env_width, env_height)

        # MIS weight for this direction (the one we got from the BSDF sample)
        #   "pBSDF" = pdf_bsdf, "pLight" = pdf_env
        w_bsdf = mis_power_heuristic(pdf_bsdf, pdf_env)

        # If this new direction sees the environment (and is above the surface):
        cosN2 = dot(scatter_dir, hit_normal)
        if cosN2 > 1e-8:
            # Check for occlusion
            shadow_dist = scene_occlusion_test(tmp_hit_point, scatter_dir,
                                               sphere_centers, sphere_radii,
                                               triangle_vertices)
            if shadow_dist < 0.0:
                # Not occluded => environment visible
                env_col = cuda.local.array(3, dtype=float32)
                eval_env_map(scatter_dir, d_env_map, env_width, env_height, env_col)

                # Add contribution from environment, multiplied by MIS weight
                # scaled by cosN2 / pdf_bsdf, times attenuation
                # (Lambert = albedo/pi, etc. included in "attenuation * mat_color"?)
                # The portion that belongs to the direct bounce
                scale_val = (cosN2 / max(pdf_bsdf, 1e-8)) * w_bsdf
                for i in range(3):
                    color_accum[i] += attenuation[i] * env_col[i] * scale_val

        # Update the path for next bounce
        # Multiply attenuation by the material's albedo or reflectivity:
        # (The sampling routine might do that, or do it here.)
        # For example, if Lambertian:  attenuation *= mat_color
        # If Metal:    attenuation *= (some reflection color)
        # If Dielectric: attenuation is always Vector3(1,1,1)...

        update_scatter_attenuation(mat_type, mat_color, attenuation)

        # Move ray origin forward a bit
        for i in range(3):
            current_origin[i] = tmp_hit_point[i]
            current_dir[i]    = scatter_dir[i]

        # --- Russian Roulette ---
        if bounce >= 3:
            lum = 0.2126*attenuation[0] + 0.7152*attenuation[1] + 0.0722*attenuation[2]
            rr_prob = min(max(lum, 0.05), 0.95)
            if xoroshiro128p_uniform_float32(rng_states, thread_id) > rr_prob:
                break
            invp = 1.0 / rr_prob
            for i in range(3):
                attenuation[i] *= invp

    # Write final color
    for i in range(3):
        out_color[i] = color_accum[i]

