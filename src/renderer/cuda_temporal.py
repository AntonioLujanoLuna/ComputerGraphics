# File: renderer/cuda_temporal.py
from numba import cuda, float32
import math

@cuda.jit
def temporal_reprojection_kernel(width, height,
                                 d_prev_accum, d_prev_sample_count, d_prev_depth,
                                 # Previous camera parameters:
                                 prev_cam_origin, prev_cam_lower_left, prev_cam_horizontal, prev_cam_vertical,
                                 prev_cam_forward, prev_cam_right, prev_cam_up, prev_focus,
                                 # Current camera parameters:
                                 curr_cam_origin, curr_cam_lower_left, curr_cam_horizontal, curr_cam_vertical,
                                 curr_cam_forward, curr_cam_right, curr_cam_up, curr_focus,
                                 # Output buffers: accumulation and sample count (to be added into)
                                 d_new_accum, d_new_sample_count):
    x, y = cuda.grid(2)
    if x >= width or y >= height:
        return

    # Only proceed if there are history samples.
    prev_count = d_prev_sample_count[x, y]
    if prev_count <= 0:
        return

    # Get the previous depth for this pixel.
    depth = d_prev_depth[x, y]
    if depth <= 0:
        return

    # --- Back-project: Compute world-space position from the previous frame ---
    u_prev = float(x) / float(width - 1)
    v_prev = 1.0 - float(y) / float(height - 1)
    
    # Construct the previous ray direction: prev_lower_left + u_prev * prev_horizontal + v_prev * prev_vertical - prev_cam_origin
    prev_ray_dir = cuda.local.array(3, dtype=float32)
    for i in range(3):
        prev_ray_dir[i] = prev_cam_lower_left[i] + u_prev * prev_cam_horizontal[i] + v_prev * prev_cam_vertical[i] - prev_cam_origin[i]
    
    norm = math.sqrt(prev_ray_dir[0]**2 + prev_ray_dir[1]**2 + prev_ray_dir[2]**2)
    if norm > 0.0:
        for i in range(3):
            prev_ray_dir[i] /= norm
    
    # Compute world position P = prev_cam_origin + depth * prev_ray_dir.
    P = cuda.local.array(3, dtype=float32)
    for i in range(3):
        P[i] = prev_cam_origin[i] + depth * prev_ray_dir[i]
    
    # --- Reproject P into the current frame ---
    # Compute vector D from current camera origin to P.
    D = cuda.local.array(3, dtype=float32)
    for i in range(3):
        D[i] = P[i] - curr_cam_origin[i]
    
    dotD_forward = D[0]*curr_cam_forward[0] + D[1]*curr_cam_forward[1] + D[2]*curr_cam_forward[2]
    if dotD_forward <= 0.0:
        return  # The point is behind the current camera.
    
    # Find the intersection with the current image plane at distance curr_focus.
    t = curr_focus / dotD_forward
    P_proj = cuda.local.array(3, dtype=float32)
    for i in range(3):
        P_proj[i] = curr_cam_origin[i] + t * D[i]
    
    # Compute the center of the current image plane.
    center = cuda.local.array(3, dtype=float32)
    for i in range(3):
        center[i] = curr_cam_origin[i] + curr_cam_forward[i] * curr_focus

    # Compute the offset vector from the center to P_proj.
    offset = cuda.local.array(3, dtype=float32)
    for i in range(3):
        offset[i] = P_proj[i] - center[i]
    
    # Normalize the horizontal and vertical vectors.
    norm_h = math.sqrt(curr_cam_horizontal[0]**2 + curr_cam_horizontal[1]**2 + curr_cam_horizontal[2]**2)
    norm_v = math.sqrt(curr_cam_vertical[0]**2 + curr_cam_vertical[1]**2 + curr_cam_vertical[2]**2)
    h_norm = cuda.local.array(3, dtype=float32)
    v_norm = cuda.local.array(3, dtype=float32)
    for i in range(3):
        h_norm[i] = curr_cam_horizontal[i] / norm_h
        v_norm[i] = curr_cam_vertical[i] / norm_v
    
    dot_h = offset[0]*h_norm[0] + offset[1]*h_norm[1] + offset[2]*h_norm[2]
    dot_v = offset[0]*v_norm[0] + offset[1]*v_norm[1] + offset[2]*v_norm[2]
    
    # Map offset to normalized image coordinates.
    u_curr = 0.5 + dot_h / norm_h
    v_curr = 0.5 + dot_v / norm_v

    # Convert normalized coordinates into pixel indices.
    x_new = int(u_curr * (width - 1))
    y_new = int((1.0 - v_curr) * (height - 1))
    
    if x_new < 0 or x_new >= width or y_new < 0 or y_new >= height:
        return

    # --- Compute dynamic blending weight based on reprojection motion ---
    dx = float(x_new) - float(x)
    dy = float(y_new) - float(y)
    motion = math.sqrt(dx*dx + dy*dy)
    # Use a threshold (in pixels) to decide how much history to trust.
    threshold = 2.0  # For example, if motion is 2 pixels or less, trust history fully.
    dynamic_alpha = 1.0 - min(motion / threshold, 1.0)
    # Optionally clamp to a minimum value, so that history is not completely discarded.
    if dynamic_alpha < 0.2:
        dynamic_alpha = 0.2

    # --- Blend the history from the previous frame into the new buffers ---
    prev_color_r = d_prev_accum[x, y, 0]
    prev_color_g = d_prev_accum[x, y, 1]
    prev_color_b = d_prev_accum[x, y, 2]
    
    cuda.atomic.add(d_new_accum, (x_new, y_new, 0), prev_color_r * dynamic_alpha)
    cuda.atomic.add(d_new_accum, (x_new, y_new, 1), prev_color_g * dynamic_alpha)
    cuda.atomic.add(d_new_accum, (x_new, y_new, 2), prev_color_b * dynamic_alpha)
    cuda.atomic.add(d_new_sample_count, (x_new, y_new), int(prev_count * dynamic_alpha))
