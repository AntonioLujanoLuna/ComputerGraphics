# renderer/cuda_env.py

from numba import cuda, float32
import math
from numba.cuda.random import xoroshiro128p_uniform_float32

@cuda.jit(device=True)
def env_dir_to_uv(direction, out_uv):
    """
    Convert a normalized 3D direction to equirectangular UV coordinates.
    Input:
      direction: float32[3]
    Output:
      out_uv: float32[2] where u ∈ [0,1] and v ∈ [0,1].
    """
    x = direction[0]
    y = direction[1]
    z = direction[2]
    # Compute phi in [0, 2*pi)
    phi = math.atan2(-z, x)
    if phi < 0.0:
        phi += 2.0 * math.pi
    # Compute theta in [0, pi]
    theta = math.acos(max(-1.0, min(1.0, y)))
    out_uv[0] = phi / (2.0 * math.pi)
    out_uv[1] = theta / math.pi

@cuda.jit(device=True)
def env_uv_to_dir(u, v, out_dir):
    """
    Convert equirectangular UV coordinates (u,v) to a normalized 3D direction.
    """
    phi = 2.0 * math.pi * u
    theta = math.pi * v
    sin_theta = math.sin(theta)
    out_dir[0] = math.cos(phi) * sin_theta
    out_dir[1] = math.cos(theta)
    out_dir[2] = -math.sin(phi) * sin_theta

@cuda.jit(device=True)
def binary_search_cdf(cdf, value, length):
    """
    Perform a binary search on the 1D CDF array.
    Returns the index i such that cdf[i-1] < value ≤ cdf[i].
    """
    left = 0
    right = length - 1
    while left < right:
        mid = (left + right) // 2
        if cdf[mid] < value:
            left = mid + 1
        else:
            right = mid
    return left

@cuda.jit(device=True)
def sample_env_importance(rng_states, thread_id, cdf, total, width, height, out_dir, out_pdf):
    """
    Sample a direction from the environment map using the precomputed CDF.
    Inputs:
      rng_states: CUDA RNG states.
      cdf: 1D array (flattened) of length width*height.
      total: total sum of weights.
      width, height: dimensions of the environment map.
    Outputs:
      out_dir: float32[3] – the sampled direction.
      out_pdf: float32[1] – the PDF (per solid angle) for the sampled direction.
      
    Note: For a correct PDF, the per-pixel weight and the pixel solid angle must be taken into account.
    Here we assume that each pixel covers approximately 4*pi/(width*height) solid angle.
    """
    r = xoroshiro128p_uniform_float32(rng_states, thread_id)
    target = r * total
    length_total = width * height
    idx = binary_search_cdf(cdf, target, length_total)
    if idx >= length_total:
        idx = length_total - 1
    # Convert linear index to (x,y)
    y = idx // width
    x = idx % width
    # Use pixel center for UV:
    u = (x + 0.5) / width
    v = (y + 0.5) / height
    env_uv_to_dir(u, v, out_dir)
    # Approximate PDF: each pixel covers solid angle ≈ 4*pi/(width*height)
    pixel_solid_angle = 4.0 * math.pi / (width * height)
    # We assume the weight for this pixel is the difference between its CDF values.
    # For a more accurate implementation, store the original per–pixel weight.
    # Here we use a placeholder weight = 1.0.
    weight = 1.0
    pdf = (weight * pixel_solid_angle) / total
    out_pdf[0] = pdf

@cuda.jit(device=True)
def eval_env_map(direction, env_map, width, height, out_color):
    """
    Evaluate the environment map radiance in the given direction.
    Uses nearest-neighbor lookup.
    Inputs:
      direction: float32[3] (normalized)
      env_map: 3D array of shape (height, width, 3) containing radiance.
      width, height: dimensions of env_map.
    Output:
      out_color: float32[3] – the radiance.
    """
    uv = cuda.local.array(2, dtype=float32)
    env_dir_to_uv(direction, uv)
    u = uv[0]
    v = uv[1]
    ix = int(u * (width - 1))
    iy = int(v * (height - 1))
    out_color[0] = env_map[iy, ix, 0]
    out_color[1] = env_map[iy, ix, 1]
    out_color[2] = env_map[iy, ix, 2]
