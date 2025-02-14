# renderer/cuda_utils.py

from numba import cuda, float32
import math
from numba.cuda.random import xoroshiro128p_uniform_float32
from .cuda_env import env_dir_to_uv
import numpy as np

INFINITY = 1e20
EPSILON = 1e-8

# Maximum number of precomputed Halton samples.
MAX_HALTON_SAMPLES = 4096

# -------------------------------------------------------------------------
# Original Halton function (retained as a fallback)
@cuda.jit(device=True)
def halton(index, base):
    """
    Compute the Halton sequence value for a given index and base.
    """
    f = 1.0
    r = 0.0
    while index > 0:
        f = f / base
        r = r + f * (index % base)
        index = index // base
    return r

# Increase the number of precomputed Halton tables by adding one for base 5.
def halton_host(index, base):
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

def precompute_halton_tables(max_samples):
    table_base2 = np.empty(max_samples, dtype=np.float32)
    table_base3 = np.empty(max_samples, dtype=np.float32)
    table_base5 = np.empty(max_samples, dtype=np.float32)  # new table for base 5
    for i in range(max_samples):
        table_base2[i] = halton_host(i, 2)
        table_base3[i] = halton_host(i, 3)
        table_base5[i] = halton_host(i, 5)
    return table_base2, table_base3, table_base5

# Modify the halton_cached function to use the extra table.
@cuda.jit(device=True)
def halton_cached(index, base, halton_table_base2, halton_table_base3, halton_table_base5):
    if index < MAX_HALTON_SAMPLES:
        if base == 2:
            return halton_table_base2[index]
        elif base == 3:
            return halton_table_base3[index]
        elif base == 5:
            return halton_table_base5[index]
    # Fallback: compute on the fly.
    f = 1.0
    r = 0.0
    while index > 0:
        f /= base
        r += f * (index % base)
        index //= base
    return r

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
def sample_cosine_hemisphere(rng_states, thread_id, normal, out_dir, out_pdf):
    """
    Cosine-weighted sampling over the hemisphere oriented by 'normal'.
    Uses two uniform random numbers.
    """
    u1 = xoroshiro128p_uniform_float32(rng_states, thread_id)
    u2 = xoroshiro128p_uniform_float32(rng_states, thread_id)
    r = math.sqrt(u1)
    theta = 2.0 * math.pi * u2
    x = r * math.cos(theta)
    y = r * math.sin(theta)
    z = math.sqrt(max(0.0, 1.0 - u1))
    # Build an orthonormal basis (normal, tangent, bitangent)
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
    # Transform local sample to world space:
    for i in range(3):
        out_dir[i] = x * tangent[i] + y * bitangent[i] + z * normal[i]
    normalize_inplace(out_dir)
    # The PDF for cosine-weighted hemisphere is: pdf = cos(theta)/pi.
    out_pdf_val = max(0.0, dot(normal, out_dir)) / math.pi
    return out_pdf_val

@cuda.jit(device=True)
def compute_env_pdf(direction, env_map, cdf, total, width, height):
    """
    Compute the PDF (per solid angle) for a given direction using the environment CDF.
    For a proper implementation, you would evaluate the pixel weight (luminance * sin(theta))
    at the corresponding pixel. Here we provide an approximate placeholder.
    """
    # Convert direction to UV:
    uv = cuda.local.array(2, dtype=float32)
    env_dir_to_uv(direction, uv)
    u = uv[0]
    v = uv[1]
    ix = int(u * (width - 1))
    iy = int(v * (height - 1))
    # Placeholder: assume constant weight 1.0 per pixel.
    weight = 1.0
    pixel_solid_angle = 4.0 * math.pi / (width * height)
    pdf = (weight * pixel_solid_angle) / total
    return pdf

@cuda.jit(device=True)
def mis_power_heuristic(pdf_a, pdf_b):
    """
    Compute the MIS weight using the power heuristic.
    """
    a2 = pdf_a * pdf_a
    b2 = pdf_b * pdf_b
    return a2 / (a2 + b2 + 1e-8)