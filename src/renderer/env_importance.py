# renderer/env_importance.py

import math
import numpy as np

def build_env_cdf(env_map):
    """
    Build a cumulative distribution function (CDF) for environment map importance sampling.
    The environment map is assumed to be a NumPy array of shape (height, width, 3) in linear radiance.
    Each pixel is weighted by its luminance and by sin(theta) (to account for the spherical area element).
    
    Returns:
      cdf: a flattened 1D np.float32 array (length = width*height) that is the prefix sum of weights.
      total: the total weight (a float).
      width, height: dimensions of the env_map.
    """
    H, W, _ = env_map.shape
    luminance = np.zeros((H, W), dtype=np.float64)
    for y in range(H):
        # theta from 0 to pi; use the pixel’s center (y+0.5)
        theta = math.pi * (y + 0.5) / H
        sin_theta = math.sin(theta)
        for x in range(W):
            r, g, b = env_map[y, x]
            # Standard luminance: 0.2126*r + 0.7152*g + 0.0722*b
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            luminance[y, x] = lum * sin_theta
    lum_flat = luminance.ravel()
    cdf = np.cumsum(lum_flat)
    total = cdf[-1]
    return cdf.astype(np.float32), float(total), W, H
