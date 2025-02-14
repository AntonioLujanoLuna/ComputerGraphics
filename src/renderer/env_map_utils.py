# renderer/env_map_utils.py

import numpy as np

def generate_gradient_env_map(width=512, height=256):
    """
    Generate a gradient environment map.
    Interpolates vertically between a zenith color and a horizon color.
    
    Args:
        width (int): The width of the generated environment map.
        height (int): The height of the generated environment map.
        
    Returns:
        np.ndarray: A (height x width x 3) array in float32 (values in [0,1]).
    """
    env_map = np.zeros((height, width, 3), dtype=np.float32)
    # Define the zenith (top) and horizon (bottom) colors.
    zenith_color = np.array([0.2, 0.4, 0.8], dtype=np.float32)   # Deep blue sky
    horizon_color = np.array([1.0, 0.8, 0.6], dtype=np.float32)    # Warm light near horizon

    for y in range(height):
        t = y / (height - 1)  # t goes from 0 at the top to 1 at the bottom
        # Interpolate between zenith and horizon colors.
        row_color = (1.0 - t) * zenith_color + t * horizon_color
        env_map[y, :, :] = row_color  # assign the same row color to all x

    return env_map