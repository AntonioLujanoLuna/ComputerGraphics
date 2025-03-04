# renderer/tone_mapping.py
from numba import cuda

def reinhard_tone_mapping(accumulated, exposure=1.0, white_point=1.0, gamma=2.2):
    """
    Apply Reinhard tone mapping to a linear radiance image.
    """
    scaled = accumulated * exposure
    mapped = scaled / (1.0 + scaled / white_point)
    mapped = mapped ** (1.0 / gamma)
    output = (mapped * 255).clip(0, 255).astype("uint8")
    return output

def auto_exposure_tone_mapping(accumulated, gamma=2.2, target_midgray=0.18):
    """
    Compute an exposure value based on the average scene luminance and then
    apply Reinhard tone mapping.
    """
    # Compute per-pixel luminance using standard coefficients.
    luminance = 0.2126 * accumulated[:,:,0] + 0.7152 * accumulated[:,:,1] + 0.0722 * accumulated[:,:,2]
    avg_lum = luminance.mean() + 1e-5  # avoid division by zero
    exposure = target_midgray / avg_lum
    return reinhard_tone_mapping(accumulated, exposure=exposure, white_point=1.0, gamma=gamma)

@cuda.jit
def tone_mapping_kernel(linear_image, output_image, exposure, white_point, gamma):
    x, y = cuda.grid(2)
    if x < output_image.shape[0] and y < output_image.shape[1]:
        # Apply Reinhard tone mapping
        r = linear_image[x, y, 0] * exposure
        g = linear_image[x, y, 1] * exposure
        b = linear_image[x, y, 2] * exposure
        
        r = r / (1.0 + r / white_point)
        g = g / (1.0 + g / white_point)
        b = b / (1.0 + b / white_point)
        
        # Apply gamma correction
        r = r ** (1.0 / gamma)
        g = g ** (1.0 / gamma)
        b = b ** (1.0 / gamma)
        
        # Convert to 8-bit
        output_image[x, y, 0] = min(255, max(0, int(r * 255)))
        output_image[x, y, 1] = min(255, max(0, int(g * 255)))
        output_image[x, y, 2] = min(255, max(0, int(b * 255)))
