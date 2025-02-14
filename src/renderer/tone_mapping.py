# renderer/tone_mapping.py

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
