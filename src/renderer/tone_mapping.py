# renderer/tone_mapping.py

def reinhard_tone_mapping(accumulated, exposure=1.0, white_point=1.0, gamma=2.2):
    """
    Apply Reinhard tone mapping to a linear radiance image.
    
    Parameters:
      accumulated: NumPy array with shape (width, height, 3) holding linear radiance.
      exposure: Scalar multiplier for exposure.
      white_point: The maximum luminance value that maps to white.
      gamma: Gamma value for correction.
    
    Returns:
      A NumPy uint8 array with tone mapped and gamma-corrected values.
    """
    # Scale radiance by exposure.
    scaled = accumulated * exposure
    # Reinhard operator: L_out = L_in / (1 + L_in / white_point)
    mapped = scaled / (1.0 + scaled / white_point)
    # Gamma correction.
    mapped = mapped ** (1.0 / gamma)
    # Convert to 8-bit.
    output = (mapped * 255).clip(0, 255).astype("uint8")
    return output
