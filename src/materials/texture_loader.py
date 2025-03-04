# materials/texture_loader.py
import os
from PIL import Image
import numpy as np
from core.vector import Vector3
from materials.textures import ImageTexture

def load_texture(image_path: str) -> ImageTexture:
    """
    Load an image file as a texture, with error handling and automatic format conversion.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        ImageTexture object
    
    Raises:
        FileNotFoundError: If the image file doesn't exist
        ValueError: If the image format is unsupported
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Texture file not found: {image_path}")
    
    try:
        # Load and convert the image
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Create the texture
            return ImageTexture(image_path)
            
    except Exception as e:
        raise ValueError(f"Error loading texture {image_path}: {str(e)}")

def create_image_material(image_path: str, material_class, **material_params):
    """
    Create a material with an image texture.
    
    Args:
        image_path: Path to the image file
        material_class: Material class to instantiate (e.g., Lambertian, Metal)
        **material_params: Additional parameters for the material (e.g., fuzz for Metal)
        
    Returns:
        Material instance with the image texture
    """
    texture = load_texture(image_path)
    return material_class(texture, **material_params)
