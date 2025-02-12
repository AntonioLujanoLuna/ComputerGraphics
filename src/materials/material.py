# materials/material.py
from typing import Optional, Tuple
from core.ray import Ray
from core.vector import Vector3
from core.uv import UV
from geometry.hittable import HitRecord
from materials.textures import Texture, SolidTexture

class Material:
    """
    Abstract material class. Subclasses must implement scatter().
    Materials can now have textures for their properties.
    """
    def __init__(self):
        self.texture = None

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        """
        Computes the scattered ray and attenuation.
        Returns a tuple (scattered_ray, attenuation) or None if no scattering occurs.
        """
        raise NotImplementedError("scatter() must be implemented by subclasses.")

    def get_texture_color(self, uv: UV, point: Vector3) -> Vector3:
        """
        Get the color from the texture at the given UV coordinates and point.
        If no texture is set, returns None.
        """
        if self.texture is None:
            return None
        return self.texture.sample(uv)