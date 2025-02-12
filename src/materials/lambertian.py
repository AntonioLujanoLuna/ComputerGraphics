# materials/lambertian.py
from typing import Tuple, Union
from core.ray import Ray
from core.vector import Vector3
from core.utils import random_unit_vector
from geometry.hittable import HitRecord
from materials.material import Material
from materials.textures import Texture, SolidTexture

class Lambertian(Material):
    """
    Lambertian diffuse material with optional texture support.
    """
    def __init__(self, albedo: Union[Vector3, Texture]):
        super().__init__()
        if isinstance(albedo, Vector3):
            self.texture = SolidTexture(albedo)
        else:
            self.texture = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[Ray, Vector3]:
        # Scatter direction is the hit normal plus a random unit vector
        scatter_direction = rec.normal + random_unit_vector()
        
        # Catch degenerate scatter direction
        if scatter_direction.length() < 1e-8:
            scatter_direction = rec.normal
            
        scattered = Ray(rec.p, scatter_direction)
        
        # Get albedo from texture if UV coordinates are available
        if hasattr(rec, 'uv'):
            attenuation = self.texture.sample(rec.uv)
        else:
            attenuation = self.texture.sample(UV(0, 0))  # Default UV if none provided
            
        return scattered, attenuation