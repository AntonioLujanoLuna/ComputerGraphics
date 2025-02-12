# materials/metal.py
from typing import Optional, Tuple, Union
from core.ray import Ray
from core.vector import Vector3
from core.utils import reflect, random_in_unit_sphere
from geometry.hittable import HitRecord
from materials.material import Material
from materials.textures import Texture, SolidTexture

class Metal(Material):
    """
    Metal material with reflective properties and optional texture support.
    """
    def __init__(self, albedo: Union[Vector3, Texture], fuzz: float):
        super().__init__()
        if isinstance(albedo, Vector3):
            self.texture = SolidTexture(albedo)
        else:
            self.texture = albedo
        self.fuzz = min(fuzz, 1)

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        reflected = reflect(ray_in.direction.normalize(), rec.normal)
        scattered = Ray(rec.p, reflected + random_in_unit_sphere() * self.fuzz)
        
        if scattered.direction.dot(rec.normal) > 0:
            # Get albedo from texture if UV coordinates are available
            if hasattr(rec, 'uv'):
                attenuation = self.texture.sample(rec.uv)
            else:
                attenuation = self.texture.sample(UV(0, 0))  # Default UV if none provided
            return scattered, attenuation
            
        return None  # Absorb the ray if it does not scatter forward