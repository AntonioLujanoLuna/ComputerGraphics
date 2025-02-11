# materials/metal.py
from typing import Optional, Tuple
from core.ray import Ray
from core.vector import Vector3
from core.utils import reflect, random_in_unit_sphere
from geometry.hittable import HitRecord
from materials.material import Material

class Metal(Material):
    """
    Metal material with reflective properties.
    """
    def __init__(self, albedo: Vector3, fuzz: float):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1)

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        reflected = reflect(ray_in.direction.normalize(), rec.normal)
        scattered = Ray(rec.p, reflected + random_in_unit_sphere() * self.fuzz)
        
        if scattered.direction.dot(rec.normal) > 0:
            attenuation = self.albedo
            return scattered, attenuation
            
        return None  # Absorb the ray if it does not scatter forward