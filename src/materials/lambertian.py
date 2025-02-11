# materials/lambertian.py
from typing import Tuple
from core.ray import Ray
from core.vector import Vector3
from core.utils import random_unit_vector
from geometry.hittable import HitRecord
from materials.material import Material

class Lambertian(Material):
    """
    Lambertian diffuse material.
    """
    def __init__(self, albedo: Vector3):
        self.albedo = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[Ray, Vector3]:
        # Scatter direction is the hit normal plus a random unit vector
        scatter_direction = rec.normal + random_unit_vector()
        
        # Catch degenerate scatter direction
        if scatter_direction.length() < 1e-8:
            scatter_direction = rec.normal
            
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return scattered, attenuation