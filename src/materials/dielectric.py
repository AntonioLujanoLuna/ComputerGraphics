# src/materials/dielectric.py
import math
import random
from typing import Optional, Tuple
from core.ray import Ray
from core.vector import Vector3
from geometry.hittable import HitRecord
from materials.material import Material

class Dielectric(Material):
    def __init__(self, ref_idx: float):
        self.ref_idx = ref_idx

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        attenuation = Vector3(1.0, 1.0, 1.0)  # Glass doesn't absorb light
        
        # Determine if we're entering or exiting the material
        ni_over_nt = 1.0 / self.ref_idx if rec.front_face else self.ref_idx
        
        unit_direction = ray_in.direction.normalize()
        
        # Calculate cosine using the angle between incoming ray and normal
        cos_theta = min(-unit_direction.dot(rec.normal), 1.0)
        sin_theta = math.sqrt(1.0 - cos_theta * cos_theta)
        
        # Check for total internal reflection
        if ni_over_nt * sin_theta > 1.0:
            # Must reflect
            reflected = reflect(unit_direction, rec.normal)
            return Ray(rec.p, reflected), attenuation
            
        # Calculate reflection probability using Schlick's approximation
        reflect_prob = schlick(cos_theta, ni_over_nt)
        
        if random.random() < reflect_prob:
            reflected = reflect(unit_direction, rec.normal)
            return Ray(rec.p, reflected), attenuation
        else:
            refracted = refract(unit_direction, rec.normal, ni_over_nt)
            if refracted:
                return Ray(rec.p, refracted), attenuation
            else:
                reflected = reflect(unit_direction, rec.normal)
                return Ray(rec.p, reflected), attenuation

# Improved refraction calculation
def refract(v: Vector3, n: Vector3, ni_over_nt: float) -> Optional[Vector3]:
    unit_v = v.normalize()
    cos_theta = min(-unit_v.dot(n), 1.0)
    r_out_perp = (unit_v + n * cos_theta) * ni_over_nt
    r_out_parallel = n * (-math.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))))
    return r_out_perp + r_out_parallel

# Improved Schlick approximation
def schlick(cos_theta: float, ref_idx: float) -> float:
    r0 = (1.0 - ref_idx) / (1.0 + ref_idx)
    r0 = r0 * r0
    return r0 + (1.0 - r0) * math.pow((1.0 - cos_theta), 5)

def reflect(v: Vector3, n: Vector3) -> Vector3:
    return v - n * 2 * v.dot(n)