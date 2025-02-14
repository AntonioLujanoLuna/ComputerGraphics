# geometry/sphere.py
import math
from typing import Optional
from core.vector import Vector3
from core.ray import Ray
from geometry.hittable import Hittable, HitRecord
from core.aabb import AABB 

class Sphere(Hittable):
    """
    Represents a sphere defined by its center, radius, and material.
    """
    def __init__(self, center: Vector3, radius: float, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        # Find the nearest root that lies in the acceptable range
        root = (-half_b - sqrt_disc) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrt_disc) / a
            if root < t_min or root > t_max:
                return None

        rec = HitRecord()
        rec.t = root
        rec.p = ray.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.material = self.material
        return rec
    
    def bounding_box(self) -> AABB:
        # The bounding box of a sphere is center ± radius
        offset = Vector3(self.radius, self.radius, self.radius)
        return AABB(self.center - offset, self.center + offset)