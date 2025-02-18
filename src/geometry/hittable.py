# geometry/hittable.py
from typing import Optional
from core.vector import Vector3
from core.ray import Ray
from core.uv import UV

class HitRecord:
    def __init__(self, p: Vector3 = None, normal: Vector3 = None,
                 t: float = 0, front_face: bool = True, material = None,
                 uv: UV = None):  # Add UV coordinates
        self.p = p
        self.normal = normal
        self.t = t
        self.front_face = front_face
        self.material = material
        self.uv = uv if uv is not None else UV(0, 0)  # Default UV coordinates

    def set_face_normal(self, ray: Ray, outward_normal: Vector3):
        """
        Ensures that the normal always points against the ray.
        """
        self.front_face = ray.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else outward_normal * -1

class Hittable:
    """
    Abstract class for objects that can be hit by a ray.
    """
    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        raise NotImplementedError("hit() must be implemented by subclasses.")