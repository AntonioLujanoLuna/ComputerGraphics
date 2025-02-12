# src/core/aabb.py
import math
from core.vector import Vector3

class AABB:
    def __init__(self, minimum: Vector3, maximum: Vector3):
        self.minimum = minimum
        self.maximum = maximum

    def hit(self, ray, t_min: float, t_max: float) -> bool:
        # Slab method: for each axis, find intersection intervals.
        for a in ['x', 'y', 'z']:
            invD = 1.0 / getattr(ray.direction, a)
            t0 = (getattr(self.minimum, a) - getattr(ray.origin, a)) * invD
            t1 = (getattr(self.maximum, a) - getattr(ray.origin, a)) * invD
            if invD < 0:
                t0, t1 = t1, t0
            t_min = t0 if t0 > t_min else t_min
            t_max = t1 if t1 < t_max else t_max
            if t_max <= t_min:
                return False
        return True

    def surface_area(self) -> float:
        d = self.maximum - self.minimum  # uses Vector3 __sub__
        return 2 * (d.x * d.y + d.x * d.z + d.y * d.z)

    @staticmethod
    def surrounding_box(box0: "AABB", box1: "AABB") -> "AABB":
        small = Vector3(
            min(box0.minimum.x, box1.minimum.x),
            min(box0.minimum.y, box1.minimum.y),
            min(box0.minimum.z, box1.minimum.z)
        )
        big = Vector3(
            max(box0.maximum.x, box1.maximum.x),
            max(box0.maximum.y, box1.maximum.y),
            max(box0.maximum.z, box1.maximum.z)
        )
        return AABB(small, big)
