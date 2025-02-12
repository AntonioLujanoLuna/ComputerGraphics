# geometry/world.py
from typing import Optional, List
from core.ray import Ray
from geometry.hittable import Hittable, HitRecord

class HittableList(Hittable):
    """
    A list of Hittable objects. The hit() method returns the closest hit.
    """
    def __init__(self):
        self.objects: List[Hittable] = []

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def clear(self):
        self.objects.clear()

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        hit_record = None
        closest_so_far = t_max

        for obj in self.objects:
            rec = obj.hit(ray, t_min, closest_so_far)
            if rec is not None:
                closest_so_far = rec.t
                hit_record = rec

        return hit_record
    
    def build_bvh(self):
        from geometry.bvh import BVHNode
        if len(self.objects) == 0:
            return None
        return BVHNode(self.objects, 0, len(self.objects))