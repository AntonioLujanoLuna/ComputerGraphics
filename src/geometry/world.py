# src/geometry/world.py
from geometry.hittable import Hittable, HitRecord
from geometry.bvh import BVHNode, flatten_bvh
from typing import Optional, List
from core.ray import Ray

class HittableList(Hittable):
    """
    A list of Hittable objects. In addition to storing the objects, we build a BVH and
    store its flattened representation for GPU use.
    """
    def __init__(self):
        self.objects: List[Hittable] = []
        self.bvh_root = None  # top-level BVH node (for CPU)
        self.bvh_flat = None  # flattened BVH (for GPU)

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def clear(self):
        self.objects.clear()
        self.bvh_root = None
        self.bvh_flat = None

    def build_bvh(self):
        if len(self.objects) == 0:
            self.bvh_root = None
            self.bvh_flat = None
            return
        self.bvh_root = BVHNode(self.objects, 0, len(self.objects))
        self.bvh_flat = flatten_bvh(self.bvh_root)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        # For CPU ray–tracing we use the tree.
        if self.bvh_root is not None:
            return self.bvh_root.hit(ray, t_min, t_max)
        else:
            hit_record = None
            closest_so_far = t_max
            for obj in self.objects:
                rec = obj.hit(ray, t_min, closest_so_far)
                if rec is not None:
                    closest_so_far = rec.t
                    hit_record = rec
            return hit_record
