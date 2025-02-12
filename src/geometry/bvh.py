# src/geometry/bvh.py
import math
from core.aabb import AABB

class BVHNode:
    def __init__(self, objects: list, start: int, end: int):
        object_span = end - start

        if object_span == 1:
            self.left = self.right = objects[start]
            self.box = objects[start].bounding_box()
            return

        best_cost = float('inf')
        best_split = None
        best_axis = None

        # Try splitting along each axis (0: x, 1: y, 2: z)
        for axis in range(3):
            # Sort the sub-list of objects by the minimum coordinate on this axis.
            objects[start:end] = sorted(objects[start:end],
                                        key=lambda obj: getattr(obj.bounding_box().minimum, "xyz"[axis]))
            # Precompute prefix bounding boxes.
            left_boxes = []
            current_box = objects[start].bounding_box()
            left_boxes.append(current_box)
            for i in range(start + 1, end):
                current_box = AABB.surrounding_box(current_box, objects[i].bounding_box())
                left_boxes.append(current_box)

            # Precompute suffix bounding boxes.
            right_boxes = [None] * object_span
            current_box = objects[end - 1].bounding_box()
            right_boxes[-1] = current_box
            for i in range(end - 2, start - 1, -1):
                current_box = AABB.surrounding_box(objects[i].bounding_box(), current_box)
                right_boxes[i - start] = current_box

            # Test each possible split (between objects[i] and objects[i+1]).
            for i in range(0, object_span - 1):
                left_area = left_boxes[i].surface_area()
                right_area = right_boxes[i + 1].surface_area()
                # Cost: (number of left objects)*left_area + (number of right objects)*right_area.
                cost = left_area * (i + 1) + right_area * (object_span - i - 1)
                if cost < best_cost:
                    best_cost = cost
                    best_split = start + i + 1
                    best_axis = axis

        # Now sort along the best axis.
        objects[start:end] = sorted(objects[start:end],
                                    key=lambda obj: getattr(obj.bounding_box().minimum, "xyz"[best_axis]))
        # Recursively build the child nodes.
        self.left = BVHNode(objects, start, best_split)
        self.right = BVHNode(objects, best_split, end)
        self.box = AABB.surrounding_box(self.left.box, self.right.box)

    def hit(self, ray, t_min: float, t_max: float):
        if not self.box.hit(ray, t_min, t_max):
            return None

        # Check for hits in the left and right children.
        hit_left = self.left.hit(ray, t_min, t_max) if hasattr(self.left, "hit") else None
        hit_right = self.right.hit(ray, t_min, t_max) if hasattr(self.right, "hit") else None

        if hit_left and hit_right:
            return hit_left if hit_left.t < hit_right.t else hit_right
        return hit_left or hit_right
