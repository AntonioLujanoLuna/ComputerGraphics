# src/geometry/bvh.py
import math
from core.aabb import AABB
import numpy as np

class BVHNode:
    def __init__(self, objects: list, start: int, end: int):
        object_span = end - start

        if object_span == 1:
            self.left = self.right = objects[start]
            self.box = objects[start].bounding_box()
            self.is_leaf = True
            self.object = objects[start]
            return

        best_cost = float('inf')
        best_split = None
        best_axis = None

        # Try splitting along each axis (0: x, 1: y, 2: z)
        for axis in range(3):
            objects[start:end] = sorted(objects[start:end],
                                        key=lambda obj: getattr(obj.bounding_box().minimum, "xyz"[axis]))
            left_boxes = []
            current_box = objects[start].bounding_box()
            left_boxes.append(current_box)
            for i in range(start + 1, end):
                current_box = AABB.surrounding_box(current_box, objects[i].bounding_box())
                left_boxes.append(current_box)

            right_boxes = [None] * object_span
            current_box = objects[end - 1].bounding_box()
            right_boxes[-1] = current_box
            for i in range(end - 2, start - 1, -1):
                current_box = AABB.surrounding_box(objects[i].bounding_box(), current_box)
                right_boxes[i - start] = current_box

            for i in range(0, object_span - 1):
                left_area = left_boxes[i].surface_area()
                right_area = right_boxes[i + 1].surface_area()
                cost = left_area * (i + 1) + right_area * (object_span - i - 1)
                if cost < best_cost:
                    best_cost = cost
                    best_split = start + i + 1
                    best_axis = axis

        # Now sort along the best axis.
        objects[start:end] = sorted(objects[start:end],
                                    key=lambda obj: getattr(obj.bounding_box().minimum, "xyz"[best_axis]))
        self.left = BVHNode(objects, start, best_split)
        self.right = BVHNode(objects, best_split, end)
        self.box = AABB.surrounding_box(self.left.box, self.right.box)
        self.is_leaf = False
        self.object = None

    def hit(self, ray, t_min: float, t_max: float):
        if not self.box.hit(ray, t_min, t_max):
            return None

        hit_left = self.left.hit(ray, t_min, t_max) if hasattr(self.left, "hit") else None
        hit_right = self.right.hit(ray, t_min, t_max) if hasattr(self.right, "hit") else None

        if hit_left and hit_right:
            return hit_left if hit_left.t < hit_right.t else hit_right
        return hit_left or hit_right

def flatten_bvh(bvh_root):
    """
    Traverse and flatten the BVH tree into NumPy arrays suitable for GPU traversal.
    Returns six arrays:
      - bbox_min: (n,3) array of minimum coordinates.
      - bbox_max: (n,3) array of maximum coordinates.
      - left_indices: (n,) array (index of left child, or -1 for a leaf).
      - right_indices: (n,) array (index of right child, or -1 for a leaf).
      - is_leaf: (n,) int array (1 if leaf, 0 otherwise).
      - object_indices: (n,) array of the object index for leaf nodes (or -1).
    """
    nodes = []

    def traverse(node):
        index = len(nodes)
        nodes.append(None)  # placeholder
        if node.is_leaf:
            flat_node = {
                'bbox_min': [node.box.minimum.x, node.box.minimum.y, node.box.minimum.z],
                'bbox_max': [node.box.maximum.x, node.box.maximum.y, node.box.maximum.z],
                'left': -1,
                'right': -1,
                'is_leaf': 1,
                # We assume that later the object will be assigned a GPU index (e.g. via a scene‐data mapping)
                'object_index': getattr(node.object, "gpu_index", -1)
            }
        else:
            left_index = traverse(node.left)
            right_index = traverse(node.right)
            flat_node = {
                'bbox_min': [node.box.minimum.x, node.box.minimum.y, node.box.minimum.z],
                'bbox_max': [node.box.maximum.x, node.box.maximum.y, node.box.maximum.z],
                'left': left_index,
                'right': right_index,
                'is_leaf': 0,
                'object_index': -1
            }
        nodes[index] = flat_node
        return index

    traverse(bvh_root)
    n = len(nodes)
    bbox_min = np.zeros((n, 3), dtype=np.float32)
    bbox_max = np.zeros((n, 3), dtype=np.float32)
    left_indices = -np.ones(n, dtype=np.int32)
    right_indices = -np.ones(n, dtype=np.int32)
    is_leaf = np.zeros(n, dtype=np.int32)
    object_indices = -np.ones(n, dtype=np.int32)

    for i, node in enumerate(nodes):
        bbox_min[i] = node['bbox_min']
        bbox_max[i] = node['bbox_max']
        left_indices[i] = node['left']
        right_indices[i] = node['right']
        is_leaf[i] = node['is_leaf']
        object_indices[i] = node['object_index']
    return bbox_min, bbox_max, left_indices, right_indices, is_leaf, object_indices
