# src/geometry/bvh.py
import math
from core.aabb import AABB
import numpy as np
from core.vector import Vector3

class BVHNode:
    def __init__(self, objects: list, start: int, end: int, max_bin_count=16):
        object_span = end - start

        if object_span == 1:
            self.left = self.right = objects[start]
            self.box = objects[start].bounding_box()
            self.is_leaf = True
            self.object = objects[start]
            return

        # Compute the bounding box of all objects for this node
        self.box = objects[start].bounding_box()
        for i in range(start + 1, end):
            self.box = AABB.surrounding_box(self.box, objects[i].bounding_box())
        
        # If the volume is too small, make it a leaf to avoid precision issues
        volume = (self.box.maximum.x - self.box.minimum.x) * \
                (self.box.maximum.y - self.box.minimum.y) * \
                (self.box.maximum.z - self.box.minimum.z)
        if volume < 1e-8:
            self.left = objects[start]
            self.right = objects[start]
            self.is_leaf = True
            self.object = objects[start]
            return
        
        # Use binned SAH method for faster, more efficient splits
        best_cost = float('inf')
        best_split = start + object_span // 2  # Default split in the middle
        best_axis = 0

        # Try splitting along each axis (0: x, 1: y, 2: z)
        for axis in range(3):
            # Find min/max along this axis
            min_val = getattr(objects[start].bounding_box().minimum, "xyz"[axis])
            max_val = getattr(objects[start].bounding_box().maximum, "xyz"[axis])
            
            for i in range(start + 1, end):
                min_val = min(min_val, getattr(objects[i].bounding_box().minimum, "xyz"[axis]))
                max_val = max(max_val, getattr(objects[i].bounding_box().maximum, "xyz"[axis]))
            
            # Skip if the extent is too small
            if max_val - min_val < 1e-4:
                continue
                
            # Create bins
            bin_count = min(max_bin_count, object_span)
            infinity = 1e20
            bins = [
                {
                    'count': 0, 
                    'box': AABB(
                        Vector3(infinity, infinity, infinity),      # Minimum (initialized large)
                        Vector3(-infinity, -infinity, -infinity)    # Maximum (initialized small)
                    )
                } 
                for _ in range(bin_count)
            ]
            
            # Place objects in bins based on centroid
            bin_width = (max_val - min_val) / bin_count
            
            for i in range(start, end):
                # Use centroid for binning
                centroid = (getattr(objects[i].bounding_box().minimum, "xyz"[axis]) + 
                          getattr(objects[i].bounding_box().maximum, "xyz"[axis])) * 0.5
                
                # Calculate bin index
                bin_idx = min(bin_count - 1, int((centroid - min_val) / bin_width))
                
                # Update bin
                bins[bin_idx]['count'] += 1
                if bins[bin_idx]['count'] == 1:
                    bins[bin_idx]['box'] = objects[i].bounding_box()
                else:
                    bins[bin_idx]['box'] = AABB.surrounding_box(bins[bin_idx]['box'], objects[i].bounding_box())
            
            # Build left/right prefix boxes and evaluate SAH cost for each split
            left_boxes = [
                AABB(
                    Vector3(infinity, infinity, infinity),
                    Vector3(-infinity, -infinity, -infinity)
                ) for _ in range(bin_count)
            ]
            right_boxes = [
                AABB(
                    Vector3(infinity, infinity, infinity),
                    Vector3(-infinity, -infinity, -infinity)
                ) for _ in range(bin_count)
            ]
            
            # Left-to-right sweep
            left_box = AABB(
                Vector3(infinity, infinity, infinity),
                Vector3(-infinity, -infinity, -infinity)
            )
            left_count = 0
            for i in range(bin_count):
                if bins[i]['count'] > 0:
                    left_count += bins[i]['count']
                    left_box = AABB.surrounding_box(left_box, bins[i]['box']) if left_count > bins[i]['count'] else bins[i]['box']
                left_boxes[i] = left_box
            
            # Right-to-left sweep
            right_box = AABB(
                Vector3(infinity, infinity, infinity),
                Vector3(-infinity, -infinity, -infinity)
            )
            right_count = 0
            for i in range(bin_count - 1, -1, -1):
                if bins[i]['count'] > 0:
                    right_count += bins[i]['count']
                    right_box = AABB.surrounding_box(right_box, bins[i]['box']) if right_count > bins[i]['count'] else bins[i]['box']
                right_boxes[i] = right_box
            
            # Evaluate SAH for each split position
            for i in range(1, bin_count):
                left_count = sum(bins[j]['count'] for j in range(i))
                right_count = sum(bins[j]['count'] for j in range(i, bin_count))
                
                if left_count == 0 or right_count == 0:
                    continue
                
                left_area = left_boxes[i-1].surface_area()
                right_area = right_boxes[i].surface_area()
                
                # SAH cost for this split
                cost = 0.125 + (left_count * left_area + right_count * right_area) / self.box.surface_area()
                
                if cost < best_cost:
                    best_cost = cost
                    best_axis = axis
                    
                    # Map bin split back to object index split
                    # Count objects up to this bin
                    count = 0
                    for j in range(i):
                        count += bins[j]['count']
                    best_split = start + count
        
        # Fallback if all splits failed or if the cost is worse than not splitting
        if best_cost >= object_span:
            # Either make this a leaf with multiple objects or use a simple median split
            if object_span <= 4:  # Arbitrary small threshold
                self.left = objects[start]
                self.right = objects[start]
                self.is_leaf = True
                self.object = objects[start]
                return
            else:
                # Simple median split
                best_axis = 0
                for axis in range(3):
                    axis_span = getattr(self.box.maximum, "xyz"[axis]) - getattr(self.box.minimum, "xyz"[axis])
                    if axis_span > getattr(self.box.maximum, "xyz"[best_axis]) - getattr(self.box.minimum, "xyz"[best_axis]):
                        best_axis = axis
                
                # Sort objects along the best axis by centroid
                objects[start:end] = sorted(objects[start:end],
                    key=lambda obj: (getattr(obj.bounding_box().minimum, "xyz"[best_axis]) + 
                                   getattr(obj.bounding_box().maximum, "xyz"[best_axis])) * 0.5)
                
                best_split = start + object_span // 2
        
        # Sort along the best axis - sort by centroid for more balanced splits
        objects[start:end] = sorted(objects[start:end],
            key=lambda obj: (getattr(obj.bounding_box().minimum, "xyz"[best_axis]) + 
                           getattr(obj.bounding_box().maximum, "xyz"[best_axis])) * 0.5)
        
        # Create child nodes
        self.left = BVHNode(objects, start, best_split, max_bin_count)
        self.right = BVHNode(objects, best_split, end, max_bin_count)
        self.is_leaf = False
        self.object = None

    def hit(self, ray, t_min: float, t_max: float):
        if not self.box.hit(ray, t_min, t_max):
            return None

        # Use a stack-based approach to avoid recursion
        if self.is_leaf:
            if hasattr(self.object, "hit"):
                return self.object.hit(ray, t_min, t_max)
            return None
            
        hit_left = self.left.hit(ray, t_min, t_max) if hasattr(self.left, "hit") else None
        
        # Update t_max for right branch if we hit something on the left
        if hit_left:
            t_max = hit_left.t
            
        hit_right = self.right.hit(ray, t_min, t_max) if hasattr(self.right, "hit") else None

        # Return the closer hit
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
            # Ensure the object has a gpu_index attribute
            if not hasattr(node.object, "gpu_index"):
                print(f"WARNING: Object {node.object} has no gpu_index, assigning -1")
                node.object.gpu_index = -1
                
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
    
    # Preallocate arrays for better memory efficiency
    bbox_min = np.zeros((n, 3), dtype=np.float32)
    bbox_max = np.zeros((n, 3), dtype=np.float32)
    left_indices = -np.ones(n, dtype=np.int32)
    right_indices = -np.ones(n, dtype=np.int32)
    is_leaf = np.zeros(n, dtype=np.int32)
    object_indices = -np.ones(n, dtype=np.int32)

    # Fill arrays in a vectorized manner where possible
    for i, node in enumerate(nodes):
        bbox_min[i] = node['bbox_min']
        bbox_max[i] = node['bbox_max']
        left_indices[i] = node['left']
        right_indices[i] = node['right']
        is_leaf[i] = node['is_leaf']
        object_indices[i] = node['object_index']
        
    return bbox_min, bbox_max, left_indices, right_indices, is_leaf, object_indices
