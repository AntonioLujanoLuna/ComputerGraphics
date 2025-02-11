# geometry/mesh.py
from typing import List, Tuple, Optional
import numpy as np
from core.vector import Vector3
from core.ray import Ray
from geometry.hittable import Hittable, HitRecord
from materials.material import Material

class Triangle:
    """Represents a single triangle in 3D space."""
    def __init__(self, v0: Vector3, v1: Vector3, v2: Vector3,
                 n0: Optional[Vector3] = None, n1: Optional[Vector3] = None, n2: Optional[Vector3] = None):
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        
        # If normals aren't provided, calculate the face normal
        if n0 is None or n1 is None or n2 is None:
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = edge1.cross(edge2).normalize()
            self.n0 = self.n1 = self.n2 = face_normal
        else:
            self.n0 = n0
            self.n1 = n1
            self.n2 = n2

    def get_normal(self, u: float, v: float) -> Vector3:
        """Interpolate normal at the given barycentric coordinates."""
        w = 1.0 - u - v
        return (self.n0 * w + self.n1 * u + self.n2 * v).normalize()

class TriangleMesh(Hittable):
    """Represents a 3D mesh composed of triangles."""
    def __init__(self, triangles: List[Triangle], material: Material):
        self.triangles = triangles
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_hit = None
        closest_t = t_max

        for triangle in self.triangles:
            # Möller–Trumbore intersection algorithm
            edge1 = triangle.v1 - triangle.v0
            edge2 = triangle.v2 - triangle.v0
            h = ray.direction.cross(edge2)
            a = edge1.dot(h)

            # If ray is parallel to triangle
            if abs(a) < 1e-8:
                continue

            f = 1.0 / a
            s = ray.origin - triangle.v0
            u = f * s.dot(h)

            # Ray misses the triangle
            if u < 0.0 or u > 1.0:
                continue

            q = s.cross(edge1)
            v = f * ray.direction.dot(q)

            # Ray misses the triangle
            if v < 0.0 or u + v > 1.0:
                continue

            t = f * edge2.dot(q)

            # Intersection is behind ray origin or too far
            if t < t_min or t > closest_t:
                continue

            # We have a valid hit
            closest_t = t
            hit_point = ray.at(t)
            normal = triangle.get_normal(u, v)

            rec = HitRecord()
            rec.t = t
            rec.p = hit_point
            rec.set_face_normal(ray, normal)
            rec.material = self.material
            closest_hit = rec

        return closest_hit

def load_obj(filename: str, material: Material) -> TriangleMesh:
    """Load a 3D model from an OBJ file."""
    vertices: List[Vector3] = []
    normals: List[Vector3] = []
    triangles: List[Triangle] = []

    print(f"Opening file: {filename}")  # Debug print
    with open(filename, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                if line.startswith('#'):  # Skip comments
                    continue

                values = line.split()
                if not values:
                    continue

                if values[0] == 'v':  # Vertex
                    v = Vector3(float(values[1]), float(values[2]), float(values[3]))
                    vertices.append(v)
                elif values[0] == 'vn':  # Normal
                    n = Vector3(float(values[1]), float(values[2]), float(values[3]))
                    normals.append(n)
                elif values[0] == 'f':  # Face
                    # Handle different face formats
                    def get_vertex_data(vertex_str: str) -> Tuple[int, Optional[int]]:
                        indices = vertex_str.split('/')
                        v_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                        n_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                        return v_idx, n_idx

                    # Get vertex indices for all vertices in the face
                    vertex_data = [get_vertex_data(v) for v in values[1:]]

                    # Triangulate the face (assuming it's convex)
                    for i in range(1, len(vertex_data) - 1):
                        v0_idx, n0_idx = vertex_data[0]
                        v1_idx, n1_idx = vertex_data[i]
                        v2_idx, n2_idx = vertex_data[i + 1]

                        v0, v1, v2 = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]
                        
                        if n0_idx is not None and normals:
                            n0, n1, n2 = normals[n0_idx], normals[n1_idx], normals[n2_idx]
                            triangle = Triangle(v0, v1, v2, n0, n1, n2)
                        else:
                            triangle = Triangle(v0, v1, v2)
                        
                        triangles.append(triangle)
            except Exception as e:
                print(f"Error processing line {line_num}: {line.strip()}")
                print(f"Error details: {str(e)}")
                raise

    print(f"Loaded {len(vertices)} vertices, {len(normals)} normals, {len(triangles)} triangles")
    return TriangleMesh(triangles, material)