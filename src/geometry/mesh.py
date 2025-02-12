# geometry/mesh.py
from typing import List, Tuple, Optional
import numpy as np
from core.vector import Vector3
from core.uv import UV
from core.ray import Ray
from geometry.hittable import Hittable, HitRecord
from materials.material import Material
from core.aabb import AABB

class Triangle:
    """Represents a single triangle in 3D space with texture coordinates."""
    def __init__(self, 
                 v0: Vector3, v1: Vector3, v2: Vector3,
                 uv0: Optional[UV] = None, uv1: Optional[UV] = None, uv2: Optional[UV] = None,
                 n0: Optional[Vector3] = None, n1: Optional[Vector3] = None, n2: Optional[Vector3] = None):
        # Vertices
        self.v0 = v0
        self.v1 = v1
        self.v2 = v2
        
        # UV coordinates (default to basic mapping if not provided)
        self.uv0 = uv0 if uv0 is not None else UV(0.0, 0.0)
        self.uv1 = uv1 if uv1 is not None else UV(1.0, 0.0)
        self.uv2 = uv2 if uv2 is not None else UV(0.5, 1.0)
        
        # Normals
        if n0 is None or n1 is None or n2 is None:
            edge1 = v1 - v0
            edge2 = v2 - v0
            face_normal = edge1.cross(edge2).normalize()
            self.n0 = self.n1 = self.n2 = face_normal
        else:
            self.n0 = n0
            self.n1 = n1
            self.n2 = n2

    def interpolate_uv(self, u: float, v: float) -> UV:
        """Interpolate UV coordinates at the given barycentric coordinates."""
        w = 1.0 - u - v
        return UV(
            w * self.uv0.u + u * self.uv1.u + v * self.uv2.u,
            w * self.uv0.v + u * self.uv1.v + v * self.uv2.v
        )

    def get_normal(self, u: float, v: float) -> Vector3:
        """Interpolate normal at the given barycentric coordinates."""
        w = 1.0 - u - v
        return (self.n0 * w + self.n1 * u + self.n2 * v).normalize()

    def bounding_box(self) -> AABB:
        """Compute the bounding box for the triangle."""
        min_x = min(self.v0.x, self.v1.x, self.v2.x)
        min_y = min(self.v0.y, self.v1.y, self.v2.y)
        min_z = min(self.v0.z, self.v1.z, self.v2.z)
        max_x = max(self.v0.x, self.v1.x, self.v2.x)
        max_y = max(self.v0.y, self.v1.y, self.v2.y)
        max_z = max(self.v0.z, self.v1.z, self.v2.z)
        return AABB(Vector3(min_x, min_y, min_z), Vector3(max_x, max_y, max_z))

class TriangleMesh(Hittable):
    """Represents a 3D mesh composed of triangles."""
    def __init__(self, triangles: List[Triangle], material: Material):
        self.triangles = triangles
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_hit = None
        closest_t = t_max
        closest_u = 0.0
        closest_v = 0.0

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
            closest_u = u
            closest_v = v
            hit_point = ray.at(t)
            normal = triangle.get_normal(u, v)
            uv = triangle.interpolate_uv(u, v)

            rec = HitRecord()
            rec.t = t
            rec.p = hit_point
            rec.set_face_normal(ray, normal)
            rec.material = self.material
            rec.uv = uv
            closest_hit = rec

        return closest_hit

def load_obj(filename: str, material: Material) -> TriangleMesh:
    """Load a 3D model from an OBJ file with UV support."""
    vertices: List[Vector3] = []
    normals: List[Vector3] = []
    uvs: List[UV] = []  # Add UV coordinates list
    triangles: List[Triangle] = []

    print(f"Opening file: {filename}")
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
                elif values[0] == 'vt':  # Texture coordinate
                    uv = UV(float(values[1]), float(values[2]))
                    uvs.append(uv)
                elif values[0] == 'f':  # Face
                    # Handle different face formats
                    def get_vertex_data(vertex_str: str) -> Tuple[int, Optional[int], Optional[int]]:
                        indices = vertex_str.split('/')
                        v_idx = int(indices[0]) - 1  # OBJ indices are 1-based
                        t_idx = int(indices[1]) - 1 if len(indices) > 1 and indices[1] else None
                        n_idx = int(indices[2]) - 1 if len(indices) > 2 and indices[2] else None
                        return v_idx, t_idx, n_idx

                    # Get vertex indices for all vertices in the face
                    vertex_data = [get_vertex_data(v) for v in values[1:]]

                    # Triangulate the face (assuming it's convex)
                    for i in range(1, len(vertex_data) - 1):
                        v0_idx, t0_idx, n0_idx = vertex_data[0]
                        v1_idx, t1_idx, n1_idx = vertex_data[i]
                        v2_idx, t2_idx, n2_idx = vertex_data[i + 1]

                        v0, v1, v2 = vertices[v0_idx], vertices[v1_idx], vertices[v2_idx]
                        
                        # Get UVs if available
                        uv0 = uvs[t0_idx] if t0_idx is not None and uvs else None
                        uv1 = uvs[t1_idx] if t1_idx is not None and uvs else None
                        uv2 = uvs[t2_idx] if t2_idx is not None and uvs else None
                        
                        # Get normals if available
                        if n0_idx is not None and normals:
                            n0, n1, n2 = normals[n0_idx], normals[n1_idx], normals[n2_idx]
                            triangle = Triangle(v0, v1, v2, uv0, uv1, uv2, n0, n1, n2)
                        else:
                            triangle = Triangle(v0, v1, v2, uv0, uv1, uv2)
                        
                        triangles.append(triangle)

            except Exception as e:
                print(f"Error processing line {line_num}: {line.strip()}")
                print(f"Error details: {str(e)}")
                raise

    print(f"Loaded {len(vertices)} vertices, {len(normals)} normals, {len(uvs)} UVs, {len(triangles)} triangles")
    return TriangleMesh(triangles, material)