# core/ray.py
from core.vector import Vector3

class Ray:
    """
    Represents a ray in 3D space with an origin and direction.
    """
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> Vector3:
        """
        Returns the point along the ray at parameter t.
        """
        return self.origin + self.direction * t