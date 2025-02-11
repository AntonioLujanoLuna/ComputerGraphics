# materials/material.py
from typing import Optional, Tuple
from core.ray import Ray
from core.vector import Vector3
from geometry.hittable import HitRecord

class Material:
    """
    Abstract material class. Subclasses must implement scatter().
    """
    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        """
        Computes the scattered ray and attenuation.
        Returns a tuple (scattered_ray, attenuation) or None if no scattering occurs.
        """
        raise NotImplementedError("scatter() must be implemented by subclasses.")