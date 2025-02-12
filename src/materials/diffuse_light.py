# materials/diffuse_light.py
from typing import Optional, Tuple
from core.ray import Ray
from core.vector import Vector3
from geometry.hittable import HitRecord
from materials.material import Material

class DiffuseLight(Material):
    """
    Emissive material that provides constant radiance.
    
    Attributes:
        emit (Vector3): The emission color (radiance) of the material.
    """
    def __init__(self, emit: Vector3):
        """
        Initialize the DiffuseLight with an emission color.
        
        Args:
            emit (Vector3): The emission color.
        """
        self.emit = emit

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        """
        Emissive materials do not scatter rays.
        
        Returns:
            None always.
        """
        return None

    def emitted(self, u: float, v: float, p: Vector3) -> Vector3:
        """
        Return the emitted radiance. In this simple model, it is constant.
        
        Args:
            u (float): The horizontal texture coordinate.
            v (float): The vertical texture coordinate.
            p (Vector3): The hit point.
            
        Returns:
            Vector3: The emission color.
        """
        return self.emit
