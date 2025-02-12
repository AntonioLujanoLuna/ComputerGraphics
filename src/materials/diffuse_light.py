# materials/diffuse_light.py
from typing import Optional, Tuple, Union
from core.ray import Ray
from core.vector import Vector3
from core.uv import UV
from geometry.hittable import HitRecord
from materials.material import Material
from materials.textures import Texture, SolidTexture

class DiffuseLight(Material):
    """
    Emissive material that provides constant radiance with optional texture support.
    
    The texture can be used to create patterns in the emitted light.
    """
    def __init__(self, emit: Union[Vector3, Texture]):
        super().__init__()
        if isinstance(emit, Vector3):
            self.texture = SolidTexture(emit)
        else:
            self.texture = emit

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Optional[Tuple[Ray, Vector3]]:
        """
        Emissive materials do not scatter rays.
        """
        return None

    def emitted(self, u: float, v: float, p: Vector3) -> Vector3:
        """
        Return the emitted radiance, which can now be textured.
        
        Args:
            u (float): The horizontal texture coordinate.
            v (float): The vertical texture coordinate.
            p (Vector3): The hit point.
            
        Returns:
            Vector3: The emission color from the texture.
        """
        return self.texture.sample(UV(u, v))