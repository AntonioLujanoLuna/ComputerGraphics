# materials/lambertian.py

import math
from typing import Tuple, Union
from core.ray import Ray
from core.vector import Vector3
from core.utils import random_unit_vector
from geometry.hittable import HitRecord
from materials.material import Material
from materials.textures import Texture, SolidTexture
from core.uv import UV  # Ensure we can use UV(0,0) if needed.

class Lambertian(Material):
    """
    Lambertian diffuse material with optional texture support.
    """

    def __init__(self, albedo: Union[Vector3, Texture]):
        super().__init__()
        # Store either a solid color or a texture.
        if isinstance(albedo, Vector3):
            self.texture = SolidTexture(albedo)
        else:
            self.texture = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord) -> Tuple[Ray, Vector3]:
        """
        Scatter a ray according to a Lambertian reflection model.
        Returns (scattered_ray, attenuation).
        """
        # Pick a random scatter direction by adding a random vector to the normal.
        scatter_direction = rec.normal + random_unit_vector()

        # If scatter_direction is degenerate (very small), just use the normal.
        if scatter_direction.length() < 1e-8:
            scatter_direction = rec.normal

        # Create the scattered ray.
        scattered = Ray(rec.p, scatter_direction)

        # Fetch the base color (albedo) from the texture or solid color.
        if hasattr(rec, 'uv'):
            albedo = self.texture.sample(rec.uv)
        else:
            albedo = self.texture.sample(UV(0, 0))  # Default UV if not provided.

        # Multiply by 1/pi to properly normalize the Lambertian BRDF.
        attenuation = albedo * (1.0 / math.pi)

        return scattered, attenuation
