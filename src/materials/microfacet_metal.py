# materials/microfacet_metal.py
import math
import random
from core.vector import Vector3
from core.ray import Ray
from geometry.hittable import HitRecord
from materials.material import Material

def ggx_distribution(normal, half_vector, alpha):
    # GGX (Trowbridge-Reitz) normal distribution function
    NdotH = max(normal.dot(half_vector), 0.0)
    alpha2 = alpha * alpha
    denom = (NdotH * NdotH * (alpha2 - 1.0) + 1.0)
    return alpha2 / (math.pi * denom * denom)

def fresnel_schlick(cos_theta, F0):
    return F0 + (1.0 - F0) * math.pow(1.0 - cos_theta, 5)

class MicrofacetMetal(Material):
    def __init__(self, albedo: Vector3, roughness: float):
        super().__init__()
        self.albedo = albedo
        self.roughness = roughness
        # For metals, the Fresnel reflectance at normal incidence is roughly the albedo.
        self.F0 = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord):
        # Compute a half-vector via an importance–sampled cosine-weighted hemisphere.
        # (For brevity, here we choose a random vector in the hemisphere.)
        while True:
            random_vec = Vector3(random.uniform(-1, 1),
                                 random.uniform(-1, 1),
                                 random.uniform(-1, 1)).normalize()
            if random_vec.dot(rec.normal) > 0:
                half_vector = random_vec
                break

        # Perfect specular reflection:
        reflected = ray_in.direction - rec.normal * 2 * ray_in.direction.dot(rec.normal)
        # Compute Fresnel term (using one channel for simplicity):
        cos_theta = max(rec.normal.dot(ray_in.direction.normalize()), 0.0)
        F = fresnel_schlick(cos_theta, self.F0.x)
        # Blend the reflected direction with the half–vector based on F and roughness.
        scattered_direction = (reflected * F + half_vector * (1.0 - F) * self.roughness).normalize()
        scattered = Ray(rec.p, scattered_direction)
        attenuation = self.albedo
        return scattered, attenuation
