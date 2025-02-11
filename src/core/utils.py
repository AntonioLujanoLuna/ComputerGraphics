# core/utils.py
import random
from core.vector import Vector3

def random_in_unit_sphere() -> Vector3:
    """
    Returns a random point inside a unit sphere.
    """
    while True:
        p = Vector3(random.uniform(-1, 1),
                   random.uniform(-1, 1),
                   random.uniform(-1, 1))
        if p.dot(p) < 1.0:
            return p

def random_unit_vector() -> Vector3:
    """
    Returns a random unit vector (uniformly distributed over the sphere).
    """
    return random_in_unit_sphere().normalize()

def reflect(v: Vector3, n: Vector3) -> Vector3:
    """
    Reflects vector v about the normal n.
    """
    return v - n * 2 * v.dot(n)