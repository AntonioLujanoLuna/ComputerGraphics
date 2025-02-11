# core/vector.py
import math

class Vector3:
    """
    A simple 3D vector class supporting arithmetic, dot and cross products,
    and normalization.
    """
    def __init__(self, x: float, y: float, z: float):
        self.x = x
        self.y = y
        self.z = z

    def __add__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x + other.x, self.y + other.y, self.z + other.z)

    def __sub__(self, other: "Vector3") -> "Vector3":
        return Vector3(self.x - other.x, self.y - other.y, self.z - other.z)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        return Vector3(self.x * other.x, self.y * other.y, self.z * other.z)

    def __rmul__(self, other: float) -> "Vector3":
        return self.__mul__(other)

    def __truediv__(self, t: float) -> "Vector3":
        return Vector3(self.x / t, self.y / t, self.z / t)

    def dot(self, other: "Vector3") -> float:
        return self.x * other.x + self.y * other.y + self.z * other.z

    def cross(self, other: "Vector3") -> "Vector3":
        return Vector3(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def length(self) -> float:
        return math.sqrt(self.dot(self))

    def normalize(self) -> "Vector3":
        l = self.length()
        if l == 0:
            return Vector3(0, 0, 0)
        return self / l

    def __repr__(self) -> str:
        return f"Vector3({self.x}, {self.y}, {self.z})"