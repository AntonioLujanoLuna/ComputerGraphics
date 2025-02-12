# core/uv.py
class UV:
    """
    Represents a 2D texture coordinate.
    """
    def __init__(self, u: float, v: float):
        self.u = u
        self.v = v

    def __add__(self, other: "UV") -> "UV":
        return UV(self.u + other.u, self.v + other.v)

    def __sub__(self, other: "UV") -> "UV":
        return UV(self.u - other.u, self.v - other.v)

    def __mul__(self, t: float) -> "UV":
        return UV(self.u * t, self.v * t)

    def __truediv__(self, t: float) -> "UV":
        return UV(self.u / t, self.v / t)

    def __repr__(self) -> str:
        return f"UV({self.u}, {self.v})"