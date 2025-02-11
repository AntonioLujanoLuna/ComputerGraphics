# camera/camera.py
import math
from core.vector import Vector3
from core.ray import Ray

class Camera:
    """
    A simple camera that supports dynamic movement and rotation.
    """
    def __init__(self, position: Vector3, yaw: float, pitch: float,
                 fov: float, aspect_ratio: float):
        self.position = position
        self.yaw = yaw      # In radians
        self.pitch = pitch  # In radians
        self.fov = fov      # Field of view in radians
        self.aspect_ratio = aspect_ratio
        self.focal_length = 1.0
        self.update_camera()

    def update_camera(self):
        """
        Updates the camera's basis vectors and viewport.
        """
        global_up = Vector3(0, 1, 0)
        
        # Compute forward vector
        self.forward = Vector3(
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            -math.cos(self.yaw) * math.cos(self.pitch)
        ).normalize()
        
        # Compute right and up vectors
        self.right = self.forward.cross(global_up).normalize()
        self.up = self.right.cross(self.forward).normalize()

        # Compute viewport dimensions based on fov
        viewport_height = 2.0 * math.tan(self.fov / 2)
        viewport_width = self.aspect_ratio * viewport_height

        self.horizontal = self.right * viewport_width
        self.vertical = self.up * viewport_height

        self.lower_left_corner = (self.position +
                                self.forward * self.focal_length -
                                self.horizontal * 0.5 -
                                self.vertical * 0.5)

    def get_ray(self, u: float, v: float) -> Ray:
        """
        Generates a ray passing through the viewport coordinates (u, v).
        """
        direction = (self.lower_left_corner +
                    self.horizontal * u +
                    self.vertical * v -
                    self.position)
        return Ray(self.position, direction)