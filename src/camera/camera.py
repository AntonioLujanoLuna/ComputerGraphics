# camera/camera.py
import math
from core.vector import Vector3
from core.ray import Ray

class Camera:
    def __init__(self, position: Vector3, yaw: float, pitch: float,
                 fov: float, aspect_ratio: float, aperture: float = 0.0, focus_dist: float = 10.0):
        self.position = position
        self.yaw = yaw
        self.pitch = pitch
        self.fov = fov
        self.aspect_ratio = aspect_ratio
        self.aperture = aperture  # Lens aperture for depth of field
        self.focus_dist = focus_dist  # Distance to focus plane
        self.lens_radius = aperture / 2.0
        self.update_camera()

    def update_camera(self):
        """Updates the camera's basis vectors and viewport."""
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

        # Scale by focus distance
        self.horizontal = self.right * viewport_width * self.focus_dist
        self.vertical = self.up * viewport_height * self.focus_dist

        self.lower_left_corner = (self.position +
                               self.forward * self.focus_dist -
                               self.horizontal * 0.5 -
                               self.vertical * 0.5)

    def get_ray(self, u: float, v: float, rng) -> Ray:
        """Generates a ray with depth of field effect."""
        if self.aperture <= 0:
            direction = (self.lower_left_corner +
                      self.horizontal * u +
                      self.vertical * v -
                      self.position)
            return Ray(self.position, direction)
        
        # Generate random point on lens
        rd = self.lens_radius * random_in_unit_disk(rng)
        offset = self.right * rd.x + self.up * rd.y
        
        # Update ray origin and direction for depth of field
        ray_origin = self.position + offset
        ray_direction = (self.lower_left_corner +
                       self.horizontal * u +
                       self.vertical * v -
                       ray_origin)
        
        return Ray(ray_origin, ray_direction)

def random_in_unit_disk(rng) -> Vector3:
    """Generate random point in unit disk for DOF."""
    while True:
        p = Vector3(
            rng.uniform(-1, 1),
            rng.uniform(-1, 1),
            0
        )
        if p.dot(p) < 1:
            return p