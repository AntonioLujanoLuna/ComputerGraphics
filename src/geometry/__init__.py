import math
import random
import pygame
import numpy as np

###############################################################################
# Vector3 Class
###############################################################################
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
        # Allow scalar multiplication.
        if isinstance(other, (int, float)):
            return Vector3(self.x * other, self.y * other, self.z * other)
        # Element-wise multiplication.
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


###############################################################################
# Helper Functions for Random Sampling
###############################################################################
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


###############################################################################
# Ray Class
###############################################################################
class Ray:
    """
    Represents a ray in 3D space with an origin and direction.
    """
    def __init__(self, origin: Vector3, direction: Vector3):
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> Vector3:
        """
        Returns the point along the ray at parameter t.
        """
        return self.origin + self.direction * t


###############################################################################
# HitRecord and Hittable Base Class
###############################################################################
class HitRecord:
    """
    Records details of a ray-object intersection.
    """
    def __init__(self, p: Vector3 = None, normal: Vector3 = None,
                 t: float = 0, front_face: bool = True, material = None):
        self.p = p              # Intersection point.
        self.normal = normal    # Surface normal at intersection.
        self.t = t              # Ray parameter at intersection.
        self.front_face = front_face  # Whether the hit was on the front side.
        self.material = material

    def set_face_normal(self, ray: Ray, outward_normal: Vector3):
        """
        Ensures that the normal always points against the ray.
        """
        self.front_face = ray.direction.dot(outward_normal) < 0
        self.normal = outward_normal if self.front_face else outward_normal * -1

class Hittable:
    """
    Abstract class for objects that can be hit by a ray.
    """
    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        raise NotImplementedError("hit() must be implemented by subclasses.")


###############################################################################
# Sphere Class (implements Hittable)
###############################################################################
class Sphere(Hittable):
    """
    Represents a sphere defined by its center, radius, and material.
    """
    def __init__(self, center: Vector3, radius: float, material):
        self.center = center
        self.radius = radius
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        half_b = oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = half_b * half_b - a * c

        if discriminant < 0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        # Find the nearest root that lies in the acceptable range.
        root = (-half_b - sqrt_disc) / a
        if root < t_min or root > t_max:
            root = (-half_b + sqrt_disc) / a
            if root < t_min or root > t_max:
                return None

        rec = HitRecord()
        rec.t = root
        rec.p = ray.at(rec.t)
        outward_normal = (rec.p - self.center) / self.radius
        rec.set_face_normal(ray, outward_normal)
        rec.material = self.material
        return rec


###############################################################################
# HittableList Class
###############################################################################
class HittableList(Hittable):
    """
    A list of Hittable objects. The hit() method returns the closest hit.
    """
    def __init__(self):
        self.objects = []

    def add(self, obj: Hittable):
        self.objects.append(obj)

    def hit(self, ray: Ray, t_min: float, t_max: float) -> HitRecord:
        hit_record = None
        closest_so_far = t_max

        for obj in self.objects:
            rec = obj.hit(ray, t_min, closest_so_far)
            if rec is not None:
                closest_so_far = rec.t
                hit_record = rec

        return hit_record


###############################################################################
# Material Classes
###############################################################################
class Material:
    """
    Abstract material class. Subclasses must implement scatter().
    """
    def scatter(self, ray_in: Ray, rec: HitRecord):
        """
        Computes the scattered ray and attenuation.
        Returns a tuple (scattered_ray, attenuation) or None if no scattering occurs.
        """
        raise NotImplementedError("scatter() must be implemented by subclasses.")

class Lambertian(Material):
    """
    Lambertian diffuse material.
    """
    def __init__(self, albedo: Vector3):
        self.albedo = albedo

    def scatter(self, ray_in: Ray, rec: HitRecord):
        # Scatter direction is the hit normal plus a random unit vector.
        scatter_direction = rec.normal + random_unit_vector()
        # Catch degenerate scatter direction.
        if scatter_direction.length() < 1e-8:
            scatter_direction = rec.normal
        scattered = Ray(rec.p, scatter_direction)
        attenuation = self.albedo
        return (scattered, attenuation)

class Metal(Material):
    """
    Metal material with reflective properties.
    """
    def __init__(self, albedo: Vector3, fuzz: float):
        self.albedo = albedo
        self.fuzz = min(fuzz, 1)

    def scatter(self, ray_in: Ray, rec: HitRecord):
        reflected = reflect(ray_in.direction.normalize(), rec.normal)
        scattered = Ray(rec.p, reflected + random_in_unit_sphere() * self.fuzz)
        if scattered.direction.dot(rec.normal) > 0:
            attenuation = self.albedo
            return (scattered, attenuation)
        return None  # Absorb the ray if it does not scatter forward.


###############################################################################
# Ray Color Function (Recursive)
###############################################################################
def ray_color(ray: Ray, world: Hittable, depth: int) -> Vector3:
    """
    Returns the color seen along the ray. If the ray hits an object, the material
    scatter is computed recursively up to 'depth' bounces.
    """
    if depth <= 0:
        return Vector3(0, 0, 0)  # Exceeded recursion depth.

    rec = world.hit(ray, 0.001, float('inf'))
    if rec is not None:
        scatter_result = rec.material.scatter(ray, rec)
        if scatter_result is not None:
            scattered, attenuation = scatter_result
            return attenuation * ray_color(scattered, world, depth - 1)
        return Vector3(0, 0, 0)
    else:
        # Background gradient.
        unit_direction = ray.direction.normalize()
        t = 0.5 * (unit_direction.y + 1.0)
        return (1.0 - t) * Vector3(1.0, 1.0, 1.0) + t * Vector3(0.5, 0.7, 1.0)


###############################################################################
# Camera Class
###############################################################################
class Camera:
    """
    A simple camera that supports dynamic movement and rotation.
    """
    def __init__(self, position: Vector3, yaw: float, pitch: float,
                 fov: float, aspect_ratio: float):
        self.position = position
        self.yaw = yaw      # In radians.
        self.pitch = pitch  # In radians.
        self.fov = fov      # Field of view in radians.
        self.aspect_ratio = aspect_ratio
        self.focal_length = 1.0
        self.update_camera()

    def update_camera(self):
        """
        Updates the camera's basis vectors and viewport.
        """
        global_up = Vector3(0, 1, 0)
        # Compute forward vector.
        self.forward = Vector3(
            math.sin(self.yaw) * math.cos(self.pitch),
            math.sin(self.pitch),
            -math.cos(self.yaw) * math.cos(self.pitch)
        ).normalize()
        # Compute right and up vectors.
        self.right = self.forward.cross(global_up).normalize()
        self.up = self.right.cross(self.forward).normalize()

        # Compute viewport dimensions based on fov.
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


###############################################################################
# Main Real-Time Ray Tracing Loop Using Pygame
###############################################################################
def main():
    # Initialize Pygame.
    pygame.init()
    window_width = 640
    window_height = 360
    screen = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Improved Real-Time Ray Tracer")

    # Set render resolution (adjust for performance).
    render_width = 240
    render_height = 135
    aspect_ratio = render_width / render_height

    # Create the camera with an initial position and orientation.
    camera = Camera(
        position=Vector3(0, 0, 0),
        yaw=0.0,      # Looking toward (0, 0, -1)
        pitch=0.0,
        fov=math.radians(90),
        aspect_ratio=aspect_ratio
    )

    # Build the scene (world) with several spheres and materials.
    world = HittableList()
    # Ground sphere (large, diffuse).
    world.add(Sphere(Vector3(0, -1000, 0), 1000, Lambertian(Vector3(0.5, 0.5, 0.5))))
    # Center sphere (diffuse).
    world.add(Sphere(Vector3(0, 0, -1), 0.5, Lambertian(Vector3(0.7, 0.3, 0.3))))
    # Right sphere (metal).
    world.add(Sphere(Vector3(1, 0, -1), 0.5, Metal(Vector3(0.8, 0.6, 0.2), fuzz=0.1)))
    # Left sphere (metal, no fuzz).
    world.add(Sphere(Vector3(-1, 0, -1), 0.5, Metal(Vector3(0.8, 0.8, 0.8), fuzz=0.0)))

    clock = pygame.time.Clock()
    running = True

    # Speeds for movement and rotation.
    move_speed = 2.0            # Units per second.
    rotation_speed = math.radians(90)  # Radians per second.
    max_depth = 3               # Maximum ray bounce recursion.

    while running:
        dt = clock.tick(30) / 1000.0  # Delta time in seconds (~30 FPS).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # Process keyboard input for camera movement and rotation.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            camera.position = camera.position + camera.forward * (move_speed * dt)
        if keys[pygame.K_s]:
            camera.position = camera.position - camera.forward * (move_speed * dt)
        if keys[pygame.K_a]:
            camera.position = camera.position - camera.right * (move_speed * dt)
        if keys[pygame.K_d]:
            camera.position = camera.position + camera.right * (move_speed * dt)
        if keys[pygame.K_q]:
            camera.position = camera.position + Vector3(0, 1, 0) * (move_speed * dt)
        if keys[pygame.K_e]:
            camera.position = camera.position - Vector3(0, 1, 0) * (move_speed * dt)
        if keys[pygame.K_LEFT]:
            camera.yaw -= rotation_speed * dt
        if keys[pygame.K_RIGHT]:
            camera.yaw += rotation_speed * dt
        if keys[pygame.K_UP]:
            camera.pitch += rotation_speed * dt
            if camera.pitch > math.radians(89):
                camera.pitch = math.radians(89)
        if keys[pygame.K_DOWN]:
            camera.pitch -= rotation_speed * dt
            if camera.pitch < math.radians(-89):
                camera.pitch = math.radians(-89)

        # Update the camera with new position/orientation.
        camera.update_camera()

        # Create a NumPy array to store the render (shape: [width, height, 3]).
        image = np.zeros((render_width, render_height, 3), dtype=np.uint8)

        # Loop over each pixel in the low-res render.
        for x in range(render_width):
            for y in range(render_height):
                # Compute normalized viewport coordinates.
                u = x / (render_width - 1)
                v = (render_height - 1 - y) / (render_height - 1)
                ray = camera.get_ray(u, v)
                col = ray_color(ray, world, max_depth)
                # Clamp and convert to [0, 255].
                r = int(255.999 * max(0.0, min(col.x, 1.0)))
                g = int(255.999 * max(0.0, min(col.y, 1.0)))
                b = int(255.999 * max(0.0, min(col.z, 1.0)))
                image[x, y] = [r, g, b]

        # Convert the rendered NumPy array to a Pygame surface.
        surf = pygame.surfarray.make_surface(image)
        # Scale the low-res render to fill the window.
        surf = pygame.transform.scale(surf, (window_width, window_height))
        screen.blit(surf, (0, 0))
        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
