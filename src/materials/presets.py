# materials/presets.py
from core.vector import Vector3
from materials.metal import Metal
from materials.lambertian import Lambertian
from materials.dielectric import Dielectric
from materials.diffuse_light import DiffuseLight
from materials.textures import CheckerTexture, MarbleTexture

class MetalPresets:
    """Predefined metal materials with realistic properties."""
    
    @staticmethod
    def gold() -> Metal:
        return Metal(Vector3(1.0, 0.78, 0.34), fuzz=0.1)
    
    @staticmethod
    def silver() -> Metal:
        return Metal(Vector3(0.95, 0.93, 0.88), fuzz=0.05)
    
    @staticmethod
    def copper() -> Metal:
        return Metal(Vector3(0.95, 0.64, 0.54), fuzz=0.1)
    
    @staticmethod
    def aluminum() -> Metal:
        return Metal(Vector3(0.91, 0.92, 0.92), fuzz=0.08)
    
    @staticmethod
    def chrome() -> Metal:
        return Metal(Vector3(0.9, 0.9, 0.9), fuzz=0.05)
    
    @staticmethod
    def brushed_metal() -> Metal:
        return Metal(Vector3(0.8, 0.8, 0.8), fuzz=0.3)

class DielectricPresets:
    """Predefined dielectric materials with realistic refractive indices."""
    
    @staticmethod
    def glass() -> Dielectric:
        return Dielectric(1.52)  # Common glass
    
    @staticmethod
    def water() -> Dielectric:
        return Dielectric(1.33)
    
    @staticmethod
    def diamond() -> Dielectric:
        return Dielectric(2.42)
    
    @staticmethod
    def ice() -> Dielectric:
        return Dielectric(1.31)
    
    @staticmethod
    def sapphire() -> Dielectric:
        return Dielectric(1.77)

class LightPresets:
    """Predefined light sources with different colors and intensities."""
    
    @staticmethod
    def warm_light(intensity: float = 1.0) -> DiffuseLight:
        return DiffuseLight(Vector3(1.0, 0.95, 0.9) * intensity)
    
    @staticmethod
    def cool_light(intensity: float = 1.0) -> DiffuseLight:
        return DiffuseLight(Vector3(0.9, 0.95, 1.0) * intensity)
    
    @staticmethod
    def daylight(intensity: float = 1.0) -> DiffuseLight:
        return DiffuseLight(Vector3(1.0, 1.0, 1.0) * intensity)
    
    @staticmethod
    def sunset_light(intensity: float = 1.0) -> DiffuseLight:
        return DiffuseLight(Vector3(1.0, 0.6, 0.3) * intensity)

class ColorPresets:
    """Common color presets for materials."""
    
    # Warm colors
    RED = Vector3(0.9, 0.2, 0.2)
    ORANGE = Vector3(0.9, 0.6, 0.1)
    YELLOW = Vector3(0.9, 0.9, 0.1)
    
    # Cool colors
    BLUE = Vector3(0.2, 0.3, 0.9)
    GREEN = Vector3(0.2, 0.8, 0.2)
    PURPLE = Vector3(0.6, 0.2, 0.8)
    
    # Neutral colors
    WHITE = Vector3(0.9, 0.9, 0.9)
    GRAY = Vector3(0.5, 0.5, 0.5)
    BLACK = Vector3(0.1, 0.1, 0.1)
    
    @staticmethod
    def matte(color: Vector3) -> Lambertian:
        """Create a matte material with the given color."""
        return Lambertian(color)

class TexturePresets:
    """Predefined texture presets."""
    
    @staticmethod
    def checkerboard(color1: Vector3 = None, color2: Vector3 = None, scale: float = 4.0) -> CheckerTexture:
        """Create a checkerboard texture with default or custom colors."""
        if color1 is None:
            color1 = ColorPresets.WHITE
        if color2 is None:
            color2 = ColorPresets.BLACK
        return CheckerTexture(color1, color2, scale)
    
    @staticmethod
    def marble(scale: float = 5.0, turbulence: float = 5.0) -> MarbleTexture:
        """Create a marble texture with the given scale and turbulence."""
        return MarbleTexture(scale, turbulence)