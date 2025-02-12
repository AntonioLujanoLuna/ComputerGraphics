# materials/textures.py
from typing import Optional
import numpy as np
from PIL import Image
from core.vector import Vector3
from core.uv import UV

class Texture:
    """Base class for all textures."""
    def sample(self, uv: UV) -> Vector3:
        """Sample the texture at given UV coordinates."""
        raise NotImplementedError("sample() must be implemented by texture subclasses.")

class SolidTexture(Texture):
    """A solid color texture."""
    def __init__(self, color: Vector3):
        self.color = color

    def sample(self, uv: UV) -> Vector3:
        return self.color

class CheckerTexture(Texture):
    """A checker pattern texture."""
    def __init__(self, color1: Vector3, color2: Vector3, scale: float = 1.0):
        self.color1 = color1
        self.color2 = color2
        self.scale = scale

    def sample(self, uv: UV) -> Vector3:
        x = int(uv.u * self.scale)
        y = int(uv.v * self.scale)
        is_even = (x + y) % 2 == 0
        return self.color1 if is_even else self.color2

class ImageTexture(Texture):
    """A texture from an image file."""
    def __init__(self, image_path: str):
        # Load image using PIL
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                # Convert to numpy array for faster access
                self.data = np.array(img) / 255.0  # Normalize to [0,1]
                self.width = img.width
                self.height = img.height
        except Exception as e:
            print(f"Error loading texture {image_path}: {e}")
            # Create a simple error texture
            self.data = np.zeros((2, 2, 3))
            self.width = 2
            self.height = 2

    def sample(self, uv: UV) -> Vector3:
        # Handle texture wrapping
        u = uv.u % 1.0
        v = 1.0 - (uv.v % 1.0)  # Flip V coordinate for OpenGL-style UV

        # Convert to pixel coordinates
        x = min(int(u * self.width), self.width - 1)
        y = min(int(v * self.height), self.height - 1)

        # Sample the color
        color = self.data[y, x]
        return Vector3(color[0], color[1], color[2])

class ProceduralTexture(Texture):
    """Base class for procedural textures."""
    def noise(self, x: float, y: float) -> float:
        """Simple 2D Perlin-like noise function."""
        # Hash function
        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        
        # Integer parts
        xi = int(x) & 255
        yi = int(y) & 255
        
        # Fractional parts
        xf = x - int(x)
        yf = y - int(y)
        
        # Fade curves
        u = fade(xf)
        v = fade(yf)
        
        # Hash coordinates of cube corners
        aa = self.p[self.p[xi    ] + yi    ]
        ab = self.p[self.p[xi    ] + yi + 1]
        ba = self.p[self.p[xi + 1] + yi    ]
        bb = self.p[self.p[xi + 1] + yi + 1]
        
        # Blend
        return (1 + ((aa + (ba - aa) * u) + ((ab - aa) * v + (bb - ba) * u * v))) / 2

    def __init__(self):
        # Initialize permutation table
        self.p = np.random.permutation(256).tolist() * 2

class MarbleTexture(ProceduralTexture):
    """A marble-like procedural texture."""
    def __init__(self, scale: float = 5.0, turbulence: float = 5.0):
        super().__init__()
        self.scale = scale
        self.turbulence = turbulence

    def sample(self, uv: UV) -> Vector3:
        x = uv.u * self.scale
        y = uv.v * self.scale
        
        # Add turbulence
        t = 0
        freq = 1.0
        amp = 1.0
        for _ in range(5):
            t += self.noise(x * freq, y * freq) * amp
            freq *= 2.0
            amp *= 0.5

        # Create marble pattern
        value = (1 + np.sin(x + self.turbulence * t)) / 2
        
        # Mix between two colors based on the value
        color1 = Vector3(0.8, 0.8, 0.8)  # Light color
        color2 = Vector3(0.2, 0.2, 0.2)  # Dark color
        return color1 * value + color2 * (1 - value)