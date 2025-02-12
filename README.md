# Real-Time GPU Ray Tracer

A high-performance real-time ray tracer implemented in Python using CUDA acceleration through Numba. This application demonstrates physically-based rendering with features like global illumination, material systems, and dynamic camera controls.

## Features

### Rendering
- Real-time path tracing with CUDA acceleration
- Global illumination with multiple light bounces
- Adaptive resolution scaling for consistent performance
- Temporal accumulation for noise reduction
- Physically-based materials (Lambertian, Metal)
- Support for both spheres and triangle meshes (OBJ files)
- Depth of field effects
- Dynamic lighting with emissive materials
- Environment lighting with sky and sun

### Materials
- Lambertian diffuse surfaces
- Metallic surfaces with configurable roughness
- Emissive materials for light sources

### Camera System
- Interactive camera controls
- Configurable field of view
- Depth of field with adjustable aperture
- Motion-controlled accumulation reset

### User Interface
- Real-time FPS display
- Sample count indicator
- Adaptive resolution scaling
- Mouse-look camera control
- Toggle-able mouse capture

## Controls

- **WASD:** Move camera forward/left/backward/right
- **Space/Left Shift:** Move camera up/down
- **Mouse:** Look around
- **Tab:** Toggle mouse capture
- **Escape:** Exit application

## Requirements

- Python 3.8 or higher
- CUDA-capable NVIDIA GPU
- Required Python packages:
  - pygame
  - numpy
  - numba
  - cudatoolkit

## Installation

1. Ensure you have CUDA toolkit installed on your system
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install required packages:
   ```bash
   pip install pygame numpy numba cudatoolkit
   ```

## Project Structure

```
src/
├── camera/
│   ├── __init__.py
│   └── camera.py          # Camera system implementation
├── core/
│   ├── __init__.py
│   ├── ray.py            # Ray class definition
│   ├── utils.py          # Utility functions
│   └── vector.py         # Vector3 class implementation
├── geometry/
│   ├── __init__.py
│   ├── hittable.py       # Abstract base class for hittable objects
│   ├── mesh.py           # Triangle mesh implementation
│   ├── sphere.py         # Sphere primitive implementation
│   └── world.py          # Scene management
├── materials/
│   ├── __init__.py
│   ├── diffuse_light.py  # Emissive material
│   ├── lambertian.py     # Diffuse material
│   ├── material.py       # Base material class
│   └── metal.py          # Metallic material
├── renderer/
│   ├── __init__.py
│   └── raytracer.py      # CUDA-accelerated renderer
└── main.py               # Application entry point
```

## Performance Tips

1. The renderer automatically adjusts resolution to maintain target frame rate
2. Higher sample counts provide better image quality but require more processing power
3. The depth of field effect can be adjusted by modifying the aperture and focus distance
4. Scene complexity (number of objects and light bounces) affects performance

## Implementation Details

- Uses Numba CUDA for GPU acceleration
- Implements physically-based path tracing
- Features next-event estimation for efficient light sampling
- Supports OBJ file loading for mesh geometry
- Uses temporal accumulation for noise reduction
- Implements Russian Roulette path termination

## Known Limitations

1. Currently only supports triangulated meshes
2. Limited to static scenes (no animation support)
3. Requires CUDA-capable NVIDIA GPU
4. No texture mapping support

## Future Improvements

- Texture mapping support
- BVH acceleration structure
- Motion blur
- Additional material types
- Scene file format support
- Animation system