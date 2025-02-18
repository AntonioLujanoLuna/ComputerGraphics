# main.py
from doctest import debug
import math
import pygame
import numpy as np
from core.vector import Vector3
from camera.camera import Camera
from geometry.world import HittableList
from geometry.sphere import Sphere
from geometry.mesh import TriangleMesh, load_obj
from materials.lambertian import Lambertian
from materials.metal import Metal
from materials.dielectric import Dielectric
from materials.diffuse_light import DiffuseLight
from renderer.tone_mapping import reinhard_tone_mapping
from renderer.raytracer import MAX_BOUNCES, Renderer

class Application:
    def __init__(self):
        # Initialize Pygame
        pygame.init()
        
        # Get the display info to set up a good window size
        display_info = pygame.display.Info()
        self.window_width = min(1280, display_info.current_w - 100)
        self.window_height = min(720, display_info.current_h - 100)
        
        # Mouse control settings
        pygame.mouse.set_visible(False)  # Hide the cursor
        pygame.event.set_grab(True)      # Capture and lock the mouse
        pygame.mouse.set_pos(self.window_width // 2, self.window_height // 2)
        self.mouse_sensitivity = 0.001    # More precise mouse control
        self.mouse_locked = True         # Track mouse lock state
        
        # Set initial render scale for adaptive resolution
        self.render_scale = 1.0 
        self.target_fps = 30
        self.fps_history = []
        
        self.render_width = self.window_width
        self.render_height = self.window_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Real-Time Ray Tracer")
        
        # Initialize camera with depth of field
        self.aspect_ratio = self.window_width / self.window_height
        self.camera = Camera(
            position=Vector3(0, 2, 10),
            yaw=0.0,
            pitch=-0.2,
            fov=math.radians(50),  # Slightly narrower FOV
            aspect_ratio=self.aspect_ratio,
            aperture=0.01,  # Small aperture for subtle depth of field
            focus_dist=8.0   # Focus on the scene center
        )
        
        # Initialize renderer with higher bounce depth
        self.renderer = Renderer(
            self.render_width,
            self.render_height,
            max_depth=MAX_BOUNCES
        )
        
        # Movement and rotation speeds
        self.move_speed = 3.0  # Slower for more precise control
        self.rotation_speed = math.radians(60)
        
        # Setup the world
        self.world = self.create_world()
        
        # FPS tracking
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        self.frame_count = 0
    
    def cleanup(self):
        if hasattr(self, 'renderer'):
            self.renderer.cleanup()

    def handle_input(self, dt: float) -> bool:
        """
        Process input and update the camera. Returns True if the camera has changed.
        """
        moved = False
        keys = pygame.key.get_pressed()
        
        # Handle mouse movement for rotation
        if self.mouse_locked:
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            if abs(mouse_dx) > 0 or abs(mouse_dy) > 0:
                moved = True
            self.camera.yaw += mouse_dx * self.mouse_sensitivity
            self.camera.pitch -= mouse_dy * self.mouse_sensitivity
            
            # Keep mouse centered when locked
            mouse_x, mouse_y = pygame.mouse.get_pos()
            if (mouse_x, mouse_y) != (self.window_width // 2, self.window_height // 2):
                pygame.mouse.set_pos(self.window_width // 2, self.window_height // 2)
        
        # Clamp pitch to prevent camera flip
        self.camera.pitch = max(min(self.camera.pitch, math.radians(89)), math.radians(-89))
        
        # Calculate forward and right vectors for movement
        forward = Vector3(
            math.sin(self.camera.yaw) * math.cos(self.camera.pitch),
            math.sin(self.camera.pitch),
            -math.cos(self.camera.yaw) * math.cos(self.camera.pitch)
        ).normalize()
        
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        
        # Movement with WASD and vertical adjustments
        move_dir = Vector3(0, 0, 0)
        if keys[pygame.K_w]: 
            move_dir = move_dir + forward
        if keys[pygame.K_s]: 
            move_dir = move_dir - forward
        if keys[pygame.K_a]: 
            move_dir = move_dir - right
        if keys[pygame.K_d]: 
            move_dir = move_dir + right
        if keys[pygame.K_SPACE]: 
            move_dir = move_dir + Vector3(0, 1, 0)  # Up
        if keys[pygame.K_LSHIFT]: 
            move_dir = move_dir - Vector3(0, 1, 0)  # Down
        
        # Apply movement if any key was pressed
        if move_dir.length() > 0:
            moved = True
            move_dir = move_dir.normalize() * self.move_speed * dt
            self.camera.position = self.camera.position + move_dir
        
        # Update camera orientation and viewport
        self.camera.update_camera()
        
        return moved

    def run(self):
        try:
            # Update renderer's scene data before rendering.
            self.renderer.update_scene_data(self.world)

            running = True
            last_scene_update = 0
            
            # Reset mouse position for initial delta
            pygame.mouse.get_rel()
            
            while running:
                frame_start = pygame.time.get_ticks()
                dt = self.clock.tick(60) / 1000.0
                
                # Handle events
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_TAB:
                            self.mouse_locked = not self.mouse_locked
                            if self.mouse_locked:
                                pygame.event.set_grab(True)
                                pygame.mouse.set_visible(False)
                                pygame.mouse.set_pos(self.window_width // 2, self.window_height // 2)
                            else:
                                pygame.event.set_grab(False)
                                pygame.mouse.set_visible(True)
                        elif event.key == pygame.K_SPACE:
                            # Reset accumulation on space press
                            self.renderer.reset_accumulation()
                
                # Only update camera if mouse/keyboard input detected
                if self.handle_input(dt):
                    self.renderer.reset_accumulation()
                
                # Render frame
                image = self.renderer.render_frame_adaptive(self.camera, self.world)
                #mapped_image = reinhard_tone_mapping(image, exposure=1.0, white_point=1.0, gamma=2.2)
                mapped_image = reinhard_tone_mapping(image, exposure=2.0, white_point=2.0, gamma=2.2)
                surf = pygame.surfarray.make_surface(mapped_image)
                
                if self.render_width != self.window_width or self.render_height != self.window_height:
                    surf = pygame.transform.scale(surf, (self.window_width, self.window_height))
                self.screen.blit(surf, (0, 0))
                
                # Show sample count, update display, etc.
                pygame.display.flip()
                self.frame_count += 1
                
        finally:
            self.cleanup()
            pygame.quit()
        
    def update_render_resolution(self):
        """Update render resolution based on current scale."""
        self.render_width = max(32, int(self.window_width * self.render_scale))
        self.render_height = max(32, int(self.window_height * self.render_scale))
        if hasattr(self, 'renderer'):
            self.renderer = Renderer(
                self.render_width,
                self.render_height,
                self.renderer.max_depth
            )
    
    def adjust_render_scale(self, current_fps):
        """Dynamically adjust render scale based on performance."""
        self.fps_history.append(current_fps)
        if len(self.fps_history) > 30:  # Average over last 30 frames
            self.fps_history.pop(0)
            avg_fps = sum(self.fps_history) / len(self.fps_history)
            
            if avg_fps < self.target_fps * 0.8:  # Too slow
                self.render_scale = max(0.25, self.render_scale - 0.05)
                self.update_render_resolution()
            elif avg_fps > self.target_fps * 1.2:  # Too fast
                self.render_scale = min(1.0, self.render_scale + 0.05)
                self.update_render_resolution()
    
    def create_world(self) -> HittableList:
        world = HittableList()
        
        # Import presets
        from materials.presets import MetalPresets, DielectricPresets, LightPresets, ColorPresets, TexturePresets
        
        # Ground plane with checkerboard pattern
        ground_texture = TexturePresets.checkerboard(
            ColorPresets.WHITE * 0.8,  # Slightly dimmer white
            ColorPresets.BLACK * 0.2   # Very dark gray
        )
        world.add(Sphere(
            Vector3(0, -1000, 0), 1000,
            Lambertian(ground_texture)
        ))
        
        # Marble sphere
        world.add(Sphere(
            Vector3(-3, 1, 2), 1.0,
            Lambertian(TexturePresets.marble(scale=3.0, turbulence=6.0))
        ))
        
        # Central red cube
        cube_material = ColorPresets.matte(ColorPresets.RED)
        try:
            import os
            model_path = os.path.join(os.path.dirname(__file__), "models", "cube.obj")
            if os.path.exists(model_path):
                cube_mesh = load_obj(model_path, cube_material)
                # Scale and position the cube
                for triangle in cube_mesh.triangles:
                    scale = 2.0
                    triangle.v0 = triangle.v0 * scale
                    triangle.v1 = triangle.v1 * scale
                    triangle.v2 = triangle.v2 * scale
                    position = Vector3(0, 1, -3)
                    triangle.v0 = triangle.v0 + position
                    triangle.v1 = triangle.v1 + position
                    triangle.v2 = triangle.v2 + position
                world.add(cube_mesh)
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        
        # Gold sphere
        world.add(Sphere(
            Vector3(2, 1, 0), 1.0,
            MetalPresets.gold()
        ))
        
        # Glass sphere
        world.add(Sphere(
            Vector3(-2, 1, 0), 1.0,
            DielectricPresets.glass()
        ))
        
        # Add a water drop (hollow sphere)
        world.add(Sphere(
            Vector3(0, 1, 2), 0.7,
            DielectricPresets.glass()
        ))
        world.add(Sphere(
            Vector3(0, 1, 2), -0.65,  # Slightly thicker water shell
            DielectricPresets.water()
        ))

        # Main light source (warm light)
        world.add(Sphere(
            Vector3(0, 6, 2), 0.5,
            LightPresets.warm_light(5.0)  # Higher intensity
        ))

        # Build the BVH over everything:
        world.build_bvh()
        
        return world
    
    def display_performance_metrics(self, fps: float):
        """Display FPS and other performance information."""
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        res_text = self.font.render(
            f"Resolution: {self.render_width}x{self.render_height} ({int(self.render_scale*100)}%)", 
            True, (255, 255, 255)
        )
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(res_text, (10, 50))

def main():
    # Force CUDA initialization and check
    import numba.cuda
    if not numba.cuda.is_available():
        print("CUDA is not available. Check your GPU and drivers.")
        return
    
    # Print CUDA device info
    device = numba.cuda.get_current_device()
    print(f"Using CUDA device: {device.name}")
    
    app = Application()
    app.run()

if __name__ == "__main__":
    main()