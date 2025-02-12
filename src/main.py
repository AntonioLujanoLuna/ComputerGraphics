# main.py
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
from materials.diffuse_light import DiffuseLight
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
            aperture=0.05,  # Small aperture for subtle depth of field
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

    def handle_input(self, dt: float):
        keys = pygame.key.get_pressed()
        
        if self.mouse_locked:
            # Get mouse movement
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            
            # Update camera rotation based on mouse movement
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
        
        # Movement with WASD
        move_dir = Vector3(0, 0, 0)
        if keys[pygame.K_w]: move_dir = move_dir + forward
        if keys[pygame.K_s]: move_dir = move_dir - forward
        if keys[pygame.K_a]: move_dir = move_dir - right
        if keys[pygame.K_d]: move_dir = move_dir + right
        if keys[pygame.K_SPACE]: move_dir = move_dir + Vector3(0, 1, 0)  # Up
        if keys[pygame.K_LSHIFT]: move_dir = move_dir - Vector3(0, 1, 0)  # Down
        
        # Apply movement
        if move_dir.length() > 0:
            move_dir = move_dir.normalize() * self.move_speed * dt
            self.camera.position = self.camera.position + move_dir

        # Reset accumulation when camera moves
        if any([keys[k] for k in [pygame.K_w, pygame.K_s, pygame.K_a, pygame.K_d, 
                                pygame.K_SPACE, pygame.K_LSHIFT]]):
            self.renderer.reset_accumulation()
        
        # Update camera orientation
        self.camera.update_camera()

    def run(self):
        try:
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
                image = self.renderer.render_frame(self.camera, self.world)
                
                # Display
                surf = pygame.surfarray.make_surface(image)
                if self.render_width != self.window_width or self.render_height != self.window_height:
                    surf = pygame.transform.scale(surf, (self.window_width, self.window_height))
                self.screen.blit(surf, (0, 0))
                
                # Show sample count
                samples_text = self.font.render(
                    f"Samples: {self.renderer.samples}", True, (255, 255, 255))
                self.screen.blit(samples_text, (10, 130))
                
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
        
        # Ground plane (darker and less reflective)
        world.add(Sphere(
            Vector3(0, -1000, 0), 1000,
            Metal(Vector3(0.3, 0.3, 0.3), fuzz=0.2)
        ))
        
        # Central red cube (more saturated color)
        cube_material = Lambertian(Vector3(0.9, 0.2, 0.2))
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
        
        # Metallic spheres with different colors and properties
        world.add(Sphere(
            Vector3(2, 1, 0), 1.0,
            Metal(Vector3(0.8, 0.6, 0.2), fuzz=0.1)  # Gold-like
        ))
        
        world.add(Sphere(
            Vector3(-2, 1, 0), 1.0,
            Metal(Vector3(0.9, 0.9, 0.9), fuzz=0.05)  # Chrome-like
        ))

        # Main light source (reduced intensity and moved)
        world.add(Sphere(
            Vector3(0, 6, 2), 0.5,
            DiffuseLight(Vector3(2.0, 1.9, 1.8))  # Slightly warm light
        ))
        
        # Add a second, dimmer light for fill
        world.add(Sphere(
            Vector3(-3, 4, 3), 0.3,
            DiffuseLight(Vector3(0.5, 0.5, 0.6))  # Slightly cool fill light
        ))
        
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
    app = Application()
    app.run()

if __name__ == "__main__":
    main()