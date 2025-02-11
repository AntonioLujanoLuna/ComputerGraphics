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
from renderer.raytracer import Renderer

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
        pygame.mouse.set_pos(self.window_width // 2, self.window_height // 2)  # Center the mouse
        self.mouse_sensitivity = 0.002    # Adjust this to change mouse sensitivity
        self.mouse_locked = True         # Track mouse lock state
        
        # Set initial render scale for adaptive resolution
        self.render_scale = 0.5  # Start at 50% resolution
        self.target_fps = 30
        self.fps_history = []
        
        # Calculate render resolution
        self.update_render_resolution()
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Real-Time Ray Tracer")
        
        # Initialize camera with better FOV and position
        self.aspect_ratio = self.window_width / self.window_height
        self.camera = Camera(
            position=Vector3(0, 2, 10),  # Further back and higher up
            yaw=0.0,
            pitch=-0.2,  # Look down slightly
            fov=math.radians(60),  # Narrower FOV for less distortion
            aspect_ratio=self.aspect_ratio
        )
        
        # Initialize renderer
        self.renderer = Renderer(
            self.render_width,
            self.render_height,
            max_depth=2  # Reduced bounce depth for better performance
        )
        
        # Movement and rotation speeds
        self.move_speed = 5.0  # Increased for better control
        self.rotation_speed = math.radians(90)
        
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
                        elif event.key == pygame.K_TAB:  # Toggle mouse capture
                            self.mouse_locked = not self.mouse_locked
                            if self.mouse_locked:
                                pygame.event.set_grab(True)
                                pygame.mouse.set_visible(False)
                                # Re-center mouse when locking
                                pygame.mouse.set_pos(self.window_width // 2, self.window_height // 2)
                            else:
                                pygame.event.set_grab(False)
                                pygame.mouse.set_visible(True)

                # Handle input
                self.handle_input(dt)
                
                # Update scene data in renderer periodically
                current_time = pygame.time.get_ticks()
                if current_time - last_scene_update > 1000:
                    self.renderer.update_scene_data(self.world)
                    last_scene_update = current_time

                # Render frame
                image = self.renderer.render_frame(self.camera, self.world)

                # Convert and scale the image
                surf = pygame.surfarray.make_surface(image)
                if self.render_width != self.window_width or self.render_height != self.window_height:
                    surf = pygame.transform.scale(surf, (self.window_width, self.window_height))
                self.screen.blit(surf, (0, 0))

                # Calculate and display FPS
                frame_time = (pygame.time.get_ticks() - frame_start) / 1000.0
                fps = 1.0 / frame_time if frame_time > 0 else 0
                self.adjust_render_scale(fps)
                
                # Display performance metrics
                fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
                res_text = self.font.render(
                    f"Resolution: {self.render_width}x{self.render_height} ({int(self.render_scale*100)}%)", 
                    True, (255, 255, 255)
                )
                pos_text = self.font.render(
                    f"Pos: ({self.camera.position.x:.1f}, {self.camera.position.y:.1f}, {self.camera.position.z:.1f})",
                    True, (255, 255, 255)
                )
                
                self.screen.blit(fps_text, (10, 10))
                self.screen.blit(res_text, (10, 50))
                self.screen.blit(pos_text, (10, 90))

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
        
        # Ground (metallic for interesting reflections)
        world.add(Sphere(
            Vector3(0, -1000, 0), 1000,
            Metal(Vector3(0.5, 0.5, 0.5), fuzz=0.05)
        ))
        
        # Add a mesh object
        cube_material = Metal(Vector3(0.8, 0.6, 0.2), fuzz=0.1)
        try:
            import os
            model_path = os.path.join(os.path.dirname(__file__), "models", "cube.obj")
            print(f"Loading model from: {model_path}")
            
            if not os.path.exists(model_path):
                print(f"Error: File does not exist at {model_path}")
                all_files = os.listdir(os.path.join(os.path.dirname(__file__), "models"))
                print(f"Files in models directory: {all_files}")
            else:
                cube_mesh = load_obj(model_path, cube_material)
                # Scale and position the cube
                for triangle in cube_mesh.triangles:
                    # Scale
                    scale = 0.5
                    triangle.v0 = triangle.v0 * scale
                    triangle.v1 = triangle.v1 * scale
                    triangle.v2 = triangle.v2 * scale
                    # Position
                    position = Vector3(0, 1, 0)
                    triangle.v0 = triangle.v0 + position
                    triangle.v1 = triangle.v1 + position
                    triangle.v2 = triangle.v2 + position
                world.add(cube_mesh)
                print("Successfully loaded and added cube mesh")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            import traceback
            traceback.print_exc()
        
        # Add some spheres around the cube
        world.add(Sphere(
            Vector3(2, 1, 0), 1.0,
            Metal(Vector3(0.8, 0.6, 0.2), fuzz=0.1)
        ))
        
        world.add(Sphere(
            Vector3(-2, 1, 0), 1.0,
            Metal(Vector3(0.8, 0.8, 0.8), fuzz=0.0)
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