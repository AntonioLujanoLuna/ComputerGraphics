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
        
        # Add quality settings
        self.quality_levels = {
            "interactive": {"samples": 2, "bounces": 2, "scale": 0.5},
            "balanced": {"samples": 4, "bounces": 4, "scale": 0.75},
            "high_quality": {"samples": 16, "bounces": 8, "scale": 1.0}
        }
        self.current_quality = "interactive"  # Start with interactive mode
        
        # Initialize camera with depth of field
        self.aspect_ratio = self.window_width / self.window_height
        
        # Adjust camera position to better view the scene
        # Position the camera further back and higher to see more of the scene
        self.camera = Camera(
            position=Vector3(0, 3, 10),  # Move back to z=10 and up to y=3
            yaw=0.0,
            pitch=-0.2,  # Look slightly downward
            fov=math.radians(60),  # Wider FOV to see more
            aspect_ratio=self.aspect_ratio,
            aperture=0.01,  # Small aperture for subtle depth of field
            focus_dist=10.0   # Focus on the scene center
        )
        
        # Initialize with quality settings
        quality = self.quality_levels[self.current_quality]
        self.render_scale = quality["scale"]
        self.update_render_resolution()
        self.renderer = Renderer(
            self.render_width,
            self.render_height,
            N=quality["samples"],
            max_depth=quality["bounces"]
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
    
    def apply_quality_settings(self):
        """
        Apply the current quality settings by updating render scale and renderer parameters.
        This resets the renderer accumulation to start a fresh render.
        """
        quality = self.quality_levels[self.current_quality]
        self.render_scale = quality["scale"]
        self.update_render_resolution()
        
        # If renderer already exists, update its parameters rather than recreating
        if hasattr(self, 'renderer'):
            old_width = self.renderer.width
            old_height = self.renderer.height
            
            # Create new renderer with quality settings
            self.renderer = Renderer(
                self.render_width,
                self.render_height,
                N=quality["samples"],
                max_depth=quality["bounces"]
            )
            
            # Important: Update scene data in the new renderer
            self.renderer.update_scene_data(self.world)
            print(f"Scene data updated after quality change")
        else:
            # First-time initialization
            self.renderer = Renderer(
                self.render_width,
                self.render_height,
                N=quality["samples"],
                max_depth=quality["bounces"]
            )
            self.renderer.update_scene_data(self.world)
        
        # Reset accumulation to start fresh render
        self.renderer.reset_accumulation()
        print(f"Quality changed to: {self.current_quality}")

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

        # Add quality toggle with number keys
        keys = pygame.key.get_pressed()
        if keys[pygame.K_1] and self.current_quality != "interactive":
            self.current_quality = "interactive"
            self.apply_quality_settings()
            return True
        elif keys[pygame.K_2] and self.current_quality != "balanced":
            self.current_quality = "balanced"
            self.apply_quality_settings()
            return True
        elif keys[pygame.K_3] and self.current_quality != "high_quality":
            self.current_quality = "high_quality"
            self.apply_quality_settings()
            return True
        
        return moved

    def run(self):
        try:
            # Update renderer's scene data before rendering.
            print("\n=== Initializing Renderer ===")
            print(f"Render resolution: {self.render_width}x{self.render_height}")
            print(f"Quality settings: {self.current_quality}")
            print(f"Samples per pixel: {self.quality_levels[self.current_quality]['samples']}")
            print(f"Max bounces: {self.quality_levels[self.current_quality]['bounces']}")
            print(f"Render scale: {self.render_scale}")
            
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
                        elif event.key == pygame.K_f:
                            # Quick low quality mode for faster navigation
                            print("Switching to fast navigation mode")
                            old_quality = self.current_quality
                            self.current_quality = "interactive"
                            self.apply_quality_settings()
                            
                            # Reduce resolution even further for very fast preview
                            temp_scale = self.render_scale
                            self.render_scale = 0.25
                            self.update_render_resolution()
                            self.renderer = Renderer(
                                self.render_width,
                                self.render_height,
                                N=1,  # Minimal samples
                                max_depth=1  # Minimal bounces
                            )
                            self.renderer.update_scene_data(self.world)
                    
                # Only update camera if mouse/keyboard input detected
                if self.handle_input(dt):
                    self.renderer.reset_accumulation()
                
                # Render frame
                image = self.renderer.render_frame_adaptive(self.camera, self.world)
                mapped_image = reinhard_tone_mapping(image, exposure=2.0, white_point=2.0, gamma=2.2)
                surf = pygame.surfarray.make_surface(mapped_image)
                
                if self.render_width != self.window_width or self.render_height != self.window_height:
                    surf = pygame.transform.scale(surf, (self.window_width, self.window_height))
                self.screen.blit(surf, (0, 0))
                
                # Calculate and display FPS
                current_fps = self.clock.get_fps()
                self.display_performance_metrics(current_fps)
                
                # Adjust render resolution based on performance
                if self.frame_count % 30 == 0:  # Check every 30 frames
                    self.adjust_render_scale(current_fps)
                
                # Show sample count, update display, etc.
                pygame.display.flip()
                self.frame_count += 1
                
        finally:
            self.cleanup()
            pygame.quit()
        
    def update_render_resolution(self):
        """Update render resolution based on current scale."""
        old_width = self.render_width if hasattr(self, 'render_width') else 0 
        old_height = self.render_height if hasattr(self, 'render_height') else 0
        
        # Calculate new resolution
        self.render_width = max(32, int(self.window_width * self.render_scale))
        self.render_height = max(32, int(self.window_height * self.render_scale))
        
        # Only recreate renderer if resolution actually changed
        if self.render_width != old_width or self.render_height != old_height:
            print(f"Updating render resolution: {old_width}x{old_height} -> {self.render_width}x{self.render_height}")
            if hasattr(self, 'renderer'):
                # Store current settings
                old_N = self.renderer.N if hasattr(self.renderer, 'N') else 4
                old_max_depth = self.renderer.max_depth
                
                # Create new renderer with the same settings but new resolution
                self.renderer = Renderer(
                    self.render_width,
                    self.render_height,
                    N=old_N,
                    max_depth=old_max_depth
                )
                
                # Important: Update scene data in the new renderer
                self.renderer.update_scene_data(self.world)
                print(f"Scene data updated after resolution change")
    
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
        
        print("\n=== Creating World ===")
        print(f"Camera position: {self.camera.position}")
        print(f"Camera forward: {self.camera.forward}")
        print(f"Camera focus distance: {self.camera.focus_dist}")
        
        # Ground plane with checkerboard pattern
        ground_texture = TexturePresets.checkerboard(
            ColorPresets.WHITE * 0.8,  # Slightly dimmer white
            ColorPresets.BLACK * 0.2   # Very dark gray
        )
        world.add(Sphere(
            Vector3(0, -1000, 0), 1000,
            Lambertian(ground_texture)
        ))
        print(f"Added ground plane at y=-1000 with radius 1000")
        
        # Marble sphere
        world.add(Sphere(
            Vector3(-3, 1, 2), 1.0,
            Lambertian(TexturePresets.marble(scale=3.0, turbulence=6.0))
        ))
        print(f"Added marble sphere at (-3, 1, 2) with radius 1.0")
        
        # Central red cube
        cube_material = ColorPresets.matte(ColorPresets.RED)
        try:
            import os
            model_path = os.path.join(os.path.dirname(__file__), "models", "cube.obj")
            if os.path.exists(model_path):
                cube_mesh = load_obj(model_path, cube_material)
                # Scale and position the cube
                for triangle in cube_mesh.triangles:
                    # Apply scale first
                    scale = 1.0  # Reduced scale to make the cube more proportional
                    triangle.v0 = triangle.v0 * scale
                    triangle.v1 = triangle.v1 * scale
                    triangle.v2 = triangle.v2 * scale
                    
                    # Then apply position
                    position = Vector3(0, 1, -2)  # Moved closer to camera
                    triangle.v0 = triangle.v0 + position
                    triangle.v1 = triangle.v1 + position
                    triangle.v2 = triangle.v2 + position
                    
                    # Recalculate normals to ensure they're correct
                    edge1 = triangle.v1 - triangle.v0
                    edge2 = triangle.v2 - triangle.v0
                    face_normal = edge1.cross(edge2).normalize()
                    triangle.n0 = triangle.n1 = triangle.n2 = face_normal
                
                world.add(cube_mesh)
                print(f"Added red cube at (0, 1, -2) with scale 1.0")
            else:
                print(f"Cube model not found at {model_path}")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
        
        # Gold microfacet sphere
        world.add(Sphere(
            Vector3(2, 1, 0), 1.0,
            MetalPresets.microfacet_gold()  # Use microfacet gold instead
        ))
        print(f"Added gold sphere at (2, 1, 0) with radius 1.0")
        
        # Glass sphere
        world.add(Sphere(
            Vector3(-2, 1, 0), 1.0,
            DielectricPresets.glass()
        ))
        print(f"Added glass sphere at (-2, 1, 0) with radius 1.0")
        
        # Add a water drop (hollow sphere)
        world.add(Sphere(
            Vector3(0, 1, 2), 0.7,
            DielectricPresets.glass()
        ))
        print(f"Added outer water drop at (0, 1, 2) with radius 0.7")
        
        world.add(Sphere(
            Vector3(0, 1, 2), -0.65,  # Slightly thicker water shell
            DielectricPresets.water()
        ))
        print(f"Added inner water drop at (0, 1, 2) with radius -0.65")

        # Main light source (warm light)
        world.add(Sphere(
            Vector3(0, 6, 2), 0.5,
            LightPresets.warm_light(5.0)  # Higher intensity
        ))
        print(f"Added light source at (0, 6, 2) with radius 0.5")

        # Build the BVH over everything:
        print(f"Building BVH for {len(world.objects)} objects...")
        world.build_bvh()
        print(f"BVH built successfully: {world.bvh_root is not None}")
        if world.bvh_flat is not None:
            bbox_min, bbox_max, left_indices, right_indices, is_leaf, object_indices = world.bvh_flat
            print(f"BVH nodes: {len(object_indices)}")
        
        return world
    
    def display_performance_metrics(self, fps: float):
        """
        Display FPS, resolution, and quality setting information on screen.
        
        Args:
            fps: Current frames per second value
        """
        fps_text = self.font.render(f"FPS: {int(fps)}", True, (255, 255, 255))
        res_text = self.font.render(
            f"Resolution: {self.render_width}x{self.render_height} ({int(self.render_scale*100)}%)", 
            True, (255, 255, 255)
        )
        quality_text = self.font.render(
            f"Quality: {self.current_quality} (Press 1/2/3 to change)", 
            True, (255, 255, 255)
        )
        sample_text = self.font.render(
            f"Samples: {self.quality_levels[self.current_quality]['samples']}, Bounces: {self.quality_levels[self.current_quality]['bounces']}", 
            True, (255, 255, 255)
        )
        
        self.screen.blit(fps_text, (10, 10))
        self.screen.blit(res_text, (10, 50))
        self.screen.blit(quality_text, (10, 90))
        self.screen.blit(sample_text, (10, 130))

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