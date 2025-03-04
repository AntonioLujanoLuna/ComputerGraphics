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
        self.fps_update_interval = 60  # Increased from 30 to reduce overhead
        self.frame_count_since_last_fps_update = 0
        
        # Initialize frame count early - needed for update_render_resolution
        self.frame_count = 0
        
        self.render_width = self.window_width
        self.render_height = self.window_height
        
        # Create window
        self.screen = pygame.display.set_mode((self.window_width, self.window_height))
        pygame.display.set_caption("Real-Time Ray Tracer")
        
        # Add quality settings with optimized parameters for better performance
        self.quality_levels = {
            "interactive": {"samples": 1, "bounces": 2, "scale": 0.5},
            "balanced": {"samples": 4, "bounces": 4, "scale": 0.67},  # Adjusted from 0.75 to 0.67 for better performance
            "high_quality": {"samples": 8, "bounces": 6, "scale": 1.0}  # Reduced max bounces from 8 to 6
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
        self.move_speed = 3.0
        self.rotation_speed = math.radians(60)
        
        # Setup the world
        self.world = self.create_world()
        self.world.build_bvh()  # Precompute BVH for scene
        self.renderer.update_scene_data(self.world)
        
        # FPS tracking
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)
        
        # Frame timing stats
        self.render_times = []
        self.max_render_time_history = 10
        
        # Flags for optimization
        self.needs_reset = False
        # Initialize with current camera position to prevent immediate reset
        self.last_camera_position = Vector3(
            self.camera.position.x,
            self.camera.position.y,
            self.camera.position.z
        )
        self.movement_threshold = 0.1  # Increased from 0.05 to reduce unnecessary resets
        
        # Cache surfaces for faster rendering
        self.cached_fps_surface = None
        self.fps_display_value = 0
        self.last_fps_update_time = 0
        self.fps_update_interval_ms = 1000  # Update FPS display once per second (reduced from twice per second)
        
        # Optimization: Pre-cache key state lookups
        self.key_map = {
            'w': pygame.K_w,
            's': pygame.K_s,
            'a': pygame.K_a,
            'd': pygame.K_d,
            'space': pygame.K_SPACE,
            'shift': pygame.K_LSHIFT,
            '1': pygame.K_1,
            '2': pygame.K_2,
            '3': pygame.K_3
        }
        
        # Performance optimization: Track whether GPU memory warnings have been issued
        self.gpu_memory_warning_shown = False
    
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

    def handle_input(self, dt: float, current_frame: int = 0) -> bool:
        """
        Process input and update the camera. Returns True if the camera has changed.
        
        Args:
            dt: Delta time since last frame in seconds
            current_frame: Current frame number for debug output
        
        Returns:
            bool: True if camera has moved, False otherwise
        """
        moved = False
        keys = pygame.key.get_pressed()
        
        # Handle mouse movement for rotation
        if self.mouse_locked:
            mouse_dx, mouse_dy = pygame.mouse.get_rel()
            # Only process mouse movement if it's significant - reduces small jitter movement
            if abs(mouse_dx) > 1 or abs(mouse_dy) > 1:
                moved = True
                self.camera.yaw += mouse_dx * self.mouse_sensitivity
                self.camera.pitch -= mouse_dy * self.mouse_sensitivity
            
            # Only recenter mouse when it gets too far from center to reduce unnecessary events
            mouse_x, mouse_y = pygame.mouse.get_pos()
            center_x, center_y = self.window_width // 2, self.window_height // 2
            if abs(mouse_x - center_x) > 50 or abs(mouse_y - center_y) > 50:
                pygame.mouse.set_pos(center_x, center_y)
        
        # Clamp pitch to prevent camera flip
        self.camera.pitch = max(min(self.camera.pitch, math.radians(89)), math.radians(-89))
        
        # Calculate forward and right vectors for movement
        forward = Vector3(
            math.sin(self.camera.yaw) * math.cos(self.camera.pitch),
            math.sin(self.camera.pitch),
            -math.cos(self.camera.yaw) * math.cos(self.camera.pitch)
        ).normalize()
        
        right = forward.cross(Vector3(0, 1, 0)).normalize()
        
        # Movement with WASD and vertical adjustments - cache the movement direction
        move_dir = Vector3(0, 0, 0)
        
        # Use direct key state checks for better performance
        # Optimization: check most common keys first
        if keys[self.key_map['w']]: 
            move_dir = move_dir + forward
        if keys[self.key_map['s']]: 
            move_dir = move_dir - forward
        if keys[self.key_map['a']]: 
            move_dir = move_dir - right
        if keys[self.key_map['d']]: 
            move_dir = move_dir + right
        if keys[self.key_map['space']]: 
            move_dir = move_dir + Vector3(0, 1, 0)  # Up
        if keys[self.key_map['shift']]: 
            move_dir = move_dir - Vector3(0, 1, 0)  # Down
        
        # Apply movement if any key was pressed
        if move_dir.length() > 0:
            moved = True
            move_dir = move_dir.normalize() * self.move_speed * dt
            self.camera.position = self.camera.position + move_dir
            
            # Check if movement exceeds threshold for accumulation reset
            # Create a new Vector3 object for the old position to avoid reference issues
            old_pos = self.last_camera_position
            current_pos = self.camera.position
            
            # Calculate movement vector
            dx = current_pos.x - old_pos.x
            dy = current_pos.y - old_pos.y
            dz = current_pos.z - old_pos.z
            
            # Calculate squared distance (faster than using length() which computes square root)
            movement_squared = dx*dx + dy*dy + dz*dz
            movement_threshold_squared = self.movement_threshold * self.movement_threshold
            
            # Debug output less frequently to reduce overhead
            if current_frame % 120 == 0:
                print(f"Movement: sqrt({movement_squared:.6f}) = {math.sqrt(movement_squared):.6f}, threshold: {self.movement_threshold}")
            
            if movement_squared > movement_threshold_squared:
                self.needs_reset = True
                # Make a deep copy by creating a new Vector3 with the current values
                self.last_camera_position = Vector3(
                    current_pos.x,
                    current_pos.y,
                    current_pos.z
                )
                
                if current_frame % 30 == 0:  # Reduced frequency of debug messages
                    print(f"Resetting accumulation due to movement: {math.sqrt(movement_squared):.6f} > {self.movement_threshold}")
        
        # Update camera orientation and viewport
        self.camera.update_camera()

        # Add quality toggle with number keys - use a dictionary for cleaner code
        # Only check quality keys if moved to avoid unnecessary quality changes
        if moved:
            quality_key_map = {
                self.key_map['1']: "interactive",
                self.key_map['2']: "balanced",
                self.key_map['3']: "high_quality"
            }
            
            for key, quality in quality_key_map.items():
                if keys[key] and self.current_quality != quality:
                    self.current_quality = quality
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
            
            # Precompute common resources
            self.last_camera_position = Vector3(
                self.camera.position.x,
                self.camera.position.y,
                self.camera.position.z
            )
            
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
                            # Toggle fullscreen
                            pygame.display.toggle_fullscreen()
                        # Add F11 as an alternative fullscreen toggle
                        elif event.key == pygame.K_F11:
                            pygame.display.toggle_fullscreen()
                
                # Handle continuous input (movement, rotation)
                moved = self.handle_input(dt, self.frame_count)
                
                # Reset accumulation if camera moved and we need to
                if moved and self.needs_reset:
                    self.renderer.reset_accumulation()
                    self.needs_reset = False
                
                # Adaptively adjust render scale based on FPS
                self.frame_count_since_last_fps_update += 1
                if self.frame_count_since_last_fps_update >= self.fps_update_interval:
                    current_fps = self.clock.get_fps()
                    self.fps_history.append(current_fps)
                    if len(self.fps_history) > 5:
                        self.fps_history.pop(0)  # Keep only 5 most recent readings
                    
                    # Only adjust render scale in interactive mode
                    if self.current_quality == "interactive":
                        self.adjust_render_scale(current_fps)
                    self.frame_count_since_last_fps_update = 0
                
                # Render the scene
                render_start = pygame.time.get_ticks()
                
                # Perform ray tracing
                frame = self.renderer.render_frame_adaptive(self.camera, self.world)
                
                # Track render time
                render_time = pygame.time.get_ticks() - render_start
                self.render_times.append(render_time)
                if len(self.render_times) > self.max_render_time_history:
                    self.render_times.pop(0)
                
                # Convert the frame to a pygame surface and upscale if needed
                frame_surface = pygame.surfarray.make_surface(frame * 255)
                if self.render_width != self.window_width or self.render_height != self.window_height:
                    frame_surface = pygame.transform.scale(frame_surface, (self.window_width, self.window_height))
                
                # Draw the frame
                self.screen.blit(frame_surface, (0, 0))
                
                # Update FPS display at fixed intervals to avoid text rendering overhead
                current_time = pygame.time.get_ticks()
                if current_time - self.last_fps_update_time > self.fps_update_interval_ms:
                    self.fps_display_value = self.clock.get_fps()
                    self.last_fps_update_time = current_time
                    # Pre-render the FPS text 
                    self.cached_fps_surface = self.font.render(
                        f"FPS: {self.fps_display_value:.1f} | Quality: {self.current_quality} | Scale: {self.render_scale:.2f}",
                        True, (255, 255, 255))
                
                # Display the cached FPS surface
                if self.cached_fps_surface:
                    self.screen.blit(self.cached_fps_surface, (10, 10))
                
                # Update the display
                pygame.display.flip()
                
                # Count frames
                self.frame_count += 1
                
            
        finally:
            # Clean up resources
            print("Cleaning up...")
            self.cleanup()
            pygame.quit()

    def update_render_resolution(self):
        """Update the render resolution based on the current render scale."""
        # Ensure the render size is at least 8x8 to avoid CUDA kernel errors
        self.render_width = max(8, int(self.window_width * self.render_scale))
        self.render_height = max(8, int(self.window_height * self.render_scale))
        
        # Ensure dimensions are divisible by 16 for better CUDA performance (was 8)
        self.render_width = (self.render_width // 16) * 16
        self.render_height = (self.render_height // 16) * 16
        
        # Double-check minimums after making divisible
        if self.render_width < 16:
            self.render_width = 16
        if self.render_height < 16:
            self.render_height = 16
            
        # Print update message - safely handle frame_count attribute
        if hasattr(self, 'frame_count') and self.frame_count % 60 == 0:
            print(f"Render resolution updated to {self.render_width}x{self.render_height} (scale: {self.render_scale:.2f})")
        else:
            # Always print during initialization
            print(f"Render resolution updated to {self.render_width}x{self.render_height} (scale: {self.render_scale:.2f})")

    def adjust_render_scale(self, current_fps):
        """Dynamically adjust render scale to maintain target FPS."""
        # Use exponential moving average for smoother adjustments
        if len(self.fps_history) == 0:
            avg_fps = current_fps
        else:
            # Weight recent FPS higher (70% newest, 30% history)
            avg_fps = current_fps * 0.7 + sum(self.fps_history) * 0.3 / len(self.fps_history)
        
        # Adjust scale based on how far we are from target FPS
        # More aggressive downscaling when FPS drops too low
        if avg_fps < self.target_fps * 0.7:  # More than 30% below target
            new_scale = self.render_scale * 0.85  # Reduce scale by 15%
        elif avg_fps < self.target_fps * 0.9:  # 10-30% below target
            new_scale = self.render_scale * 0.95  # Reduce scale by 5%
        elif avg_fps > self.target_fps * 1.3:  # More than 30% above target
            new_scale = min(self.render_scale * 1.05, 1.0)  # Increase scale up to 1.0
        elif avg_fps > self.target_fps * 1.1:  # 10-30% above target
            new_scale = min(self.render_scale * 1.02, 1.0)  # Small increase up to 1.0
        else:
            return  # Don't adjust if we're close to target
        
        # Clamp scale to reasonable range
        new_scale = max(0.2, min(new_scale, 1.0))
        
        # Only update if the change is significant to avoid thrashing
        if abs(new_scale - self.render_scale) > 0.02:
            old_scale = self.render_scale
            self.render_scale = new_scale
            self.update_render_resolution()
            self.renderer.width = self.render_width
            self.renderer.height = self.render_height
            self.renderer.reset_accumulation()
            
            # Adjust blockspergrid for the new resolution
            self.renderer.blockspergrid_x = math.ceil(self.render_width / self.renderer.threadsperblock[0])
            self.renderer.blockspergrid_y = math.ceil(self.render_height / self.renderer.threadsperblock[1])
            self.renderer.blockspergrid = (self.renderer.blockspergrid_x, self.renderer.blockspergrid_y)
            
            print(f"Adjusted render scale: {old_scale:.2f} → {new_scale:.2f} (FPS: {avg_fps:.1f})")

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
                    
                    # We want to keep original normals since they define the intended face orientations
                    # No need to transform normals as they only define orientation, not position
                    # They will be normalized during rendering anyway
                
                print(f"Added red cube with {len(cube_mesh.triangles)} triangles at (0, 1, -2) with scale 1.0")
                world.add(cube_mesh)
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
        
        # Modify water drop for better transparency - use a single sphere with glass material
        water_sphere = Sphere(Vector3(0, 1, 2), 0.7, DielectricPresets.water())
        world.add(water_sphere)
        print(f"Added water sphere at (0, 1, 2) with radius 0.7, IOR: 1.33")

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
    try:
        from numba import cuda
        cuda.detect()  # Ensure CUDA is available
        
        # Print GPU information
        device = cuda.get_current_device()
        compute_capability = device.compute_capability
        print(f"\nUsing GPU: {device.name}")
        print(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        
        # Safely get memory info - this was causing an error
        try:
            gpu_memory = device.total_memory / (1024**3)  # Convert to GB
            print(f"Memory: {gpu_memory:.2f} GB\n")
        except AttributeError:
            # Some CUDA configurations may not expose total_memory attribute
            print("Warning: Could not access GPU memory information.")
            
        # Don't try to use MemoryManager as it's not available in this version of numba
        print("Using default CUDA memory settings")
        
    except Exception as e:
        print(f"CUDA initialization error: {e}")
        print("Defaulting to CPU mode (will be much slower)")
    
    # Create and run the application
    app = Application()
    
    try:
        app.run()
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Ensure proper cleanup
        if hasattr(app, 'renderer'):
            print("Cleaning up renderer resources...")
            app.renderer.cleanup()
        
        # Force CUDA context cleanup
        try:
            cuda.close()
        except:
            pass
        
        pygame.quit()

if __name__ == "__main__":
    main()