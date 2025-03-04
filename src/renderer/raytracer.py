# renderer/raytracer.py
import numpy as np
from numba import cuda, float32
from numba.cuda.random import create_xoroshiro128p_states
import math
from materials.metal import Metal
from materials.dielectric import Dielectric
from materials.textures import Texture
from materials.microfacet_metal import MicrofacetMetal
from .cuda_kernels import adaptive_render_kernel, compute_variance_mask_kernel
from renderer.cuda_temporal import temporal_reprojection_kernel
from renderer.env_importance import build_env_cdf
from core.uv import UV
from .env_map_utils import generate_gradient_env_map
from renderer.cuda_utils import MAX_HALTON_SAMPLES, precompute_halton_tables

# CUDA device constants
INFINITY = float32(1e20)
EPSILON = float32(1e-20)
MAX_BOUNCES = 4

class Renderer:    
    def __init__(self, width: int, height: int, N: int = 16, max_depth: int = 4, debug_mode: bool = False):
        self.width = width
        self.height = height
        self.max_depth = max_depth
        self.debug_mode = debug_mode
        self.frame_number = 0
        self.accumulation_buffer = np.zeros((width, height, 3), dtype=np.float32)
        self.accumulation_buffer_sq = np.zeros((width, height, 3), dtype=np.float32)
        self.samples = 0

        self.threadsperblock = (16,8)
        self.blockspergrid_x = math.ceil(width / self.threadsperblock[0])
        self.blockspergrid_y = math.ceil(height / self.threadsperblock[1])
        self.blockspergrid = (self.blockspergrid_x, self.blockspergrid_y)
        n_states = width * height * N
        self.rng_states = create_xoroshiro128p_states(n_states, seed=42)

        # --- Use a real (generated) environment map ---
        default_env = generate_gradient_env_map(512, 256)
        cdf, total, w, h = build_env_cdf(default_env)
        self.d_env_map = cuda.to_device(default_env)
        self.d_env_cdf = cuda.to_device(cdf)
        self.env_total = total
        self.env_width = w
        self.env_height = h
        self.N = N

        # (The rest of your scene and buffer initialization follows...)
        self.d_sphere_centers = None
        self.d_sphere_radii = None
        self.d_sphere_materials = None
        self.d_sphere_material_types = None
        self.d_triangle_vertices = None
        self.d_triangle_uvs = None
        self.d_triangle_materials = None
        self.d_triangle_material_types = None
        self.d_texture_data = None
        self.d_texture_indices = None
        self.texture_dimensions = None

        # Adaptive sampling buffers
        self.d_accum_buffer = None
        self.d_accum_buffer_sq = None
        self.d_sample_count = None
        self.d_mask = None

        # Buffers and previous camera parameters for temporal reprojection
        self.d_prev_accum = None
        self.d_prev_sample_count = None
        self.d_prev_depth = None
        self.prev_camera_origin = None
        self.prev_camera_lower_left = None
        self.prev_camera_horizontal = None
        self.prev_camera_vertical = None
        self.prev_camera_forward = None
        self.prev_camera_right = None
        self.prev_camera_up = None
        self.prev_focus = None

        # Precompute three Halton tables on the host.
        halton_table_base2_host, halton_table_base3_host, halton_table_base5_host = precompute_halton_tables(MAX_HALTON_SAMPLES)

        # Upload the precomputed tables to the device.
        self.d_halton_table_base2 = cuda.to_device(halton_table_base2_host)
        self.d_halton_table_base3 = cuda.to_device(halton_table_base3_host)
        self.d_halton_table_base5 = cuda.to_device(halton_table_base5_host)

        # Pre-allocate CUDA buffers for camera parameters
        self.d_camera_origin = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_lower_left = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_horizontal = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_vertical = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_forward = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_right = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_camera_up = cuda.to_device(np.zeros(3, dtype=np.float32))
        self.d_curr_focus = cuda.to_device(np.zeros(1, dtype=np.float32))

        # Pre-allocate output buffer
        self.d_frame_output = cuda.to_device(np.zeros((width, height, 3), dtype=np.float32))
        self.frame_output = cuda.pinned_array((width, height, 3), dtype=np.float32)

        self.reset_accumulation()

    def load_textures(self, textures: list):
        texture_data = []
        texture_dimensions = []
        for texture in textures:
            # If using an ImageTexture, assume texture.data is a NumPy array.
            if hasattr(texture, 'data'):
                flat_data = texture.data.reshape(-1)
                texture_data.extend(flat_data)
                texture_dimensions.append((texture.data.shape[1], texture.data.shape[0]))
            elif hasattr(texture, 'color'):
                texture_data.extend([texture.color.x, texture.color.y, texture.color.z])
                texture_dimensions.append((1, 1))
        self.d_texture_data = cuda.to_device(np.array(texture_data, dtype=np.float32))
        self.texture_dimensions = cuda.to_device(np.array(texture_dimensions, dtype=np.int32))

    def cleanup(self):
        try:
            if self.d_sphere_centers is not None:
                del self.d_sphere_centers
                del self.d_sphere_radii
                del self.d_sphere_materials
                del self.d_sphere_material_types
                del self.d_triangle_vertices
                del self.d_triangle_uvs
                del self.d_triangle_materials
                del self.d_triangle_material_types
                if self.d_texture_data is not None:
                    del self.d_texture_data
                    del self.d_texture_indices
            if hasattr(self, 'd_sphere_roughness') and self.d_sphere_roughness is not None:
                del self.d_sphere_roughness
            if hasattr(self, 'd_triangle_roughness') and self.d_triangle_roughness is not None:
                del self.d_triangle_roughness
            self.d_sphere_centers = None
            self.d_sphere_radii = None
            self.d_sphere_materials = None
            self.d_sphere_material_types = None
            self.d_sphere_roughness = None
            self.d_triangle_vertices = None
            self.d_triangle_uvs = None
            self.d_triangle_materials = None
            self.d_triangle_material_types = None
            self.d_triangle_roughness = None
            self.d_texture_data = None
            self.d_texture_indices = None
            self.texture_dimensions = None
            self.d_accum_buffer = None
            self.d_accum_buffer_sq = None
            self.d_sample_count = None
            self.d_mask = None

            # Clean up pre-allocated buffers
            if hasattr(self, 'd_camera_origin') and self.d_camera_origin is not None:
                del self.d_camera_origin
                del self.d_camera_lower_left
                del self.d_camera_horizontal
                del self.d_camera_vertical
                del self.d_camera_forward
                del self.d_camera_right
                del self.d_camera_up
                del self.d_curr_focus
                del self.d_frame_output

        except Exception as e:
            print(f"Warning: Error during CUDA cleanup: {str(e)}")

    def update_scene_data(self, world) -> None:
        # Count spheres and triangles.
        sphere_count = sum(1 for obj in world.objects if hasattr(obj, 'radius'))
        mesh_objects = [obj for obj in world.objects if hasattr(obj, 'triangles')]
        triangle_count = sum(len(mesh.triangles) for mesh in mesh_objects)

        self.cleanup()

        centers = np.zeros((max(1, sphere_count), 3), dtype=np.float32)
        radii = np.zeros(max(1, sphere_count), dtype=np.float32)
        sphere_materials = np.zeros(max(1, sphere_count) * 3, dtype=np.float32)
        sphere_material_types = np.zeros(max(1, sphere_count), dtype=np.int32)
        sphere_roughness = np.zeros(max(1, sphere_count), dtype=np.float32)

        # Initialize texture arrays with default values even if no textures exist
        texture_data = np.zeros(1, dtype=np.float32)  # Minimal default texture
        texture_dimensions = np.array([[1, 1]], dtype=np.int32)  # Default dimensions
        
        self.d_texture_data = cuda.to_device(texture_data)
        self.texture_dimensions = cuda.to_device(texture_dimensions)
        
        # Initialize texture indices with -1 (no texture)
        num_texture_entries = sphere_count + triangle_count
        default_texture_indices = -1 * np.ones((max(1, num_texture_entries),), dtype=np.int32)
        self.d_texture_indices = cuda.to_device(default_texture_indices)

        sphere_idx = 0
        for obj in world.objects:
            if hasattr(obj, 'radius'):
                centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
                radii[sphere_idx] = obj.radius
                mat_idx = sphere_idx * 3
                mat = obj.material
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    emission = mat.emitted(0, 0, obj.center)
                    sphere_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    sphere_material_types[sphere_idx] = 2  # Emissive
                    sphere_roughness[sphere_idx] = 0.0
                elif isinstance(mat, Metal):
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        sphere_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                    sphere_material_types[sphere_idx] = 1  # Metal
                    sphere_roughness[sphere_idx] = mat.fuzz if hasattr(mat, 'fuzz') else 0.1
                elif isinstance(mat, Dielectric):
                    sphere_materials[mat_idx:mat_idx+3] = [mat.ref_idx, 0.0, 0.0]
                    sphere_material_types[sphere_idx] = 3  # Dielectric
                    sphere_roughness[sphere_idx] = 0.0
                elif isinstance(mat, MicrofacetMetal):
                    sphere_materials[mat_idx:mat_idx+3] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]
                    sphere_material_types[sphere_idx] = 4  # MicrofacetMetal
                    sphere_roughness[sphere_idx] = mat.roughness
                else:
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):
                            sphere_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                        else:
                            color = mat.texture.sample(UV(0.5, 0.5))
                            sphere_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                    sphere_material_types[sphere_idx] = 0  # Lambertian or other
                    sphere_roughness[sphere_idx] = 0.0
                sphere_idx += 1

        triangle_vertices = np.zeros((max(1, triangle_count), 9), dtype=np.float32)
        triangle_uvs = np.zeros((max(1, triangle_count), 6), dtype=np.float32)
        triangle_materials = np.zeros((max(1, triangle_count) * 3), dtype=np.float32)
        triangle_material_types = np.zeros((max(1, triangle_count)), dtype=np.int32)
        triangle_roughness = np.zeros(max(1, triangle_count), dtype=np.float32)

        triangle_idx = 0
        for mesh in mesh_objects:
            for triangle in mesh.triangles:
                triangle_vertices[triangle_idx, 0:3] = [triangle.v0.x, triangle.v0.y, triangle.v0.z]
                triangle_vertices[triangle_idx, 3:6] = [triangle.v1.x, triangle.v1.y, triangle.v1.z]
                triangle_vertices[triangle_idx, 6:9] = [triangle.v2.x, triangle.v2.y, triangle.v2.z]
                triangle_uvs[triangle_idx, 0:2] = [triangle.uv0.u, triangle.uv0.v]
                triangle_uvs[triangle_idx, 2:4] = [triangle.uv1.u, triangle.uv1.v]
                triangle_uvs[triangle_idx, 4:6] = [triangle.uv2.u, triangle.uv2.v]
                mat_idx = triangle_idx * 3
                mat = mesh.material
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    emission = mat.emitted(0, 0, triangle.v0)
                    triangle_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    triangle_material_types[triangle_idx] = 2
                elif isinstance(mat, Metal):
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        triangle_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                    triangle_material_types[triangle_idx] = 1
                    triangle_roughness[triangle_idx] = mat.roughness
                else:
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):
                            triangle_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                        else:
                            color = mat.texture.sample(UV(0.5, 0.5))
                            triangle_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                            triangle_roughness[triangle_idx] = mat.roughness
                    triangle_material_types[triangle_idx] = 0
                triangle_idx += 1

        self.d_sphere_centers = cuda.to_device(np.ascontiguousarray(centers))
        self.d_sphere_radii = cuda.to_device(np.ascontiguousarray(radii))
        self.d_sphere_materials = cuda.to_device(np.ascontiguousarray(sphere_materials))
        self.d_sphere_material_types = cuda.to_device(np.ascontiguousarray(sphere_material_types))
        self.d_sphere_roughness = cuda.to_device(np.ascontiguousarray(sphere_roughness))
        self.d_triangle_vertices = cuda.to_device(np.ascontiguousarray(triangle_vertices))
        self.d_triangle_uvs = cuda.to_device(np.ascontiguousarray(triangle_uvs))
        self.d_triangle_materials = cuda.to_device(np.ascontiguousarray(triangle_materials))
        self.d_triangle_material_types = cuda.to_device(np.ascontiguousarray(triangle_material_types))
        self.d_triangle_roughness = cuda.to_device(np.ascontiguousarray(triangle_roughness))

        textures = set()
        for obj in world.objects:
            if hasattr(obj, 'material') and hasattr(obj.material, 'texture'):
                textures.add(obj.material.texture)
        self.load_textures(list(textures))

        # Allocate adaptive sampling buffers.
        self.d_accum_buffer = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
        self.d_accum_buffer_sq = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
        self.d_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
        self.d_mask = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))

        # Build the BVH over the world objects.
        world.build_bvh()

        # If the BVH was built, upload the flattened arrays to device memory.
        if world.bvh_flat is not None:
            bbox_min, bbox_max, left_indices, right_indices, is_leaf, object_indices = world.bvh_flat
            self.d_bvh_bbox_min = cuda.to_device(bbox_min)
            self.d_bvh_bbox_max = cuda.to_device(bbox_max)
            self.d_bvh_left = cuda.to_device(left_indices)
            self.d_bvh_right = cuda.to_device(right_indices)
            self.d_bvh_is_leaf = cuda.to_device(is_leaf)
            self.d_bvh_object_indices = cuda.to_device(object_indices)
        else:
            self.d_bvh_bbox_min = None
            self.d_bvh_bbox_max = None
            self.d_bvh_left = None
            self.d_bvh_right = None
            self.d_bvh_is_leaf = None
            self.d_bvh_object_indices = None

    def render_frame_adaptive(self, camera, world) -> np.ndarray:
        """
        Render a frame with adaptive sampling, BVH acceleration, and temporal reprojection.
        
        This method:
        1. Prepares camera and scene parameters for GPU rendering
        2. Performs temporal reprojection to reuse data from previous frames
        3. Launches the adaptive rendering kernel with BVH acceleration
        4. Computes a convergence mask to focus computation on noisy areas
        5. Updates history buffers for the next frame
        
        Parameters:
            camera: Camera object with position, orientation, and projection parameters
            world: Scene containing objects to render
            
        Returns:
            np.ndarray: Rendered image of shape (width, height, 3) with float32 values
        """
        # Create CUDA stream for asynchronous operations
        stream = cuda.stream()

        # Set up current camera parameters
        camera_origin = np.array(
            [camera.position.x, camera.position.y, camera.position.z],
            dtype=np.float32
        )
        camera_lower_left = np.array(
            [camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z],
            dtype=np.float32
        )
        camera_horizontal = np.array(
            [camera.horizontal.x, camera.horizontal.y, camera.horizontal.z],
            dtype=np.float32
        )
        camera_vertical = np.array(
            [camera.vertical.x, camera.vertical.y, camera.vertical.z],
            dtype=np.float32
        )
        camera_forward = np.array(
            [camera.forward.x, camera.forward.y, camera.forward.z],
            dtype=np.float32
        )
        camera_right = np.array(
            [camera.right.x, camera.right.y, camera.right.z],
            dtype=np.float32
        )
        camera_up = np.array(
            [camera.up.x, camera.up.y, camera.up.z],
            dtype=np.float32
        )
        curr_focus = camera.focus_dist

        # Transfer camera parameters to pre-allocated device memory
        cuda.to_device(camera_origin, to=self.d_camera_origin, stream=stream)
        cuda.to_device(camera_lower_left, to=self.d_camera_lower_left, stream=stream)
        cuda.to_device(camera_horizontal, to=self.d_camera_horizontal, stream=stream)
        cuda.to_device(camera_vertical, to=self.d_camera_vertical, stream=stream)
        cuda.to_device(camera_forward, to=self.d_camera_forward, stream=stream)
        cuda.to_device(camera_right, to=self.d_camera_right, stream=stream)
        cuda.to_device(camera_up, to=self.d_camera_up, stream=stream)
        cuda.to_device(np.array([curr_focus], dtype=np.float32), to=self.d_curr_focus, stream=stream)
        
        # Ensure depth buffer is allocated
        if self.d_prev_depth is None:
            host_depth = np.zeros((self.width, self.height), dtype=np.float32)
            self.d_prev_depth = cuda.to_device(host_depth, stream=stream)

        # Allocate current frame's depth buffer
        d_depth_buffer = cuda.to_device(np.zeros((self.width, self.height), dtype=np.float32), stream=stream)

        # Reset output buffer - no need to reallocate
        self.d_frame_output.copy_to_device(np.zeros((self.width, self.height, 3), dtype=np.float32), stream=stream)

        # --- Temporal Reprojection Step ---
        # If we have a previous frame and camera parameters, reproject data
        if self.frame_number > 0 and self.prev_camera_origin is not None:
            d_prev_cam_origin = cuda.to_device(self.prev_camera_origin, stream=stream)
            d_prev_cam_lower_left = cuda.to_device(self.prev_camera_lower_left, stream=stream)
            d_prev_cam_horizontal = cuda.to_device(self.prev_camera_horizontal, stream=stream)
            d_prev_cam_vertical = cuda.to_device(self.prev_camera_vertical, stream=stream)
            d_prev_cam_forward = cuda.to_device(self.prev_camera_forward, stream=stream)
            d_prev_cam_right = cuda.to_device(self.prev_camera_right, stream=stream)
            d_prev_cam_up = cuda.to_device(self.prev_camera_up, stream=stream)
            d_prev_focus = cuda.to_device(np.array([self.prev_focus], dtype=np.float32), stream=stream)

            # Allocate temporary buffers for the blended accumulation and sample count
            temp_accum = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32), stream=stream)
            temp_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32), stream=stream)

            # Launch temporal reprojection kernel
            temporal_reprojection_kernel[self.blockspergrid, self.threadsperblock, stream](
                self.width, self.height,
                self.d_prev_accum, self.d_prev_sample_count, self.d_prev_depth,
                d_prev_cam_origin, d_prev_cam_lower_left, d_prev_cam_horizontal, d_prev_cam_vertical,
                d_prev_cam_forward, d_prev_cam_right, d_prev_cam_up, d_prev_focus[0],
                self.d_camera_origin, self.d_camera_lower_left, self.d_camera_horizontal, self.d_camera_vertical,
                self.d_camera_forward, self.d_camera_right, self.d_camera_up, self.d_curr_focus[0],
                temp_accum, temp_sample_count
            )
            stream.synchronize()
            
            # Replace current accumulation buffers with reprojected ones
            self.d_accum_buffer = temp_accum
            self.d_sample_count = temp_sample_count

        # --- Determine BVH parameters ---
        # Get BVH node count if available
        num_nodes = 0
        if hasattr(self, 'd_bvh_bbox_min') and self.d_bvh_bbox_min is not None:
            num_nodes = self.d_bvh_bbox_min.shape[0]

        # --- Launch Adaptive Render Kernel ---
        adaptive_render_kernel[self.blockspergrid, self.threadsperblock, stream](
            self.width, self.height,
            self.d_camera_origin, self.d_camera_lower_left,
            self.d_camera_horizontal, self.d_camera_vertical,
            self.d_sphere_centers, self.d_sphere_radii, 
            self.d_sphere_materials, self.d_sphere_material_types,
            self.d_sphere_roughness,  
            self.d_triangle_vertices, self.d_triangle_uvs, 
            self.d_triangle_materials, self.d_triangle_material_types,
            self.d_triangle_roughness, 
            self.d_texture_data, self.texture_dimensions, self.d_texture_indices,
            self.d_accum_buffer, self.d_accum_buffer_sq, 
            self.d_sample_count, self.d_mask, self.d_frame_output, d_depth_buffer,
            self.frame_number, self.rng_states, self.N,
            self.d_env_map, self.d_env_cdf, self.env_total, self.env_width, self.env_height,
            self.d_halton_table_base2, self.d_halton_table_base3, self.d_halton_table_base5,
            self.d_bvh_bbox_min if num_nodes > 0 else None,
            self.d_bvh_bbox_max if num_nodes > 0 else None,
            self.d_bvh_left if num_nodes > 0 else None,
            self.d_bvh_right if num_nodes > 0 else None,
            self.d_bvh_is_leaf if num_nodes > 0 else None,
            self.d_bvh_object_indices if num_nodes > 0 else None,
            num_nodes
        )
        
        # --- Update Convergence Mask ---
        # Every 8 frames, update which pixels have converged
        if self.frame_number % 8 == 0:
            compute_variance_mask_kernel[self.blockspergrid, self.threadsperblock, stream](
                self.d_accum_buffer, self.d_accum_buffer_sq, 
                self.d_sample_count, self.d_mask, 
                0.001,  # Variance threshold
                32      # Minimum samples before considering convergence
            )
        
        stream.synchronize()

        # Copy the result to host memory
        self.d_frame_output.copy_to_host(self.frame_output, stream=stream)
        stream.synchronize()
        
        # Increment frame counter
        self.frame_number += 1

        # --- Update Previous Frame Data for Next Reprojection ---
        self.prev_camera_origin = camera_origin.copy()
        self.prev_camera_lower_left = camera_lower_left.copy()
        self.prev_camera_horizontal = camera_horizontal.copy()
        self.prev_camera_vertical = camera_vertical.copy()
        self.prev_camera_forward = camera_forward.copy()
        self.prev_camera_right = camera_right.copy()
        self.prev_camera_up = camera_up.copy()
        self.prev_focus = curr_focus

        # Store current state for next frame's temporal reprojection
        self.d_prev_accum = self.d_accum_buffer
        self.d_prev_sample_count = self.d_sample_count
        self.d_prev_depth = d_depth_buffer

        return self.frame_output

    def reset_accumulation(self):
        """Reset all accumulation buffers (current and previous) when the scene or camera changes."""
        host_accum = np.zeros((self.width, self.height, 3), dtype=np.float32)
        host_accum_sq = np.zeros((self.width, self.height, 3), dtype=np.float32)
        host_sample_count = np.zeros((self.width, self.height), dtype=np.int32)
        host_mask = np.zeros((self.width, self.height), dtype=np.int32)
        host_depth = np.zeros((self.width, self.height), dtype=np.float32)

        # Reset our host buffers.
        self.accumulation_buffer.fill(0)
        self.accumulation_buffer_sq.fill(0)
        self.samples = 0

        # Allocate (or reallocate) the device buffers.
        self.d_accum_buffer = cuda.to_device(host_accum)
        self.d_accum_buffer_sq = cuda.to_device(host_accum_sq)
        self.d_sample_count = cuda.to_device(host_sample_count)
        self.d_mask = cuda.to_device(host_mask)

        # Also allocate the previous frame buffers.
        self.d_prev_accum = cuda.to_device(host_accum)
        self.d_prev_sample_count = cuda.to_device(host_sample_count)
        self.d_prev_depth = cuda.to_device(host_depth)

        # Reset previous camera parameters.
        self.prev_camera_origin = None
        self.prev_camera_lower_left = None
        self.prev_camera_horizontal = None
        self.prev_camera_vertical = None
        self.prev_camera_forward = None
        self.prev_camera_right = None
        self.prev_camera_up = None
        self.prev_focus = None

    def update_environment(self, env_map):
        """
        Load an environment map and build its CDF for importance sampling.
        env_map is a NumPy array (H x W x 3) in linear radiance.
        """
        cdf, total, width, height = build_env_cdf(env_map)
        self.d_env_map = cuda.to_device(env_map.astype(np.float32))
        self.d_env_cdf = cuda.to_device(cdf)
        self.env_total = total
        self.env_width = width
        self.env_height = height

