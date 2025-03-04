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
from .tone_mapping import reinhard_tone_mapping

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

        # Initialize empty placeholders for scene data
        self.d_sphere_centers = None
        self.d_sphere_radii = None
        self.d_sphere_materials = None
        self.d_sphere_material_types = None
        self.d_sphere_roughness = None  # Explicitly initialize this attribute
        self.d_triangle_vertices = None
        self.d_triangle_uvs = None
        self.d_triangle_materials = None
        self.d_triangle_material_types = None
        self.d_triangle_roughness = None  # Explicitly initialize this attribute
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

        # Initialize depth buffer
        self.d_depth_buffer = cuda.to_device(np.ones((width, height), dtype=np.float32) * 1e20)
        
        # Initialize previous frame buffers for temporal reprojection
        self.d_prev_accum = cuda.to_device(np.zeros((width, height, 3), dtype=np.float32))
        self.d_prev_sample_count = cuda.to_device(np.zeros((width, height), dtype=np.int32))
        self.d_prev_depth = cuda.to_device(np.ones((width, height), dtype=np.float32) * 1e20)
        
        # These BVH attributes should be explicitly initialized too
        self.d_bvh_bbox_min = None
        self.d_bvh_bbox_max = None
        self.d_bvh_left = None
        self.d_bvh_right = None
        self.d_bvh_is_leaf = None
        self.d_bvh_object_indices = None

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

        except Exception as e:
            print(f"Warning: Error during CUDA cleanup: {str(e)}")

    def update_scene_data(self, world) -> None:
        """
        Update the scene data on the GPU.
        
        This method:
        1. Extracts all spheres and triangles from the world
        2. Prepares arrays for their properties (positions, materials, etc.)
        3. Uploads these arrays to the GPU
        4. Builds and uploads the BVH acceleration structure
        
        Parameters:
            world: The scene containing objects to render
        """
        print("\n=== Updating Scene Data ===")
        print(f"World contains {len(world.objects)} objects")
        
        # Extract all spheres and triangles from the world
        sphere_objects = []
        mesh_objects = []
        
        for obj in world.objects:
            if hasattr(obj, 'radius'):
                sphere_objects.append(obj)
            elif hasattr(obj, 'triangles'):
                mesh_objects.append(obj)
        
        print(f"Found {len(sphere_objects)} spheres and {len(mesh_objects)} meshes")
        
        # Count total triangles
        triangle_count = sum(len(mesh.triangles) for mesh in mesh_objects)
        print(f"Total triangles: {triangle_count}")
        
        # Prepare arrays for sphere data
        centers = np.zeros((len(sphere_objects), 3), dtype=np.float32)
        radii = np.zeros(len(sphere_objects), dtype=np.float32)
        sphere_materials = np.zeros(len(sphere_objects) * 3, dtype=np.float32)
        sphere_material_types = np.zeros(len(sphere_objects), dtype=np.int32)
        sphere_roughness = np.zeros(len(sphere_objects), dtype=np.float32)
        
        # Prepare arrays for triangle data
        triangle_vertices = np.zeros((triangle_count, 9), dtype=np.float32)  # 3 vertices * 3 coordinates
        triangle_uvs = np.zeros((triangle_count, 6), dtype=np.float32)  # 3 vertices * 2 UV coordinates
        triangle_materials = np.zeros(triangle_count * 3, dtype=np.float32)
        triangle_material_types = np.zeros(triangle_count, dtype=np.int32)
        triangle_roughness = np.zeros(triangle_count, dtype=np.float32)
        
        # Fill sphere data
        sphere_idx = 0
        for obj in sphere_objects:
            # Assign a GPU index for BVH
            obj.gpu_index = sphere_idx
            
            # Fill position and radius
            centers[sphere_idx] = [obj.center.x, obj.center.y, obj.center.z]
            radii[sphere_idx] = obj.radius
            
            # Fill material data
            mat_idx = sphere_idx * 3
            mat = obj.material
            
            # Determine material type and properties
            if hasattr(mat, 'emitted') and callable(mat.emitted):
                emission = mat.emitted(0, 0, obj.center)
                sphere_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                sphere_material_types[sphere_idx] = 2  # Emissive
                sphere_roughness[sphere_idx] = 0.0
                print(f"    Sphere {sphere_idx}: Emissive material: ({emission.x}, {emission.y}, {emission.z})")
            elif isinstance(mat, Metal):
                if hasattr(mat, 'texture') and mat.texture is not None:
                    if hasattr(mat.texture, 'color'):
                        sphere_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                        print(f"    Sphere {sphere_idx}: Metal material: ({mat.texture.color.x}, {mat.texture.color.y}, {mat.texture.color.z})")
                    else:
                        color = mat.texture.sample(UV(0.5, 0.5))
                        sphere_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                        print(f"    Sphere {sphere_idx}: Metal material with texture: ({color.x}, {color.y}, {color.z})")
                else:
                    # Default metal color if no texture
                    sphere_materials[mat_idx:mat_idx+3] = [0.8, 0.8, 0.8]  # Default to silver
                    print(f"    Sphere {sphere_idx}: Metal material with default color (0.8, 0.8, 0.8)")
                sphere_material_types[sphere_idx] = 1  # Metal
                sphere_roughness[sphere_idx] = mat.fuzz if hasattr(mat, 'fuzz') else 0.1
            elif isinstance(mat, Dielectric):
                sphere_materials[mat_idx:mat_idx+3] = [mat.ref_idx, 0.0, 0.0]
                sphere_material_types[sphere_idx] = 3  # Dielectric
                sphere_roughness[sphere_idx] = 0.0
                print(f"    Sphere {sphere_idx}: Dielectric material: ref_idx = {mat.ref_idx}")
            elif isinstance(mat, MicrofacetMetal):
                sphere_materials[mat_idx:mat_idx+3] = [mat.albedo.x, mat.albedo.y, mat.albedo.z]
                sphere_material_types[sphere_idx] = 4  # MicrofacetMetal
                sphere_roughness[sphere_idx] = mat.roughness
                print(f"    Sphere {sphere_idx}: Microfacet material: ({mat.albedo.x}, {mat.albedo.y}, {mat.albedo.z}) roughness={mat.roughness}")
            else:
                # Default to Lambertian
                if hasattr(mat, 'texture') and mat.texture is not None:
                    if hasattr(mat.texture, 'color'):
                        sphere_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                        print(f"    Sphere {sphere_idx}: Lambertian material: ({mat.texture.color.x}, {mat.texture.color.y}, {mat.texture.color.z})")
                    else:
                        color = mat.texture.sample(UV(0.5, 0.5))
                        sphere_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                        print(f"    Sphere {sphere_idx}: Lambertian material with texture: ({color.x}, {color.y}, {color.z})")
                else:
                    # Default lambertian color if no texture
                    sphere_materials[mat_idx:mat_idx+3] = [0.5, 0.5, 0.5]  # Default to gray
                    print(f"    Sphere {sphere_idx}: Lambertian material with default color (0.5, 0.5, 0.5)")
                sphere_material_types[sphere_idx] = 0  # Lambertian or other
                sphere_roughness[sphere_idx] = 0.0
            
            sphere_idx += 1
        
        # Fill triangle data
        triangle_idx = 0
        for mesh_idx, mesh in enumerate(mesh_objects):
            print(f"  Adding mesh {mesh_idx} with {len(mesh.triangles)} triangles")
            
            # Assign GPU indices to the mesh itself
            mesh.gpu_index = len(sphere_objects) + triangle_idx
            
            for tri_idx, triangle in enumerate(mesh.triangles):
                # Assign GPU indices to each triangle
                triangle.gpu_index = len(sphere_objects) + triangle_idx
                
                # Fill triangle vertex data
                triangle_vertices[triangle_idx, 0:3] = [triangle.v0.x, triangle.v0.y, triangle.v0.z]
                triangle_vertices[triangle_idx, 3:6] = [triangle.v1.x, triangle.v1.y, triangle.v1.z]
                triangle_vertices[triangle_idx, 6:9] = [triangle.v2.x, triangle.v2.y, triangle.v2.z]
                
                # Fill UV data if available
                if hasattr(triangle, 'uv0') and triangle.uv0 is not None:
                    triangle_uvs[triangle_idx, 0:2] = [triangle.uv0.u, triangle.uv0.v]
                    triangle_uvs[triangle_idx, 2:4] = [triangle.uv1.u, triangle.uv1.v]
                    triangle_uvs[triangle_idx, 4:6] = [triangle.uv2.u, triangle.uv2.v]
                
                # Fill material data
                mat_idx = triangle_idx * 3
                mat = mesh.material
                
                if hasattr(mat, 'emitted') and callable(mat.emitted):
                    emission = mat.emitted(0, 0, triangle.v0)
                    triangle_materials[mat_idx:mat_idx+3] = [emission.x, emission.y, emission.z]
                    triangle_material_types[triangle_idx] = 2  # Emissive
                    triangle_roughness[triangle_idx] = 0.0
                    print(f"    Triangle {triangle_idx}: Emissive material: ({emission.x}, {emission.y}, {emission.z})")
                elif isinstance(mat, Metal):
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):
                            triangle_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                            print(f"    Triangle {triangle_idx}: Metal material: ({mat.texture.color.x}, {mat.texture.color.y}, {mat.texture.color.z})")
                        else:
                            color = mat.texture.sample(UV(0.5, 0.5))
                            triangle_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                            print(f"    Triangle {triangle_idx}: Metal material with texture")
                    else:
                        # Default metal color if no texture
                        triangle_materials[mat_idx:mat_idx+3] = [0.8, 0.8, 0.8]  # Default to silver
                        print(f"    Triangle {triangle_idx}: Metal material with default color")
                    triangle_material_types[triangle_idx] = 1  # Metal
                    triangle_roughness[triangle_idx] = mat.fuzz if hasattr(mat, 'fuzz') else 0.1
                elif isinstance(mat, Dielectric):
                    triangle_materials[mat_idx:mat_idx+3] = [mat.ref_idx, 0.0, 0.0]
                    triangle_material_types[triangle_idx] = 3  # Dielectric
                    triangle_roughness[triangle_idx] = 0.0
                    print(f"    Triangle {triangle_idx}: Dielectric material: ref_idx = {mat.ref_idx}")
                else:
                    # Default to Lambertian
                    if hasattr(mat, 'texture') and mat.texture is not None:
                        if hasattr(mat.texture, 'color'):
                            triangle_materials[mat_idx:mat_idx+3] = [mat.texture.color.x, mat.texture.color.y, mat.texture.color.z]
                            print(f"    Triangle {triangle_idx}: Lambertian material: ({mat.texture.color.x}, {mat.texture.color.y}, {mat.texture.color.z})")
                        else:
                            color = mat.texture.sample(UV(0.5, 0.5))
                            triangle_materials[mat_idx:mat_idx+3] = [color.x, color.y, color.z]
                            print(f"    Triangle {triangle_idx}: Lambertian material with texture")
                    else:
                        # Default lambertian color if no texture
                        triangle_materials[mat_idx:mat_idx+3] = [0.5, 0.5, 0.5]  # Default to gray
                        print(f"    Triangle {triangle_idx}: Lambertian material with default color")
                    triangle_material_types[triangle_idx] = 0  # Lambertian or other
                    triangle_roughness[triangle_idx] = 0.0
                
                triangle_idx += 1
        
        # Upload data to GPU
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

        # Process texture data
        textures = set()
        for obj in world.objects:
            if hasattr(obj, 'material') and hasattr(obj.material, 'texture'):
                textures.add(obj.material.texture)
        
        # Ensure we have at least a default texture
        if not textures:
            print("No textures found, creating default texture")
            default_texture_data = np.array([0.5, 0.5, 0.5], dtype=np.float32)  # Default gray
            self.d_texture_data = cuda.to_device(default_texture_data)
            self.texture_dimensions = cuda.to_device(np.array([[1, 1]], dtype=np.int32))
            self.d_texture_indices = cuda.to_device(np.zeros(1, dtype=np.int32))
        else:
            self.load_textures(list(textures))
            
            # Ensure texture indices is initialized
            if not hasattr(self, 'd_texture_indices') or self.d_texture_indices is None:
                print("Initializing texture indices")
                num_objects = len(sphere_objects) + triangle_count
                self.d_texture_indices = cuda.to_device(np.zeros(num_objects, dtype=np.int32))

        # Allocate adaptive sampling buffers
        self.d_accum_buffer = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
        self.d_accum_buffer_sq = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
        self.d_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
        self.d_mask = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))

        # Build the BVH over the world objects
        print(f"Building BVH for {len(world.objects)} objects...")
        
        # Ensure all objects have a gpu_index attribute before building the BVH
        for i, obj in enumerate(world.objects):
            if not hasattr(obj, 'gpu_index') or obj.gpu_index < 0:
                print(f"Setting GPU index for object {i}")
                if hasattr(obj, 'radius'):
                    # It's a sphere
                    # Find the object in sphere_objects by comparing object identity, not just type
                    sphere_idx = -1
                    for j, s in enumerate(sphere_objects):
                        if s is obj:  # Use identity comparison (is) instead of equality (==)
                            sphere_idx = j
                            break
                    
                    # Only assign if we found a valid index
                    if sphere_idx >= 0:
                        obj.gpu_index = sphere_idx
                        print(f"  Assigned sphere index {sphere_idx} to object {i}")
                    else:
                        obj.gpu_index = -1
                        print(f"  WARNING: Could not find matching sphere for object {i}")
                        
                elif hasattr(obj, 'triangles'):
                    # It's a mesh
                    mesh_idx = -1
                    for j, m in enumerate(mesh_objects):
                        if m is obj:  # Use identity comparison
                            mesh_idx = j
                            break
                    
                    if mesh_idx >= 0:
                        # Calculate correct base index for the first triangle of this mesh
                        base_idx = len(sphere_objects)
                        for k in range(mesh_idx):
                            base_idx += len(mesh_objects[k].triangles)
                        
                        obj.gpu_index = base_idx
                        print(f"  Assigned mesh base index {base_idx} to object {i} with {len(obj.triangles)} triangles")
                        
                        # Also assign indices to individual triangles
                        for t_idx, triangle in enumerate(obj.triangles):
                            triangle.gpu_index = base_idx + t_idx
                    else:
                        obj.gpu_index = -1
                        print(f"  WARNING: Could not find matching mesh for object {i}")
                else:
                    obj.gpu_index = -1
                    print(f"  WARNING: Unknown object type for object {i}")
        
        # Verify all objects have valid GPU indices
        invalid_indices = [i for i, obj in enumerate(world.objects) if not hasattr(obj, 'gpu_index') or obj.gpu_index < 0]
        if invalid_indices:
            print(f"  WARNING: {len(invalid_indices)} objects still have invalid GPU indices: {invalid_indices}")
        else:
            print(f"  All {len(world.objects)} objects have valid GPU indices")
            
        # Build the BVH after assigning all GPU indices
        world.build_bvh()
        print(f"  BVH built successfully: {world.bvh_root is not None}")

        # Upload BVH data to GPU
        if world.bvh_flat is not None:
            bbox_min, bbox_max, left_indices, right_indices, is_leaf, object_indices = world.bvh_flat
            self.d_bvh_bbox_min = cuda.to_device(bbox_min)
            self.d_bvh_bbox_max = cuda.to_device(bbox_max)
            self.d_bvh_left = cuda.to_device(left_indices)
            self.d_bvh_right = cuda.to_device(right_indices)
            self.d_bvh_is_leaf = cuda.to_device(is_leaf)
            self.d_bvh_object_indices = cuda.to_device(object_indices)
            print(f"  BVH data uploaded: {len(object_indices)} nodes")
            
            # Print some BVH node data for debugging
            print(f"  BVH root bounding box: min=({bbox_min[0][0]}, {bbox_min[0][1]}, {bbox_min[0][2]}), max=({bbox_max[0][0]}, {bbox_max[0][1]}, {bbox_max[0][2]})")
            
            # Check if any objects are assigned to leaf nodes
            leaf_count = sum(1 for i in range(len(is_leaf)) if is_leaf[i] == 1)
            valid_objects = sum(1 for i in range(len(is_leaf)) if is_leaf[i] == 1 and object_indices[i] >= 0)
            print(f"  BVH leaf nodes: {leaf_count}, with valid objects: {valid_objects}")
            
            # Check if object indices match the number of objects
            if valid_objects != len(world.objects):
                print(f"  WARNING: Mismatch between valid BVH object indices ({valid_objects}) and world objects ({len(world.objects)})")
                
                # Print the first few object indices for debugging
                print(f"  First 10 object indices: {object_indices[:min(10, len(object_indices))]}")
        else:
            print("  WARNING: No BVH data available to upload")
            self.d_bvh_bbox_min = None
            self.d_bvh_bbox_max = None
            self.d_bvh_left = None
            self.d_bvh_right = None
            self.d_bvh_is_leaf = None
            self.d_bvh_object_indices = None
            
        # Verify all required buffers are initialized
        print("Scene data initialization complete.")
        print(f"  d_sphere_centers: {self.d_sphere_centers is not None}")
        print(f"  d_sphere_roughness: {self.d_sphere_roughness is not None}")
        print(f"  d_triangle_vertices: {self.d_triangle_vertices is not None}")
        print(f"  d_triangle_roughness: {self.d_triangle_roughness is not None}")
        print(f"  d_accum_buffer: {self.d_accum_buffer is not None}")

    def render_frame_adaptive(self, camera, world) -> np.ndarray:
        """
        Renders a frame using adaptive sampling based on temporal reprojection and variance analysis.
        
        This version adaptively samples pixels with high variance, and reuses pixel data 
        from the previous frame through temporal reprojection when the camera motion is small.
        """
        # Update frame count
        self.frame_number += 1
        
        # Store camera parameters for temporal reprojection
        camera_origin_host = np.array([camera.position.x, camera.position.y, camera.position.z], dtype=np.float32)
        
        # ---- Performance optimization: Cache last frame's camera state in class variables ----
        if not hasattr(self, 'prev_camera_origin_host'):
            self.prev_camera_origin_host = np.zeros(3, dtype=np.float32)
            self.prev_camera_forward_host = np.zeros(3, dtype=np.float32)
        
        # ---- Reuse arrays rather than allocating new ones each frame ----
        camera_motion = False
        camera_forward_host = np.array([camera.forward.x, camera.forward.y, camera.forward.z], dtype=np.float32)
        
        # Calculate camera motion - check both position and orientation changes
        camera_origin_diff = np.linalg.norm(camera_origin_host - self.prev_camera_origin_host)
        forward_diff = np.linalg.norm(camera_forward_host - self.prev_camera_forward_host)
        
        # Detect any significant change in camera position or orientation
        position_changed = camera_origin_diff > 0.0001  # Position threshold
        rotation_changed = forward_diff > 0.0001        # Rotation threshold
        camera_motion = position_changed or rotation_changed
        
        # Only print debug info occasionally to reduce CPU overhead
        if self.frame_number % 60 == 0:
            print(f"Movement: sqrt({camera_origin_diff**2:.6f}) = {camera_origin_diff:.6f}, rotation: {forward_diff:.6f}, threshold: 0.05")
        
        if camera_motion and rotation_changed and self.frame_number % 10 == 0:
            print(f"Resetting accumulation due to rotation: {forward_diff:.6f} > 0.0001")
        elif camera_motion and position_changed and self.frame_number % 10 == 0:
            print(f"Resetting accumulation due to movement: {camera_origin_diff:.6f} > 0.0001")
        
        # Store current camera parameters for next frame's comparison
        np.copyto(self.prev_camera_forward_host, camera_forward_host)
        np.copyto(self.prev_camera_origin_host, camera_origin_host)
        
        # ---- Minimize host-device transfers ----
        # Calculate viewport corners and upload to device in a single batch if possible
        if not hasattr(self, 'camera_params_host'):
            # First-time allocation
            self.camera_params_host = np.zeros((4, 3), dtype=np.float32)  # [origin, lower_left, horizontal, vertical]
        
        # Fill the host array
        self.camera_params_host[0, :] = camera_origin_host  
        self.camera_params_host[1, :] = [camera.lower_left_corner.x, camera.lower_left_corner.y, camera.lower_left_corner.z]
        self.camera_params_host[2, :] = [camera.horizontal.x, camera.horizontal.y, camera.horizontal.z]
        self.camera_params_host[3, :] = [camera.vertical.x, camera.vertical.y, camera.vertical.z]
        
        # Transfer to device (only the used parts)
        self.d_camera_origin.copy_to_device(self.camera_params_host[0])
        self.d_camera_lower_left.copy_to_device(self.camera_params_host[1])
        self.d_camera_horizontal.copy_to_device(self.camera_params_host[2])
        self.d_camera_vertical.copy_to_device(self.camera_params_host[3])
        
        # Similarly batch camera orientation vectors together
        if not hasattr(self, 'camera_orientation_host'):
            self.camera_orientation_host = np.zeros((3, 3), dtype=np.float32)  # [forward, right, up]
        
        self.camera_orientation_host[0, :] = camera_forward_host
        self.camera_orientation_host[1, :] = [camera.right.x, camera.right.y, camera.right.z]
        self.camera_orientation_host[2, :] = [camera.up.x, camera.up.y, camera.up.z]
        
        self.d_camera_forward.copy_to_device(self.camera_orientation_host[0])
        self.d_camera_right.copy_to_device(self.camera_orientation_host[1]) 
        self.d_camera_up.copy_to_device(self.camera_orientation_host[2])
        
        # Set current focus distance
        curr_focus_host = np.array([camera.focus_dist], dtype=np.float32)
        self.d_curr_focus.copy_to_device(curr_focus_host)
        
        # Initialize buffers if they don't exist
        if self.d_mask is None:
            self.d_mask = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
        
        if self.d_accum_buffer is None:
            # If buffers don't exist, create them
            self.d_accum_buffer = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.d_accum_buffer_sq = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.d_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
            self.d_mask = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
        
        # Only initialize depth buffer for first frame or after reset
        if self.d_depth_buffer is None:
            self.d_depth_buffer = cuda.to_device(np.ones((self.width, self.height), dtype=np.float32) * 1e20)
            
        # Allocate buffers for temporal reprojection
        if self.d_prev_accum is None:
            self.d_prev_accum = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.d_prev_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
            self.d_prev_depth = cuda.to_device(np.ones((self.width, self.height), dtype=np.float32) * 1e20)
        
        # Swap for next frame
        # Store current camera parameters for next frame's reprojection
        if self.prev_camera_origin is None:
            self.prev_camera_origin = cuda.to_device(camera_origin_host)
            self.prev_camera_forward = cuda.to_device(camera_forward_host)
            self.prev_camera_right = cuda.to_device(self.camera_orientation_host[1])
            self.prev_camera_up = cuda.to_device(self.camera_orientation_host[2])
            self.prev_focus = cuda.to_device(curr_focus_host)
        else:
            # Update only when camera actually moved to avoid unnecessary transfers
            if camera_motion:
                self.prev_camera_origin.copy_to_device(camera_origin_host)
                self.prev_camera_forward.copy_to_device(camera_forward_host)
                self.prev_camera_right.copy_to_device(self.camera_orientation_host[1])
                self.prev_camera_up.copy_to_device(self.camera_orientation_host[2])
                self.prev_focus.copy_to_device(curr_focus_host)
        
        # ---- Optimize buffer management ----
        # Swap pointers to avoid data copying
        if self.frame_number > 1:
            # Swap pointers to avoid data copying
            self.d_prev_accum, self.d_accum_buffer = self.d_accum_buffer, self.d_prev_accum
            self.d_prev_sample_count, self.d_sample_count = self.d_sample_count, self.d_prev_sample_count
            self.d_prev_depth, self.d_depth_buffer = self.d_depth_buffer, self.d_prev_depth
            
            # Clear accumulation buffers for current frame - only if camera moved
            if camera_motion:
                # Use CUDA kernels to clear buffers directly on GPU - much faster than transferring from host
                # These operations can be streamed for parallelism
                
                # Define a generic kernel to clear arrays
                threads_per_block = (16, 16)
                blocks_per_grid_x = (self.width + threads_per_block[0] - 1) // threads_per_block[0]
                blocks_per_grid_y = (self.height + threads_per_block[1] - 1) // threads_per_block[1]
                
                # Create and launch kernels to clear buffers
                @cuda.jit
                def clear_float_buffer(d_buffer, value):
                    x, y = cuda.grid(2)
                    if x < d_buffer.shape[0] and y < d_buffer.shape[1]:
                        for c in range(d_buffer.shape[2]):
                            d_buffer[x, y, c] = value
                
                @cuda.jit
                def clear_int_buffer(d_buffer, value):
                    x, y = cuda.grid(2)
                    if x < d_buffer.shape[0] and y < d_buffer.shape[1]:
                        d_buffer[x, y] = value
                
                @cuda.jit
                def reset_depth_buffer(d_depth, far_value):
                    x, y = cuda.grid(2)
                    if x < d_depth.shape[0] and y < d_depth.shape[1]:
                        d_depth[x, y] = far_value
                
                # Clear all buffers
                clear_float_buffer[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
                    self.d_accum_buffer, 0.0
                )
                clear_float_buffer[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
                    self.d_accum_buffer_sq, 0.0
                )
                clear_int_buffer[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
                    self.d_sample_count, 0
                )
                reset_depth_buffer[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
                    self.d_depth_buffer, 1e20
                )
                clear_int_buffer[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](
                    self.d_mask, 0
                )
        
        # Compute variance and create adaptive sampling mask
        # Only update the mask every few frames to amortize cost
        if self.frame_number % 4 == 0:
            compute_variance_mask_kernel[self.blockspergrid, self.threadsperblock](
                self.d_accum_buffer, self.d_accum_buffer_sq,
                self.d_sample_count, self.d_mask,
                0.005,  # variance threshold - lower means more refinement
                10      # min samples before checking convergence
            )
        
        # BVH acceleration structure parameters
        num_nodes = 0
        
        if hasattr(world, 'bvh') and world.bvh is not None:
            num_nodes = world.bvh.num_nodes
        
        # ---- Optimize kernel launch strategy ----
        # Process samples in batches to keep kernel execution time reasonable
        # Use smaller batch sizes when quality is higher to prevent CUDA timeouts
        batch_size = 4  # Starting batch size
        if self.N > 8:
            batch_size = 2  # Smaller batch size for higher quality settings to avoid timeouts
        
        # Store streams for asynchronous execution
        if not hasattr(self, 'cuda_streams'):
            self.cuda_streams = [cuda.stream() for _ in range(2)]
        
        # Launch the adaptive rendering kernel in batches with stream synchronization
        for batch_start in range(0, self.N, batch_size):
            batch_end = min(batch_start + batch_size, self.N)
            batch_samples = batch_end - batch_start
            
            # Alternate between streams for overlapped execution
            stream_idx = (batch_start // batch_size) % len(self.cuda_streams)
            
            # Call the kernel with the current batch and explicitly name the BVH parameters
            adaptive_render_kernel[self.blockspergrid, self.threadsperblock, self.cuda_streams[stream_idx]](
                self.width, self.height,
                self.d_camera_origin, self.d_camera_lower_left,
                self.d_camera_horizontal, self.d_camera_vertical,
                self.d_sphere_centers, self.d_sphere_radii,
                self.d_sphere_materials, self.d_sphere_material_types, self.d_sphere_roughness,
                self.d_triangle_vertices, self.d_triangle_uvs,
                self.d_triangle_materials, self.d_triangle_material_types, self.d_triangle_roughness,
                self.d_texture_data, self.texture_dimensions, self.d_texture_indices,
                self.d_accum_buffer, self.d_accum_buffer_sq,
                self.d_sample_count, self.d_mask, self.d_frame_output, self.d_depth_buffer,
                self.frame_number + batch_start, self.rng_states, batch_samples,
                self.d_env_map, self.d_env_cdf, self.env_total, self.env_width, self.env_height,
                self.d_halton_table_base2, self.d_halton_table_base3, self.d_halton_table_base5,
                self.d_bvh_bbox_min if hasattr(self, 'd_bvh_bbox_min') and self.d_bvh_bbox_min is not None else None,
                self.d_bvh_bbox_max if hasattr(self, 'd_bvh_bbox_max') and self.d_bvh_bbox_max is not None else None,
                self.d_bvh_left if hasattr(self, 'd_bvh_left') and self.d_bvh_left is not None else None,
                self.d_bvh_right if hasattr(self, 'd_bvh_right') and self.d_bvh_right is not None else None,
                self.d_bvh_is_leaf if hasattr(self, 'd_bvh_is_leaf') and self.d_bvh_is_leaf is not None else None,
                self.d_bvh_object_indices if hasattr(self, 'd_bvh_object_indices') and self.d_bvh_object_indices is not None else None,
                num_nodes
            )
        
        # Synchronize all streams before proceeding
        cuda.synchronize()
        
        # Asynchronously copy the frame output to host memory
        self.d_frame_output.copy_to_host(self.frame_output)
        
        # Apply tone mapping to the frame
        tone_mapped = reinhard_tone_mapping(self.frame_output)
        
        # Increment sample count
        self.samples += self.N
        
        return tone_mapped

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

