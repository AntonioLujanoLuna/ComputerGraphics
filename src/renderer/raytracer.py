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
            if not hasattr(obj, 'gpu_index'):
                print(f"WARNING: Object {i} has no gpu_index, assigning one")
                if hasattr(obj, 'radius'):
                    # It's a sphere
                    sphere_idx = next((j for j, s in enumerate(sphere_objects) if s is obj), -1)
                    obj.gpu_index = sphere_idx
                elif hasattr(obj, 'triangles'):
                    # It's a mesh
                    mesh_idx = next((j for j, m in enumerate(mesh_objects) if m is obj), -1)
                    if mesh_idx >= 0:
                        obj.gpu_index = len(sphere_objects) + sum(len(mesh_objects[k].triangles) for k in range(mesh_idx))
                    else:
                        obj.gpu_index = -1
                else:
                    obj.gpu_index = -1
                    
        world.build_bvh()
        print(f"  BVH built successfully: {world.bvh_root is not None}")

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
            if not hasattr(obj, 'gpu_index'):
                print(f"WARNING: Object {i} has no gpu_index, assigning one")
                if hasattr(obj, 'radius'):
                    # It's a sphere
                    sphere_idx = next((j for j, s in enumerate(sphere_objects) if s is obj), -1)
                    obj.gpu_index = sphere_idx
                elif hasattr(obj, 'triangles'):
                    # It's a mesh
                    mesh_idx = next((j for j, m in enumerate(mesh_objects) if m is obj), -1)
                    if mesh_idx >= 0:
                        obj.gpu_index = len(sphere_objects) + sum(len(mesh_objects[k].triangles) for k in range(mesh_idx))
                    else:
                        obj.gpu_index = -1
                else:
                    obj.gpu_index = -1
                    
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

        # Check world data to diagnose why we might not be seeing objects
        sphere_count = sum(1 for obj in world.objects if hasattr(obj, 'radius'))
        mesh_objects = [obj for obj in world.objects if hasattr(obj, 'triangles')]
        triangle_count = sum(len(mesh.triangles) for mesh in mesh_objects)
        
        print(f"\n=== Rendering Frame {self.frame_number} ===")
        print(f"Scene contains {len(world.objects)} objects: {sphere_count} spheres, {len(mesh_objects)} meshes with {triangle_count} triangles")
        print(f"Camera position: ({camera.position.x}, {camera.position.y}, {camera.position.z})")
        print(f"Camera direction: ({camera.forward.x}, {camera.forward.y}, {camera.forward.z})")
        
        # More specific check for scene data initialization with better diagnostics
        scene_init_needed = False
        if not hasattr(self, 'd_sphere_centers') or self.d_sphere_centers is None:
            print(f"Scene data missing: d_sphere_centers")
            scene_init_needed = True
        elif not hasattr(self, 'd_sphere_roughness') or self.d_sphere_roughness is None:
            print(f"Scene data missing: d_sphere_roughness")
            scene_init_needed = True
        
        if scene_init_needed:
            print(f"Reinitializing scene with {len(world.objects)} objects ({sphere_count} spheres, {triangle_count} triangles)")
            self.update_scene_data(world)
            # After update, verify we have the data
            if not hasattr(self, 'd_sphere_centers') or self.d_sphere_centers is None:
                print("ERROR: d_sphere_centers still missing after scene update!")
            if not hasattr(self, 'd_sphere_roughness') or self.d_sphere_roughness is None:
                print("ERROR: d_sphere_roughness still missing after scene update!")

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

        # Initialize camera buffers if they don't exist
        if not hasattr(self, 'd_camera_origin') or self.d_camera_origin is None:
            self.d_camera_origin = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_lower_left = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_horizontal = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_vertical = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_forward = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_right = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_camera_up = cuda.to_device(np.zeros(3, dtype=np.float32))
            self.d_curr_focus = cuda.to_device(np.zeros(1, dtype=np.float32))
            self.d_frame_output = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.frame_output = cuda.pinned_array((self.width, self.height, 3), dtype=np.float32)

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
        if not hasattr(self, 'd_prev_depth') or self.d_prev_depth is None:
            host_depth = np.zeros((self.width, self.height), dtype=np.float32)
            self.d_prev_depth = cuda.to_device(host_depth, stream=stream)

        # Check if adaptive sampling buffers exist
        if not hasattr(self, 'd_accum_buffer') or self.d_accum_buffer is None:
            self.d_accum_buffer = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.d_accum_buffer_sq = cuda.to_device(np.zeros((self.width, self.height, 3), dtype=np.float32))
            self.d_sample_count = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))
            self.d_mask = cuda.to_device(np.zeros((self.width, self.height), dtype=np.int32))

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
        has_bvh = False
        if hasattr(self, 'd_bvh_bbox_min') and self.d_bvh_bbox_min is not None:
            num_nodes = self.d_bvh_bbox_min.shape[0]
            has_bvh = True
            print(f"Using BVH with {num_nodes} nodes for acceleration")
        else:
            print("WARNING: No BVH data available for acceleration")

        # Check for any None values in the parameters
        print("Checking for None values in parameters:")
        if self.d_sphere_centers is None: print("d_sphere_centers is None")
        if self.d_sphere_radii is None: print("d_sphere_radii is None")
        if self.d_sphere_materials is None: print("d_sphere_materials is None")
        if self.d_sphere_material_types is None: print("d_sphere_material_types is None")
        if self.d_sphere_roughness is None: print("d_sphere_roughness is None")
        if self.d_triangle_vertices is None: print("d_triangle_vertices is None")
        if self.d_triangle_uvs is None: print("d_triangle_uvs is None")
        if self.d_triangle_materials is None: print("d_triangle_materials is None")
        if self.d_triangle_material_types is None: print("d_triangle_material_types is None")
        if self.d_triangle_roughness is None: print("d_triangle_roughness is None")
        if self.d_texture_data is None: print("d_texture_data is None")
        if self.texture_dimensions is None: print("texture_dimensions is None")
        if self.d_texture_indices is None: print("d_texture_indices is None")
        if self.d_accum_buffer is None: print("d_accum_buffer is None")
        if self.d_accum_buffer_sq is None: print("d_accum_buffer_sq is None")
        if self.d_sample_count is None: print("d_sample_count is None")
        if self.d_mask is None: print("d_mask is None")
        if self.d_frame_output is None: print("d_frame_output is None")
        if d_depth_buffer is None: print("d_depth_buffer is None")
        if self.rng_states is None: print("rng_states is None")
        if self.d_env_map is None: print("d_env_map is None")
        if self.d_env_cdf is None: print("d_env_cdf is None")
        if self.d_halton_table_base2 is None: print("d_halton_table_base2 is None")
        if self.d_halton_table_base3 is None: print("d_halton_table_base3 is None")
        if self.d_halton_table_base5 is None: print("d_halton_table_base5 is None")
        
        # --- Launch Adaptive Render Kernel ---
        # Create empty arrays if BVH is not available to avoid passing None to CUDA
        if not has_bvh:
            empty_array_3d = cuda.to_device(np.zeros((1, 3), dtype=np.float32))
            empty_array_1d = cuda.to_device(np.zeros(1, dtype=np.int32))
            
            # Initialize texture indices if it's None
            if not hasattr(self, 'd_texture_indices') or self.d_texture_indices is None:
                print("Creating empty texture indices array")
                self.d_texture_indices = cuda.to_device(np.zeros(1, dtype=np.int32))
            
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
                empty_array_3d, empty_array_3d, empty_array_1d, empty_array_1d, empty_array_1d, empty_array_1d, 0
            )
        else:
            # Initialize texture indices if it's None
            if not hasattr(self, 'd_texture_indices') or self.d_texture_indices is None:
                print("Creating empty texture indices array")
                self.d_texture_indices = cuda.to_device(np.zeros(1, dtype=np.int32))
                
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
                self.d_bvh_bbox_min, self.d_bvh_bbox_max, self.d_bvh_left, self.d_bvh_right, 
                self.d_bvh_is_leaf, self.d_bvh_object_indices, num_nodes
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

