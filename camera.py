"""Camera.
Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
    - Auto-position camera based on object bounds
    - Multi-view rendering with normalized objects
"""
import argparse
import os
import sapien as sp
import sapien.core as sapien
import numpy as np
from PIL import Image

def get_actor_bounds(actor):
    """Get the axis-aligned bounding box of an actor using collision shapes"""
    min_bound = np.array([np.inf, np.inf, np.inf])
    max_bound = np.array([-np.inf, -np.inf, -np.inf])
    
    # Method 1: Try to use collision shapes to get bounds
    for link in actor.get_links():
        link_pose = link.get_pose()
        
        # Get collision shapes
        for collision_shape in link.get_collision_shapes():
            # Get the geometry bounds
            geometry = collision_shape.geometry
            
            # Different geometry types have different ways to get bounds
            if hasattr(geometry, 'get_bounding_box'):
                try:
                    local_min, local_max = geometry.get_bounding_box()
                except:
                    # Fallback: estimate bounds based on geometry type
                    if hasattr(geometry, 'vertices'):
                        vertices = geometry.vertices
                        local_min = vertices.min(axis=0)
                        local_max = vertices.max(axis=0)
                    else:
                        # Use a default small bounds
                        local_min = np.array([-0.1, -0.1, -0.1])
                        local_max = np.array([0.1, 0.1, 0.1])
            else:
                # Fallback for unknown geometry types
                local_min = np.array([-0.1, -0.1, -0.1])
                local_max = np.array([0.1, 0.1, 0.1])
            
            # Transform bounds to world coordinates
            shape_pose = collision_shape.get_local_pose()
            transform = link_pose.to_transformation_matrix() @ shape_pose.to_transformation_matrix()
            
            # Transform the 8 corners of the bounding box
            corners = np.array([
                [local_min[0], local_min[1], local_min[2]],
                [local_min[0], local_min[1], local_max[2]],
                [local_min[0], local_max[1], local_min[2]],
                [local_min[0], local_max[1], local_max[2]],
                [local_max[0], local_min[1], local_min[2]],
                [local_max[0], local_min[1], local_max[2]],
                [local_max[0], local_max[1], local_min[2]],
                [local_max[0], local_max[1], local_max[2]]
            ])
            
            # Transform corners to world coordinates
            corners_homogeneous = np.hstack([corners, np.ones((8, 1))])
            corners_world = (transform @ corners_homogeneous.T)[:3, :].T
            
            # Update global bounds
            min_bound = np.minimum(min_bound, corners_world.min(axis=0))
            max_bound = np.maximum(max_bound, corners_world.max(axis=0))
    
    # If no valid bounds found, use default
    if np.any(np.isinf(min_bound)) or np.any(np.isinf(max_bound)):
        min_bound = np.array([-1, -1, -1])
        max_bound = np.array([1, 1, 1])
    
    return min_bound, max_bound

def normalize_and_center_object(actor):
    """Center the object and return normalization info for camera positioning"""
    # Get object bounds
    min_bound, max_bound = get_actor_bounds(actor)
    
    # Calculate object center and size
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    max_extent = np.max(size)
    
    # Calculate diagonal length
    diagonal_length = np.linalg.norm(size)
    
    # Move object to be centered at origin (0, 0, 0)
    target_center = np.array([0, 0, 0])  # Center at origin for unit sphere sampling
    translation = target_center - center
    
    # Apply translation to actor
    current_pose = actor.get_pose()
    new_pose = sapien.Pose(p=current_pose.p + translation, q=current_pose.q)
    actor.set_pose(new_pose)
    
    # Calculate scale factor to fit in unit cube
    # We want the largest dimension to be 1.0
    scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
    
    return {
        'center': target_center,
        'size': size,
        'scaled_size': size * scale_factor,  # What the size would be after scaling
        'max_extent': max_extent,
        'diagonal_length': diagonal_length,
        'scale_factor': scale_factor,  # This is what we need to apply via camera distance
        'translation': translation
    }

def position_camera_for_object(scene, camera, camera_mount_actor, normalization_info, azimuth=45, elevation=30, distance_factor=2.5):
    """Position camera on sphere around normalized object"""
    center = normalization_info['center']
    diagonal_length = normalization_info['diagonal_length']
    
    # Use diagonal length for consistent camera distance
    cam_distance = diagonal_length * distance_factor
    
    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    
    # Calculate camera position on sphere
    x = cam_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = cam_distance * np.cos(elevation_rad) * np.sin(azimuth_rad) 
    z = cam_distance * np.sin(elevation_rad)
    
    cam_pos = center + np.array([x, y, z])
    
    # Compute camera orientation to look at object center
    forward = (center - cam_pos)
    forward = forward / np.linalg.norm(forward)
    
    # Use world up as reference
    world_up = np.array([0, 0, 1])
    
    # Right-handed coordinate system
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:  # Handle edge case when forward is parallel to world_up
        right = np.array([1, 0, 0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    
    # SAPIEN expects [forward, left, up]
    left = -right
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    engine = sapien.Engine()
    renderer = sapien.SapienRenderer()
    engine.set_renderer(renderer)
    
    scene = engine.create_scene()
    scene.set_timestep(1 / 100.0)
    
    # Load URDF
    loader = scene.create_urdf_loader()
    loader.fix_root_link = True
    
    token = os.getenv("TOKEN_SAPIEN_STR", None)
    assert token is not None, "Please set TOKEN_SAPIEN_STR environment variable"
    
    urdf_file = sp.asset.download_partnet_mobility(args.object_id, token)
    asset = loader.load_kinematic(urdf_file)
    assert asset, "Failed to load URDF."
    
    # Normalize and center the object
    normalization_info = normalize_and_center_object(asset)
    
    # Set up lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([2, 2, 2], [1, 1, 1])
    scene.add_point_light([2, -2, 2], [1, 1, 1])
    scene.add_point_light([-2, 0, 2], [1, 1, 1])
    
    # ---------------------------------------------------------------------------- #
    # Camera Setup
    # ---------------------------------------------------------------------------- #
    near, far = 0.1, 100
    width, height = args.image_shape[0], args.image_shape[1]
    
    camera = scene.add_camera(
        name="camera",
        width=width,
        height=height,
        fovy=np.deg2rad(35),
        near=near,
        far=far,
    )
    
    # Create camera mount
    camera_mount_actor = scene.create_actor_builder().build_kinematic()
    camera.set_parent(parent=camera_mount_actor, keep_pose=False)
    
    # Position camera based on normalized object
    position_camera_for_object(scene, camera, camera_mount_actor, normalization_info,
                             azimuth=args.azimuth,
                             elevation=args.elevation)
    
    # Update scene
    scene.step()
    scene.update_render()
    camera.take_picture()
    
    ## ---------------------------------------------------------------------------- #
    ## Render RGBA
    ## ---------------------------------------------------------------------------- #
    #rgba = camera.get_float_texture('Color')  # [H, W, 4]
    #rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    #rgba_pil = Image.fromarray(rgba_img)
    
    ## Create output filename
    #output_name = f'color_obj{args.object_id}_az{args.azimuth}_el{args.elevation}.png'
    #rgba_pil.save(output_name)
    
    # ---------------------------------------------------------------------------- #
    # Render Depth
    # ---------------------------------------------------------------------------- #
    position = camera.get_float_texture('Position')  # [H, W, 4]
    depth = -position[..., 2]  # OpenGL convention: -z is forward
    
    ## Convert to millimeters and save as 16-bit PNG
    #depth_image = (depth * 1000.0).astype(np.uint16)
    #depth_pil = Image.fromarray(depth_image)
    #depth_name = f'depth_obj{args.object_id}_az{args.azimuth}_el{args.elevation}.png'
    #depth_pil.save(depth_name)
    
    # Optional: Save normalized depth for visualization
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_vis_pil = Image.fromarray(depth_normalized)
    vis_name = f'depth_normalized_obj{args.object_id}_az{args.azimuth}_el{args.elevation}.png'
    depth_vis_pil.save(os.path.join(args.output_dir, vis_name))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--object_id", type=int, default=179)
    parser.add_argument("--image_shape",
                        type=lambda x: [int(y) for y in x.split(",")],
                        default=[512, 512])
    parser.add_argument("--azimuth", type=float, default=45,
                        help="Camera azimuth angle in degrees")
    parser.add_argument("--elevation", type=float, default=30,
                        help="Camera elevation angle in degrees")
    args = parser.parse_args()
    main(args)