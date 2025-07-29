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
        print("Warning: Could not determine object bounds, using default")
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
    
    print(f"Object center: {center}")
    print(f"Object size: {size}")
    print(f"Max extent: {max_extent}")
    
    # Move object to be centered at origin (0, 0, 0)
    target_center = np.array([0, 0, 0])  # Center at origin for unit sphere sampling
    translation = target_center - center
    
    # Apply translation to actor
    current_pose = actor.get_pose()
    new_pose = sapien.Pose(p=current_pose.p + translation, q=current_pose.q)
    actor.set_pose(new_pose)
    
    print(f"Translated object by: {translation}")
    print(f"New center should be at: {target_center}")
    
    # Calculate scale factor to fit in unit cube
    # We want the largest dimension to be 1.0
    scale_factor = 1.0 / max_extent if max_extent > 0 else 1.0
    
    print(f"Scale factor needed for unit cube: {scale_factor}")
    print(f"After scaling, size would be: {size * scale_factor}")
    
    return {
        'center': target_center,
        'size': size,
        'scaled_size': size * scale_factor,  # What the size would be after scaling
        'max_extent': max_extent,
        'scale_factor': scale_factor,  # This is what we need to apply via camera distance
        'translation': translation
    }

def position_camera_for_object(scene, camera, camera_mount_actor, normalization_info, azimuth=45, elevation=30, distance_factor=2.5):
    """Position camera on sphere around normalized object"""
    center = normalization_info['center']
    scale_factor = normalization_info['scale_factor']
    
    # IMPORTANT: Use the scale factor to simulate unit cube normalization
    # Since we can't scale geometry, we scale the camera distance inversely
    # If object is large (scale_factor small), camera should be far
    # If object is small (scale_factor large), camera should be close
    cam_distance = distance_factor / scale_factor
    
    print(f"Object scale_factor: {scale_factor:.3f}")
    print(f"Base distance_factor: {distance_factor}")
    print(f"Final camera distance: {cam_distance:.3f}")
    print(f"This simulates the object being scaled to unit cube")
    
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
    
    # Use world up as reference for left vector
    world_up = np.array([0, 0, 1])
    left = np.cross(world_up, forward)
    left = left / np.linalg.norm(left)
    up = np.cross(forward, left)
    
    # Create transformation matrix
    mat44 = np.eye(4)
    mat44[:3, :3] = np.stack([forward, left, up], axis=1)
    mat44[:3, 3] = cam_pos
    
    camera_mount_actor.set_pose(sapien.Pose.from_transformation_matrix(mat44))
    
    print(f"Camera positioned at: {cam_pos}")

def main(args):
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
                             azimuth=getattr(args, 'azimuth', 45),
                             elevation=getattr(args, 'elevation', 30))
    
    # Update scene
    scene.step()
    scene.update_render()
    camera.take_picture()
    
    print('Intrinsic matrix\n', camera.get_intrinsic_matrix())
    
    # ---------------------------------------------------------------------------- #
    # Render RGBA
    # ---------------------------------------------------------------------------- #
    rgba = camera.get_float_texture('Color')  # [H, W, 4]
    rgba_img = (rgba * 255).clip(0, 255).astype("uint8")
    rgba_pil = Image.fromarray(rgba_img)
    
    # Create output filename
    azimuth = getattr(args, 'azimuth', 45)
    elevation = getattr(args, 'elevation', 30)
    output_name = f'color_obj{args.object_id}_az{azimuth}_el{elevation}.png'
    rgba_pil.save(output_name)
    print(f"Saved {output_name}")
    
    # ---------------------------------------------------------------------------- #
    # Render Depth
    # ---------------------------------------------------------------------------- #
    position = camera.get_float_texture('Position')  # [H, W, 4]
    depth = -position[..., 2]  # OpenGL convention: -z is forward
    
    # Convert to millimeters and save as 16-bit PNG
    depth_image = (depth * 1000.0).astype(np.uint16)
    depth_pil = Image.fromarray(depth_image)
    depth_name = f'depth_obj{args.object_id}_az{azimuth}_el{elevation}.png'
    depth_pil.save(depth_name)
    print(f"Saved {depth_name}")
    
    # Optional: Save normalized depth for visualization (YOUR ORIGINAL CODE!)
    depth_normalized = ((depth - depth.min()) / (depth.max() - depth.min()) * 255).astype(np.uint8)
    depth_vis_pil = Image.fromarray(depth_normalized)
    vis_name = f'depth_normalized_obj{args.object_id}_az{azimuth}_el{elevation}.png'
    depth_vis_pil.save(vis_name)
    print(f"Saved {vis_name}")
    
    # ---------------------------------------------------------------------------- #
    # Point Cloud (Optional)
    # ---------------------------------------------------------------------------- #
    # Extract point cloud in world coordinates
    points_opengl = position[..., :3][position[..., 3] < 1]
    
    # Transform to world coordinates
    model_matrix = camera.get_model_matrix()
    points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
    
    print(f"Generated point cloud with {len(points_world)} points")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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