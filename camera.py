"""Camera.
Concepts:
    - Create and mount cameras
    - Render RGB images, point clouds, segmentation masks
    - Normalize URDF to unit cube for consistent multi-view rendering
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
    
    for link in actor.get_links():
        link_pose = link.get_pose()
        
        # Try collision shapes first
        collision_shapes = link.get_collision_shapes()
        if len(collision_shapes) > 0:
            for collision_shape in collision_shapes:
                try:
                    geometry = collision_shape.geometry
                    
                    # Handle different geometry types
                    if hasattr(geometry, 'half_lengths'):  # Box
                        half = geometry.half_lengths
                        local_min, local_max = -half, half
                    elif hasattr(geometry, 'radius'):  # Sphere/Cylinder
                        r = geometry.radius
                        if hasattr(geometry, 'half_length'):  # Cylinder
                            h = geometry.half_length
                            local_min = np.array([-r, -r, -h])
                            local_max = np.array([r, r, h])
                        else:  # Sphere
                            local_min = np.array([-r, -r, -r])
                            local_max = np.array([r, r, r])
                    else:
                        # Default bounds
                        local_min = np.array([-0.1, -0.1, -0.1])
                        local_max = np.array([0.1, 0.1, 0.1])
                    
                    # Transform to world coordinates
                    try:
                        shape_pose = collision_shape.get_local_pose()
                        transform = link_pose.to_transformation_matrix() @ shape_pose.to_transformation_matrix()
                    except:
                        transform = link_pose.to_transformation_matrix()
                    
                    # Transform corners
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
                    
                    corners_homogeneous = np.hstack([corners, np.ones((8, 1))])
                    corners_world = (transform @ corners_homogeneous.T)[:3, :].T
                    
                    min_bound = np.minimum(min_bound, corners_world.min(axis=0))
                    max_bound = np.maximum(max_bound, corners_world.max(axis=0))
                    
                except Exception as e:
                    print(f"Warning: Error processing collision shape: {e}")
                    continue
        else:
            # Fallback: use link position with small bounds
            pos = link_pose.p
            link_size = 0.1
            min_bound = np.minimum(min_bound, pos - link_size)
            max_bound = np.maximum(max_bound, pos + link_size)
    
    # If no valid bounds found, use reasonable defaults
    if np.any(np.isinf(min_bound)) or np.any(np.isinf(max_bound)):
        print("Warning: Could not determine object bounds, using default")
        min_bound = np.array([-0.5, -0.5, 0])
        max_bound = np.array([0.5, 0.5, 1])
    
    return min_bound, max_bound

def normalize_actor_to_unit_cube(actor, target_size=1.0):
    """
    Normalize the URDF actor to fit within a unit cube centered at origin
    Returns the transformation parameters for consistent multi-view rendering
    """
    # Get current bounds
    min_bound, max_bound = get_actor_bounds(actor)
    
    # Calculate current center and size
    current_center = (min_bound + max_bound) / 2
    current_size = max_bound - min_bound
    max_extent = np.max(current_size)
    
    print(f"Original bounds: min={min_bound}, max={max_bound}")
    print(f"Original center: {current_center}")
    print(f"Original size: {current_size}")
    print(f"Max extent: {max_extent}")
    
    # Calculate scaling factor to fit in unit cube
    if max_extent > 0:
        scale_factor = target_size / max_extent
    else:
        scale_factor = 1.0
    
    # Calculate translation to center at origin
    # Move to origin first, then scale, then move to desired center (0,0,0.5 to sit on ground)
    target_center = np.array([0, 0, target_size/2])  # Place bottom at z=0
    
    # Get the root link and apply transformation
    root_link = actor.get_links()[0]  # Assuming first link is root
    current_pose = root_link.get_pose()
    
    # Create transformation: translate to origin, scale, then translate to target
    translation_to_origin = -current_center
    final_translation = target_center
    
    # Apply transformation to the actor
    # Note: SAPIEN doesn't support direct scaling, so we work with positioning
    new_position = (current_pose.p + translation_to_origin) * scale_factor + final_translation
    new_pose = sapien.Pose(p=new_position, q=current_pose.q)
    
    # For kinematic actors, we can set the pose directly
    root_link.set_pose(new_pose)
    
    print(f"Applied normalization:")
    print(f"  Scale factor: {scale_factor}")
    print(f"  New position: {new_position}")
    print(f"  Target center: {target_center}")
    
    return {
        'scale_factor': scale_factor,
        'original_center': current_center,
        'target_center': target_center,
        'original_size': current_size,
        'max_extent': max_extent
    }

def sample_camera_pose_on_sphere(radius=2.0, elevation_range=(15, 75), azimuth=0):
    """
    Sample camera pose on a sphere around the origin
    elevation_range: (min_deg, max_deg) elevation from horizontal plane
    azimuth: azimuth angle in degrees
    """
    # Convert to radians
    elevation = np.random.uniform(elevation_range[0], elevation_range[1])
    elevation_rad = np.deg2rad(elevation)
    azimuth_rad = np.deg2rad(azimuth)
    
    # Spherical to Cartesian coordinates
    x = radius * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = radius * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = radius * np.sin(elevation_rad)
    
    cam_pos = np.array([x, y, z])
    
    # Look at origin (where object is centered)
    target = np.array([0, 0, 0.5])  # Look at center of unit cube
    
    return cam_pos, target

def position_camera_on_sphere(camera_mount_actor, radius=2.0, azimuth=45, elevation=30):
    """Position camera on sphere around normalized object"""
    cam_pos, target = sample_camera_pose_on_sphere(radius, (elevation, elevation), azimuth)
    
    # Compute camera orientation
    forward = (target - cam_pos)
    forward = forward / np.linalg.norm(forward)
    
    # Use world up as reference
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
    print(f"Looking at: {target}")
    print(f"Distance: {np.linalg.norm(cam_pos - target):.2f}")

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
    
    # Normalize object to unit cube - THIS IS THE KEY STEP
    normalization_info = normalize_actor_to_unit_cube(asset, target_size=1.0)
    
    # Set up lighting (positioned for normalized object)
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
    
    # Position camera on sphere around normalized object
    position_camera_on_sphere(
        camera_mount_actor, 
        radius=2.5,  # Distance from center
        azimuth=args.camera_azimuth,  # Horizontal angle
        elevation=args.camera_elevation  # Vertical angle
    )
    
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
    
    # Save with descriptive filename
    output_name = f'color_obj{args.object_id}_az{args.camera_azimuth}_el{args.camera_elevation}.png'
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
    depth_name = f'depth_obj{args.object_id}_az{args.camera_azimuth}_el{args.camera_elevation}.png'
    depth_pil.save(depth_name)
    print(f"Saved {depth_name}")
    
    # Optional: Save normalized depth for visualization
    valid_depth = depth[depth > 0]
    if len(valid_depth) > 0:
        depth_normalized = np.zeros_like(depth)
        depth_normalized[depth > 0] = ((depth[depth > 0] - valid_depth.min()) / 
                                      (valid_depth.max() - valid_depth.min()) * 255)
        depth_vis_pil = Image.fromarray(depth_normalized.astype(np.uint8))
        vis_name = f'depth_vis_obj{args.object_id}_az{args.camera_azimuth}_el{args.camera_elevation}.png'
        depth_vis_pil.save(vis_name)
        print(f"Saved {vis_name}")
    
    # Print normalization info for debugging
    print("\nNormalization Summary:")
    print(f"  Original max extent: {normalization_info['max_extent']:.3f}")
    print(f"  Scale factor applied: {normalization_info['scale_factor']:.3f}")
    print(f"  Object now centered at: {normalization_info['target_center']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--object_id", type=int, default=179)
    parser.add_argument("--image_shape",
                        type=lambda x: [int(y) for y in x.split(",")],
                        default=[512, 512])
    parser.add_argument("--camera_azimuth", type=float, default=45, 
                        help="Camera azimuth angle in degrees")
    parser.add_argument("--camera_elevation", type=float, default=30,
                        help="Camera elevation angle in degrees")
    args = parser.parse_args()
    main(args)