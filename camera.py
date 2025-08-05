"""Camera normalization using bounding_box.json from URDF folder"""
import argparse
import os
import json
import sapien as sp
import sapien.core as sapien
import numpy as np
from PIL import Image
import tempfile
import shutil
import subprocess

def load_bounding_box(urdf_path):
    """Load bounding box from bounding_box.json in the URDF directory"""
    urdf_dir = os.path.dirname(urdf_path)
    bbox_file = os.path.join(urdf_dir, 'bounding_box.json')
    
    if not os.path.exists(bbox_file):
        print(f"Warning: bounding_box.json not found at {bbox_file}")
        return None
    
    with open(bbox_file, 'r') as f:
        bbox_data = json.load(f)
    
    # Extract min and max bounds
    min_bound = np.array(bbox_data['min'])
    max_bound = np.array(bbox_data['max'])
    
    print(f"Loaded bounding box: min={min_bound}, max={max_bound}")
    
    return min_bound, max_bound

def normalize_and_center_object(actor, min_bound, max_bound):
    """Center the object at origin and return normalization info"""
    # Calculate object center and size
    center = (min_bound + max_bound) / 2
    size = max_bound - min_bound
    max_extent = np.max(size)
    diagonal_length = np.linalg.norm(size)
    
    print(f"Object center: {center}")
    print(f"Object size: {size}")
    print(f"Max extent: {max_extent}")
    print(f"Diagonal length: {diagonal_length}")
    
    # Move object to be centered at origin
    target_center = np.array([0, 0, 0])
    translation = target_center - center
    
    # Apply translation to actor
    current_pose = actor.get_pose()
    new_pose = sapien.Pose(p=current_pose.p + translation, q=current_pose.q)
    actor.set_pose(new_pose)
    
    print(f"Translated object by: {translation}")
    
    return {
        'center': target_center,  # Now at origin
        'original_center': center,
        'size': size,
        'max_extent': max_extent,
        'diagonal_length': diagonal_length,
        'translation': translation
    }

def position_camera_for_object(scene, camera, camera_mount_actor, normalization_info, 
                               azimuth=45, elevation=30, distance_factor=2.5):
    """Position camera on sphere around normalized object"""
    center = normalization_info['center']  # Should be [0, 0, 0]
    diagonal_length = normalization_info['diagonal_length']
    
    # Camera distance based on diagonal length
    cam_distance = diagonal_length * distance_factor
    
    print(f"Camera distance: {cam_distance}")
    
    # Convert angles to radians
    azimuth_rad = np.deg2rad(azimuth)
    elevation_rad = np.deg2rad(elevation)
    
    # Calculate camera position on sphere
    x = cam_distance * np.cos(elevation_rad) * np.cos(azimuth_rad)
    y = cam_distance * np.cos(elevation_rad) * np.sin(azimuth_rad)
    z = cam_distance * np.sin(elevation_rad)
    
    cam_pos = center + np.array([x, y, z])
    
    print(f"Camera position: {cam_pos}")
    
    # Compute camera orientation to look at object center
    forward = (center - cam_pos)
    forward = forward / np.linalg.norm(forward)
    
    # Use world up as reference
    world_up = np.array([0, 0, 1])
    
    # Right-handed coordinate system
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
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
    
    # Load bounding box from JSON
    bbox_result = load_bounding_box(urdf_file)
    if bbox_result is None:
        print("Error: Could not load bounding_box.json")
        return
    
    min_bound, max_bound = bbox_result
    
    # Load the URDF
    asset = loader.load_kinematic(urdf_file)
    assert asset, "Failed to load URDF."
    
    # Normalize and center the object
    normalization_info = normalize_and_center_object(asset, min_bound, max_bound)
    
    # Get joints information
    all_joints = asset.get_joints()
    
    print("\nAll joints info:")
    for i, joint in enumerate(all_joints):
        print(f"Joint {i}: {joint.name}")
        print(f"  Type: {joint.type}")
        if joint.type != "fixed":
            print(f"  Limits: {joint.get_limits()}")
    
    movable_joints = [j for j in all_joints if j.type != "fixed"]
    
    print(f"\nFound {len(movable_joints)} movable joints")
    
    if len(movable_joints) == 0:
        print("No movable joints found in this object.")
        return
    
    if args.joint_index >= len(movable_joints):
        print(f"Error: joint_index {args.joint_index} out of range. Found {len(movable_joints)} movable joints:")
        for i, j in enumerate(movable_joints):
            print(f"  {i}: {j.name} (type: {j.type})")
        return
    
    target_joint = movable_joints[args.joint_index]
    limits = target_joint.get_limits()[0]
    
    if np.isinf(limits[0]) or np.isinf(limits[1]):
        print(f"\nJoint {target_joint.name} has infinite limits, using range [0, 1] for animation")
        limits = [0.0, 1.0]
    
    print(f"\nAnimating joint: {target_joint.name} (type: {target_joint.type}), limits: {limits}")
    
    # Set up lighting
    scene.set_ambient_light([0.5, 0.5, 0.5])
    scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5], shadow=True)
    scene.add_point_light([2, 2, 2], [1, 1, 1])
    scene.add_point_light([2, -2, 2], [1, 1, 1])
    scene.add_point_light([-2, 0, 2], [1, 1, 1])
    
    # Camera Setup
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
                               elevation=args.elevation,
                               distance_factor=args.distance_factor)
    
    # Get initial depth range for consistent normalization
    scene.step()
    scene.update_render()
    camera.take_picture()
    position = camera.get_float_texture('Position')
    initial_depth = -position[..., 2]
    depth_min, depth_max = initial_depth.min(), initial_depth.max() * 1.5
    
    print(f"\nDepth range: [{depth_min:.3f}, {depth_max:.3f}]")
    
    temp_dir = tempfile.mkdtemp()
    print(f"Saving frames to temporary directory: {temp_dir}")
    
    for frame in range(args.num_frames):
        # Interpolate joint position
        t = frame / (args.num_frames - 1)
        qpos = limits[0] + t * (limits[1] - limits[0])
        
        # Set joint position
        articulation = target_joint.articulation
        current_qpos = articulation.get_qpos()
        
        # Debug for first frame only
        if False:#frame == 0:
            print(f"\nDEBUG: Setting joint at movable_joints[{args.joint_index}] = {target_joint.name}")
            print(f"DEBUG: qpos array length = {len(current_qpos)}")
            print(f"DEBUG: qpos before = {current_qpos}")
        
        # The key fix: qpos only contains movable joints, so the index is direct
        for i in range(len(current_qpos)):
            if i == args.joint_index:
                current_qpos[i] = qpos
            else:
                current_qpos[i] = 0
        articulation.set_qpos(current_qpos)
        
        # Update scene
        scene.step()
        scene.update_render()
        camera.take_picture()
        
        # Render Depth
        position = camera.get_float_texture('Position')
        depth = -position[..., 2]
        
        # Normalize depth using global range
        depth_normalized = ((depth - depth_min) / (depth_max - depth_min) * 255).clip(0, 255).astype(np.uint8)
        depth_vis_pil = Image.fromarray(depth_normalized)
        vis_name = f'depth_frame_{frame:04d}.png'
        depth_vis_pil.save(os.path.join(temp_dir, vis_name))
    
    # Generate video using ffmpeg
    output_video = os.path.join(args.output_dir, f'depth_obj{args.object_id}_joint{args.joint_index}_az{args.azimuth}_el{args.elevation}_df{args.distance_factor}.mp4')
    ffmpeg_cmd = [
        'ffmpeg',
        '-y',
        '-framerate', '30',
        '-i', os.path.join(temp_dir, 'depth_frame_%04d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-crf', '23',
        output_video
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        print(f"Video saved to: {output_video}")
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir)
        print("Temporary frames deleted")
    except subprocess.CalledProcessError as e:
        print(f"Error generating video with ffmpeg: {e}")
        print(f"Frames are still available in: {temp_dir}")
    except FileNotFoundError:
        print("ffmpeg not found. Please install ffmpeg to generate video.")
        print(f"Frames saved in: {temp_dir}")

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
    parser.add_argument("--joint_index", type=int, default=0,
                        help="Which movable joint to animate (0-indexed)")
    parser.add_argument("--num_frames", type=int, default=30,
                        help="Number of frames to generate")
    parser.add_argument("--distance_factor", type=float, default=2.5,
                        help="Camera distance multiplier relative to object diagonal")
    args = parser.parse_args()
    main(args)