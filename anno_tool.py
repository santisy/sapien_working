"""
Updated Annotation Server for Articulated Objects
Run with: python anno_tool.py
"""

from flask import Flask, request, jsonify, send_file, send_from_directory
import os
import json
import numpy as np
import tempfile
import shutil
import subprocess
from pathlib import Path
import glob

# Import SAPIEN related libraries
try:
    import sapien as sp
    import sapien.core as sapien
    from PIL import Image
    SAPIEN_AVAILABLE = True
except ImportError:
    SAPIEN_AVAILABLE = False
    print("Warning: SAPIEN not installed. Install with: pip install sapien")

app = Flask(__name__)

# Global variables to store current scene
current_scene = None
current_engine = None
current_renderer = None

# Configuration
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_html():
    """Load HTML content from anno.html file"""
    try:
        with open('anno.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return """
        <html><body>
        <h1>Error: anno.html file not found</h1>
        <p>Please ensure anno.html is in the same directory as this server file.</p>
        </body></html>
        """

@app.route('/')
def index():
    """Serve the main HTML page"""
    return load_html()

@app.route('/load_object', methods=['POST'])
def load_object():
    """Load object and return joint information"""
    if not SAPIEN_AVAILABLE:
        return jsonify({'error': 'SAPIEN not installed'}), 500
    
    data = request.json
    object_id = data.get('object_id')
    
    if not object_id:
        return jsonify({'error': 'No object_id provided'}), 400
    
    try:
        # Check for SAPIEN token
        token = os.getenv("TOKEN_SAPIEN_STR", None)
        if not token:
            return jsonify({'error': 'Please set TOKEN_SAPIEN_STR environment variable'}), 500
        
        print(f"Loading object {object_id}...")
        
        # Download URDF
        urdf_file = sp.asset.download_partnet_mobility(object_id, token)
        
        # Create temporary scene to load object
        global current_engine, current_renderer, current_scene
        
        if current_engine is None:
            current_engine = sapien.Engine()
            current_renderer = sapien.SapienRenderer()
            current_engine.set_renderer(current_renderer)
        
        if current_scene:
            current_scene = None
        
        current_scene = current_engine.create_scene()
        current_scene.set_timestep(1 / 100.0)
        
        # Load URDF
        loader = current_scene.create_urdf_loader()
        loader.fix_root_link = True
        asset = loader.load_kinematic(urdf_file)
        
        if not asset:
            return jsonify({'error': 'Failed to load URDF'}), 500
        
        # Get joints information
        all_joints = asset.get_joints()
        joints_info = []
        movable_joints_info = []
        
        for i, joint in enumerate(all_joints):
            joint_data = {
                'index': i,
                'name': joint.name,
                'type': str(joint.type)
            }
            
            if joint.type != "fixed":
                limits = joint.get_limits()[0]
                if not (np.isinf(limits[0]) or np.isinf(limits[1])):
                    joint_data['limits'] = limits.tolist()
                else:
                    joint_data['limits'] = [0.0, 1.0]
                movable_joints_info.append(joint_data)
            
            joints_info.append(joint_data)
        
        return jsonify({
            'object_id': object_id,
            'joints': joints_info,
            'movable_joints': movable_joints_info,
            'urdf_path': urdf_file
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/generate_video', methods=['POST'])
def generate_video():
    """Generate both depth and color videos for specified joint"""
    if not SAPIEN_AVAILABLE:
        return jsonify({'error': 'SAPIEN not installed'}), 500
    
    data = request.json
    object_id = data.get('object_id')
    joint_index = data.get('joint_index', 0)
    azimuth = data.get('azimuth', 180)
    elevation = data.get('elevation', 20)
    distance_factor = data.get('distance_factor', 3)
    
    try:
        # Run the updated camera.py script with parameters
        cmd = [
            'python', 'camera.py',
            '--object_id', str(object_id),
            '--joint_index', str(joint_index),
            '--azimuth', str(azimuth),
            '--elevation', str(elevation),
            '--distance_factor', str(distance_factor),
            '--num_frames', '30',
            '--output_dir', RESULTS_DIR,
            '--generate_both'  # Generate both depth and color videos
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            return jsonify({'error': f'Video generation failed: {result.stderr}'}), 500
        
        # Find the generated video files - match the actual format from camera.py
        # camera.py uses float formatting which adds .0 to integers
        base_filename = f'obj{object_id}_joint{joint_index}_az{float(azimuth)}_el{float(elevation)}_df{float(distance_factor)}'
        depth_filename = f'depth_{base_filename}.mp4'
        color_filename = f'color_{base_filename}.mp4'
        
        depth_path = os.path.join(RESULTS_DIR, depth_filename)
        color_path = os.path.join(RESULTS_DIR, color_filename)
        
        response_data = {}
        
        # Check if files exist and are readable
        if os.path.exists(depth_path) and os.access(depth_path, os.R_OK):
            response_data['depth_video_path'] = f'/results/{depth_filename}'
            print(f"Depth video available at: {depth_path}")
        else:
            print(f"Depth video not found or not readable: {depth_path}")
        
        if os.path.exists(color_path) and os.access(color_path, os.R_OK):
            response_data['color_video_path'] = f'/results/{color_filename}'
            print(f"Color video available at: {color_path}")
        else:
            print(f"Color video not found or not readable: {color_path}")
        
        if not response_data:
            return jsonify({'error': 'No video files were generated or accessible'}), 500
        
        print(f"Returning video paths: {response_data}")
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/results/<filename>')
def serve_video(filename):
    """Serve video files from results directory"""
    try:
        file_path = os.path.join(RESULTS_DIR, filename)
        print(f"Attempting to serve: {file_path}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return "File not found", 404
        
        if not os.access(file_path, os.R_OK):
            print(f"File not readable: {file_path}")
            return "File not accessible", 403
            
        print(f"Serving file: {file_path}")
        return send_from_directory(RESULTS_DIR, filename)
        
    except Exception as e:
        print(f"Error serving file {filename}: {e}")
        return f"Error serving file: {e}", 500

# Also add the old route for backward compatibility
@app.route('/video/<filename>')
def serve_video_old(filename):
    """Serve video files (old route for compatibility)"""
    return serve_video(filename)

@app.route('/clear_test_videos', methods=['POST'])
def clear_test_videos():
    """Clear all test videos from results directory"""
    try:
        # Find all video files in results directory
        video_files = glob.glob(os.path.join(RESULTS_DIR, '*.mp4'))
        
        deleted_count = 0
        for video_file in video_files:
            try:
                os.remove(video_file)
                deleted_count += 1
                print(f"Deleted: {video_file}")
            except Exception as e:
                print(f"Error deleting {video_file}: {e}")
        
        return jsonify({
            'message': f'Cleared {deleted_count} test videos',
            'deleted_count': deleted_count
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/export_annotations', methods=['POST'])
def export_annotations():
    """Export annotations to JSON file"""
    data = request.json
    annotations = data.get('annotations', [])
    
    if not annotations:
        return jsonify({'error': 'No annotations to export'}), 400
    
    # Save to file
    output_file = os.path.join(RESULTS_DIR, f'annotations_{len(annotations)}_items.json')
    with open(output_file, 'w') as f:
        json.dump(annotations, f, indent=2)
    
    return jsonify({
        'message': 'Annotations exported successfully',
        'file': output_file,
        'count': len(annotations)
    })

@app.route('/save_to_feishu', methods=['POST'])
def save_to_feishu():
    """Save annotations to Feishu table (placeholder - implement based on Feishu API)"""
    data = request.json
    annotations = data.get('annotations', [])
    
    # TODO: Implement Feishu API integration
    # This is a placeholder that saves to CSV for now
    import csv
    
    csv_file = os.path.join(RESULTS_DIR, 'annotations.csv')
    
    with open(csv_file, 'w', newline='') as f:
        if annotations:
            writer = csv.DictWriter(f, fieldnames=annotations[0].keys())
            writer.writeheader()
            writer.writerows(annotations)
    
    return jsonify({
        'message': f'Saved {len(annotations)} annotations to CSV',
        'file': csv_file
    })

# 配置部分添加
ANNOTATIONS_FILE = os.path.join(RESULTS_DIR, 'annotations_db.json')

@app.route('/load_annotations', methods=['GET'])
def load_annotations_from_server():
    """从服务器加载所有注释"""
    try:
        if os.path.exists(ANNOTATIONS_FILE):
            with open(ANNOTATIONS_FILE, 'r', encoding='utf-8') as f:
                annotations = json.load(f)
            return jsonify({'annotations': annotations})
        else:
            return jsonify({'annotations': []})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/save_annotations', methods=['POST'])
def save_annotations_to_server():
    """保存所有注释到服务器"""
    try:
        data = request.json
        annotations = data.get('annotations', [])
        
        with open(ANNOTATIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(annotations, f, indent=2, ensure_ascii=False)
        
        return jsonify({'message': f'Saved {len(annotations)} annotations to server'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("Articulated Object Annotation Server")
    print("=" * 60)
    print("\nMake sure you have:")
    print("1. Set TOKEN_SAPIEN_STR environment variable")
    print("2. Installed sapien: pip install sapien")
    print("3. The camera.py script in the same directory")
    print("4. The anno.html file in the same directory")
    print(f"\nResults directory: {os.path.abspath(RESULTS_DIR)}")
    print("Server starting at http://localhost:5000")
    print("=" * 60)
    
    # Check for required files
    missing_files = []
    if not os.path.exists('camera.py'):
        missing_files.append('camera.py')
    if not os.path.exists('anno.html'):
        missing_files.append('anno.html')
    
    if missing_files:
        print(f"\nWARNING: Missing files: {', '.join(missing_files)}")
        print("Please ensure all required files are in the same folder as this server.")
    
    # Set proper permissions for results directory
    try:
        os.chmod(RESULTS_DIR, 0o755)
        print(f"Set permissions for {RESULTS_DIR}")
    except Exception as e:
        print(f"Warning: Could not set permissions for {RESULTS_DIR}: {e}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
