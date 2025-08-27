import argparse
import json
import os
import random
import subprocess

from augment_sentence import augment_video_description


parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", required=True, type=str)
parser.add_argument("--prompt_n", default=20, type=int)
parser.add_argument("--view_n", default=20, type=int)
parser.add_argument("--aug_aspect_n", default=5, type=int)
parser.add_argument("--output_dir", default="outputs", type=str)

args = parser.parse_args()

# Make output directory
os.makedirs(args.output_dir, exist_ok=True)

# Read partnet annotation
with open(args.csv_path, "r") as f:
    metadata = json.load(f)

# Main loop --------------------
for data in metadata:
    uid = data["uid"]
    uid_out_root = os.path.join(args.output_dir, str(uid))
    os.makedirs(uid_out_root, exist_ok=True)
    elevation_range  = data["elevation_range"]
    azimuth_range = data["azimuth_range"]
    joint_indices = data["joint_indices"]
    joint_prompts = data["joint_prompts"]
    distance_factor = data["distance_factor"]
    base_prompt = data["base_prompt"]

    f = open(os.path.join(uid_out_root, "prompts.txt"), "w")
    for joint_id, joint_prompt in zip(joint_indices, joint_prompts):
        for i in range(args.prompt_n):
            complete_desc = augment_video_description(joint_prompt, base_prompt)
            f.write(str(joint_id) + "\t" + f"{base_prompt}; {joint_prompt}" + "\t" + complete_desc + "\n")
    f.close()

    for joint_id in joint_indices:
        for j in range(args.view_n):
            for aug_id in range(args.aug_aspect_n):
                elevation = elevation_range[0] + random.random() * (elevation_range[1] - elevation_range[0])
                azimuth = azimuth_range[0] + random.random() * (azimuth_range[1] - azimuth_range[0])
                cmd = [
                    "python", "camera.py",
                    "--object_id", str(uid),
                    "--joint_index", str(joint_id),
                    "--azimuth", str(azimuth),
                    "--elevation", str(elevation),
                    "--distance_factor", str(distance_factor),
                    "--output_dir", uid_out_root,
                    "--aspect_augmentation",
                    "--aspect_seed", str(aug_id),
                    "--generate_both"
                ]
                subprocess.run(cmd)                
        