import argparse
import pandas as pd
import os
import random
import subprocess

from augment_sentence import augment_video_description

from utils import read_metadata

parser = argparse.ArgumentParser()
parser.add_argument("--csv_path", required=True, type=str)
parser.add_argument("--prompt_n", default=10, type=int)
parser.add_argument("--view_n", default=100, type=int)
parser.add_argument("--output_dir", default="outputs", type=str)

args = parser.parse_args()

# Make output directory
os.makedirs(args.output_dir, exist_ok=True)

# Read partnet annotation
metadata = read_metadata(args.csv_path)
metadata = metadata.to_dict(orient="records")

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
            f.write(str(joint_id) + "\t" + complete_desc + "\n")
    f.close()

    for joint_id in joint_indices:
        for j in range(args.view_n):
            elevation = elevation_range[0] + random.random() * (elevation_range[1] - elevation_range[0])
            azimuth = azimuth_range[0] + random.random() * (azimuth_range[1] - azimuth_range[0])
            subprocess.run(f"python camera.py --object_id {uid} --joint_index {joint_id} --azimuth {azimuth} --elevation {elevation} --distance_factor {distance_factor} --output_dir {uid_out_root}", shell=True)
        