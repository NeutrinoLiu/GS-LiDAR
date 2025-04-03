import json
import yaml
import numpy as np
import os
import re

base_dir = os.path.join("/", "nobackup", "chengpo", "LiDAR", "GS-LiDAR", "data", "opv2v")

mode = "test"
mode_transform = "test_transform"

os.makedirs(os.path.join(base_dir, mode_transform), exist_ok=True)

def rotation_matrix(roll, yaw, pitch):
    R = np.array([[np.cos(yaw)*np.cos(pitch), 
                   np.cos(yaw)*np.sin(pitch)*np.sin(roll)-np.sin(yaw)*np.cos(roll), 
                   np.cos(yaw)*np.sin(pitch)*np.cos(roll)+np.sin(yaw)*np.sin(roll)],
                  [np.sin(yaw)*np.cos(pitch), 
                   np.sin(yaw)*np.sin(pitch)*np.sin(roll)+np.cos(yaw)*np.cos(roll), 
                   np.sin(yaw)*np.sin(pitch)*np.cos(roll)-np.cos(yaw)*np.sin(roll)],
                  [-np.sin(pitch), 
                   np.cos(pitch)*np.sin(roll), 
                   np.cos(pitch)*np.cos(roll)]])
    return R

def get_c2w_matrix(calib):
    R = rotation_matrix(*(np.array(calib["lidar_pose"][3:]) * np.pi / 180))
    
    translation_vector = np.array(calib["lidar_pose"][:3])
    
    c2w_matrix = np.eye(4)
    c2w_matrix[:3, :3] = R
    c2w_matrix[:3, 3] = translation_vector
    
    return c2w_matrix

for scenario in os.listdir(os.path.join(base_dir, mode)):
    os.makedirs(os.path.join(base_dir, mode_transform, scenario), exist_ok=True)
    for vehicle in os.listdir(os.path.join(base_dir, mode, scenario)):
        if vehicle.endswith(".yaml"):
            continue
        else:
            resulted_frames = []
            sorted_frames = sorted(os.listdir(os.path.join(base_dir, mode, scenario, vehicle)), key=lambda x: int(re.search(r'\d+', x).group()))
            for frame in sorted_frames:
                if frame.endswith(".yaml"):
                    with open(os.path.join(base_dir, mode, scenario, vehicle, frame), "r") as f:
                        calib = yaml.load(f, Loader=yaml.Loader)
                    c2w_matrix = get_c2w_matrix(calib)
                    resulted_frames.append({
                        "idx": int(frame.replace(".yaml", "")),
                        "lidar2world": c2w_matrix.tolist()
                    })
            
            json_dict = {
                "frames": resulted_frames
            }
            json_path = os.path.join(base_dir, mode_transform, scenario, f"transforms_{scenario}_{vehicle}.json")
            with open(json_path, "w") as f:
                json.dump(json_dict, f, indent=2)
                print(f"Saved {json_path}")
    print(f"Processed scene {scenario}")
print("Done")