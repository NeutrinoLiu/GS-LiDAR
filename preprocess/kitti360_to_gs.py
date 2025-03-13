import os
from pathlib import Path
from kitti360_loader import KITTI360Loader
import camtools as ct
import numpy as np
import json


def normalize_Ts(Ts):
    # New Cs.
    Cs = np.array([ct.convert.T_to_C(T) for T in Ts])
    normalize_mat = ct.normalize.compute_normalize_mat(Cs)
    Cs_new = ct.project.homo_project(Cs.reshape((-1, 3)), normalize_mat)

    # New Ts.
    Ts_new = []
    for T, C_new in zip(Ts, Cs_new):
        pose = ct.convert.T_to_pose(T)
        pose[:3, 3] = C_new
        T_new = ct.convert.pose_to_T(pose)
        Ts_new.append(T_new)

    return Ts_new


def main(sequence_id):
    project_root = Path(__file__).parent.parent
    kitti_360_root = project_root / "data" / "kitti360" / "KITTI-360"
    kitti_360_parent_dir = kitti_360_root.parent

    # Specify frames and splits.
    sequence_name = "2013_05_28_drive_0000"

    if sequence_id == "1538":
        print("Using sqequence 1538-1601")
        s_frame_id = 1538
        e_frame_id = 1601  # Inclusive
        val_frame_ids = [1551, 1564, 1577, 1590]
    elif sequence_id == "1728":
        print("Using sqequence 1728-1791")
        s_frame_id = 1728
        e_frame_id = 1791  # Inclusive
        val_frame_ids = [1741, 1754, 1767, 1780]
    elif sequence_id == "1908":
        print("Using sqequence 1908-1971")
        s_frame_id = 1908
        e_frame_id = 1971  # Inclusive
        val_frame_ids = [1921, 1934, 1947, 1960]
    elif sequence_id == "3353":
        print("Using sqequence 3353-3416")
        s_frame_id = 3353
        e_frame_id = 3416  # Inclusive
        val_frame_ids = [3366, 3379, 3392, 3405]

    elif sequence_id == "2350":
        s_frame_id = 2350
        e_frame_id = 2400  # Inclusive
        val_frame_ids = [2360, 2370, 2380, 2390]
    elif sequence_id == "4950":
        s_frame_id = 4950
        e_frame_id = 5000  # Inclusive
        val_frame_ids = [4960, 4970, 4980, 4990]
    elif sequence_id == "8120":
        s_frame_id = 8120
        e_frame_id = 8170  # Inclusive
        val_frame_ids = [8130, 8140, 8150, 8160]
    elif sequence_id == "10200":
        s_frame_id = 10200
        e_frame_id = 10250  # Inclusive
        val_frame_ids = [10210, 10220, 10230, 10240]
    elif sequence_id == "10750":
        s_frame_id = 10750
        e_frame_id = 10800  # Inclusive
        val_frame_ids = [10760, 10770, 10780, 10790]
    elif sequence_id == "11400":
        s_frame_id = 11400
        e_frame_id = 11450  # Inclusive
        val_frame_ids = [11410, 11420, 11430, 11440]
    else:
        raise ValueError(f"Invalid sequence id: {sequence_id}")

    frame_ids = list(range(s_frame_id, e_frame_id + 1))

    # Load KITTI-360 dataset.
    k3 = KITTI360Loader(kitti_360_root)

    # Get lidar2world.
    lidar2world = k3.load_lidars(sequence_name, frame_ids)
    all_indices = [i - s_frame_id for i in frame_ids]

    split_to_all_indices = {
        "all": all_indices
    }
    for split, indices in split_to_all_indices.items():
        print(f"Split {split} has {len(indices)} frames.")
        lidar2world_split = [lidar2world[i] for i in indices]

        json_dict = {
            "w_lidar": 1030,
            "h_lidar": 66,
            "aabb_scale": 2,
            "frames": [{"idx": idx + s_frame_id,
                        "lidar2world": lidar2world.tolist()}
                       for idx, lidar2world in enumerate(lidar2world_split)],
        }
        os.makedirs(kitti_360_parent_dir / f"{sequence_id}", exist_ok=True)
        json_path = kitti_360_parent_dir / f"{sequence_id}" / f"transforms_{sequence_id}_{split}.json"

        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)
            print(f"Saved {json_path}.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        type=str,
        default="1908",
        help="Sequence to use.",
    )
    args = parser.parse_args()
    main(args.seq)
