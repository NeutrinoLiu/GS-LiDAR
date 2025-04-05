import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AdvCollaborativePerception")))

from attack import GeneralAttacker

dest_dir = os.path.join("/", "nobackup", "CPV2V", "LiDAR", "GS-LiDAR", "data", "opv2v")

ga = GeneralAttacker()
normal_lidar, attack_lidar, general_info, attack_info = ga.attack(
    attacker_type="spoof",
    dense=0,
    sync=0,
    attack_id=0,
    attack_frame_ids=[9])