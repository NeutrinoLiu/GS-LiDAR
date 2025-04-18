import os,sys
import yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "AdvCollaborativePerception")))
from attack import GeneralAttacker

# 创建输出文件夹
output_dir = "all_configs"
os.makedirs(output_dir, exist_ok=True)

# 初始化 GeneralAttacker
ga = GeneralAttacker()

# 基础模板配置（除了下面会替换的内容）
base_config = {
    'exhaust_test': False,
    'frame_stride': 2,
    'attacker_type': "spoof",
    'dense': 0,
    'sync': 0,
    'advshape': 0,
    'scene_type': "OPV2V_SR",
    'sequence_id': "all",
    'resolution_scales': [1],
    'scale_increase_interval': 5000,
    'cam_num': 2,
    'eval': True,
    'separate_scaling_t': 0.2,
    'vfov': [-25.0, 2.0],
    'hfov': [-90, 90],
    'hw': [66, 515],
    'iterations': 30000,
    'densify_until_iter': 15000,
    'densify_grad_threshold': 1e-4,
    'sh_increase_interval': 2000,
    'lambda_lidar': 1e1,
    'lambda_lidar_median': 1e1,
    'lidar_decay': 0,
    'only_velodyne': True,
    'lambda_intensity': 0.05,
    'lambda_intensity_sh': 0.05,
    'lambda_raydrop': 0.05,
    'lambda_smooth': 1e-3,
    'lambda_chamfer': 1e-1,
    'lambda_distortion': 1e-2,
    'lambda_normal_consistency': 1e-1,
    'thresh_opa_prune': 0.1,
    'test_iterations': [3000, 7000, 15000, 30000],
    'random_init_point': 0,
    'dynamic': True,
    'sky_depth': False,
    'opacity_lr': 0.05,
    'velocity_lr': 0.001,
    'time_split_frac': 1.0,
    'lambda_self_supervision': 0.5,
    'lambda_v_reg': 1.0,
    't_init': 0.006,
    'num_pts': 3000000,
    'densify_until_num_points': 6000000,
}

# 遍历 0-299 的 attack_id
for attack_id in range(300):
    spoof_info = ga.get_spoof_attack_info(attack_id)

    config = base_config.copy()
    config['attack_id'] = attack_id
    config['attack_frame_ids'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    config['attacker_vehicle_id'] = spoof_info['attacker_vehicle_id']
    config['val_frames'] = [spoof_info['frame_ids'][-1]]

    output_path = os.path.join(output_dir, f"opv2v_spoof_attack_{attack_id}.yaml")
    with open(output_path, "w") as f:
        yaml.dump(config, f)

print(f"✅ All spoof attack configs saved to: {output_dir}")