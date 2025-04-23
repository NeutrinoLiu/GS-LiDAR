import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os, sys
import traceback # Keep for detailed error reporting

# --- Path Setup and Import ---
# Adjust the path according to your project structure if necessary
script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
adv_collab_path = os.path.abspath(os.path.join(script_dir, "..", "AdvCollaborativePerception"))
if adv_collab_path not in sys.path:
    sys.path.append(adv_collab_path)

# Import GeneralAttacker (assuming it's always available)
from attack import GeneralAttacker

# --- Plotting Function ---
def plot_bev_for_frame(frame_num,
                       real_vehicles_data_current, spoof_world_pose_current,
                       real_vehicles_data_next, spoof_world_pose_next,
                       attacker_id, victim_id, participant_ids,
                       attack_id, save_dir):
    """
    Generates and saves a Bird's-Eye View plot for a single frame,
    showing motion vectors based on the next frame and distinguishing vehicle roles.
    """
    plot_data = []
    spoof_was_plotted = False

    # Prepare data for current frame objects
    if spoof_world_pose_current:
        try:
            plot_data.append({
                'id': 'Spoof', 'x': spoof_world_pose_current[0], 'y': spoof_world_pose_current[1],
                'length': spoof_world_pose_current[3], 'width': spoof_world_pose_current[4],
                'yaw': spoof_world_pose_current[2], 'role': 'spoof'
            })
            spoof_was_plotted = True
        except (IndexError, TypeError):
            # Corrected indentation
             print(f"Warn: Invalid format for spoof_world_pose_current in frame {frame_num}.")

    if real_vehicles_data_current:
        for vehicle_id, data in real_vehicles_data_current.items():
            if not isinstance(data, dict) or not all(k in data for k in ['location', 'extent', 'angle']):
                continue
            try:
                if vehicle_id == attacker_id: role = 'attacker'
                elif victim_id is not None and vehicle_id == victim_id: role = 'victim'
                elif vehicle_id in participant_ids: role = 'participant'
                else: role = 'background'

                x, y = data['location'][0], data['location'][1]
                length, width = data['extent'][0] * 2, data['extent'][1] * 2
                yaw = data['angle'][1] * np.pi / 180
                plot_data.append({
                    'id': vehicle_id, 'x': x, 'y': y, 'length': length,
                    'width': width, 'yaw': yaw, 'role': role
                })
            except (IndexError, TypeError, KeyError) as e:
                 # Corrected indentation
                 # Keep minimal warnings in final version
                 # print(f"Warn: Error processing data for vehicle {vehicle_id} in frame {frame_num}: {e}")
                 continue

    if not plot_data:
        # print(f"Info: No vehicles (real or spoof) to plot for frame {frame_num}.")
        return # Nothing to plot
    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    color_map = {'spoof': 'red', 'attacker': 'orange', 'victim': 'green', 'participant': 'blue', 'background': 'gray'}
    default_color = 'black'; all_x, all_y = [], []

    # Plot Boxes and Labels
    for index, row in df.iterrows():
        x_center, y_center = row['x'], row['y']; length, width = row['length'], row['width']; yaw = row['yaw']; role = row['role']; current_id = row['id']
        all_x.append(x_center); all_y.append(y_center)
        half_length, half_width = length / 2, width / 2; corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]]); corners_world = corners_local @ rotation.T + np.array([x_center, y_center])
        face_color = color_map.get(role, default_color)
        polygon = patches.Polygon(corners_world, closed=True, edgecolor=face_color, facecolor=face_color, alpha=0.6); ax.add_patch(polygon)
        ax.text(x_center, y_center, str(current_id), ha='center', va='center', fontsize=6, color='black') # No background

    # --- Plot Motion Arrows ---
    visual_arrow_length = 2.0; arrow_head_width = 0.6; arrow_head_length = 0.8; arrows_drawn = 0
    for index, row in df.iterrows():
        current_id = row['id']; current_pos = (row['x'], row['y']); next_pos = None
        if current_id == 'Spoof':
            if spoof_world_pose_next:
                 try: next_pos = (spoof_world_pose_next[0], spoof_world_pose_next[1])
                 except (IndexError, TypeError): pass # Corrected indentation
        elif real_vehicles_data_next:
            next_vehicle_data = real_vehicles_data_next.get(current_id)
            if next_vehicle_data and 'location' in next_vehicle_data:
                try: next_pos = (next_vehicle_data['location'][0], next_vehicle_data['location'][1])
                except (IndexError, TypeError): pass # Corrected indentation

        if next_pos:
            dx = next_pos[0] - current_pos[0]; dy = next_pos[1] - current_pos[1]; magnitude = np.sqrt(dx**2 + dy**2)
            if magnitude > 1e-6: ux = dx / magnitude; uy = dy / magnitude
            else: ux, uy = 0, 0
            if ux != 0 or uy != 0:
                 arrow_dx = visual_arrow_length * ux; arrow_dy = visual_arrow_length * uy
                 ax.arrow(current_pos[0], current_pos[1], arrow_dx, arrow_dy, head_width=arrow_head_width, head_length=arrow_head_length, length_includes_head=True, fc=color_map.get(row['role'], default_color), ec=color_map.get(row['role'], default_color), alpha=0.7)
                 arrows_drawn += 1

    # --- Final Touches ---
    if not all_x or not all_y: x_min, x_max, y_min, y_max = -50, 50, -50, 50
    else: padding = 20; x_min, x_max = min(all_x) - padding, max(all_x) + padding; y_min, y_max = min(all_y) - padding, max(all_y) + padding
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max); ax.set_xlabel("X coordinate (m)"); ax.set_ylabel("Y coordinate (m)")
    ax.set_title(f"Bird's-Eye View - Attack {attack_id} - Frame {frame_num}"); ax.set_aspect('equal', adjustable='box'); ax.grid(True)
    legend_handles = [ patches.Patch(color=color_map['attacker'], label=f'Attacker ({attacker_id})', alpha=0.6),
                       patches.Patch(color=color_map['victim'], label=f'Victim ({victim_id})', alpha=0.6) if victim_id is not None else None,
                       patches.Patch(color=color_map['participant'], label='Participant', alpha=0.6), patches.Patch(color=color_map['background'], label='Background', alpha=0.6),
                       patches.Patch(color=color_map['spoof'], label='Spoof Vehicle', alpha=0.6) if spoof_was_plotted else None,
                       plt.Line2D([0], [0], marker='>', color='black', label='Motion Vector', markersize=5, linestyle='None') if arrows_drawn > 0 else None ]
    ax.legend(handles=[h for h in legend_handles if h is not None], fontsize='small')

    # Save the plot
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError as e:
            # Corrected indentation
            print(f"Error creating dir {save_dir}: {e}")
            plt.close(fig)
            return
    save_path = os.path.join(save_dir, f"attack_{attack_id}_frame_{frame_num}.png")
    try:
        plt.savefig(save_path, dpi=300)
    except Exception as e:
        # Corrected indentation
        print(f"Error saving plot for frame {frame_num}: {e}")
    plt.close(fig)


# --- Main Function ---
def generate_attack_bev_plots(attack_id, offset_file="spoof_offsets.npy", save_dir_base="bev_plots"):
    """
    Generates Bird's-Eye View plots for all frames of a given attack ID,
    applying stored offsets to the spoof vehicle position.
    """
    save_dir = os.path.join(save_dir_base, f"attack_{attack_id}")
    print(f"Processing attack ID: {attack_id}")
    print(f"Attempting to load offsets from: {offset_file}")
    print(f"Plots will be saved in: {save_dir}")

    offset_array = None
    try:
        offset_array = np.load(offset_file)
        print(f"Loaded offsets shape: {offset_array.shape}")
        if offset_array.shape != (300, 2):
             print(f"Warning: Offset file shape is not (300, 2). Assuming zero offsets.")
             offset_array = np.zeros((300, 2))
    except FileNotFoundError:
        print(f"Info: Offset file '{offset_file}' not found. Assuming zero offsets.")
        offset_array = np.zeros((300, 2))
    except Exception as e:
        # Corrected indentation
        print(f"Error loading offset file {offset_file}: {e}. Assuming zero offsets.")
        offset_array = np.zeros((300, 2))

    if not (0 <= attack_id < 300):
         print(f"Error: attack_id {attack_id} is out of range (0-299). Cannot get offset.")
         current_offset = np.array([0.0, 0.0])
    else:
         current_offset = offset_array[attack_id]
    print(f"Using offset for attack {attack_id}: [{current_offset[0]:.2f}, {current_offset[1]:.2f}]")

    try:
        ga = GeneralAttacker()
        attack_details = ga.get_spoof_attack_details(attack_id)
        if not attack_details:
            print(f"Error: No details for attack ID {attack_id}")
            return

        attack_meta = attack_details.get('attack_meta', {})
        attack_opts = attack_details.get('attack_opts', {})
        scenario_id = attack_meta.get('scenario_id')
        original_frame_ids = attack_meta.get('frame_ids')
        attacker_id = attack_meta.get('attacker_vehicle_id')
        victim_id = attack_meta.get('victim_vehicle_id', None)
        participant_ids = set(attack_meta.get('vehicle_ids', []))
        spoof_positions_relative_array = attack_opts.get('positions')

        if not all([scenario_id, original_frame_ids is not None, attacker_id is not None, spoof_positions_relative_array is not None]):
             print(f"Error: Missing essential data for attack ID {attack_id}")
             return
        if not isinstance(spoof_positions_relative_array, np.ndarray):
             try:
                 spoof_positions_relative_array = np.array(spoof_positions_relative_array)
             except Exception as e:
                 # Corrected indentation
                 print(f"Error converting spoof positions: {e}")
                 return
        if spoof_positions_relative_array.ndim != 2 or spoof_positions_relative_array.shape[1] != 7:
             print(f"Error: Spoof positions array shape {spoof_positions_relative_array.shape}.")
             return
        if len(original_frame_ids) != len(spoof_positions_relative_array):
            print(f"Error: Mismatch frames/spoof positions len.")
            return

        print(f"Pre-loading data for scenario {scenario_id} and applying offset...")
        if not hasattr(ga, 'dataset') or not hasattr(ga.dataset, 'meta'): print("Error: Dataset structure missing."); return
        if scenario_id not in ga.dataset.meta: print(f"Error: Scenario ID '{scenario_id}' not found."); return
        scene = ga.dataset.meta[scenario_id]
        if 'label' not in scene: print(f"Error: 'label' key missing for scenario '{scenario_id}'."); return

        all_real_vehicles_data = {}
        spoof_world_poses = {} # frame_num -> (x, y, yaw, l, w)
        offset_dx, offset_dy = current_offset

        frame_count_with_labels = 0
        for i, frame_num in enumerate(original_frame_ids):
            if frame_num in scene['label']:
                all_real_vehicles_data[frame_num] = scene['label'][frame_num]
                frame_count_with_labels += 1
            else: continue

            attacker_data = all_real_vehicles_data[frame_num].get(attacker_id)
            if attacker_data:
                try:
                    ax, ay = attacker_data['location'][0], attacker_data['location'][1]
                    ayaw_rad = attacker_data['angle'][1] * np.pi / 180
                    spoof_rel_bbox = spoof_positions_relative_array[i]
                    orig_rx, orig_ry = spoof_rel_bbox[0], spoof_rel_bbox[1]
                    orig_ryaw = spoof_rel_bbox[6]
                    spoof_l, spoof_w = spoof_rel_bbox[3], spoof_rel_bbox[4]
                    adj_rx = orig_rx + offset_dx; adj_ry = orig_ry + offset_dy
                    cos_a, sin_a = np.cos(ayaw_rad), np.sin(ayaw_rad)
                    swx = ax + adj_rx * cos_a - adj_ry * sin_a; swy = ay + adj_rx * sin_a + adj_ry * cos_a
                    swyaw = ayaw_rad + orig_ryaw
                    spoof_world_poses[frame_num] = (swx, swy, swyaw, spoof_l, spoof_w)
                except Exception as e:
                    # Corrected indentation
                    print(f"Warn: Could not calc spoof pose for frame {frame_num}. Err: {e}")

        print(f"Found label data for {frame_count_with_labels}/{len(original_frame_ids)} frames. Calculated {len(spoof_world_poses)} spoof poses.")
        print(f"Participant vehicle IDs: {participant_ids}")

        print(f"Plotting frames...")
        num_frames = len(original_frame_ids)
        plotted_count = 0
        for i, frame_num in enumerate(original_frame_ids):
            real_vehicles_data_current = all_real_vehicles_data.get(frame_num)
            spoof_world_pose_current = spoof_world_poses.get(frame_num)
            if not real_vehicles_data_current: continue

            real_vehicles_data_next = None
            spoof_world_pose_next = None
            if i < num_frames - 1:
                next_frame_num = original_frame_ids[i+1]
                real_vehicles_data_next = all_real_vehicles_data.get(next_frame_num)
                spoof_world_pose_next = spoof_world_poses.get(next_frame_num)

            plot_bev_for_frame( # Pass calculated poses
                frame_num=frame_num,
                real_vehicles_data_current=real_vehicles_data_current,
                spoof_world_pose_current=spoof_world_pose_current,
                real_vehicles_data_next=real_vehicles_data_next,
                spoof_world_pose_next=spoof_world_pose_next,
                attacker_id=attacker_id, victim_id=victim_id,
                participant_ids=participant_ids,
                attack_id=attack_id, save_dir=save_dir
            )
            plotted_count += 1

        print(f"Finished processing attack ID: {attack_id}. Plotted {plotted_count} frames saved in {save_dir}")

    except ImportError:
        # Corrected indentation
        print("Error: Could not import GeneralAttacker.")
    except Exception as e:
        # Corrected indentation
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bird\'s-Eye View plots for spoof attacks with motion vectors, applying stored offsets.')
    parser.add_argument('--attack_id', type=int, required=True, help='Attack ID (0-299) to process')
    parser.add_argument('--offset_file', type=str, default="spoof_offsets.npy", help='Path to the NumPy file storing offsets (default: spoof_offsets.npy)')
    parser.add_argument('--save_dir_base', type=str, default="bev_plots", help='Base directory to save plots (default: bev_plots)')
    args = parser.parse_args()

    if not (0 <= args.attack_id < 300):
         print("Error: attack_id must be between 0 and 299.")
    else:
        generate_attack_bev_plots(
            attack_id=args.attack_id,
            offset_file=args.offset_file, # Pass offset file path
            save_dir_base=args.save_dir_base
        )