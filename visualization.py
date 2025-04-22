import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os, sys
# Adjust the path according to your project structure if necessary
script_dir = os.path.dirname(__file__) if "__file__" in locals() else os.getcwd()
sys.path.append(os.path.abspath(os.path.join(script_dir, "..", "AdvCollaborativePerception")))
try:
    from attack import GeneralAttacker # Assuming this class is defined elsewhere as described
except ImportError:
    print("Error: Could not import GeneralAttacker. Make sure 'attack.py' is accessible via sys.path.")
    print(f"Current sys.path includes: {sys.path}")
    # As a fallback for testing structure without the actual class:
    class GeneralAttacker: # Placeholder if import fails
         def __init__(self): self.dataset = type('obj', (object,), {'meta': {}})()
         def get_spoof_attack_details(self, aid): print("Warning: Using Placeholder GeneralAttacker"); return None

# --- Plotting Function ---
# Added arguments for next frame's data
def plot_bev_for_frame(frame_num,
                       real_vehicles_data_current, spoof_world_pose_current,
                       real_vehicles_data_next, spoof_world_pose_next,
                       attacker_id, victim_id, participant_ids,
                       attack_id, save_dir="bev_plots"):
    """
    Generates BEV plot for a single frame, showing motion vectors based on the next frame.

    Args:
        frame_num (int): The current frame number.
        real_vehicles_data_current (dict): Vehicle states for the current frame.
        spoof_world_pose_current (tuple): (x, y, yaw) for spoof vehicle in the current frame. Can be None.
        real_vehicles_data_next (dict | None): Vehicle states for the next frame, or None if last frame.
        spoof_world_pose_next (tuple | None): (x, y, yaw) for spoof vehicle in the next frame, or None.
        attacker_id (int): Attacker vehicle ID.
        victim_id (int): Victim vehicle ID.
        participant_ids (set): Set of participant vehicle IDs.
        attack_id (int or str): Attack identifier.
        save_dir (str): Directory to save plots.
    """
    plot_data = []

    # --- Prepare data for current frame boxes ---
    # Add calculated spoof world pose for current frame (if available)
    if spoof_world_pose_current:
        plot_data.append({
            'id': 'Spoof', 'x': spoof_world_pose_current[0], 'y': spoof_world_pose_current[1],
            'length': spoof_world_pose_current[3], # Assuming length/width stored here now
            'width': spoof_world_pose_current[4],
            'yaw': spoof_world_pose_current[2], 'role': 'spoof'
        })

    # Process real vehicles for current frame
    if real_vehicles_data_current:
        for vehicle_id, data in real_vehicles_data_current.items():
            # ... (validation and role assignment as before) ...
            try:
                if vehicle_id == attacker_id: role = 'attacker'
                elif vehicle_id == victim_id: role = 'victim'
                elif vehicle_id in participant_ids: role = 'participant'
                else: role = 'background'

                x = data['location'][0]; y = data['location'][1]
                length = data['extent'][0] * 2; width = data['extent'][1] * 2
                yaw = data['angle'][1] * np.pi / 180
                plot_data.append({
                    'id': vehicle_id, 'x': x, 'y': y, 'length': length,
                    'width': width, 'yaw': yaw, 'role': role
                })
            except (IndexError, TypeError, KeyError) as e:
                 print(f"Warning: Error processing vehicle {vehicle_id} in frame {frame_num}. Data: {data}. Error: {e}")
                 continue

    if not plot_data:
        print(f"Warning: No valid vehicles to plot boxes for frame {frame_num}.")
        return # Cant plot if nothing is there
    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    color_map = {'spoof': 'red', 'attacker': 'orange', 'victim': 'green', 'participant': 'blue', 'background': 'gray'}
    default_color = 'black'
    all_x, all_y = [], []

    # Plot Boxes and Labels
    for index, row in df.iterrows():
        x_center, y_center = row['x'], row['y']
        length, width = row['length'], row['width']
        yaw = row['yaw']
        role = row['role']
        current_id = row['id']
        all_x.append(x_center); all_y.append(y_center)

        # ... (calculate corners_world as before) ...
        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = corners_local @ rotation.T + np.array([x_center, y_center])

        face_color = color_map.get(role, default_color)
        polygon = patches.Polygon(corners_world, closed=True, edgecolor=face_color, facecolor=face_color, alpha=0.6)
        ax.add_patch(polygon)
        ax.text(x_center, y_center, str(current_id), ha='center', va='center', fontsize=6, color='black',
                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.3, ec='none'))

    # --- Plot Motion Arrows (using next frame data) ---
    arrow_scale_factor = 1.0 # Adjust this to control visual arrow length
    min_move_threshold = 0.1 # meters

    for index, row in df.iterrows():
        current_id = row['id']
        current_pos = (row['x'], row['y'])
        next_pos = None

        # Find position in next frame
        if current_id == 'Spoof':
            if spoof_world_pose_next is not None:
                next_pos = (spoof_world_pose_next[0], spoof_world_pose_next[1])
        elif real_vehicles_data_next is not None:
            next_vehicle_data = real_vehicles_data_next.get(current_id)
            if next_vehicle_data and 'location' in next_vehicle_data:
                try:
                    next_pos = (next_vehicle_data['location'][0], next_vehicle_data['location'][1])
                except (IndexError, TypeError): pass # Ignore malformed next data

        # If next position found, calculate and draw arrow
        if next_pos is not None:
            dx = next_pos[0] - current_pos[0]
            dy = next_pos[1] - current_pos[1]
            magnitude = np.sqrt(dx**2 + dy**2)

            if magnitude > min_move_threshold:
                 # Draw arrow using quiver
                 ax.quiver(current_pos[0], current_pos[1], dx, dy,
                           angles='xy', scale_units='xy', scale=1/arrow_scale_factor, # Use scale factor
                           color=color_map.get(row['role'], default_color),
                           width=0.005, headwidth=3, headlength=5, headaxislength=4.5) # Adjust appearance


    # Determine plot limits (based on current frame positions)
    if not all_x or not all_y: x_min, x_max, y_min, y_max = -50, 50, -50, 50
    else:
        padding = 20
        x_min, x_max = min(all_x) - padding, max(all_x) + padding
        y_min, y_max = min(all_y) - padding, max(all_y) + padding

    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_xlabel("X coordinate (m)"); ax.set_ylabel("Y coordinate (m)")
    ax.set_title(f"Bird's-Eye View - Attack {attack_id} - Frame {frame_num}")
    ax.set_aspect('equal', adjustable='box'); ax.grid(True)

    # Create legend (adjusted)
    legend_handles = [
        patches.Patch(color=color_map['attacker'], label=f'Attacker ({attacker_id})', alpha=0.6),
        patches.Patch(color=color_map['victim'], label=f'Victim ({victim_id})', alpha=0.6) if victim_id is not None else None,
        patches.Patch(color=color_map['participant'], label='Participant Vehicle', alpha=0.6),
        patches.Patch(color=color_map['background'], label='Background Vehicle', alpha=0.6),
        patches.Patch(color=color_map['spoof'], label='Spoof Vehicle', alpha=0.6) if spoof_world_pose_current else None # Check if spoof was plotted
    ]
    # Add a pseudo-handle for the arrow if needed, or explain in title/caption
    # legend_handles.append(plt.Line2D([0], [0], marker='>', color='black', label='Motion Vector (to next frame)', linestyle='None'))
    ax.legend(handles=[h for h in legend_handles if h is not None])

    # Save the plot
    # ... (saving logic as before, using dpi=300) ...
    if not os.path.exists(save_dir):
        try: os.makedirs(save_dir)
        except OSError as e: print(f"Error creating dir {save_dir}: {e}"); return
    save_path = os.path.join(save_dir, f"attack_{attack_id}_frame_{frame_num}.png")
    try:
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot for frame {frame_num} to {save_path} (300 DPI).")
    except Exception as e: print(f"Error saving plot for frame {frame_num}: {e}")
    plt.close(fig)


# --- Main Function ---
def generate_attack_bev_plots(attack_id, save_dir_base="bev_plots"):
    """
    Generates Bird's-Eye View plots for all frames of a given attack ID,
    showing motion vectors based on the next frame's position.
    """
    save_dir = os.path.join(save_dir_base, f"attack_{attack_id}")
    print(f"Processing attack ID: {attack_id}")
    print(f"Plots will be saved in: {save_dir}")

    try:
        ga = GeneralAttacker()
        attack_details = ga.get_spoof_attack_details(attack_id)
        # ... (extract attack_meta, attack_opts, scenario_id, etc. as before) ...
        attack_meta = attack_details.get('attack_meta', {})
        attack_opts = attack_details.get('attack_opts', {})
        scenario_id = attack_meta.get('scenario_id')
        original_frame_ids = attack_meta.get('frame_ids')
        attacker_id = attack_meta.get('attacker_vehicle_id')
        victim_id = attack_meta.get('victim_vehicle_id', None)
        participant_ids = set(attack_meta.get('vehicle_ids', []))
        spoof_positions_relative_array = attack_opts.get('positions')

        # --- Validation ---
        if not all([scenario_id, original_frame_ids is not None, attacker_id is not None, spoof_positions_relative_array is not None]):
             print(f"Error: Missing essential data for attack ID {attack_id}"); return
        if not isinstance(spoof_positions_relative_array, np.ndarray):
             try: spoof_positions_relative_array = np.array(spoof_positions_relative_array)
             except Exception as e: print(f"Error converting spoof positions: {e}"); return
        if spoof_positions_relative_array.ndim != 2 or spoof_positions_relative_array.shape[1] != 7:
             print(f"Error: Spoof positions array shape {spoof_positions_relative_array.shape}."); return
        if len(original_frame_ids) != len(spoof_positions_relative_array):
            print(f"Error: Mismatch frames/spoof positions len."); return

        # --- Load ALL necessary data first ---
        print("Pre-loading scene and calculating spoof world poses...")
        if not hasattr(ga, 'dataset') or not hasattr(ga.dataset, 'meta'):
             print("Error: Dataset structure missing."); return
        if scenario_id not in ga.dataset.meta:
            print(f"Error: Scenario ID '{scenario_id}' not found."); return
        scene = ga.dataset.meta[scenario_id]
        if 'label' not in scene:
             print(f"Error: 'label' key missing for scenario '{scenario_id}'."); return

        all_real_vehicles_data = {}
        spoof_world_poses = {} # Store calculated world poses: frame_num -> (x, y, yaw, l, w)

        for i, frame_num in enumerate(original_frame_ids):
            if frame_num not in scene['label']:
                 print(f"Warning: Frame {frame_num} not found in scene labels. Will skip processing this frame index.")
                 continue
            real_data = scene['label'][frame_num]
            all_real_vehicles_data[frame_num] = real_data

            # Calculate and store spoof world pose for this frame
            attacker_data = real_data.get(attacker_id)
            if attacker_data:
                try:
                    ax, ay = attacker_data['location'][0], attacker_data['location'][1]
                    ayaw_rad = attacker_data['angle'][1] * np.pi / 180
                    spoof_rel_bbox = spoof_positions_relative_array[i]
                    rx, ry = spoof_rel_bbox[0], spoof_rel_bbox[1]
                    ryaw_rad = spoof_rel_bbox[6]
                    spoof_l, spoof_w = spoof_rel_bbox[3], spoof_rel_bbox[4]
                    cos_a, sin_a = np.cos(ayaw_rad), np.sin(ayaw_rad)
                    swx = ax + rx * cos_a - ry * sin_a
                    swy = ay + rx * sin_a + ry * cos_a
                    swyaw = ayaw_rad + ryaw_rad
                    # Store world pose along with dimensions needed later
                    spoof_world_poses[frame_num] = (swx, swy, swyaw, spoof_l, spoof_w)
                except Exception as e:
                    print(f"Warning: Could not calculate spoof world pose for frame {frame_num}. Error: {e}")
            else:
                 print(f"Warning: Attacker {attacker_id} not found in frame {frame_num}, cannot calculate spoof world pose.")


        # --- Loop through frames and plot ---
        print(f"Plotting {len(original_frame_ids)} frames...")
        num_frames = len(original_frame_ids)
        for i, frame_num in enumerate(original_frame_ids):
            print(f"Plotting frame {frame_num} (index {i})...")

            # Get data for current frame
            real_vehicles_data_current = all_real_vehicles_data.get(frame_num)
            spoof_world_pose_current = spoof_world_poses.get(frame_num)

            if not real_vehicles_data_current:
                 print(f"Skipping frame {frame_num} due to missing real vehicle data.")
                 continue # Skip if no real data was loaded

            # Get data for next frame (if not the last frame)
            real_vehicles_data_next = None
            spoof_world_pose_next = None
            if i < num_frames - 1:
                next_frame_num = original_frame_ids[i+1]
                real_vehicles_data_next = all_real_vehicles_data.get(next_frame_num)
                spoof_world_pose_next = spoof_world_poses.get(next_frame_num)

            # Plot and save
            plot_bev_for_frame(
                frame_num=frame_num,
                real_vehicles_data_current=real_vehicles_data_current,
                spoof_world_pose_current=spoof_world_pose_current, # Pass calculated pose
                real_vehicles_data_next=real_vehicles_data_next,   # Pass next frame data
                spoof_world_pose_next=spoof_world_pose_next,     # Pass next spoof pose
                attacker_id=attacker_id,
                victim_id=victim_id,
                participant_ids=participant_ids,
                attack_id=attack_id,
                save_dir=save_dir
            )

        print(f"Finished processing attack ID: {attack_id}")

    # ... (except blocks as before) ...
    except ImportError:
         print("Error: Could not import GeneralAttacker from attack.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bird\'s-Eye View plots for spoof attacks with motion vectors.')
    parser.add_argument('--attack_id', type=int, required=True, help='Attack ID to process')
    parser.add_argument('--save_dir_base', type=str, default="bev_plots_with_motion", help='Base directory to save plots')
    args = parser.parse_args()
    generate_attack_bev_plots(attack_id=args.attack_id, save_dir_base=args.save_dir_base)