import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import argparse
import os,sys
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
# Added participant_ids argument
def plot_bev_for_frame(frame_num, real_vehicles_data, spoof_bbox_relative,
                       attacker_id, victim_id, participant_ids,
                       attack_id, save_dir="bev_plots"):
    """
    Generates and saves a Bird's-Eye View plot for a single frame,
    calculating the spoof vehicle's world position relative to the attacker
    and distinguishing participant vs background vehicles.

    Args:
        frame_num (int): The original frame number.
        real_vehicles_data (dict): Dictionary mapping vehicle IDs to their state dicts.
        spoof_bbox_relative (list or np.array): 7-element list [rx, ry, rz, l, w, h, relative_yaw_rad] for spoof.
        attacker_id (int): ID of the attacker vehicle.
        victim_id (int): ID of the victim vehicle.
        participant_ids (set): Set of vehicle IDs that are considered participants (provided point clouds).
        attack_id (int or str): Identifier for the attack.
        save_dir (str): Directory to save the plot images.
    """
    plot_data = []
    spoof_world_calculated = False

    # --- Calculate Spoof World Pose ---
    attacker_data = real_vehicles_data.get(attacker_id)
    if attacker_data is not None:
        try:
            ax, ay = attacker_data['location'][0], attacker_data['location'][1]
            ayaw_rad = attacker_data['angle'][1] * np.pi / 180
            rx, ry = spoof_bbox_relative[0], spoof_bbox_relative[1]
            ryaw_rad = spoof_bbox_relative[6]
            spoof_l, spoof_w = spoof_bbox_relative[3], spoof_bbox_relative[4]
            cos_a, sin_a = np.cos(ayaw_rad), np.sin(ayaw_rad)
            spoof_world_x = ax + rx * cos_a - ry * sin_a
            spoof_world_y = ay + rx * sin_a + ry * cos_a
            spoof_world_yaw = ayaw_rad + ryaw_rad

            plot_data.append({
                'id': 'Spoof', 'x': spoof_world_x, 'y': spoof_world_y,
                'length': spoof_l, 'width': spoof_w, 'yaw': spoof_world_yaw,
                'role': 'spoof' # Assign role
            })
            spoof_world_calculated = True
            print(f"Frame {frame_num}: Spoof world pose calculated (x={spoof_world_x:.2f}, y={spoof_world_y:.2f}, yaw={spoof_world_yaw:.4f})")
        except (IndexError, TypeError, KeyError) as e:
            print(f"Warning: Error processing attacker/spoof data in frame {frame_num}. Error: {e}")
    else:
         print(f"Warning: Attacker vehicle {attacker_id} not found in frame {frame_num}. Cannot calculate spoof world position.")


    # Process real vehicles and assign roles
    if real_vehicles_data:
        for vehicle_id, data in real_vehicles_data.items():
            if not isinstance(data, dict) or 'location' not in data or 'extent' not in data or 'angle' not in data:
                print(f"Warning: Skipping invalid data for vehicle {vehicle_id} in frame {frame_num}")
                continue
            try:
                # Determine role for real vehicles
                if vehicle_id == attacker_id:
                    role = 'attacker'
                elif vehicle_id == victim_id:
                    role = 'victim'
                elif vehicle_id in participant_ids: # Check if it's a participant
                    role = 'participant'
                else:
                    role = 'background' # Otherwise it's background

                x = data['location'][0]
                y = data['location'][1]
                length = data['extent'][0] * 2
                width = data['extent'][1] * 2
                yaw = data['angle'][1] * np.pi / 180
                plot_data.append({
                    'id': vehicle_id, 'x': x, 'y': y, 'length': length,
                    'width': width, 'yaw': yaw, 'role': role # Store the role
                })
            except (IndexError, TypeError, KeyError) as e:
                 print(f"Warning: Error processing vehicle {vehicle_id} in frame {frame_num}. Data: {data}. Error: {e}")
                 continue

    if not plot_data:
        print(f"Warning: No valid vehicles to plot for frame {frame_num}.")
        return
    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))

    # Define colors based on roles
    color_map = {
        'spoof': 'red',
        'attacker': 'orange',
        'victim': 'green',
        'participant': 'blue', # Participants are blue
        'background': 'gray'   # Background vehicles are gray
    }
    default_color = 'black' # Fallback color

    all_x = []
    all_y = []

    for index, row in df.iterrows():
        x_center, y_center = row['x'], row['y']
        length, width = row['length'], row['width']
        yaw = row['yaw']
        role = row['role']
        current_id = row['id']

        all_x.append(x_center)
        all_y.append(y_center)

        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = corners_local @ rotation.T + np.array([x_center, y_center])

        # Determine color based on role
        face_color = color_map.get(role, default_color)
        edge_color = face_color # Use same color for edge

        polygon = patches.Polygon(corners_world, closed=True, edgecolor=edge_color, facecolor=face_color, alpha=0.6)
        ax.add_patch(polygon)

        # Display ID for ALL vehicles
        ax.text(x_center, y_center, str(current_id),
                ha='center', va='center', fontsize=6, color='black',
                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.3, ec='none'))

    # Determine plot limits
    if not all_x or not all_y:
        print(f"Warning: No vehicle coordinates found for frame {frame_num} limits calculation.")
        x_min, x_max, y_min, y_max = -50, 50, -50, 50
    else:
        padding = 20
        x_min, x_max = min(all_x) - padding, max(all_x) + padding
        y_min, y_max = min(all_y) - padding, max(all_y) + padding

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set labels and title
    ax.set_xlabel("X coordinate (m)")
    ax.set_ylabel("Y coordinate (m)")
    ax.set_title(f"Bird's-Eye View - Attack {attack_id} - Frame {frame_num}")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)

    # Create custom legend handles based on the new roles
    legend_handles = [
        patches.Patch(color=color_map['attacker'], label=f'Attacker ({attacker_id})', alpha=0.6),
        # Only add victim legend if victim_id is valid
        patches.Patch(color=color_map['victim'], label=f'Victim ({victim_id})', alpha=0.6) if victim_id is not None else None,
        patches.Patch(color=color_map['participant'], label='Participant Vehicle', alpha=0.6),
        patches.Patch(color=color_map['background'], label='Background Vehicle', alpha=0.6),
        # Only add spoof legend if it was successfully plotted
        patches.Patch(color=color_map['spoof'], label='Spoof Vehicle', alpha=0.6) if spoof_world_calculated else None
    ]
    ax.legend(handles=[h for h in legend_handles if h is not None])


    # Save the plot
    if not os.path.exists(save_dir):
        try:
            os.makedirs(save_dir)
        except OSError as e:
            print(f"Error creating directory {save_dir}: {e}")
            return

    save_path = os.path.join(save_dir, f"attack_{attack_id}_frame_{frame_num}.png")
    try:
        # Save with higher DPI
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot for frame {frame_num} to {save_path} with higher resolution (300 DPI).")
    except Exception as e:
        print(f"Error saving plot for frame {frame_num}: {e}")

    plt.close(fig) # Close the plot to free memory

# --- Main Function ---
def generate_attack_bev_plots(attack_id, save_dir_base="bev_plots"):
    """
    Generates Bird's-Eye View plots for all frames of a given attack ID.
    """
    save_dir = os.path.join(save_dir_base, f"attack_{attack_id}")

    print(f"Processing attack ID: {attack_id}")
    print(f"Plots will be saved in: {save_dir}")

    try:
        ga = GeneralAttacker()
        attack_details = ga.get_spoof_attack_details(attack_id)
        if not attack_details:
            print(f"Error: Could not retrieve details for attack ID {attack_id}")
            return

        attack_meta = attack_details.get('attack_meta', {})
        attack_opts = attack_details.get('attack_opts', {})

        scenario_id = attack_meta.get('scenario_id')
        original_frame_ids = attack_meta.get('frame_ids')
        attacker_id = attack_meta.get('attacker_vehicle_id')
        victim_id = attack_meta.get('victim_vehicle_id', None)
        # --- MODIFICATION: Get participant IDs ---
        # Use set for efficient lookup, default to empty set if key missing
        participant_ids = set(attack_meta.get('vehicle_ids', []))
        print(f"Participant vehicle IDs: {participant_ids}")

        spoof_positions_array = attack_opts.get('positions')

        # --- Validation (remains mostly the same) ---
        if not all([scenario_id, original_frame_ids is not None, attacker_id is not None, spoof_positions_array is not None]):
             print(f"Error: Missing essential data in attack_details for attack ID {attack_id}")
             return
        if victim_id is None:
             print(f"Warning: victim_vehicle_id not found in attack_meta for attack ID {attack_id}.")
        if not isinstance(spoof_positions_array, np.ndarray):
             try: spoof_positions_array = np.array(spoof_positions_array)
             except Exception as e: print(f"Error converting spoof positions: {e}"); return
        if spoof_positions_array.ndim != 2 or spoof_positions_array.shape[1] != 7:
             print(f"Error: Spoof positions array shape {spoof_positions_array.shape}, expected (N, 7)."); return
        if len(original_frame_ids) != len(spoof_positions_array):
            print(f"Error: Mismatch frames ({len(original_frame_ids)}) vs spoof positions ({len(spoof_positions_array)})"); return

        # --- Load Scene Data (remains the same) ---
        if not hasattr(ga, 'dataset') or not hasattr(ga.dataset, 'meta'):
             print("Error: GeneralAttacker instance missing dataset.meta."); return
        if scenario_id not in ga.dataset.meta:
            print(f"Error: Scenario ID '{scenario_id}' not found in dataset metadata."); return
        scene = ga.dataset.meta[scenario_id]
        if 'label' not in scene:
             print(f"Error: 'label' key not found for scenario '{scenario_id}'."); return

        # --- Loop through frames and plot ---
        print(f"Found {len(original_frame_ids)} frames to process: {original_frame_ids}")
        for i, frame_num in enumerate(original_frame_ids):
            print(f"Processing frame {frame_num} (index {i})...")
            if frame_num not in scene['label']:
                 print(f"Warning: Frame {frame_num} not found in scene labels. Skipping."); continue
            real_vehicles_data_this_frame = scene['label'][frame_num]
            spoof_bbox_relative_this_frame = spoof_positions_array[i]

            # --- MODIFICATION: Pass participant_ids to plotting function ---
            plot_bev_for_frame(
                frame_num=frame_num,
                real_vehicles_data=real_vehicles_data_this_frame,
                spoof_bbox_relative=spoof_bbox_relative_this_frame,
                attacker_id=attacker_id,
                victim_id=victim_id,
                participant_ids=participant_ids, # Pass the set here
                attack_id=attack_id,
                save_dir=save_dir
            )

        print(f"Finished processing attack ID: {attack_id}")

    except ImportError:
         print("Error: Could not import GeneralAttacker from attack.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bird\'s-Eye View plots for spoof attacks.')
    parser.add_argument('--attack_id', type=int, required=True, help='Attack ID to process')
    parser.add_argument('--save_dir_base', type=str, default="bev_plots", help='Base directory to save plots (default: bev_plots)')
    args = parser.parse_args()

    generate_attack_bev_plots(attack_id=args.attack_id, save_dir_base=args.save_dir_base)