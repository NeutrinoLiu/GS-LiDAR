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
def plot_bev_for_frame(frame_num, real_vehicles_data, spoof_bbox_relative, attacker_id, victim_id, attack_id, save_dir="bev_plots"):
    """
    Generates and saves a Bird's-Eye View plot for a single frame,
    calculating the spoof vehicle's world position relative to the attacker.

    Args:
        frame_num (int): The original frame number (for title and filename).
        real_vehicles_data (dict): Dictionary mapping vehicle IDs to their state dicts for this frame.
        spoof_bbox_relative (list or np.array): 7-element list [rx, ry, rz, l, w, h, relative_yaw_rad]
                                               for the spoof vehicle relative to the attacker.
        attacker_id (int): ID of the attacker vehicle for highlighting and relative positioning.
        victim_id (int): ID of the victim vehicle for highlighting.
        attack_id (int or str): Identifier for the attack (used in filename).
        save_dir (str): Directory to save the plot images.
    """
    plot_data = []
    spoof_world_calculated = False # Flag to check if spoof pose was calculated

    # --- Calculate Spoof World Pose ---
    attacker_data = real_vehicles_data.get(attacker_id)
    if attacker_data is None:
        print(f"Warning: Attacker vehicle {attacker_id} not found in frame {frame_num} data. Cannot calculate spoof world position.")
        # Optionally, decide how to handle this: skip frame, plot without spoof, etc.
        # For now, we'll proceed without plotting the spoof vehicle if attacker is missing.
    else:
        try:
            # 1. Get Attacker State
            ax, ay = attacker_data['location'][0], attacker_data['location'][1]
            ayaw_deg = attacker_data['angle'][1]
            ayaw_rad = ayaw_deg * np.pi / 180

            # 2. Get Spoof Relative State
            rx, ry = spoof_bbox_relative[0], spoof_bbox_relative[1]
            ryaw_rad = spoof_bbox_relative[6] # Relative yaw assumed
            spoof_l, spoof_w = spoof_bbox_relative[3], spoof_bbox_relative[4]

            # 3. Calculate Spoof World Position
            cos_a = np.cos(ayaw_rad)
            sin_a = np.sin(ayaw_rad)
            spoof_world_x = ax + rx * cos_a - ry * sin_a
            spoof_world_y = ay + rx * sin_a + ry * cos_a

            # 4. Calculate Spoof World Yaw
            spoof_world_yaw = ayaw_rad + ryaw_rad
            # Optional: Normalize yaw angle to be within [-pi, pi]
            # spoof_world_yaw = np.arctan2(np.sin(spoof_world_yaw), np.cos(spoof_world_yaw))

            # Add CORRECTED spoof vehicle data to plot list
            plot_data.append({
                'id': 'Spoof',
                'x': spoof_world_x,
                'y': spoof_world_y,
                'length': spoof_l,
                'width': spoof_w,
                'yaw': spoof_world_yaw,
                'type': 'spoof'
            })
            spoof_world_calculated = True
            print(f"Frame {frame_num}: Spoof world pose calculated (x={spoof_world_x:.2f}, y={spoof_world_y:.2f}, yaw={spoof_world_yaw:.4f})")

        except (IndexError, TypeError, KeyError) as e:
            print(f"Warning: Error processing attacker {attacker_id} or relative spoof data in frame {frame_num}. Error: {e}")
            # Proceed without plotting spoof if calculation fails

    # Process real vehicles
    if real_vehicles_data:
        for vehicle_id, data in real_vehicles_data.items():
            if not isinstance(data, dict) or 'location' not in data or 'extent' not in data or 'angle' not in data:
                print(f"Warning: Skipping invalid data for vehicle {vehicle_id} in frame {frame_num}")
                continue
            try:
                x = data['location'][0]
                y = data['location'][1]
                length = data['extent'][0] * 2
                width = data['extent'][1] * 2
                yaw = data['angle'][1] * np.pi / 180
                plot_data.append({
                    'id': vehicle_id, 'x': x, 'y': y, 'length': length, 'width': width, 'yaw': yaw, 'type': 'real'
                })
            except (IndexError, TypeError, KeyError) as e:
                 print(f"Warning: Error processing vehicle {vehicle_id} in frame {frame_num}. Data: {data}. Error: {e}")
                 continue

    # Convert to DataFrame
    if not plot_data:
        print(f"Warning: No valid vehicles (real or spoof) to plot for frame {frame_num}.")
        return
    df = pd.DataFrame(plot_data)

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(12, 12))
    color_map = {'real': 'blue', 'spoof': 'red'}
    attacker_color = 'orange'
    victim_color = 'green'
    default_real_color = 'blue'
    all_x = []
    all_y = []

    # Plot each vehicle as a rotated rectangle
    for index, row in df.iterrows():
        x_center, y_center = row['x'], row['y']
        length, width = row['length'], row['width']
        yaw = row['yaw']
        all_x.append(x_center)
        all_y.append(y_center)
        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = corners_local @ rotation.T + np.array([x_center, y_center])

        # Determine color
        current_id = row['id']
        if row['type'] == 'spoof':
            face_color, edge_color = color_map['spoof'], color_map['spoof']
        else: # type == 'real'
            edge_color = default_real_color
            if current_id == attacker_id: face_color = attacker_color
            elif current_id == victim_id: face_color = victim_color
            else: face_color = default_real_color

        polygon = patches.Polygon(corners_world, closed=True, edgecolor=edge_color, facecolor=face_color, alpha=0.6)
        ax.add_patch(polygon)

        # --- MODIFICATION: Display ID for ALL vehicles ---
        ax.text(x_center, y_center, str(current_id),
                ha='center', va='center', fontsize=6, color='black',  # Reduced font size
                bbox=dict(boxstyle='round,pad=0.1', fc='yellow', alpha=0.3, ec='none')) # Added background box

    # Determine plot limits
    if not all_x or not all_y:
        print(f"Warning: No vehicle coordinates found for frame {frame_num} limits calculation.")
        # Set arbitrary limits if no points found at all
        x_min, x_max, y_min, y_max = -50, 50, -50, 50
    else:
        padding = 20 # meters padding
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

    # Create custom legend handles
    legend_handles = [
        patches.Patch(color=default_real_color, label='Real Vehicle', alpha=0.6),
        patches.Patch(color=attacker_color, label=f'Attacker ({attacker_id})', alpha=0.6),
        # Only add victim legend if victim_id is valid
        patches.Patch(color=victim_color, label=f'Victim ({victim_id})', alpha=0.6) if victim_id is not None else None,
        # Only add spoof legend if it was successfully plotted
        patches.Patch(color=color_map['spoof'], label='Spoof Vehicle', alpha=0.6) if spoof_world_calculated else None
    ]
    # Filter out None handles before creating the legend
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
        plt.savefig(save_path, dpi=300)
        print(f"Saved plot for frame {frame_num} to {save_path}")
    except Exception as e:
        print(f"Error saving plot for frame {frame_num}: {e}")

    plt.close(fig) # Close the plot to free memory

# --- Main Function ---
def generate_attack_bev_plots(attack_id, save_dir_base="bev_plots"):
    """
    Generates Bird's-Eye View plots for all frames of a given attack ID.

    Args:
        attack_id (int or str): The identifier of the attack to process.
        save_dir_base (str): The base directory to save the plot images.
                             A subdirectory for the attack_id will be created.
    """
    save_dir = os.path.join(save_dir_base, f"attack_{attack_id}")

    print(f"Processing attack ID: {attack_id}")
    print(f"Plots will be saved in: {save_dir}")

    try:
        # Initialize the GeneralAttacker
        ga = GeneralAttacker()

        # 1. Get Attack Details
        attack_details = ga.get_spoof_attack_details(attack_id)
        if not attack_details:
            print(f"Error: Could not retrieve details for attack ID {attack_id}")
            return

        # 2. Extract necessary info
        attack_meta = attack_details.get('attack_meta', {})
        attack_opts = attack_details.get('attack_opts', {})

        scenario_id = attack_meta.get('scenario_id')
        original_frame_ids = attack_meta.get('frame_ids')
        attacker_id = attack_meta.get('attacker_vehicle_id')
        victim_id = attack_meta.get('victim_vehicle_id', None) # Use .get for safety

        spoof_positions_array = attack_opts.get('positions')

        # Basic validation
        if not all([scenario_id, original_frame_ids is not None, attacker_id is not None, spoof_positions_array is not None]):
             print(f"Error: Missing essential data in attack_details for attack ID {attack_id}")
             return
        if victim_id is None:
             print(f"Warning: victim_vehicle_id not found in attack_meta for attack ID {attack_id}.")

        # Convert spoof positions to numpy array if needed and validate shape
        if not isinstance(spoof_positions_array, np.ndarray):
            try:
                spoof_positions_array = np.array(spoof_positions_array)
            except Exception as e:
                print(f"Error: Could not convert spoof 'positions' to NumPy array for attack ID {attack_id}. Error: {e}")
                return
        if spoof_positions_array.ndim != 2 or spoof_positions_array.shape[1] != 7:
             print(f"Error: Spoof 'positions' array has incorrect shape {spoof_positions_array.shape}, expected (N, 7).")
             return

        if len(original_frame_ids) != len(spoof_positions_array):
            print(f"Error: Mismatch between number of frames ({len(original_frame_ids)}) and spoof positions ({len(spoof_positions_array)})")
            return

        # 3. Load Scene Data
        if not hasattr(ga, 'dataset') or not hasattr(ga.dataset, 'meta'):
             print("Error: GeneralAttacker instance does not have the expected dataset structure (ga.dataset.meta). Check GeneralAttacker initialization.")
             return
        if scenario_id not in ga.dataset.meta:
            print(f"Error: Scenario ID '{scenario_id}' not found in dataset metadata.")
            return
        scene = ga.dataset.meta[scenario_id]
        if 'label' not in scene:
             print(f"Error: 'label' key not found in scene data for scenario '{scenario_id}'.")
             return

        # 4. Loop through frames and plot
        print(f"Found {len(original_frame_ids)} frames to process: {original_frame_ids}")
        for i, frame_num in enumerate(original_frame_ids):
            print(f"Processing frame {frame_num} (index {i})...")

            # Get real vehicle data for this frame
            if frame_num not in scene['label']:
                 print(f"Warning: Frame number {frame_num} not found in scene labels for scenario {scenario_id}. Skipping frame.")
                 continue
            real_vehicles_data_this_frame = scene['label'][frame_num]

            # Get RELATIVE spoof vehicle data for this frame
            spoof_bbox_relative_this_frame = spoof_positions_array[i]

            # Plot and save - plot_bev_for_frame now handles the transformation
            plot_bev_for_frame(
                frame_num=frame_num,
                real_vehicles_data=real_vehicles_data_this_frame,
                spoof_bbox_relative=spoof_bbox_relative_this_frame, # Pass relative bbox
                attacker_id=attacker_id,
                victim_id=victim_id,
                attack_id=attack_id,
                save_dir=save_dir
            )

        print(f"Finished processing attack ID: {attack_id}")

    except ImportError:
         print("Error: Could not import GeneralAttacker from attack. Make sure the class is defined and accessible.")
         print(f"Current sys.path includes: {sys.path}")
    except Exception as e:
        print(f"An unexpected error occurred during processing attack ID {attack_id}: {e}")
        import traceback
        traceback.print_exc()


# --- Example Usage ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate Bird\'s-Eye View plots for spoof attacks.')
    parser.add_argument('--attack_id', type=int, required=True, help='Attack ID to process')
    parser.add_argument('--save_dir_base', type=str, default="bev_plots", help='Base directory to save plots (default: bev_plots)')
    args = parser.parse_args()

    generate_attack_bev_plots(attack_id=args.attack_id, save_dir_base=args.save_dir_base)