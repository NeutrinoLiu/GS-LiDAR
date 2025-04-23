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

class InteractiveOffsetAdjuster:
    def __init__(self, attack_id, offset_file='spoof_offsets.npy'):
        """Initializes the adjuster, loads data, creates plot, and connects events."""
        self.attack_id = attack_id
        self.offset_file = offset_file
        self.offset_array = self._load_offsets() # Renamed internal method

        # Load data specific to the attack and first frame
        self._load_attack_and_frame_data() # Renamed internal method

        # Initial plot setup
        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.spoof_patch = None # Store the patch for the spoof car
        self._draw_initial_plot() # Renamed internal method

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self._onclick) # Renamed internal method

        plt.show()

    def _load_offsets(self):
        """Loads offsets from file or initializes a new array."""
        try:
            offsets = np.load(self.offset_file)
            print(f"Loaded offsets from {self.offset_file}")
            if offsets.shape != (300, 2):
                 print(f"Warn: Loaded offset file shape {offsets.shape} incorrect, expected (300, 2). Reinitializing.")
                 raise FileNotFoundError
            return offsets
        except FileNotFoundError:
            print(f"Info: Offset file '{self.offset_file}' not found or invalid. Initializing default offsets (zeros).")
            return np.zeros((300, 2))
        except Exception as e:
             print(f"Error loading offset file {self.offset_file}: {e}. Initializing defaults.")
             return np.zeros((300, 2))

    def _save_offsets(self):
        """Saves the current offset array to the file."""
        try:
            np.save(self.offset_file, self.offset_array)
            print(f"Offsets saved to {self.offset_file}")
        except Exception as e:
            print(f"Error saving offsets to {self.offset_file}: {e}")

    def _load_attack_and_frame_data(self):
        """Loads attack details and the first frame's scene data."""
        print(f"Loading data for attack ID: {self.attack_id}")
        self.attacker_world_pose = None
        self.orig_spoof_relative_pose = None # [rx, ry, rz, l, w, h, ryaw]
        self.real_vehicles_data = None
        self.participant_ids = set()
        self.frame_num = None # Initialize frame number

        try:
            ga = GeneralAttacker() # Assumed to work
            attack_details = ga.get_spoof_attack_details(self.attack_id)
            if not attack_details: raise ValueError("Could not retrieve attack details.")

            attack_meta = attack_details.get('attack_meta', {})
            attack_opts = attack_details.get('attack_opts', {})

            scenario_id = attack_meta.get('scenario_id')
            original_frame_ids = attack_meta.get('frame_ids')
            self.attacker_id = attack_meta.get('attacker_vehicle_id')
            self.victim_id = attack_meta.get('victim_vehicle_id', None)
            self.participant_ids = set(attack_meta.get('vehicle_ids', []))
            spoof_positions_relative_array = attack_opts.get('positions')

            if not all([scenario_id, original_frame_ids, self.attacker_id, spoof_positions_relative_array is not None]):
                raise ValueError("Missing essential data in attack_details.")
            if not isinstance(spoof_positions_relative_array, np.ndarray):
                spoof_positions_relative_array = np.array(spoof_positions_relative_array)
            if spoof_positions_relative_array.ndim != 2 or spoof_positions_relative_array.shape[1] != 7:
                 raise ValueError("Spoof positions array incorrect shape.")
            if not original_frame_ids: raise ValueError("Frame IDs list empty.")

            self.frame_num = original_frame_ids[0]
            self.orig_spoof_relative_pose = spoof_positions_relative_array[0].copy() # Store copy

            if not hasattr(ga, 'dataset') or not hasattr(ga.dataset, 'meta') or scenario_id not in ga.dataset.meta:
                 raise ValueError(f"Scenario ID '{scenario_id}' not found in dataset.")
            scene = ga.dataset.meta[scenario_id]
            if 'label' not in scene or self.frame_num not in scene['label']:
                 raise ValueError(f"'label' data missing or frame {self.frame_num} not found.")
            self.real_vehicles_data = scene['label'][self.frame_num]

            attacker_data = self.real_vehicles_data.get(self.attacker_id)
            if not attacker_data: raise ValueError(f"Attacker {self.attacker_id} not found in frame {self.frame_num}.")

            self.attacker_world_pose = ( # Store attacker pose tuple
                attacker_data['location'][0], attacker_data['location'][1],
                attacker_data['angle'][1] * np.pi / 180
            )
            print(f"Loaded data for frame {self.frame_num}. Attacker pose (x,y,yaw): ({self.attacker_world_pose[0]:.2f}, {self.attacker_world_pose[1]:.2f}, {self.attacker_world_pose[2]:.3f})")

        except Exception as e:
            print(f"ERROR loading data for attack {self.attack_id}: {e}")
            traceback.print_exc()
            # Indicate failure
            self.attacker_world_pose = None
            self.orig_spoof_relative_pose = None
            self.real_vehicles_data = None

    def _calculate_spoof_world_pose(self, current_offset):
        """Calculates spoof world pose using current offset. Returns (x,y,yaw,l,w) or None."""
        if self.attacker_world_pose is None or self.orig_spoof_relative_pose is None: return None

        try:
            ax, ay, ayaw_rad = self.attacker_world_pose
            orig_rx, orig_ry, _, l, w, _, orig_ryaw = self.orig_spoof_relative_pose
            offset_dx, offset_dy = current_offset

            adj_rx = orig_rx + offset_dx; adj_ry = orig_ry + offset_dy
            cos_a = np.cos(ayaw_rad); sin_a = np.sin(ayaw_rad)
            world_x = ax + adj_rx * cos_a - adj_ry * sin_a
            world_y = ay + adj_rx * sin_a + adj_ry * cos_a
            world_yaw = ayaw_rad + orig_ryaw
            return world_x, world_y, world_yaw, l, w
        except Exception as e:
            print(f"Error calculating spoof world pose: {e}")
            return None

    def _draw_initial_plot(self):
        """Draws the initial BEV plot."""
        self.ax.clear()
        if self.real_vehicles_data is None:
            self.ax.set_title(f"Error loading data for Attack {self.attack_id}")
            self.fig.canvas.draw_idle()
            return

        current_offset = self.offset_array[self.attack_id]
        spoof_pose_tuple = self._calculate_spoof_world_pose(current_offset) # Gets (x,y,yaw,l,w)

        plot_items = []
        color_map = {'spoof': 'red', 'attacker': 'orange', 'victim': 'green', 'participant': 'blue', 'background': 'gray'}
        default_color = 'black'

        # Plot Real Vehicles
        for vehicle_id, data in self.real_vehicles_data.items():
            try:
                if vehicle_id == self.attacker_id: role = 'attacker'
                elif self.victim_id is not None and vehicle_id == self.victim_id: role = 'victim'
                elif vehicle_id in self.participant_ids: role = 'participant'
                else: role = 'background'
                x, y = data['location'][0], data['location'][1]
                length, width = data['extent'][0] * 2, data['extent'][1] * 2
                yaw = data['angle'][1] * np.pi / 180
                plot_items.append({'x':x, 'y':y})
                self._plot_vehicle_box(self.ax, x, y, length, width, yaw, color_map.get(role, default_color), str(vehicle_id))
            except Exception: continue # Skip malformed real vehicles

        # Plot Spoof Vehicle
        if spoof_pose_tuple:
            swx, swy, swyaw, sl, sw = spoof_pose_tuple
            plot_items.append({'x':swx, 'y':swy})
            self.spoof_patch = self._plot_vehicle_box(self.ax, swx, swy, sl, sw, swyaw, color_map['spoof'], 'Spoof')
        else:
            self.spoof_patch = None
            print("Could not plot initial spoof vehicle position.")

        # Set limits and labels
        if plot_items:
             all_x = [item['x'] for item in plot_items]; all_y = [item['y'] for item in plot_items]
             padding=20
             self.ax.set_xlim(min(all_x) - padding, max(all_x) + padding)
             self.ax.set_ylim(min(all_y) - padding, max(all_y) + padding)
        else: self.ax.set_xlim(-50, 50); self.ax.set_ylim(-50, 50)

        self.ax.set_xlabel("X coordinate (m)"); self.ax.set_ylabel("Y coordinate (m)")
        self.ax.set_title(f"Attack {self.attack_id} - Frame {self.frame_num} - Click to set Spoof Position\nCurrent Offset: [{current_offset[0]:.2f}, {current_offset[1]:.2f}]")
        self.ax.set_aspect('equal', adjustable='box'); self.ax.grid(True)
        legend_handles = [ patches.Patch(color=c, label=l.capitalize(), alpha=0.6) for l, c in color_map.items() if not (l=='spoof' and not self.spoof_patch)] # Only show spoof legend if plotted
        self.ax.legend(handles=legend_handles, fontsize='small')
        self.fig.canvas.draw_idle()


    def _plot_vehicle_box(self, ax, x, y, length, width, yaw, color, label):
        """Helper to plot a single vehicle box and label."""
        half_length, half_width = length / 2, width / 2
        corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
        corners_world = corners_local @ rotation.T + np.array([x, y])
        polygon = patches.Polygon(corners_world, closed=True, edgecolor=color, facecolor=color, alpha=0.6)
        patch = ax.add_patch(polygon) # Store added patch
        ax.text(x, y, label, ha='center', va='center', fontsize=6, color='black')
        return patch # Return the patch object


    def _onclick(self, event):
        """Handles mouse clicks on the plot."""
        if event.inaxes != self.ax: return
        if self.attacker_world_pose is None or self.orig_spoof_relative_pose is None:
            print("Error: Cannot process click - essential data missing."); return

        x_click, y_click = event.xdata, event.ydata
        if x_click is None or y_click is None: return # Click outside axes data range

        print(f"Clicked at world coordinates: ({x_click:.2f}, {y_click:.2f})")

        try:
            ax, ay, ayaw_rad = self.attacker_world_pose
            orig_rx, orig_ry = self.orig_spoof_relative_pose[0], self.orig_spoof_relative_pose[1]

            world_dx = x_click - ax; world_dy = y_click - ay
            cos_a = np.cos(-ayaw_rad); sin_a = np.sin(-ayaw_rad) # Inverse rotation
            new_rx = world_dx * cos_a - world_dy * sin_a
            new_ry = world_dx * sin_a + world_dy * cos_a
            new_offset_x = new_rx - orig_rx; new_offset_y = new_ry - orig_ry
            new_offset = [new_offset_x, new_offset_y]

            print(f"Calculated new offset: [{new_offset_x:.2f}, {new_offset_y:.2f}]")

            # Update and save
            self.offset_array[self.attack_id] = new_offset
            self._save_offsets()

            # Update plot visuals
            self._update_plot_visuals(new_offset)

        except Exception as e:
            print(f"Error during click processing: {e}")
            traceback.print_exc()


    def _update_plot_visuals(self, new_offset):
        """Updates the spoof car position and title on the plot."""
        new_spoof_pose = self._calculate_spoof_world_pose(new_offset) # Gets (x,y,yaw,l,w)

        if new_spoof_pose is None:
            print("Could not calculate new spoof pose for update.")
            return

        swx, swy, swyaw, sl, sw = new_spoof_pose

        if self.spoof_patch is None: # If it wasn't drawn initially, draw it now
            self.spoof_patch = self._plot_vehicle_box(self.ax, swx, swy, sl, sw, swyaw, 'red', 'Spoof')
            print("Added spoof patch during update.")
        else: # Update existing patch
             half_length, half_width = sl / 2, sw / 2
             corners_local = np.array([[half_length, half_width], [half_length, -half_width], [-half_length, -half_width], [-half_length, half_width]])
             rotation = np.array([[np.cos(swyaw), -np.sin(swyaw)], [np.sin(swyaw), np.cos(swyaw)]])
             corners_world = corners_local @ rotation.T + np.array([swx, swy])
             self.spoof_patch.set_xy(corners_world)
             print("Updated spoof patch position.")

        # Update title
        self.ax.set_title(f"Attack {self.attack_id} - Frame {self.frame_num} - Click to set Spoof Position\nCurrent Offset: [{new_offset[0]:.2f}, {new_offset[1]:.2f}]")
        self.fig.canvas.draw_idle() # Redraw


# --- Main Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Interactively adjust spoof vehicle offset for BEV plots.')
    parser.add_argument('--attack_id', type=int, required=True, help='Attack ID (0-299) to adjust.')
    parser.add_argument('--offset_file', type=str, default="spoof_offsets.npy", help='Path to the NumPy file for storing offsets (default: spoof_offsets.npy)')
    args = parser.parse_args()

    if not (0 <= args.attack_id < 300):
        print("Error: attack_id must be between 0 and 299.")
    else:
        # Create and run the interactive adjuster
        try:
            adjuster = InteractiveOffsetAdjuster(attack_id=args.attack_id, offset_file=args.offset_file)
        except Exception as e:
             print("\nAn error occurred during initialization or plotting.")
             traceback.print_exc()