import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
import sys
from tqdm import tqdm

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import load_flow_data, load_mask, ensure_output_dir

def interpolate_flow(u, v, x, y):
    """
    Interpolate flow at a continuous position using bilinear interpolation.
    
    Args:
        u (numpy.ndarray): Horizontal flow component grid
        v (numpy.ndarray): Vertical flow component grid
        x (float): x-coordinate
        y (float): y-coordinate
        
    Returns:
        tuple: (u_flow, v_flow) interpolated at the given position
    """
    # Get integer and fractional parts
    x0 = int(np.floor(x))
    y0 = int(np.floor(y))
    x1 = x0 + 1
    y1 = y0 + 1
    
    # Ensure within grid boundaries
    height, width = u.shape
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    
    # Get fractional parts
    dx = x - x0
    dy = y - y0
    
    # Bilinear interpolation for u
    u00 = u[y0, x0]
    u01 = u[y0, x1]
    u10 = u[y1, x0]
    u11 = u[y1, x1]
    
    u_interp = (1 - dx) * (1 - dy) * u00 + dx * (1 - dy) * u01 + \
               (1 - dx) * dy * u10 + dx * dy * u11
    
    # Bilinear interpolation for v
    v00 = v[y0, x0]
    v01 = v[y0, x1]
    v10 = v[y1, x0]
    v11 = v[y1, x1]
    
    v_interp = (1 - dx) * (1 - dy) * v00 + dx * (1 - dy) * v01 + \
               (1 - dx) * dy * v10 + dx * dy * v11
    
    return u_interp, v_interp

def nearest_flow(u, v, x, y):
    """
    Get flow at the nearest grid point.
    
    Args:
        u (numpy.ndarray): Horizontal flow component grid
        v (numpy.ndarray): Vertical flow component grid
        x (float): x-coordinate
        y (float): y-coordinate
        
    Returns:
        tuple: (u_flow, v_flow) at the nearest grid point
    """
    # Round to nearest integer
    i = int(round(x))
    j = int(round(y))
    
    # Ensure within grid boundaries
    height, width = u.shape
    i = max(0, min(i, width - 1))
    j = max(0, min(j, height - 1))
    
    return u[j, i], v[j, i]

def simulate_trajectory(start_pos, time_steps, data_dir='data', dt=0.1, interp_method='bilinear'):
    """
    Simulate the trajectory of a particle in the ocean flow.
    
    Args:
        start_pos (tuple): Starting position (x, y)
        time_steps (list): List of time steps to use for simulation
        data_dir (str): Directory containing data files
        dt (float): Time step size for integration (fraction of data time step)
        interp_method (str): 'nearest' or 'bilinear' interpolation
        
    Returns:
        tuple: (positions, times) where positions is list of (x, y) and times is list of time values
    """
    positions = [start_pos]
    times = [0.0]  # Time in hours
    
    x, y = start_pos
    current_time = 0.0
    
    # Load mask to check for land
    mask = load_mask(data_dir)
    
    # Choose interpolation function
    flow_interp_func = interpolate_flow if interp_method == 'bilinear' else nearest_flow
    
    # For each time step in the data
    for t_idx, t in enumerate(time_steps):
        # Load flow data for current time step
        u, v = load_flow_data(t, data_dir)
        
        # Number of sub-steps within this time step
        n_substeps = int(1/dt)
        
        # Simulate particle movement for this time step with smaller dt
        for substep in range(n_substeps):
            # Check if current position is on land
            grid_y, grid_x = int(round(y)), int(round(x))
            
            # Keep within bounds
            grid_y = max(0, min(grid_y, mask.shape[0] - 1))
            grid_x = max(0, min(grid_x, mask.shape[1] - 1))
            
            # If on land, stop tracking this particle
            if mask[grid_y, grid_x] == 0:
                return positions, times
            
            # Get interpolated flow at current position
            u_flow, v_flow = flow_interp_func(u, v, x, y)
            
            # Convert flow to km/h 
            # Flow values are in cm/s, convert to km/h: * (25/0.9) * 0.036
            u_km_h = u_flow * (25/0.9) * 0.036
            v_km_h = v_flow * (25/0.9) * 0.036
            
            # Update position using 4th order Runge-Kutta method
            h = dt * 3  # 3 hours per time step
            
            # First RK step
            k1_x = u_km_h
            k1_y = v_km_h
            
            # Second RK step
            u_half, v_half = flow_interp_func(u, v, x + 0.5 * h * k1_x, y + 0.5 * h * k1_y)
            k2_x = u_half * (25/0.9) * 0.036
            k2_y = v_half * (25/0.9) * 0.036
            
            # Third RK step
            u_half, v_half = flow_interp_func(u, v, x + 0.5 * h * k2_x, y + 0.5 * h * k2_y)
            k3_x = u_half * (25/0.9) * 0.036
            k3_y = v_half * (25/0.9) * 0.036
            
            # Fourth RK step
            u_full, v_full = flow_interp_func(u, v, x + h * k3_x, y + h * k3_y)
            k4_x = u_full * (25/0.9) * 0.036
            k4_y = v_full * (25/0.9) * 0.036
            
            # Update position
            x_new = x + (h/6) * (k1_x + 2*k2_x + 2*k3_x + k4_x)
            y_new = y + (h/6) * (k1_y + 2*k2_y + 2*k3_y + k4_y)
            
            # Update position for next iteration
            x, y = x_new, y_new
            
            # Update time
            current_time += dt * 3  # 3 hours per time step
            
            # Add to trajectory
            positions.append((x, y))
            times.append(current_time)
    
    return positions, times

def visualize_trajectories(data_dir='data', output_dir='output'):
    """
    Visualize particle trajectories in the ocean flow.
    
    Args:
        data_dir (str): Directory containing data files
        output_dir (str): Directory to save output figures
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    # Load mask
    mask = load_mask(data_dir)
    
    # Define several starting positions
    start_positions = [
        (100, 100),  # Bottom-left region
        (200, 200),  # Central region
        (300, 300),  # Upper-right region
        (150, 350),  # Upper-left region
        (350, 150)   # Bottom-right region
    ]
    
    # Define simulation time steps
    time_steps = range(1, 51)  # Use first 50 time steps
    
    # Define interpolation methods
    interp_methods = ['nearest', 'bilinear']
    
    # For each interpolation method
    for method in interp_methods:
        print(f"\nSimulating trajectories using {method} interpolation...")
        
        # Simulate trajectories
        all_trajectories = []
        all_times = []
        
        for i, start_pos in enumerate(start_positions):
            print(f"  Simulating trajectory {i+1} from position {start_pos}...")
            positions, times = simulate_trajectory(start_pos, time_steps, data_dir, 
                                                  interp_method=method)
            all_trajectories.append(positions)
            all_times.append(times)
            print(f"    Trajectory length: {len(positions)} positions")
            print(f"    Final time: {times[-1]:.1f} hours")
            print(f"    Final position: ({positions[-1][0]:.1f}, {positions[-1][1]:.1f})")
            
            # Calculate total distance traveled
            distance = sum(np.sqrt((positions[i+1][0] - positions[i][0])**2 + 
                                  (positions[i+1][1] - positions[i][1])**2) 
                          for i in range(len(positions)-1))
            print(f"    Total distance traveled: {distance*3:.1f} km")  # 3km per grid unit
        
        # Plot trajectories
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot land/water mask as background
        ax.imshow(mask == 0, cmap='terrain', alpha=0.5, 
                 extent=[0, mask.shape[1], 0, mask.shape[0]])
        
        # Plot trajectories
        colors = ['r', 'g', 'b', 'c', 'm']
        
        for i, (trajectory, times) in enumerate(zip(all_trajectories, all_times)):
            if trajectory:
                # Extract x and y coordinates
                x_coords, y_coords = zip(*trajectory)
                
                # Create colored scatter points based on time
                sc = ax.scatter(x_coords, y_coords, c=times, cmap='viridis', 
                              s=5, alpha=0.7, edgecolors='none')
                
                # Add colorbar
                plt.colorbar(sc, label='Time (hours)')
                
                # Add start and end points
                ax.plot(x_coords[0], y_coords[0], 'o', color=colors[i], 
                       markersize=8, label=f'Start {i+1}')
                ax.plot(x_coords[-1], y_coords[-1], 's', color=colors[i], 
                       markersize=8, label=f'End {i+1}')
        
        # Add title and labels
        ax.set_title(f'Particle Trajectories ({method} interpolation)')
        ax.set_xlabel('Grid Position (3km per unit)')
        ax.set_ylabel('Grid Position (3km per unit)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'particle_trajectories_{method}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create animation for the first trajectory
        if all_trajectories and all_trajectories[0]:
            print("  Creating animation for trajectory 1...")
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Plot land/water mask as background
            ax.imshow(mask == 0, cmap='terrain', alpha=0.5, 
                     extent=[0, mask.shape[1], 0, mask.shape[0]])
            
            # Get coordinates for first trajectory
            x_coords, y_coords = zip(*all_trajectories[0])
            times = all_times[0]
            
            # Create point for animation
            point, = ax.plot([], [], 'ro', markersize=8)
            
            # Set up animation
            def update(frame):
                point.set_data([x_coords[frame]], [y_coords[frame]])
                return point,
            
            # Create animation with fewer frames for better performance
            step = max(1, len(x_coords) // 100)  # Use at most 100 frames
            ani = animation.FuncAnimation(fig, update, frames=range(0, len(x_coords), step),
                                        interval=50, blit=True)
            
            # Add title and labels
            ax.set_title(f'Particle Trajectory Animation ({method} interpolation)')
            ax.set_xlabel('Grid Position (3km per unit)')
            ax.set_ylabel('Grid Position (3km per unit)')
            ax.grid(True, alpha=0.3)
            ax.invert_yaxis()
            
            # Save animation
            ani.save(os.path.join(output_dir, f'trajectory_animation_{method}.gif'), 
                    writer='pillow', fps=20)
            plt.close()

if __name__ == "__main__":
    visualize_trajectories()