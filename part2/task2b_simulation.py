import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import Normalize
import os
import sys
from tqdm import tqdm

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import load_flow_data, load_mask, ensure_output_dir
from part2.task2a_tracking import simulate_trajectory

def simulate_plane_crash_debris(mean_pos=(100, 350), variance_values=[100, 500, 1000], 
                              num_particles=100, data_dir='data', output_dir='output',
                              interp_method='bilinear'):
    """
    Simulate debris from a plane crash with Gaussian distributed starting positions.
    
    Args:
        mean_pos (tuple): Mean position of crash location (x, y)
        variance_values (list): List of variance values to simulate
        num_particles (int): Number of particles to simulate for each variance
        data_dir (str): Directory containing data files
        output_dir (str): Directory to save output figures
        interp_method (str): 'nearest' or 'bilinear' interpolation
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    # Load mask
    mask = load_mask(data_dir)
    
    # Time periods to analyze (in hours)
    time_periods = [48, 72, 120]  # hours
    
    # Convert to time steps (3 hours per time step)
    time_steps_dict = {}
    for period in time_periods:
        steps = period // 3
        time_steps_dict[period] = list(range(1, steps + 1))
    
    # Create a plot of the initial distribution (Sulu Sea area)
    plt.figure(figsize=(12, 10))
    
    # Show land/water mask
    plt.imshow(mask == 0, cmap='terrain', alpha=0.5, 
              extent=[0, mask.shape[1], 0, mask.shape[0]])
    
    # Mark the mean crash location
    plt.plot(mean_pos[0], mean_pos[1], 'r*', markersize=15, label=f'Mean Crash Location: {mean_pos}')
    
    # Show distribution ellipses for different variances
    for variance in variance_values:
        # Create a circle at 2-sigma (95% confidence region)
        sigma = np.sqrt(variance)
        circle = plt.Circle(mean_pos, 2 * sigma, fill=False, edgecolor='black', 
                           linestyle='--', label=f'2σ region (σ²={variance})')
        plt.gca().add_patch(circle)
    
    plt.title('Plane Crash Location Distribution in the Sulu Sea')
    plt.xlabel('Grid Position (3km per unit)')
    plt.ylabel('Grid Position (3km per unit)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'crash_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # For each variance value
    for variance in variance_values:
        print(f"\nSimulating debris with variance σ² = {variance}")
        
        # Create covariance matrix (isotropic)
        cov = np.eye(2) * variance
        
        # Sample initial positions from Gaussian distribution
        initial_positions = np.random.multivariate_normal(mean_pos, cov, num_particles)
        
        # Filter out positions on land
        valid_positions = []
        for pos in initial_positions:
            x, y = pos
            # Convert to integers for mask lookup
            grid_x, grid_y = int(round(x)), int(round(y))
            
            # Keep within bounds
            grid_x = max(0, min(grid_x, mask.shape[1] - 1))
            grid_y = max(0, min(grid_y, mask.shape[0] - 1))
            
            if mask[grid_y, grid_x] == 1:  # If on water
                valid_positions.append(pos)
        
        print(f"  {len(valid_positions)} valid initial positions (on water)")
        
        # If no valid positions, skip this variance
        if not valid_positions:
            print("  No valid positions for this variance. Skipping.")
            continue
        
        # Convert valid positions to numpy array
        valid_positions = np.array(valid_positions)
        
        # Simulate trajectories for each time period
        for period in time_periods:
            print(f"  Simulating for {period} hours...")
            
            # Get time steps for this period
            time_steps = time_steps_dict[period]
            
            # Track final positions for this time period
            final_positions = []
            final_times = []
            
            # Run simulations for each initial position
            for i, start_pos in enumerate(tqdm(valid_positions)):
                # Convert to tuple
                start_pos_tuple = (start_pos[0], start_pos[1])
                
                # Simulate trajectory
                positions, times = simulate_trajectory(
                    start_pos_tuple, time_steps, data_dir, interp_method=interp_method)
                
                if positions:
                    final_positions.append(positions[-1])
                    final_times.append(times[-1])
            
            print(f"    {len(final_positions)} particles tracked successfully")
            
            # If no final positions, skip to next period
            if not final_positions:
                print("    No particles successfully tracked. Skipping.")
                continue
            
            # Convert to numpy array for easier manipulation
            final_positions = np.array(final_positions)
            final_times = np.array(final_times)
            
            # Calculate statistics
            mean_final = np.mean(final_positions, axis=0)
            std_final = np.std(final_positions, axis=0)
            
            # Calculate covariance matrix of final positions
            cov_final = np.cov(final_positions.T)
            
            # Calculate eigenvalues and eigenvectors for plotting ellipse
            eigenvalues, eigenvectors = np.linalg.eigh(cov_final)
            
            # Sort eigenvalues and eigenvectors
            idx = eigenvalues.argsort()[::-1]
            eigenvalues = eigenvalues[idx]
            eigenvectors = eigenvectors[:, idx]
            
            # Calculate ellipse parameters
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            width, height = 2 * np.sqrt(5.991 * eigenvalues)  # 95% confidence region
            
            # Create visualization
            plt.figure(figsize=(12, 10))
            
            # Show land/water mask
            plt.imshow(mask == 0, cmap='terrain', alpha=0.5, 
                      extent=[0, mask.shape[1], 0, mask.shape[0]])
            
            # Plot initial positions
            plt.scatter(valid_positions[:, 0], valid_positions[:, 1], c='blue', s=20, alpha=0.5, 
                       label='Initial positions')
            
            # Plot mean crash position
            plt.plot(mean_pos[0], mean_pos[1], 'r*', markersize=15, 
                    label='Mean Crash Location')
            
            # Plot final positions, colored by time
            scatter = plt.scatter(final_positions[:, 0], final_positions[:, 1], 
                                c=final_times, cmap='viridis', s=30, alpha=0.7, 
                                label=f'Debris at {period} hours')
            
            # Add colorbar
            cbar = plt.colorbar(scatter)
            cbar.set_label('Actual simulation time (hours)')
            
            # Plot mean final position
            plt.plot(mean_final[0], mean_final[1], 'mo', markersize=12, 
                    label=f'Mean Debris Location: ({mean_final[0]:.1f}, {mean_final[1]:.1f})')
            
            # Draw 95% confidence ellipse
            ellipse = patches.Ellipse(xy=mean_final, width=width, height=height, angle=angle, 
                                     alpha=0.3, color='magenta', 
                                     label='95% confidence region')
            plt.gca().add_patch(ellipse)
            
            # Calculate search area in km²
            search_area = np.pi * width * height * (3**2) / 4  # 3km per grid unit
            
            # Add title and labels
            plt.title(f'Plane Crash Debris after {period} hours (σ² = {variance})')
            plt.xlabel('Grid Position (3km per unit)')
            plt.ylabel('Grid Position (3km per unit)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.gca().invert_yaxis()
            
            # Add text box with statistics
            stats_text = (f"Mean final position: ({mean_final[0]:.1f}, {mean_final[1]:.1f})\n"
                         f"Std dev: x={std_final[0]:.1f}, y={std_final[1]:.1f}\n"
                         f"95% search area: {search_area:.1f} km²\n"
                         f"Distance from crash: {np.sqrt(np.sum((mean_final - mean_pos)**2))*3:.1f} km")
            
            plt.annotate(stats_text, xy=(0.02, 0.02), xycoords='axes fraction',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'debris_var{variance}_t{period}.png'), 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"    Figure saved to {os.path.join(output_dir, f'debris_var{variance}_t{period}.png')}")
            print(f"    Mean final position: ({mean_final[0]:.1f}, {mean_final[1]:.1f})")
            print(f"    Standard deviation: x={std_final[0]:.1f}, y={std_final[1]:.1f}")
            print(f"    95% search area: {search_area:.1f} km²")
            print(f"    Distance from crash: {np.sqrt(np.sum((mean_final - mean_pos)**2))*3:.1f} km")
    
    # Create summary plot showing all variances and time periods
    fig, axes = plt.subplots(len(variance_values), len(time_periods), 
                            figsize=(5*len(time_periods), 5*len(variance_values)))
    
    # Handle case of single row or column
    if len(variance_values) == 1:
        axes = np.array([axes])
    if len(time_periods) == 1:
        axes = np.array([axes]).T
    
    # For each variance and time period, load the saved image and display it
    for i, variance in enumerate(variance_values):
        for j, period in enumerate(time_periods):
            img_path = os.path.join(output_dir, f'debris_var{variance}_t{period}.png')
            if os.path.exists(img_path):
                img = plt.imread(img_path)
                axes[i, j].imshow(img)
                axes[i, j].set_title(f'σ² = {variance}, t = {period}h')
                axes[i, j].axis('off')
            else:
                axes[i, j].text(0.5, 0.5, 'No data', ha='center', va='center', transform=axes[i, j].transAxes)
                axes[i, j].set_title(f'σ² = {variance}, t = {period}h')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'debris_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSummary figure saved to {os.path.join(output_dir, 'debris_summary.png')}")
    print("\nAnalysis complete. All figures saved to {}/".format(output_dir))

if __name__ == "__main__":
    simulate_plane_crash_debris()