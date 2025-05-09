import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import load_all_flow_data, load_mask, plot_flow_field, calculate_speed, ensure_output_dir

def visualize_average_flow(data_dir='data', output_dir='output'):
    """
    Visualize the average ocean flow as a 2D vector field.
    
    Args:
        data_dir (str): Directory containing data files
        output_dir (str): Directory to save output figures
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    # Load all flow data
    print("Loading flow data...")
    u_data, v_data = load_all_flow_data(data_dir)
    
    # Load mask
    mask = load_mask(data_dir)
    
    # Calculate average flow
    print("Calculating average flow...")
    u_avg = np.mean(u_data, axis=0)
    v_avg = np.mean(v_data, axis=0)
    
    # Calculate average speed
    speed_avg = calculate_speed(u_avg, v_avg)
    
    # Find locations of strongest currents
    strongest_idx = np.unravel_index(np.argmax(speed_avg), speed_avg.shape)
    strongest_y, strongest_x = strongest_idx
    strongest_speed = speed_avg[strongest_idx]
    
    print(f"Strongest average flow current location: ({strongest_x}, {strongest_y})")
    print(f"Strongest average flow speed: {strongest_speed:.2f} cm/s")
    print(f"Flow direction at strongest point: ({u_avg[strongest_idx]:.4f}, {v_avg[strongest_idx]:.4f})")
    
    # Plot average flow field with high density for visualization
    print("Plotting average flow field...")
    fig, ax = plot_flow_field(u_avg, v_avg, mask=(mask==0), scale=30, 
                             title="Average Ocean Flow in Philippine Archipelago", 
                             density=8)  # Sample every 8th point for clearer visualization
    
    # Mark the strongest current on the plot
    ax.plot(strongest_x, strongest_y, 'ro', markersize=10, label=f'Strongest Current: {strongest_speed:.1f} cm/s')
    ax.legend()
    
    # Save the figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'average_flow.png'), dpi=300, bbox_inches='tight')
    
    # Create a more detailed view around the strongest current
    margin = 50
    x_min = max(0, strongest_x - margin)
    x_max = min(u_avg.shape[1], strongest_x + margin)
    y_min = max(0, strongest_y - margin)
    y_max = min(u_avg.shape[0], strongest_y + margin)
    
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    
    # Plot zoomed region
    u_zoom = u_avg[y_min:y_max, x_min:x_max]
    v_zoom = v_avg[y_min:y_max, x_min:x_max]
    mask_zoom = mask[y_min:y_max, x_min:x_max] if mask is not None else None
    
    y_indices, x_indices = np.mgrid[:u_zoom.shape[0]:4, :u_zoom.shape[1]:4]
    
    # Calculate speed for color mapping
    speed_zoom = calculate_speed(u_zoom, v_zoom)
    
    quiver = ax2.quiver(x_indices + x_min, y_indices + y_min, 
                       u_zoom[::4, ::4], v_zoom[::4, ::4], 
                       speed_zoom[::4, ::4], cmap='viridis', scale=20, 
                       pivot='mid', alpha=0.8)
    
    # Add colorbar
    cbar = fig2.colorbar(quiver, ax=ax2)
    cbar.set_label('Flow Speed (cm/s)')
    
    # Plot land/water mask if provided
    if mask_zoom is not None:
        ax2.imshow(mask_zoom == 0, cmap='terrain', alpha=0.3, 
                 extent=[x_min, x_max, y_min, y_max])
    
    # Mark the strongest current
    ax2.plot(strongest_x, strongest_y, 'ro', markersize=10, 
           label=f'Strongest Current: {strongest_speed:.1f} cm/s')
    
    ax2.set_title('Detailed View of Strongest Current Region')
    ax2.set_xlabel('Grid Position (3km per unit)')
    ax2.set_ylabel('Grid Position (3km per unit)')
    ax2.legend()
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'strongest_current_detail.png'), dpi=300, bbox_inches='tight')
    
    # Find other strong currents
    # Make a copy of speed_avg to avoid modifying the original
    speed_avg_copy = speed_avg.copy()
    
    # Zero out the strongest current and its immediate vicinity
    radius = 10
    y_min_local = max(0, strongest_y - radius)
    y_max_local = min(speed_avg_copy.shape[0], strongest_y + radius + 1)
    x_min_local = max(0, strongest_x - radius)
    x_max_local = min(speed_avg_copy.shape[1], strongest_x + radius + 1)
    
    speed_avg_copy[y_min_local:y_max_local, x_min_local:x_max_local] = 0
    
    # Find the next 5 strongest currents
    top_currents = []
    for _ in range(5):
        idx = np.unravel_index(np.argmax(speed_avg_copy), speed_avg_copy.shape)
        y, x = idx
        speed = speed_avg_copy[idx]
        if speed > 0:
            top_currents.append((x, y, speed))
            
            # Zero out this current and its vicinity
            y_min_local = max(0, y - radius)
            y_max_local = min(speed_avg_copy.shape[0], y + radius + 1)
            x_min_local = max(0, x - radius)
            x_max_local = min(speed_avg_copy.shape[1], x + radius + 1)
            
            speed_avg_copy[y_min_local:y_max_local, x_min_local:x_max_local] = 0
    
    # Plot an overall map showing the top currents
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    
    # Plot a colored map of the average speed
    im = ax3.imshow(speed_avg, cmap='viridis', 
                  extent=[0, speed_avg.shape[1], 0, speed_avg.shape[0]])
    cbar = fig3.colorbar(im, ax=ax3)
    cbar.set_label('Average Speed (cm/s)')
    
    # Add land/water mask
    if mask is not None:
        ax3.imshow(mask == 0, cmap='terrain', alpha=0.3, 
                 extent=[0, mask.shape[1], 0, mask.shape[0]])
    
    # Mark all top currents
    ax3.plot(strongest_x, strongest_y, 'ro', markersize=12, 
           label=f'Strongest: {strongest_speed:.1f} cm/s')
    
    for i, (x, y, speed) in enumerate(top_currents):
        ax3.plot(x, y, 'mo', markersize=8, 
               label=f'Current {i+2}: {speed:.1f} cm/s')
    
    ax3.set_title('Top Strongest Currents in the Philippine Archipelago')
    ax3.set_xlabel('Grid Position (3km per unit)')
    ax3.set_ylabel('Grid Position (3km per unit)')
    ax3.legend()
    ax3.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_currents.png'), dpi=300, bbox_inches='tight')
    
    print(f"\nAdditional strong currents:")
    for i, (x, y, speed) in enumerate(top_currents):
        print(f"Current {i+2}: Location ({x}, {y}), Speed: {speed:.2f} cm/s")
    
    # Close all plots
    plt.close('all')
    print(f"\nAnalysis complete. All figures saved to {output_dir}/")
    
    return strongest_idx, top_currents

if __name__ == "__main__":
    visualize_average_flow()