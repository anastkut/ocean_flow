import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import load_all_flow_data, load_mask, calculate_speed, ensure_output_dir

def visualize_average_speed(data_dir='data', output_dir='output'):
    """
    Visualize the average speed of ocean flow as a 2D graph.
    
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
    
    # Calculate average speed at each grid point
    print("Calculating average speed...")
    
    # Method 1: Average of speeds (speed at each timestep, then average)
    speed_at_each_timestep = np.array([calculate_speed(u_data[t], v_data[t]) for t in range(len(u_data))])
    avg_of_speeds = np.mean(speed_at_each_timestep, axis=0)
    
    # Method 2: Speed of average flow (average u and v, then calculate speed)
    u_avg = np.mean(u_data, axis=0)
    v_avg = np.mean(v_data, axis=0)
    speed_of_avg_flow = calculate_speed(u_avg, v_avg)
    
    # Create comparison plot
    print("Creating comparison of average speed vs. speed of average flow...")
    fig1, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Plot average of speeds
    im1 = axes[0].imshow(avg_of_speeds, cmap='viridis', 
                       extent=[0, avg_of_speeds.shape[1], 0, avg_of_speeds.shape[0]])
    axes[0].set_title('Average of Speeds (Method 1)')
    fig1.colorbar(im1, ax=axes[0], label='Speed (cm/s)')
    
    # Overlay land mask
    if mask is not None:
        axes[0].imshow(mask == 0, cmap='binary', alpha=0.3, 
                     extent=[0, mask.shape[1], 0, mask.shape[0]])
    
    # Plot speed of average flow
    im2 = axes[1].imshow(speed_of_avg_flow, cmap='viridis', 
                       extent=[0, speed_of_avg_flow.shape[1], 0, speed_of_avg_flow.shape[0]])
    axes[1].set_title('Speed of Average Flow (Method 2)')
    fig1.colorbar(im2, ax=axes[1], label='Speed (cm/s)')
    
    # Overlay land mask
    if mask is not None:
        axes[1].imshow(mask == 0, cmap='binary', alpha=0.3, 
                     extent=[0, mask.shape[1], 0, mask.shape[0]])
    
    # Plot ratio of the two
    # To avoid division by zero
    epsilon = 1e-10
    ratio = avg_of_speeds / (speed_of_avg_flow + epsilon)
    
    # For visualization, clip outliers
    ratio_clipped = np.clip(ratio, 0, 5)
    
    im3 = axes[2].imshow(ratio_clipped, cmap='coolwarm', 
                       extent=[0, ratio.shape[1], 0, ratio.shape[0]], 
                       vmin=1, vmax=3)
    axes[2].set_title('Ratio: Avg of Speeds / Speed of Avg Flow')
    fig1.colorbar(im3, ax=axes[2], label='Ratio')
    
    # Overlay land mask
    if mask is not None:
        axes[2].imshow(mask == 0, cmap='binary', alpha=0.3, 
                     extent=[0, mask.shape[1], 0, mask.shape[0]])
    
    # Adjust axes for all plots
    for ax in axes:
        ax.set_xlabel('Grid Position (3km per unit)')
        ax.set_ylabel('Grid Position (3km per unit)')
        ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'speed_comparison.png'), dpi=300, bbox_inches='tight')
    
    # Identify interesting areas
    print("Identifying areas with interesting behavior...")
    
    # Define high and low thresholds
    high_ratio_threshold = np.percentile(ratio[mask == 1], 90)  # 90th percentile, water only
    
    # Identify areas with high average speed but low average flow
    high_avg_speed_low_avg_flow = (avg_of_speeds > np.percentile(avg_of_speeds[mask == 1], 75)) & \
                                 (speed_of_avg_flow < np.percentile(speed_of_avg_flow[mask == 1], 25)) & \
                                 (mask == 1)  # Only consider water points
    
    # Identify areas with low average speed but high average flow
    low_avg_speed_high_avg_flow = (avg_of_speeds < np.percentile(avg_of_speeds[mask == 1], 25)) & \
                                 (speed_of_avg_flow > np.percentile(speed_of_avg_flow[mask == 1], 75)) & \
                                 (mask == 1)  # Only consider water points
    
    # Create a figure to visualize these areas
    fig2, ax = plt.subplots(figsize=(12, 10))
    
    # Create a composite image
    interest_map = np.zeros_like(avg_of_speeds)
    interest_map[high_avg_speed_low_avg_flow] = 1  # High speed, low flow
    interest_map[low_avg_speed_high_avg_flow] = 2  # Low speed, high flow
    
    # Apply mask
    interest_map[mask == 0] = -1  # Land
    
    # Plot the map
    cmap = plt.cm.get_cmap('coolwarm', 3)
    im = ax.imshow(interest_map, cmap=cmap, interpolation='nearest',
                 extent=[0, interest_map.shape[1], 0, interest_map.shape[0]],
                 vmin=-1, vmax=2)
    
    # Add colorbar
    cbar = fig2.colorbar(im, ax=ax, ticks=[-1, 0, 1, 2])
    cbar.set_label('Area Type')
    cbar.ax.set_yticklabels(['Land', 'Normal', 'High Avg Speed,\nLow Avg Flow', 'Low Avg Speed,\nHigh Avg Flow'])
    
    # Calculate statistics
    num_high_low = np.sum(high_avg_speed_low_avg_flow)
    num_low_high = np.sum(low_avg_speed_high_avg_flow)
    total_water_points = np.sum(mask == 1)
    
    print(f"Number of points with high average speed but low average flow: {num_high_low} ({num_high_low/total_water_points*100:.2f}% of water points)")
    print(f"Number of points with low average speed but high average flow: {num_low_high} ({num_low_high/total_water_points*100:.2f}% of water points)")
    
    # Add title and labels
    ax.set_title('Areas with Interesting Speed Properties')
    ax.set_xlabel('Grid Position (3km per unit)')
    ax.set_ylabel('Grid Position (3km per unit)')
    ax.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'interesting_areas.png'), dpi=300, bbox_inches='tight')
    
    # Extract some example points from each category
    def extract_example_points(condition, n=5):
        y_indices, x_indices = np.where(condition)
        if len(y_indices) > 0:
            # Take n evenly spaced samples
            step = max(1, len(y_indices) // n)
            indices = list(range(0, len(y_indices), step))[:n]
            return [(x_indices[i], y_indices[i]) for i in indices]
        return []
    
    high_low_examples = extract_example_points(high_avg_speed_low_avg_flow)
    low_high_examples = extract_example_points(low_avg_speed_high_avg_flow)
    
    # Plot time series for these example points
    def plot_time_series(points, title, filename):
        fig, axes = plt.subplots(len(points), 1, figsize=(12, 3*len(points)))
        
        if len(points) == 1:
            axes = [axes]
        
        for i, (x, y) in enumerate(points):
            # Get time series of speed at this point
            speed_series = [speed_at_each_timestep[t, y, x] for t in range(len(speed_at_each_timestep))]
            
            # Get time series of u, v at this point
            u_series = [u_data[t, y, x] for t in range(len(u_data))]
            v_series = [v_data[t, y, x] for t in range(len(v_data))]
            
            # Plot
            time = np.arange(1, len(speed_series)+1) * 3  # 3 hours per time step
            
            ax1 = axes[i]
            line1, = ax1.plot(time, speed_series, 'b-', label='Speed')
            ax1.set_ylabel('Speed (cm/s)')
            ax1.set_title(f'Time Series at Point ({x}, {y})')
            
            # Create twin axis for u, v
            ax2 = ax1.twinx()
            line2, = ax2.plot(time, u_series, 'r-', label='U Component')
            line3, = ax2.plot(time, v_series, 'g-', label='V Component')
            ax2.set_ylabel('Flow Components')
            
            # Add average values as horizontal lines
            avg_speed = np.mean(speed_series)
            avg_u = np.mean(u_series)
            avg_v = np.mean(v_series)
            
            ax1.axhline(y=avg_speed, color='b', linestyle='--', alpha=0.5)
            ax2.axhline(y=avg_u, color='r', linestyle='--', alpha=0.5)
            ax2.axhline(y=avg_v, color='g', linestyle='--', alpha=0.5)
            
            # Add text annotations for averages
            ax1.text(time[-1] * 1.02, avg_speed, f'Avg: {avg_speed:.2f}', color='b', va='center')
            ax2.text(time[-1] * 1.02, avg_u, f'Avg: {avg_u:.2f}', color='r', va='center')
            ax2.text(time[-1] * 1.02, avg_v, f'Avg: {avg_v:.2f}', color='g', va='center')
            
            # Add legend
            lines = [line1, line2, line3]
            labels = [l.get_label() for l in lines]
            ax1.legend(lines, labels, loc='upper right')
            
            if i == len(points) - 1:
                ax1.set_xlabel('Time (hours)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    if high_low_examples:
        print(f"Plotting time series for {len(high_low_examples)} example points with high average speed but low average flow...")
        plot_time_series(high_low_examples, 'Example Points with High Average Speed but Low Average Flow', 'high_speed_low_flow_timeseries.png')
    
    if low_high_examples:
        print(f"Plotting time series for {len(low_high_examples)} example points with low average speed but high average flow...")
        plot_time_series(low_high_examples, 'Example Points with Low Average Speed but High Average Flow', 'low_speed_high_flow_timeseries.png')
    
    # Close all plots
    plt.close('all')
    print(f"\nAnalysis complete. All figures saved to {output_dir}/")
    
    return avg_of_speeds, speed_of_avg_flow, high_avg_speed_low_avg_flow, low_avg_speed_high_avg_flow

if __name__ == "__main__":
    visualize_average_speed()