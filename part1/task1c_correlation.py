import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import binned_statistic

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import load_all_flow_data, load_mask, calculate_speed, ensure_output_dir

def compute_correlation_with_point(data, ref_point, max_distance=200):
    """
    Compute correlation between a reference point and all other points within max_distance.
    
    Args:
        data (numpy.ndarray): Array with shape (time_steps, height, width)
        ref_point (tuple): Reference point coordinates (x, y)
        max_distance (int): Maximum L1 distance to consider
        
    Returns:
        tuple: (distances, correlations) arrays
    """
    ref_x, ref_y = ref_point
    
    # Get time series at reference point
    ref_timeseries = data[:, ref_y, ref_x]
    
    # Initialize arrays for distances and correlations
    distances = []
    correlations = []
    
    # Get array dimensions
    time_steps, height, width = data.shape
    
    # Loop through all points in the grid
    for y in range(height):
        for x in range(width):
            # Calculate L1 distance (Manhattan distance)
            l1_dist = abs(x - ref_x) + abs(y - ref_y)
            
            if l1_dist <= max_distance and (x != ref_x or y != ref_y):
                # Get time series at current point
                current_timeseries = data[:, y, x]
                
                # Compute correlation
                corr = np.corrcoef(ref_timeseries, current_timeseries)[0, 1]
                
                # Store results if correlation is valid (not NaN)
                if not np.isnan(corr):
                    distances.append(l1_dist)
                    correlations.append(corr)
    
    return np.array(distances), np.array(correlations)

def analyze_spatial_correlation(data_dir='data', output_dir='output'):
    """
    Analyze spatial correlation in ocean flow data.
    
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
    
    # Calculate speed at each time step
    print("Calculating speed data...")
    speed_data = np.array([calculate_speed(u_data[t], v_data[t]) for t in range(len(u_data))])
    
    # Points to analyze as specified in the problem
    points = [(140, 115), (400, 400)]  # (x, y) coordinates
    max_distance = 200
    
    # For pretty plotting
    point_names = ["Point 1 (140, 115)", "Point 2 (400, 400)"]
    variables = ["Speed", "Horizontal Flow (u)", "Vertical Flow (v)"]
    data_arrays = [speed_data, u_data, v_data]
    
    # Create a big figure with subplots for both points and all variables
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Process each point
    for i, (point, point_name) in enumerate(zip(points, point_names)):
        print(f"Analyzing correlation for {point_name}...")
        
        # Analyze correlation for each variable
        for j, (var_name, data_array) in enumerate(zip(variables, data_arrays)):
            print(f"  Processing {var_name}...")
            
            # Check if point is on water
            x, y = point
            if mask[y, x] == 0:
                print(f"  Warning: {point_name} is on land. Skipping.")
                continue
            
            # Compute correlation
            distances, correlations = compute_correlation_with_point(data_array, point, max_distance)
            
            # Bin the data for clearer visualization
            bin_size = 5
            bin_edges = np.arange(0, max_distance + bin_size, bin_size)
            bin_centers = bin_edges[:-1] + bin_size/2
            
            # Compute binned statistics
            binned_corr_mean, _, _ = binned_statistic(distances, correlations, statistic='mean', bins=bin_edges)
            binned_corr_std, _, _ = binned_statistic(distances, correlations, statistic='std', bins=bin_edges)
            
            # Plot scatter of raw data (semi-transparent)
            axes[i, j].scatter(distances, correlations, s=2, alpha=0.1, color='gray')
            
            # Plot binned statistics
            axes[i, j].errorbar(bin_centers, binned_corr_mean, yerr=binned_corr_std, 
                              fmt='o-', color='blue', capsize=3, elinewidth=1, markersize=5)
            
            # Add exponential fit (simple Gaussian process-like model)
            valid_bins = ~np.isnan(binned_corr_mean)
            if np.sum(valid_bins) > 3:  # Need at least 3 points for fitting
                # Initial guess: correlation drops to 0.1 at distance of 40
                initial_length_scale = 40 / np.sqrt(-2 * np.log(0.1))
                
                # Function to fit
                def exp_decay(x, sigma_sq, length_scale):
                    return sigma_sq * np.exp(-x**2 / (2 * length_scale**2))
                
                # Fit using scipy's curve_fit
                from scipy.optimize import curve_fit
                try:
                    popt, _ = curve_fit(exp_decay, bin_centers[valid_bins], binned_corr_mean[valid_bins], 
                                       p0=[1.0, initial_length_scale], bounds=([0, 0], [2, 500]))
                    
                    # Plot fitted curve
                    x_fit = np.linspace(0, max_distance, 100)
                    y_fit = exp_decay(x_fit, *popt)
                    axes[i, j].plot(x_fit, y_fit, 'r-', label=f'RBF fit: σ²={popt[0]:.2f}, ℓ={popt[1]:.2f}')
                    
                    # Print fitted parameters
                    print(f"  RBF fit parameters: σ²={popt[0]:.2f}, ℓ={popt[1]:.2f}")
                    
                except RuntimeError as e:
                    print(f"  Could not fit RBF curve: {e}")
            
            # Set plot properties
            axes[i, j].set_title(f'{point_name}: {var_name} Correlation')
            axes[i, j].set_xlabel('L1 Distance (grid units = 3km)')
            axes[i, j].set_ylabel('Correlation')
            axes[i, j].grid(True, alpha=0.3)
            axes[i, j].set_ylim(-0.2, 1.0)
            
            # Add legend if there's a fitted curve
            if 'popt' in locals():
                axes[i, j].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'spatial_correlation.png'), dpi=300, bbox_inches='tight')
    
    # Create 2D correlation maps
    print("\nCreating 2D correlation maps...")
    
    # For each point, create a visualization showing correlation with all other points
    for i, (point, point_name) in enumerate(zip(points, point_names)):
        x, y = point
        
        # Skip if point is on land
        if mask[y, x] == 0:
            print(f"  Warning: {point_name} is on land. Skipping 2D map.")
            continue
        
        # Create figure with subplots for each variable
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
        
        for j, (var_name, data_array) in enumerate(zip(variables, data_arrays)):
            # Initialize correlation map
            corr_map = np.zeros_like(mask, dtype=float)
            corr_map.fill(np.nan)  # Fill with NaN for points we don't compute
            
            # Get reference time series
            ref_timeseries = data_array[:, y, x]
            
            # Compute correlation with all points
            for y2 in range(mask.shape[0]):
                for x2 in range(mask.shape[1]):
                    # Only consider water points within max_distance
                    if mask[y2, x2] == 1:  # Water point
                        l1_dist = abs(x2 - x) + abs(y2 - y)
                        if l1_dist <= max_distance:
                            current_timeseries = data_array[:, y2, x2]
                            corr = np.corrcoef(ref_timeseries, current_timeseries)[0, 1]
                            corr_map[y2, x2] = corr
            
            # Plot correlation map
            im = axes2[j].imshow(corr_map, cmap='coolwarm', vmin=-1, vmax=1,
                               extent=[0, corr_map.shape[1], 0, corr_map.shape[0]])
            
            # Add colorbar
            cbar = fig2.colorbar(im, ax=axes2[j])
            cbar.set_label('Correlation')
            
            # Mark reference point
            axes2[j].plot(x, y, 'ko', markersize=8)
            
            # Add contour for L1 distance
            # Create distance map
            y_grid, x_grid = np.mgrid[:mask.shape[0], :mask.shape[1]]
            l1_distance_map = np.abs(x_grid - x) + np.abs(y_grid - y)
            
            # Plot contours
            contour_levels = [50, 100, 150, 200]
            contour = axes2[j].contour(x_grid, y_grid, l1_distance_map, levels=contour_levels, 
                                     colors='black', alpha=0.5, linestyles='dashed')
            axes2[j].clabel(contour, inline=True, fontsize=8)
            
            # Set plot properties
            axes2[j].set_title(f'{point_name}: {var_name} Correlation')
            axes2[j].set_xlabel('Grid Position (3km per unit)')
            axes2[j].set_ylabel('Grid Position (3km per unit)')
            axes2[j].invert_yaxis()
            
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'correlation_map_{i+1}.png'), dpi=300, bbox_inches='tight')
    
    # Close all plots
    plt.close('all')
    print(f"\nAnalysis complete. All figures saved to {output_dir}/")

if __name__ == "__main__":
    analyze_spatial_correlation()