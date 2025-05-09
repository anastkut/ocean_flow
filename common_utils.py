import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize

def load_flow_data(time_step, data_dir='data'):
    """
    Load u and v components for a specific time step.
    
    Args:
        time_step (int): Time step to load (1-100)
        data_dir (str): Directory containing data files
        
    Returns:
        tuple: (u_component, v_component) as numpy arrays
    """
    u_file = os.path.join(data_dir, f"{time_step}u.csv")
    v_file = os.path.join(data_dir, f"{time_step}v.csv")
    
    u_component = np.loadtxt(u_file, delimiter=',')
    v_component = np.loadtxt(v_file, delimiter=',')
    
    return u_component, v_component

def load_mask(data_dir='data'):
    """
    Load land/water mask.
    
    Args:
        data_dir (str): Directory containing data files
        
    Returns:
        numpy.ndarray: Mask with 0 for land, 1 for water
    """
    mask_file = os.path.join(data_dir, "mask.csv")
    return np.loadtxt(mask_file, delimiter=',')

def calculate_speed(u, v):
    """
    Calculate flow speed from u and v components.
    
    Args:
        u (numpy.ndarray): Horizontal flow component
        v (numpy.ndarray): Vertical flow component
        
    Returns:
        numpy.ndarray: Flow speed (magnitude) in cm/s
    """
    return np.sqrt(u**2 + v**2) * (25/0.9)  # Convert to cm/s

def load_all_flow_data(data_dir='data'):
    """
    Load all flow data for times 1-100.
    
    Args:
        data_dir (str): Directory containing data files
        
    Returns:
        tuple: (u_data, v_data) where each is a list of arrays
    """
    u_data = []
    v_data = []
    
    for t in range(1, 101):
        u, v = load_flow_data(t, data_dir)
        u_data.append(u)
        v_data.append(v)
    
    return np.array(u_data), np.array(v_data)

def plot_flow_field(u, v, mask=None, scale=50, title="Ocean Flow Field", 
                   colormap='viridis', figsize=(12, 10), density=1):
    """
    Plot a vector field of ocean flow.
    
    Args:
        u (numpy.ndarray): Horizontal flow component
        v (numpy.ndarray): Vertical flow component
        mask (numpy.ndarray, optional): Land/water mask
        scale (float): Scale factor for arrow size
        title (str): Plot title
        colormap (str): Matplotlib colormap name
        figsize (tuple): Figure size (width, height)
        density (int): Density of arrows (1 = all points, 2 = every 2nd point, etc.)
        
    Returns:
        tuple: (fig, ax) matplotlib figure and axis objects
    """
    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate speed for color mapping
    speed = calculate_speed(u, v)
    
    # Create a grid of positions, accounting for density
    y_indices, x_indices = np.mgrid[:u.shape[0]:density, :u.shape[1]:density]
    
    # Sample the flow field at specified density
    u_sampled = u[::density, ::density]
    v_sampled = v[::density, ::density]
    speed_sampled = speed[::density, ::density]
    
    # Plot the flow field
    quiver = ax.quiver(x_indices, y_indices, u_sampled, v_sampled, 
                     speed_sampled, cmap=colormap, scale=scale, 
                     pivot='mid', alpha=0.8, norm=Normalize(vmin=0, vmax=np.percentile(speed, 95)))
    
    # Add colorbar
    cbar = fig.colorbar(quiver, ax=ax)
    cbar.set_label('Flow Speed (cm/s)')
    
    # Plot land/water mask if provided
    if mask is not None:
        # Use masked array to show land areas
        masked_speed = np.ma.array(speed, mask=mask)
        ax.imshow(masked_speed, cmap='terrain', alpha=0.3, 
                 extent=[0, speed.shape[1], 0, speed.shape[0]])
    
    # Set title and labels
    ax.set_title(title)
    ax.set_xlabel('Grid Position (3km per unit)')
    ax.set_ylabel('Grid Position (3km per unit)')
    
    # Invert y-axis to match problem description (0,0 at bottom-left)
    ax.invert_yaxis()
    
    return fig, ax

def ensure_output_dir(output_dir):
    """
    Ensure the output directory exists.
    
    Args:
        output_dir (str): Directory path
    """
    os.makedirs(output_dir, exist_ok=True)