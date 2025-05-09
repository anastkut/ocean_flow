import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
import sys

# Import common utilities
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common_utils import ensure_output_dir

def rbf_kernel(x1, x2, sigma_squared=1.0, length_scale=1.0):
    """
    Compute the RBF/squared exponential kernel between points x1 and x2.
    
    Args:
        x1 (numpy.ndarray): First point or array of points (n1, d)
        x2 (numpy.ndarray): Second point or array of points (n2, d)
        sigma_squared (float): Signal variance parameter
        length_scale (float): Length scale parameter
        
    Returns:
        numpy.ndarray: Kernel matrix (n1, n2)
    """
    # Ensure inputs are 2D arrays
    if len(x1.shape) == 1:
        x1 = x1.reshape(-1, 1)
    if len(x2.shape) == 1:
        x2 = x2.reshape(-1, 1)
    
    # Compute squared Euclidean distances
    sq_dist = np.sum(x1**2, 1).reshape(-1, 1) + np.sum(x2**2, 1) - 2 * np.dot(x1, x2.T)
    
    # Apply RBF kernel function
    return sigma_squared * np.exp(-sq_dist / (2 * length_scale**2))

def demo_kernel_parameters(output_dir='output'):
    """
    Demonstrate the effect of changing kernel parameters in Gaussian processes.
    
    Args:
        output_dir (str): Directory to save output figures
    """
    # Ensure output directory exists
    ensure_output_dir(output_dir)
    
    print("Demonstrating effects of RBF kernel parameters...")
    
    # Set up the domain
    x = np.linspace(-3, 3, 100)
    
    # Create figure for kernel visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test different sigma_squared values
    sigma_values = [0.5, 1.0, 2.0]
    length_scale = 1.0
    
    for i, sigma_squared in enumerate(sigma_values):
        # Select reference point at x=0
        x0 = np.array([0])
        
        # Compute kernel values between x0 and all points in x
        k = rbf_kernel(x.reshape(-1, 1), x0.reshape(-1, 1), sigma_squared, length_scale)
        
        # Plot
        axes[0, i].plot(x, k, 'b-')
        axes[0, i].set_title(f'σ² = {sigma_squared}, ℓ = {length_scale}')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('k(x, 0)')
        axes[0, i].set_ylim(0, 2.5)
        axes[0, i].grid(True)
    
    # Test different length_scale values
    length_values = [0.3, 1.0, 3.0]
    sigma_squared = 1.0
    
    for i, length_scale in enumerate(length_values):
        # Compute kernel values
        k = rbf_kernel(x.reshape(-1, 1), x0.reshape(-1, 1), sigma_squared, length_scale)
        
        # Plot
        axes[1, i].plot(x, k, 'r-')
        axes[1, i].set_title(f'σ² = {sigma_squared}, ℓ = {length_scale}')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('k(x, 0)')
        axes[1, i].set_ylim(0, 1.2)
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rbf_kernel_parameters.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create figure for GP function samples
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Fix random seed for reproducibility
    np.random.seed(42)
    
    # Sample from GP with different sigma_squared values
    for i, sigma_squared in enumerate(sigma_values):
        # Compute full kernel matrix
        K = rbf_kernel(x.reshape(-1, 1), x.reshape(-1, 1), sigma_squared, length_scale=1.0)
        
        # Add small jitter to diagonal for numerical stability
        K += 1e-10 * np.eye(len(x))
        
        # Generate 5 sample functions
        for _ in range(5):
            sample = np.random.multivariate_normal(np.zeros(len(x)), K)
            axes[0, i].plot(x, sample, alpha=0.7)
        
        axes[0, i].set_title(f'σ² = {sigma_squared}, ℓ = 1.0')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('f(x)')
        axes[0, i].grid(True)
    
    # Sample from GP with different length_scale values
    for i, length_scale in enumerate(length_values):
        # Compute kernel matrix
        K = rbf_kernel(x.reshape(-1, 1), x.reshape(-1, 1), sigma_squared=1.0, length_scale=length_scale)
        
        # Add small jitter to diagonal for numerical stability
        K += 1e-10 * np.eye(len(x))
        
        # Generate 5 sample functions
        for _ in range(5):
            sample = np.random.multivariate_normal(np.zeros(len(x)), K)
            axes[1, i].plot(x, sample, alpha=0.7)
        
        axes[1, i].set_title(f'σ² = 1.0, ℓ = {length_scale}')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('f(x)')
        axes[1, i].grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gp_samples.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Figures saved to output directory")
    
    # Create a 2D visualization of the kernel
    print("Creating 2D visualization of the kernel...")
    
    # Create a 2D grid of points
    n = 20
    x1 = np.linspace(-3, 3, n)
    x2 = np.linspace(-3, 3, n)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.column_stack((X1.ravel(), X2.ravel()))
    
    # Create figure for 2D kernel visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test different sigma_squared values in 2D
    for i, sigma_squared in enumerate(sigma_values):
        # Select reference point at origin
        x0 = np.array([[0, 0]])
        
        # Compute kernel values
        k = rbf_kernel(X_grid, x0, sigma_squared, length_scale=1.0)
        
        # Reshape for plotting
        k_grid = k.reshape(n, n)
        
        # Plot as heatmap
        im = axes[0, i].imshow(k_grid, extent=[-3, 3, -3, 3], origin='lower',
                             interpolation='bilinear', cmap='viridis',
                             norm=Normalize(vmin=0, vmax=sigma_squared))
        
        # Add contour lines
        contour_levels = np.linspace(0, sigma_squared, 5)
        contour = axes[0, i].contour(X1, X2, k_grid, levels=contour_levels, colors='white', alpha=0.5)
        
        # Add colorbar
        fig.colorbar(im, ax=axes[0, i])
        
        axes[0, i].set_title(f'2D RBF kernel, σ² = {sigma_squared}, ℓ = 1.0')
        axes[0, i].set_xlabel('x₁')
        axes[0, i].set_ylabel('x₂')
    
    # Test different length_scale values in 2D
    for i, length_scale in enumerate(length_values):
        # Compute kernel values
        k = rbf_kernel(X_grid, x0, sigma_squared=1.0, length_scale=length_scale)
        
        # Reshape for plotting
        k_grid = k.reshape(n, n)
        
        # Plot as heatmap
        im = axes[1, i].imshow(k_grid, extent=[-3, 3, -3, 3], origin='lower',
                             interpolation='bilinear', cmap='viridis',
                             norm=Normalize(vmin=0, vmax=1))
        
        # Add contour lines
        contour_levels = np.linspace(0, 1, 5)
        contour = axes[1, i].contour(X1, X2, k_grid, levels=contour_levels, colors='white', alpha=0.5)
        
        # Add colorbar
        fig.colorbar(im, ax=axes[1, i])
        
        axes[1, i].set_title(f'2D RBF kernel, σ² = 1.0, ℓ = {length_scale}')
        axes[1, i].set_xlabel('x₁')
        axes[1, i].set_ylabel('x₂')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rbf_kernel_2d.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Demonstrate GP regression with different parameters
    print("Demonstrating GP regression...")
    
    # Generate some noisy data
    np.random.seed(123)
    n_train = 10
    
    x_train = np.sort(6 * np.random.rand(n_train) - 3)
    y_train = np.sin(x_train) + 0.1 * np.random.randn(n_train)
    
    # Test points for prediction
    x_test = np.linspace(-3, 3, 100)
    
    # Create a function for GP regression prediction
    def gp_predict(x_train, y_train, x_test, sigma_squared, length_scale, noise_var=0.01):
        """
        Perform GP regression prediction.
        
        Args:
            x_train (numpy.ndarray): Training inputs
            y_train (numpy.ndarray): Training outputs
            x_test (numpy.ndarray): Test inputs
            sigma_squared (float): Signal variance
            length_scale (float): Length scale
            noise_var (float): Noise variance
            
        Returns:
            tuple: (mean, variance) for predictions
        """
        # Compute kernel matrices
        K = rbf_kernel(x_train.reshape(-1, 1), x_train.reshape(-1, 1), 
                     sigma_squared, length_scale)
        K_s = rbf_kernel(x_train.reshape(-1, 1), x_test.reshape(-1, 1), 
                       sigma_squared, length_scale)
        K_ss = rbf_kernel(x_test.reshape(-1, 1), x_test.reshape(-1, 1), 
                        sigma_squared, length_scale)
        
        # Add noise to training kernel
        K += noise_var * np.eye(len(x_train))
        
        # Compute predictive mean and variance
        K_inv = np.linalg.inv(K)
        mean = K_s.T @ K_inv @ y_train
        var = K_ss - K_s.T @ K_inv @ K_s
        
        # Extract diagonal of variance matrix for individual variances
        std = np.sqrt(np.diag(var))
        
        return mean, std
    
    # Create figure for GP regression
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Test different sigma_squared values for regression
    for i, sigma_squared in enumerate(sigma_values):
        # Perform GP regression
        mean, std = gp_predict(x_train, y_train, x_test, sigma_squared, length_scale=1.0)
        
        # Plot training data
        axes[0, i].scatter(x_train, y_train, color='black', label='Training data')
        
        # Plot mean prediction
        axes[0, i].plot(x_test, mean, 'b-', label='Mean prediction')
        
        # Plot confidence intervals (95%)
        axes[0, i].fill_between(x_test, mean - 1.96 * std, mean + 1.96 * std, 
                              alpha=0.2, color='blue', label='95% confidence')
        
        # Plot true function
        axes[0, i].plot(x_test, np.sin(x_test), 'r--', label='True function')
        
        axes[0, i].set_title(f'GP Regression, σ² = {sigma_squared}, ℓ = 1.0')
        axes[0, i].set_xlabel('x')
        axes[0, i].set_ylabel('f(x)')
        axes[0, i].grid(True)
        axes[0, i].legend()
    
    # Test different length_scale values for regression
    for i, length_scale in enumerate(length_values):
        # Perform GP regression
        mean, std = gp_predict(x_train, y_train, x_test, sigma_squared=1.0, length_scale=length_scale)
        
        # Plot training data
        axes[1, i].scatter(x_train, y_train, color='black', label='Training data')
        
        # Plot mean prediction
        axes[1, i].plot(x_test, mean, 'g-', label='Mean prediction')
        
        # Plot confidence intervals (95%)
        axes[1, i].fill_between(x_test, mean - 1.96 * std, mean + 1.96 * std, 
                              alpha=0.2, color='green', label='95% confidence')
        
        # Plot true function
        axes[1, i].plot(x_test, np.sin(x_test), 'r--', label='True function')
        
        axes[1, i].set_title(f'GP Regression, σ² = 1.0, ℓ = {length_scale}')
        axes[1, i].set_xlabel('x')
        axes[1, i].set_ylabel('f(x)')
        axes[1, i].grid(True)
        axes[1, i].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'gp_regression.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  Effect of signal variance (σ²):")
    print("    - Controls the overall vertical scale of functions")
    print("    - Higher values allow for larger deviations from the mean")
    print("    - Affects confidence interval width")
    
    print("\n  Effect of length scale (ℓ):")
    print("    - Controls how quickly correlation drops off with distance")
    print("    - Smaller values lead to more rapid changes (higher frequency functions)")
    print("    - Larger values lead to smoother functions")
    print("    - Affects how far information is propagated from data points")
    
    print("\nGP regression with large datasets is computationally expensive because:")
    print("  - Requires inverting an nxn kernel matrix: O(n³) time complexity")
    print("  - Storing the kernel matrix requires O(n²) memory")
    print("  - For large n (e.g., n > 10,000), direct computation becomes infeasible")
    
    # Print description of efficient algorithm
    print("\nEfficient GP computation algorithms:")
    efficient_algorithm_description()
    
    print("\nAnalysis complete. All figures saved to {}/".format(output_dir))

def efficient_algorithm_description():
    """
    Describe algorithms for making GP computation more efficient.
    """
    print("""
Efficient GP Algorithms (Pseudocode):

1. Sparse Gaussian Processes with Inducing Points:

   function SparseGP(X_train, y_train, X_test, kernel_func, m):
       # Select m inducing points (m << n)
       Z = select_inducing_points(X_train, m)  # Could use k-means or random selection
       
       # Compute kernel matrices
       K_mm = kernel_func(Z, Z)
       K_nm = kernel_func(X_train, Z)
       K_mn = K_nm.T
       
       # Compute Qff = K_nm K_mm^(-1) K_mn (low-rank approximation of K_nn)
       # Instead of directly computing K_nn which is n×n
       
       # Add small jitter to diagonal for numerical stability
       K_mm += 1e-6 * I
       
       # Cholesky decomposition for stable inversion
       L = cholesky(K_mm)
       L_inv = inv(L)
       K_mm_inv = L_inv.T @ L_inv
       
       # Compute posterior over inducing points
       Sigma = inv(K_mm + K_mn @ K_nm)
       mu = Sigma @ K_mn @ y_train
       
       # Predictive distribution at test points
       K_tm = kernel_func(X_test, Z)
       mean_f = K_tm @ mu
       var_f = kernel_func(X_test, X_test) - K_tm @ (K_mm_inv - Sigma) @ K_tm.T
       
       return mean_f, var_f

   Time complexity: O(nm²) instead of O(n³)
   Space complexity: O(nm) instead of O(n²)

2. Nyström Method:

   function NystromGP(X_train, y_train, X_test, kernel_func, m):
       # Select m landmark points (m << n)
       landmarks = sample_landmarks(X_train, m)
       
       # Compute kernel matrices
       K_mm = kernel_func(landmarks, landmarks)
       K_nm = kernel_func(X_train, landmarks)
       
       # Eigendecomposition of K_mm
       eigvals, eigvecs = eig(K_mm)
       
       # Use only positive eigenvalues for numerical stability
       pos_idx = eigvals > 1e-10
       eigvals_pos = eigvals[pos_idx]
       eigvecs_pos = eigvecs[:, pos_idx]
       
       # Construct approximation to K_nn
       K_approx = K_nm @ eigvecs_pos @ diag(1/sqrt(eigvals_pos)) @ 
                 diag(1/sqrt(eigvals_pos)) @ eigvecs_pos.T @ K_nm.T
       
       # Proceed with standard GP using K_approx instead of K_nn
       # ...
       
       return mean_f, var_f

3. Structured Kernel Interpolation (SKI):

   function SKI(X_train, y_train, X_test, kernel_func, grid_size):
       # Create a regular grid in the input space
       grid = create_regular_grid(domain_bounds, grid_size)
       
       # Compute kernel between grid points
       K_gg = kernel_func(grid, grid)
       
       # Precompute sparse interpolation matrices
       W_ng = compute_interpolation_weights(X_train, grid)
       W_tg = compute_interpolation_weights(X_test, grid)
       
       # Approximate kernel matrices using interpolation
       K_nn ≈ W_ng @ K_gg @ W_ng.T
       K_nt ≈ W_ng @ K_gg @ W_tg.T
       
       # Use Kronecker and Toeplitz structure for efficient operations
       # For specific kernels on regular grids, K_gg can have special structure
       # allowing for O(n) matrix-vector products instead of O(n²)
       
       # Proceed with standard GP using approximated kernels
       # ...
       
       return mean_f, var_f

4. Local Gaussian Processes:

   function LocalGP(X_train, y_train, x_test, kernel_func, k_neighbors):
       predictions = []
       
       for each x in x_test:
           # Find k nearest neighbors to current test point
           distances = compute_distances(X_train, x)
           indices = argsort(distances)[:k_neighbors]
           
           X_local = X_train[indices]
           y_local = y_train[indices]
           
           # Standard GP on local neighborhood
           K = kernel_func(X_local, X_local)
           K_s = kernel_func(X_local, x.reshape(1,-1))
           K_ss = kernel_func(x.reshape(1,-1), x.reshape(1,-1))
           
           mean = K_s.T @ inv(K) @ y_local
           var = K_ss - K_s.T @ inv(K) @ K_s
           
           predictions.append((mean, var))
       
       return predictions

   Time complexity: O(k³) per test point, where k << n

These methods can be combined with the Sherman-Morrison-Woodbury formula:
(A + UCV)⁻¹ = A⁻¹ - A⁻¹U(C⁻¹ + VA⁻¹U)⁻¹VA⁻¹

which can reduce the complexity of matrix inversion when working with low-rank updates.
""")

if __name__ == "__main__":
    demo_kernel_parameters()