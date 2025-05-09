import os
import sys
import argparse
import time

def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def main():
    parser = argparse.ArgumentParser(description='Ocean Flow Analysis for Problem Set 5')
    parser.add_argument('--data_dir', default='data', help='Directory containing data files')
    parser.add_argument('--output_dir', default='output', help='Directory to save output files')
    parser.add_argument('--part', choices=['1', '2', 'all'], default='all', help='Which part to run (1, 2, or all)')
    parser.add_argument('--task', help='Specific task to run (e.g., 1a, 1b, 1c, 2a, 2b, 43)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    ensure_dir(args.output_dir)
    
    # Define tasks for each part
    part1_tasks = ['1a', '1b', '1c']
    part2_tasks = ['2a', '2b']
    optional_tasks = ['43']
    
    # Determine which tasks to run
    tasks_to_run = []
    
    if args.task:
        tasks_to_run = [args.task]
    elif args.part == '1':
        tasks_to_run = part1_tasks
    elif args.part == '2':
        tasks_to_run = part2_tasks
    else:  # 'all'
        tasks_to_run = part1_tasks + part2_tasks
    
    # Add optional tasks if specified
    if args.task in optional_tasks:
        tasks_to_run = [args.task]
    
    print(f"Ocean Flow Analysis - Running tasks: {', '.join(tasks_to_run)}\n")
    
    # Run selected tasks
    for task in tasks_to_run:
        print(f"="*80)
        print(f"Running task {task}")
        print(f"="*80)
        
        start_time = time.time()
        
        try:
            if task == '1a':
                from part1.task1a_avg_flow import visualize_average_flow
                visualize_average_flow(args.data_dir, args.output_dir)
            
            elif task == '1b':
                from part1.task1b_avg_speed import visualize_average_speed
                visualize_average_speed(args.data_dir, args.output_dir)
            
            elif task == '1c':
                from part1.task1c_correlation import analyze_spatial_correlation
                analyze_spatial_correlation(args.data_dir, args.output_dir)
            
            elif task == '2a':
                from part2.task2a_tracking import visualize_trajectories
                visualize_trajectories(args.data_dir, args.output_dir)
            
            elif task == '2b':
                from part2.task2b_simulation import simulate_plane_crash_debris
                simulate_plane_crash_debris(
                    mean_pos=(100, 350), 
                    variance_values=[100, 500, 1000],
                    num_particles=100,
                    data_dir=args.data_dir,
                    output_dir=args.output_dir
                )
            
            elif task == '43':
                # Optional Gaussian process analysis
                from optional.task43_gaussian_process import demo_kernel_parameters
                demo_kernel_parameters(args.output_dir)
            
            else:
                print(f"Unknown task: {task}")
                continue
            
            elapsed_time = time.time() - start_time
            print(f"\nTask {task} completed in {elapsed_time:.2f} seconds")
            
        except Exception as e:
            print(f"Error running task {task}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\nAll tasks completed!")

if __name__ == "__main__":
    main()