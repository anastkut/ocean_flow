import os

def create_directory_structure():
    """Create the project directory structure"""
    
    # Define directories to create
    directories = [
        "data",
        "output",
        "part1",
        "part2",
        "optional"
    ]
    
    # Create each directory if it doesn't exist
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")
    
    # Create empty __init__.py files for importing modules
    init_files = [
        "part1/__init__.py",
        "part2/__init__.py",
        "optional/__init__.py"
    ]
    
    for init_file in init_files:
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                pass  # Create empty file
            print(f"Created file: {init_file}")
        else:
            print(f"File already exists: {init_file}")
    
    print("\nDirectory structure setup complete!")
    print("Please extract OceanFlow.zip files into the 'data' directory.")

if __name__ == "__main__":
    create_directory_structure()