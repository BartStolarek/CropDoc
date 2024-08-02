import os
import shutil

def flatten_and_rename_images(source_dir, destination_dir):
    # Create destination directory if it doesn't exist
    os.makedirs(destination_dir, exist_ok=True)

    # Walk through the source directory
    for root, _, files in os.walk(source_dir):
        for file in files:
            
            if file.endswith(':Zone.Identifier'):
                continue
            
            # Construct the original and new paths
            original_path = os.path.join(root, file)
            new_filename = root.replace(os.path.sep, '_') + '_' + file
            new_path = os.path.join(destination_dir, new_filename)

            # Copy the file to the new directory and rename it
            shutil.copy(original_path, new_path)

            # Optional: Print progress
            print(f"Copied {original_path} to {new_path}")

    print("Flattening and renaming complete.")

# Example usage:
source_directory = 'data/Raw Data/CCMT Dataset'
destination_directory = '../data/flattened/rawdata'

root_of_project = os.path.dirname(os.path.abspath(__file__))
destination_directory = os.path.join(root_of_project, 'data', 'flattened', 'rawdata')


flatten_and_rename_images(source_directory, destination_directory)
