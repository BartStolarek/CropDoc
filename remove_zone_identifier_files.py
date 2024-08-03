import os

def remove_zone_identifier_files(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(':Zone.Identifier'):
                os.remove(os.path.join(root, file))
                count += 1
                
    print(f'Removed {count} Zone.Identifier files from {directory}')
                



def count_jpg_files_in_directory(directory):
    count = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.lower().endswith('.jpg'):
                count += 1
                
    return count

print(count_jpg_files_in_directory('data/CCMT Dataset-Augmented'))