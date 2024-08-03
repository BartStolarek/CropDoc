import os
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from pprint import pprint
import pandas as pd
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

# Define Constants
DATA_DIR = 'data/CCMT Dataset-Augmented'
OUTPUT_DIR = 'data/output/subclass_counts'
eda_transforms = transforms.Compose([
    transforms.ToTensor(),
])

# Load full dataset
def get_dataset(data_dir, transforms):
    dataset = ImageFolder(data_dir, transform=transforms)
    return dataset

# Function to count and plot subclasses dynamically
def count_and_plot_subclasses(dataset, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    totals = {}
    
    # Count subclass occurrences for each crop
    for crop in dataset.classes:
        subclass_counts = {}
        for img_path, class_idx in dataset.imgs:
            if crop in img_path:
                subclass = img_path.split('/')[-2]  # Assuming Unix-like path separator
                if subclass not in subclass_counts:
                    subclass_counts[subclass] = 0
                subclass_counts[subclass] += 1
  
        
        totals[crop] = subclass_counts
        
        # Plot bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(subclass_counts.keys(), subclass_counts.values())
        plt.title(f'Subclass Counts for {crop}')
        plt.xlabel('Subclass')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        
        # Add count numbers on top of bars
        for idx, count in enumerate(subclass_counts.values()):
            plt.text(idx, count + 0.1, str(count), ha='center', va='bottom')
        
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{crop}_subclass_counts.png'))
        plt.close()
    

    
    totals_pd = pd.DataFrame(totals)
    
    print(totals_pd)




###########################
#
# Main Script
#
###########################


dataset = get_dataset(DATA_DIR, eda_transforms)

# Perform subclass counting and plotting
count_and_plot_subclasses(dataset, OUTPUT_DIR+ '/original')
print("Subclass counting and plotting complete.")

