import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset
from torchvision.models import ResNet50_Weights
import torch
import matplotlib.pyplot as plt
import numpy as np
from pprint import pprint
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

# Constants
DATA_DIR = 'data/CCMT Dataset-Augmented'
NUM_CROPS = 4
NUM_STATES = 22
NUM_EPOCHS = 2  # Number of passes through entire training dataset
CV_FOLDS = 2  # Number of cross-validation folds
BATCH_SIZE = 32 # Within each epoch data is split into batches
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2
CROSS_VALIDATE = True

CROPS = ['Cashew', 'Cassava', 'Maize', 'Tomato']


"""
DATA PREPERATION

- Loading the dataset
- Splitting the data into training, validation and test sets.

"""

# Initialise the data directory
train_datasets = []
test_datasets = []
all_classes = {}

# Initialise data transforms
data_standard_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor()
    ]),
    'test': transforms.Compose([
        transforms.ToTensor()
    ])
}

# For each crop, load the training and test datasets
for crop in CROPS:
    crop_dir = os.path.join(DATA_DIR, crop)
    
    # Get the crops training set
    train_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'train_set'), transform=data_standard_transforms['train'])
    
    # Get the crops test set
    test_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'test_set'), transform=data_standard_transforms['test'])
    
    # Append the training and test datasets to the lists
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    
    # Check that all train dataset classes are the same as the test dataset classes
    assert train_dataset.classes == test_dataset.classes, f"Train and test classes do not match for {crop}"
    
    crop_classes = train_dataset.classes
    all_classes[crop] = crop_classes

# Concatenate the training and test datasets
train_dataset = ConcatDataset(train_datasets)
test_dataset = ConcatDataset(test_datasets)

print("All classes:")
individual_classes = sum([len(classes) for crop, classes in all_classes.items()])
if individual_classes != NUM_STATES:
    print(f"Number of classes does not match expected number of classes (NUM_CLASSES): {individual_classes} != {NUM_STATES}")
    exit()
print(f"Number of classes: {individual_classes}")
pprint(all_classes)
print('\n')

unique_classes = {}
for crop, classes in all_classes.items():
    for class_ in classes:
        if class_ not in unique_classes:
            unique_classes[class_] = 0
        unique_classes[class_] += 1

print("Unique classes:")
pprint(unique_classes)
print('\n')

    
"""
EDA

- Counting and plotting the subclasses of each crop.

"""

os.makedirs('data/output/original', exist_ok=True)

# For each crop in train dataset, count and plot in histogram
for dataset in train_datasets:
    class_names = dataset.classes
    class_counts = np.bincount(dataset.targets)
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar(class_names, class_counts)
    plt.xlabel('Subclass')
    plt.ylabel('Count')
    plt.title(f'Subclass Counts for {dataset.root.split("/")[2]}')
    plt.xticks(rotation=45)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}',
                 ha='center', va='bottom')
        
    plt.tight_layout()
    
    # Save the plot to output directory
    plt.savefig(f'data/output/original/{dataset.root.split("/")[2]}_eda.png')
    plt.close()


"""
Feature Engineering

- The total number of images per crop is imbalanced, which needs to be addressed, and then within each crop, the subclasses are also imbalanced.
- Address class imbalance by following techniques:
    - Class weights
    - Data resampling
    - Data transformation and augmentation
    - Confirming the class imbalance is addressed

"""

total_samples = sum([sum(np.bincount(dataset.targets)) for dataset in train_datasets])

class_weight_dict = {}
for dataset in train_datasets:
    class_counts = np.bincount(dataset.targets)
    crop_name = dataset.root.split("/")[2]
    
    subclass_weight_dict = {}
    
    for class_ in dataset.classes:
        class_idx = dataset.class_to_idx[class_]
        class_weight = round(total_samples / class_counts[class_idx], 3)
        subclass_weight_dict[class_] = class_weight
    
    class_weight_dict[crop_name] = subclass_weight_dict

print('Class Weights Dict')
pprint(class_weight_dict)

# Convert class weights to PyTorch tensor
class_weights = {}
for crop, weights in class_weight_dict.items():
    class_weights[crop] = {subclass: torch.tensor(weight, dtype=torch.float32) for subclass, weight in weights.items()}
    
print('Class Weights Tensor')
pprint(class_weights)


# Define additional augmentation and transformation
data_augmentation_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
}

train_datasets = []
test_datasets = []

# For each crop, load the training and test datasets with augmented transforms
for crop in CROPS:
    crop_dir = os.path.join(DATA_DIR, crop)
    
    # Get the crops training set with augmented transforms
    train_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'train_set'), transform=data_augmentation_transforms['train'])
    
    # Get the crops test set with standard test transforms
    test_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'test_set'), transform=data_augmentation_transforms['test'])
    
    # Append the training and test datasets to the lists
    train_datasets.append(train_dataset)
    test_datasets.append(test_dataset)
    
    
"""
Training

- Split train dataset in to train and validation sets
- Define the model and set device
- Setup cross validation if opted in
- Train the model
- Track the train & validation loss and accuracy and graph it
- Check the confusion matrix

"""
# Set device (use GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load ResNet50V2 model
model = models.resnet50(weights=ResNet50_Weights.DEFAULT)

# Replace the classifier (fully connected layer)
num_ftrs = model.fc.in_features

# Separate heads for crop and disease
model.fc_crop = nn.Linear(num_ftrs, NUM_CROPS)

# Move the model to the correct device
model = model.to(device)

# Map the weights
class_to_idx = {}
idx = 0
for crop in CROPS:
    for class_ in all_classes[crop]:
        if class_ not in class_to_idx:
            class_to_idx[class_] = idx
            idx += 1

# Create a weight tensor
weights = torch.ones(NUM_STATES)
for crop in CROPS:
    for class_, weight in class_weights[crop].items():
        weights[class_to_idx[class_]] = weight
weights = weights.to(device)

# Define loss function and optimiser
criterion = nn.CrossEntropyLoss(weight=weights.to(device))
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Define cross-validation iterator
skf = StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=VAL_SPLIT, random_state=42)

# Determine the number of splits
n_splits = skf.get_n_splits(train_dataset.samples, train_dataset.targets)

train_losses = []
train_accuracies = []
# Training Loop
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Initialize tqdm for epoch progress
    epoch_progress = tqdm(total=n_splits, desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(train_dataset.samples, train_dataset.targets)):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

        # Initialize tqdm for fold progress
        fold_progress = tqdm(total=len(train_loader), desc=f'Fold {fold_idx + 1}/{CV_FOLDS}', leave=False)

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimiser.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimiser.step()

            # Compute statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Update tqdm progress description at batch level
            fold_progress.set_postfix(loss=running_loss / (batch_idx + 1), accuracy=100. * correct / total)
            fold_progress.update(1)

        # Close fold progress bar
        fold_progress.close()

        # Update tqdm progress at fold level
        epoch_progress.update(1)

    # Close epoch progress bar
    epoch_progress.close()

    # Calculate epoch-level metrics
    epoch_loss = running_loss / len(train_loader)
    epoch_accuracy = 100. * correct / total

    # Log metrics
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_accuracy)

"""
Evaluate

- Observe the model performance on the test set
- Check the confusion matrix
- confidence intervals of accuracy and loss

"""
