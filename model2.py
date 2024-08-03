# Import necessary libraries
import os  # For interacting with the operating system
import matplotlib.pyplot as plt  # Plotting library
import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network module in PyTorch
import torch.optim as optim  # Optimisation algorithms in PyTorch
from torchvision import transforms, models  # Computer vision libraries
from torch.utils.data import Dataset, DataLoader  # PyTorch data utilities
from sklearn.model_selection import StratifiedShuffleSplit  # Data splitting utility
from tqdm import tqdm  # Progress bar utility

# Constants
DATA_DIR = 'CropDoc/data/datasets/CCMT Dataset-Augmented'  # Directory containing dataset
NUM_EPOCHS = 1  # Number of training epochs
CV_FOLDS = 2  # Number of cross-validation folds
BATCH_SIZE = 32  # Batch size for training
LEARNING_RATE = 0.001  # Learning rate for optimizer
VAL_SPLIT = 0.2  # Validation split ratio

# Labels and classes information
labels_classes = {
    'crop': ['Cashew', 'Cassava', 'Maize', 'Tomato'],  # Types of crops
    'state': {
        'Cashew': ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust'],  # States for Cashew crop
        'Cassava': ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic'],  # States for Cassava crop
        'Maize': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus'],  # States for Maize crop
        'Tomato': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt']  # States for Tomato crop
    }
}


# Custom Dataset class for loading images and labels
class CropDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        self.crop_to_idx = {crop: idx for idx, crop in enumerate(labels_classes['crop'])}
        self.state_to_idx = {state: idx for idx, state in enumerate(sum(labels_classes['state'].values(), []))}
        
        # Iterate over crop and state directories to collect image paths and labels
        for crop in os.listdir(root):
            crop_path = os.path.join(root, crop)
            split_path = os.path.join(crop_path, f'{split}_set')
            for state in os.listdir(split_path):
                state_path = os.path.join(split_path, state)
                for img in os.listdir(state_path):
                    img_path = os.path.join(state_path, img)
                    self.samples.append((img_path, self.crop_to_idx[crop], self.state_to_idx[state]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, crop_label, state_label = self.samples[idx]
        img = plt.imread(img_path)  # Read image using matplotlib
        if self.transform:
            img = self.transform(img)  # Apply transformations if specified
        return img, crop_label, state_label


# Data transforms for augmentation and normalisation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.RandomResizedCrop(224),  # Random crop and resize to 224x224
        transforms.RandomHorizontalFlip(),  # Random horizontal flip
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomRotation(20),  # Random rotation within 20 degrees
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalise image tensor
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),  # Convert numpy array to PIL image
        transforms.Resize(256),  # Resize image to 256x256
        transforms.CenterCrop(224),  # Center crop to 224x224
        transforms.ToTensor(),  # Convert PIL image to tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalise image tensor
    ])
}

# Load datasets for training and testing
train_dataset = CropDataset(DATA_DIR, split='train', transform=data_transforms['train'])
test_dataset = CropDataset(DATA_DIR, split='test', transform=data_transforms['test'])

# Define the MultiHeadResNet model class
class MultiHeadResNet(nn.Module):
    def __init__(self, num_crops, num_states):
        super(MultiHeadResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Replace fully connected layer with an identity layer
        self.fc_crop = nn.Linear(num_ftrs, num_crops)  # Linear layer for crop classification
        self.fc_state = nn.Linear(num_ftrs, num_states)  # Linear layer for state classification

    def forward(self, x):
        features = self.resnet(x)  # Extract features using ResNet
        crop_out = self.fc_crop(features)  # Output for crop classification
        state_out = self.fc_state(features)  # Output for state classification
        return crop_out, state_out


# Initialise the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check for GPU availability
model = MultiHeadResNet(len(labels_classes['crop']), len(sum(labels_classes['state'].values(), [])))  # Create model instance
model = model.to(device)  # Move model to GPU if available

# Loss function and optimiser
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE)  # Adam optimiser for model parameters


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, epoch, fold):
    model.train()
    running_loss = 0.0
    correct_crop = 0
    correct_state = 0
    total = 0

    # Iterate over training batches
    batch_progress = tqdm(train_loader, desc=f'Epoch {epoch+1} - Fold {fold+1} - Training', leave=False)
    for batch_idx, (inputs, crop_labels, state_labels) in enumerate(batch_progress):
        inputs, crop_labels, state_labels = inputs.to(device), crop_labels.to(device), state_labels.to(device)

        optimizer.zero_grad()
        crop_outputs, state_outputs = model(inputs)
        loss_crop = criterion(crop_outputs, crop_labels)
        loss_state = criterion(state_outputs, state_labels)
        loss = loss_crop + loss_state
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted_crop = torch.max(crop_outputs, 1)
        _, predicted_state = torch.max(state_outputs, 1)
        total += crop_labels.size(0)
        correct_crop += (predicted_crop == crop_labels).sum().item()
        correct_state += (predicted_state == state_labels).sum().item()

        # Update progress bar
        batch_progress.set_postfix({
            'loss': f'{running_loss/(batch_idx+1):.4f}',
            'crop_acc': f'{100.*correct_crop/total:.2f}%',
            'state_acc': f'{100.*correct_state/total:.2f}%'
        })

    train_loss = running_loss / len(train_loader)
    train_acc_crop = 100. * correct_crop / total
    train_acc_state = 100. * correct_state / total

    # Validation
    model.eval()
    val_loss = 0.0
    correct_crop = 0
    correct_state = 0
    total = 0

    with torch.no_grad():
        batch_progress = tqdm(val_loader, desc=f'Epoch {epoch+1} - Fold {fold+1} - Validation', leave=False)
        for inputs, crop_labels, state_labels in batch_progress:
            inputs, crop_labels, state_labels = inputs.to(device), crop_labels.to(device), state_labels.to(device)
            crop_outputs, state_outputs = model(inputs)
            loss_crop = criterion(crop_outputs, crop_labels)
            loss_state = criterion(state_outputs, state_labels)
            loss = loss_crop + loss_state
            val_loss += loss.item()

            _, predicted_crop = torch.max(crop_outputs, 1)
            _, predicted_state = torch.max(state_outputs, 1)
            total += crop_labels.size(0)
            correct_crop += (predicted_crop == crop_labels).sum().item()
            correct_state += (predicted_state == state_labels).sum().item()

            # Update progress bar
            batch_progress.set_postfix({
                'loss': f'{val_loss/(batch_idx+1):.4f}',
                'crop_acc': f'{100.*correct_crop/total:.2f}%',
                'state_acc': f'{100.*correct_state/total:.2f}%'
            })

    val_loss /= len(val_loader)
    val_acc_crop = 100. * correct_crop / total
    val_acc_state = 100. * correct_state / total

    return train_loss, train_acc_crop, train_acc_state, val_loss, val_acc_crop, val_acc_state


# Perform cross-validation training and validation
sss = StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=VAL_SPLIT, random_state=0)

for epoch in range(NUM_EPOCHS):
    epoch_progress = tqdm(total=CV_FOLDS, desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')
    
    for fold, (train_idx, val_idx) in enumerate(sss.split(train_dataset.samples, [label[1] for label in train_dataset.samples])):
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
        val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)
        
        # Initialize a new model for each fold
        model = MultiHeadResNet(len(labels_classes['crop']), len(sum(labels_classes['state'].values(), [])))
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
        
        # Train the model
        train_loss, train_acc_crop, train_acc_state, val_loss, val_acc_crop, val_acc_state = train_model(
            model, train_loader, val_loader, criterion, optimizer, epoch, fold
        )
        
        # Print fold results
        print(f'Epoch {epoch+1}/{NUM_EPOCHS} - Fold {fold+1}/{CV_FOLDS}:')
        print(f'Train Loss: {train_loss:.4f}, Crop Acc: {train_acc_crop:.2f}%, State Acc: {train_acc_state:.2f}%')
        print(f'Val Loss: {val_loss:.4f}, Crop Acc: {val_acc_crop:.2f}%, State Acc: {val_acc_state:.2f}%')
        
        epoch_progress.update(1)
    
    epoch_progress.close()

# Save the model after training
os.makedirs('CropDoc/data/output/model')
torch.save(model.state_dict(), 'CropDoc/data/output/model/MultiHeadResNet.pth')
print('Training complete. Model saved.')