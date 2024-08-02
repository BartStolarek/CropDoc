import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm import tqdm

# Constants
DATA_DIR = 'CropDoc/data/datasets/CCMT Dataset-Augmented'
NUM_EPOCHS = 1
CV_FOLDS = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VAL_SPLIT = 0.2

labels_classes = {
    'crop': ['Cashew', 'Cassava', 'Maize', 'Tomato'],
    'state': {
        'Cashew': ['anthracnose', 'gumosis', 'healthy', 'leaf miner', 'red rust'],
        'Cassava': ['bacterial blight', 'brown spot', 'green mite', 'healthy', 'mosaic'],
        'Maize': ['fall armyworm', 'grasshoper', 'healthy', 'leaf beetle', 'leaf blight', 'leaf spot', 'streak virus'],
        'Tomato': ['healthy', 'leaf blight', 'leaf curl', 'septoria leaf spot', 'verticulium wilt']
    }
}

# Custom Dataset
class CropDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.samples = []
        self.crop_to_idx = {crop: idx for idx, crop in enumerate(labels_classes['crop'])}
        self.state_to_idx = {state: idx for idx, state in enumerate(sum(labels_classes['state'].values(), []))}
        
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
        img = plt.imread(img_path)
        if self.transform:
            img = self.transform(img)
        return img, crop_label, state_label

# Data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomRotation(20),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# Load datasets
train_dataset = CropDataset(DATA_DIR, split='train', transform=data_transforms['train'])
test_dataset = CropDataset(DATA_DIR, split='test', transform=data_transforms['test'])

# Model definition
class MultiHeadResNet(nn.Module):
    def __init__(self, num_crops, num_states):
        super(MultiHeadResNet, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc_crop = nn.Linear(num_ftrs, num_crops)
        self.fc_state = nn.Linear(num_ftrs, num_states)

    def forward(self, x):
        features = self.resnet(x)
        crop_out = self.fc_crop(features)
        state_out = self.fc_state(features)
        return crop_out, state_out

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MultiHeadResNet(len(labels_classes['crop']), len(sum(labels_classes['state'].values(), [])))
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_crop = 0
        correct_state = 0
        total = 0

        for inputs, crop_labels, state_labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
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

        epoch_loss = running_loss / len(train_loader)
        epoch_acc_crop = 100 * correct_crop / total
        epoch_acc_state = 100 * correct_state / total

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Crop Acc: {epoch_acc_crop:.2f}%, State Acc: {epoch_acc_state:.2f}%')

        # Validation
        model.eval()
        val_loss = 0.0
        correct_crop = 0
        correct_state = 0
        total = 0

        with torch.no_grad():
            for inputs, crop_labels, state_labels in val_loader:
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

        val_loss = val_loss / len(val_loader)
        val_acc_crop = 100 * correct_crop / total
        val_acc_state = 100 * correct_state / total

        print(f'Validation Loss: {val_loss:.4f}, Crop Acc: {val_acc_crop:.2f}%, State Acc: {val_acc_state:.2f}%')

# Prepare data loaders
skf = StratifiedShuffleSplit(n_splits=CV_FOLDS, test_size=VAL_SPLIT, random_state=42)
train_indices, val_indices = next(skf.split(train_dataset.samples, [s[1] for s in train_dataset.samples]))

train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler)
val_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=val_sampler)

# Train the model
train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS)

# Evaluation
def evaluate_model(model, test_loader):
    model.eval()
    correct_crop = 0
    correct_state = 0
    total = 0

    with torch.no_grad():
        for inputs, crop_labels, state_labels in tqdm(test_loader, desc='Evaluating'):
            inputs, crop_labels, state_labels = inputs.to(device), crop_labels.to(device), state_labels.to(device)
            crop_outputs, state_outputs = model(inputs)

            _, predicted_crop = torch.max(crop_outputs, 1)
            _, predicted_state = torch.max(state_outputs, 1)
            total += crop_labels.size(0)
            correct_crop += (predicted_crop == crop_labels).sum().item()
            correct_state += (predicted_state == state_labels).sum().item()

    test_acc_crop = 100 * correct_crop / total
    test_acc_state = 100 * correct_state / total
    print(f'Test Accuracy: Crop: {test_acc_crop:.2f}%, State: {test_acc_state:.2f}%')

# Evaluate the model
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
evaluate_model(model, test_loader)
