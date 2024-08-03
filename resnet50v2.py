import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, models, datasets
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader, ConcatDataset
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image

# Define constants
DATA_DIR = 'data/CCMT Dataset-Augmented'
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Define data transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Load datasets
def get_datasets(data_dir, transforms):
    train_datasets = []
    test_datasets = []
    all_classes = set()
    crops = [
        'Cashew',
        'Cassava',
        'Maize',
        'Tomato'
    ]
    for crop in crops:
        crop_dir = os.path.join(data_dir, crop)
        train_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'train_set'), transform=transforms['train'])
        test_dataset = datasets.ImageFolder(os.path.join(crop_dir, 'test_set'), transform=transforms['test'])
        train_datasets.append(train_dataset)
        test_datasets.append(test_dataset)
        all_classes.update(train_dataset.classes)
    
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    return train_dataset, test_dataset, list(all_classes)

# Get datasets and dataloaders
train_dataset, test_dataset, class_names = get_datasets(DATA_DIR, data_transforms)
dataloaders = {
    'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4),
    'test': DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
}
dataset_sizes = {'train': len(train_dataset), 'test': len(test_dataset)}
NUM_CLASSES = len(class_names)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load pre-trained ResNet50v2 model and modify for our task
def load_model():
    model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
    return model.to(device)

# Training function
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'test' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        print()

    print(f'Best test Acc: {best_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model

# Prediction function
def predict_image(model, image_path):
    img = Image.open(image_path)
    img_tensor = data_transforms['test'](img).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        outputs = model(img_tensor)
        _, preds = torch.max(outputs, 1)
        
    return class_names[preds[0]]

# Main execution
if __name__ == '__main__':
    # Load the model
    model = load_model()

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)
    
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # Train the model
    model = train_model(model, criterion, optimizer, scheduler, NUM_EPOCHS)

    # Save the model
    torch.save(model.state_dict(), 'crop_health_model.pth')

    # Example of using the model for prediction
    image_path = 'path/to/test/image.jpg'
    prediction = predict_image(model, image_path)
    print(f'Predicted class: {prediction}')