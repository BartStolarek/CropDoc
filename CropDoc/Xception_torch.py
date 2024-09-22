# Libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
#from data import DatasetManager, TransformerManager
import Xception_class
from Xception_class import xception
from datetime import datetime
import json
from loguru import logger

# Set up logging
logger.add(f"{datetime.now().strftime("%Y-%m-%d %H%M")}_Training_Log.log", rotation="500 MB", retention="100 days", compression="zip")
logger.info(f"Training Session {datetime.now()} - Log Status, Metrics & Hopefully Soon Stats")

# Define data transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the entire training/validation dataset
root_dir = '/scratch/Az/Dataset/CCMT-Dataset-Augmented/train_Data/'
train_valid_dataset = ImageFolder(root=root_dir, transform=transform)
#train_valid_dataset = DatasetManager(root_dir, transform=transforms)

# Save the class labels from this dataset
class_labels = train_valid_dataset.classes
with open('class_labels.json', 'w') as f:
    json.dump(class_labels, f)
logger.info(f"Saved class labels as {class_labels}")

# Load the original labels and combine them with the new class labels
# Calculate the new number of total classes
# Then... Modify the model.. uhm,  the fully connected layer has to accommodate the new number of classes while retaining the old weights for the old classes.

# Definitions
NUM_EPOCHS = 10
BATCH_SIZE = 64
SPLIT_SIZE = 0.9
LEARN_RATE = 0.001
NUM_CLASSES = len(train_valid_dataset.classes)  # Get the number of classes

# Calculate split sizes for 90% train and 10% validation
train_size = int(SPLIT_SIZE * len(train_valid_dataset))
val_size = len(train_valid_dataset) - train_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(train_valid_dataset, [train_size, val_size])

# Load training and validation datasets
train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False)

# Create model
model = xception(num_classes=NUM_CLASSES)

# Print out the number of classes detected by ImageFolder
logger.info(f"Number of classes detected: {len(train_valid_dataset.classes)}")
logger.info(f"Classes: {train_valid_dataset.classes}")

# Move to GPU if available - to all possible GPUs available XD
device_ids = [i for i in range(torch.cuda.device_count())]
xcpt = Xception_class.xception(NUM_CLASSES)
model = nn.DataParallel(xcpt, device_ids=device_ids)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Loss function and optimizer
criterion_crop = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), LEARN_RATE)

# Training Loop
training_start_time = datetime.now()
logger.info('Training started @ ' + str(training_start_time))

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # labels = labels - 1   # If labels are 1-based (1, 2, 3, 4)
        # print('Train: ' + str(labels))
        inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass: compute the model output
        outputs = model(inputs)
        # print(f"Yo Outputs #1 shape: {outputs.shape}")

        # Compute the loss
        loss = criterion_crop(outputs, labels)
               
        # Backwards pass: compute gradients
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Calculate training accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update running loss
        running_loss += loss.item()

    # Calculate the average loss and accuracy for this epoch
    epoch_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    training_finish_time = datetime.now()
    training_duration = training_finish_time - training_start_time
    logger.info(f'Finished training @ {training_finish_time}, total training duration: {training_duration}')

    # Validation 
    validation_start_time = datetime.now()
    logger.info('Starting Validation @ ' + str(validation_start_time))
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            # labels = labels - 1  # Adjust the labels, derr Az get some sleep yo
            # print('Validation: ' + str(labels))
            inputs, labels = inputs.to('cuda' if torch.cuda.is_available() else 'cpu'), labels.to('cuda' if torch.cuda.is_available() else 'cpu')

            outputs = model(inputs)
            loss = criterion_crop(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    logger.info(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    validation_finish_time = datetime.now()
    validation_duration = validation_finish_time - validation_start_time
    logger.info(f'Finished validation, duration: {validation_duration}')
    epoch_duration = validation_finish_time - training_start_time
    logger.info(f'Epoch duration: {epoch_duration}')

# Save the trained model
torch.save(model.state_dict(), 'xception_trained_model.pth')
logger.info('Saved model...')

