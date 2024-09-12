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
from loguru import logger

# Set up logging
logger.add(f"{datetime.now()}_Training_Log.log", rotation="500 MB", retention="100 days", compression="zip")
logger.info(f"Training Started @ {datetime.now()} - Log Metrics & Stats")

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

# Definitions
NUM_EPOCHS = 1
NUM_CLASSES = len(train_valid_dataset.classes)  # Get the number of classes

# Calculate split sizes for 90% train and 10% validation
train_size = int(0.9 * len(train_valid_dataset))
val_size = len(train_valid_dataset) - train_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(train_valid_dataset, [train_size, val_size])

# Load training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Create model
model = xception(num_classes=NUM_CLASSES)

# Modify the final fully connected layer to match the number of classes
# num_classes = NUM_CLASSES
# model.fc = torch.nn.Linear(2048, num_classes)

# Print out the number of classes detected by ImageFolder
print(f"Number of classes detected: {len(train_valid_dataset.classes)}")
logger.info(f"Number of classes detected: {len(train_valid_dataset.classes)}")
print(f"Classes: {train_valid_dataset.classes}")
logger.info(f"Classes: {train_valid_dataset.classes}")

# Move to GPU if available - to all possible GPUs available XD
device_ids = [i for i in range(torch.cuda.device_count())]
xcpt = Xception_class.xception(True, NUM_CLASSES)
model = nn.DataParallel(xcpt, device_ids=device_ids)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Test the model with a random input tensor
# input_tensor = torch.randn(1, 3, 299, 299)  # Batch size 1, 3 channels (RGB), 299x299 image === Need to check what our image sizes are btw
# output = model(input_tensor)
# print(f"Output shape: {output.shape}")


# Loss function and optimizer
criterion_crop = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop ==========================================
training_start_time = datetime.now()
print('Starting Training :D at ' + str(training_start_time))
# num_epochs = 1
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        # labels = labels - 1   # If your labels are 1-based (1, 2, 3, 4)
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
    print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    logger.info(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    training_finish_time = datetime.now()
    training_duration = training_finish_time - training_start_time
    print(f'Finished training @ {training_finish_time}, total training duration: {training_duration}')
    logger.info(f'Total training duration: {training_duration}')

    # Validation =================================
    validation_start_time = datetime.now()
    print('Starting Validation :D at ' + str(validation_start_time))
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
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    logger.info(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    validation_finish_time = datetime.now()
    validation_duration = validation_finish_time - validation_start_time
    print(f'Finished validation @ {validation_finish_time}')
    logger.info(f'Total training duration: {validation_duration}')

# Save the trained model
model_stamp = 'CL-' + str(NUM_CLASSES) + '_EP-'  + str(NUM_EPOCHS)
torch.save(model.state_dict(), f'xception_trained_model_{model_stamp}.pth')
