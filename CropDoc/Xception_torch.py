# Libraries
import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from Xception_class import xception
# import datetime
from datetime import datetime
# from loguru import logger

# Define data transformations (resize, normalize, etc.)
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the entire training/validation dataset
train_valid_dir = 'P:/OneDrive/Studies/University of New England/COSC592 Technology Project/CropDoc/Dataset/CCMT Dataset-Augmented/Cashew/train_set'
train_valid_dataset = ImageFolder(root=train_valid_dir, transform=transform)

# Calculate split sizes for 90% train and 10% validation
train_size = int(0.9 * len(train_valid_dataset))
val_size = len(train_valid_dataset) - train_size

# Split the dataset into train and validation subsets
train_dataset, val_dataset = random_split(train_valid_dataset, [train_size, val_size])

# Load training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Create model
model = xception(pretrained=True, num_classes=5)

# Modify the final fully connected layer to match the number of classes
# num_classes = 4
# model.fc = torch.nn.Linear(2048, num_classes)

# Print out the number of classes detected by ImageFolder
print(f"Number of classes detected: {len(train_valid_dataset.classes)}")
print(f"Classes: {train_valid_dataset.classes}")

# Move to GPU if available
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Test the model with a random input tensor
# input_tensor = torch.randn(1, 3, 299, 299)  # Batch size 1, 3 channels (RGB), 299x299 image === Need to check what our image sizes are btw
# output = model(input_tensor)
# print(f"Output shape: {output.shape}")


# Loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop ==========================================
training_start_time = datetime.now()
print('Starting Training :D at ' + str(training_start_time))
num_epochs = 2
for epoch in range(num_epochs):
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
        loss = criterion(outputs, labels)
               
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
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    training_finish_time = datetime.now()
    print(f'Total Training Duration: ({training_finish_time} - {training_start_time})')

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
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    val_epoch_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total
    print(f"Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    validation_finish_time = datetime.now()
    print(f'Total Validation Duration: ({validation_finish_time} - {validation_start_time})')
    print(f'Total Duration: ({validation_finish_time} - {training_start_time})')

# Save the trained model
timestamp = datetime.strptime(str(validation_finish_time), "%Y-%m-%d %H:%M:%S")
filestamp = timestamp.strftime("%Y-%m-%d_%H%M")
torch.save(model.state_dict(), f'xception_trained_model_{filestamp}.pth')
