import torch
import torchvision


class MultiHeadResNetModel(torch.nn.Module):
    """A multi-head ResNet model for the CropCCMT dataset

    Args:
        torch (torch.nn.Module): The PyTorch module
    """
    def __init__(self, num_classes_crop, num_classes_state):
        """ Initialise a multi-head ResNet50 model with;
        - A ResNet50 backbone and pre-trained weights
        - A crop head
        - A state head
        
        Also move the model to the GPU if available

        Args:
            num_classes_crop (int): The number of unique classes for the crop head
            num_classes_state (int): The number of unique classes for the state head
        """
        super(MultiHeadResNetModel, self).__init__()
        
        # Check if GPU is available
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the ResNet50 model with pre-trained weights
        self.resnet = torchvision.models.resnet50(
            weights=torchvision.models.ResNet50_Weights.DEFAULT)  # TODO: Add to report why we used ResNet50 default weights and benefits
        
        # Modify the model to remove the final fully connected layer
        num_ftres = self.resnet.fc.in_features
        self.resnet.fc = torch.nn.Identity()
        
        # Add the crop and state heads
        self.crop_fc = torch.nn.Linear(num_ftres, num_classes_crop) 
        self.state_fc = torch.nn.Linear(num_ftres, num_classes_state)  # TODO: Add to report that we replaced the final layer and why, and how it is multihead

        # Move all parts of the model to the same device
        self.resnet = self.resnet.to(self.device)
        self.crop_fc = self.crop_fc.to(self.device)
        self.state_fc = self.state_fc.to(self.device)

        # Wrap only the resnet part in DataParallel
        self.resnet = torch.nn.DataParallel(self.resnet)

    def forward(self, x):
        """ Forward pass through the model, and return the tensors for the crop and state heads as a tuple

        Args:
            x (torch.Tensor): The input tensor, where x.shape is torch.Size(<batch_size>, <num_features>)

        Returns:
            tuple: A tuple containing the crop and state tensors
        """
        x = x.to(self.device)  # Move input to GPU if available
        
        # Forward pass through the ResNet backbone
        x = self.resnet(x)
        
        # Forward pass through the crop and state heads
        crop_out = self.crop_fc(x)
        state_out = self.state_fc(x)
        
        # Return the crop and state tensors
        return crop_out, state_out  # TODO: Add to report that the forward pass will return the crop and state tensors
