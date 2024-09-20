from PIL import Image
import torch
from loguru import logger

class PredictManager():
    def __init__(self, config, output_directory, eval, model_manager, transformers, image_path):
        self.config = config
        self.output_directory = output_directory
        self.eval = eval
        self.model_manager = model_manager
        self.transformers = transformers
        self.image_path = image_path
        
        self.image = Image.open(self.image_path).convert('RGB')
        self.image = self.transformers(self.image)
        self.image = self.image.unsqueeze(0).to(self.model_manager.model.device)
        
        
    def predict(self):
        
        with torch.no_grad():
            crop_predictions, state_predictions = self.model_manager.model(
                self.image
            )  # Get the pytorch tensors (multi dimensional arrays) with raw scores (not probabilities), shape: (1, num_classes)

        # Get the predicted classes
        crop_confidence, predicted_crop = torch.max(
            crop_predictions, 1
        )  # Tensor with highest probability class, shape: (1), Tensor with index of highest probability class, shape: (1)
        state_confidence, predicted_state = torch.max(state_predictions, 1)

        # Apply softmax to get probabilities
        crop_probabilities = torch.nn.functional.softmax(
            crop_predictions, dim=1
        )  # Tensor with probabilities for each class summing to 1, shape: (1, num_classes)
        state_probabilities = torch.nn.functional.softmax(
            state_predictions, dim=1
        )  # Tensor with probabilities for each class summing to 1, shape: (1, num_classes)

        # Convert to numpy for easier handling
        crop_confidence = crop_confidence.cpu().numpy()
        state_confidence = state_confidence.cpu().numpy()
        predicted_crop = predicted_crop.cpu().numpy()
        predicted_state = predicted_state.cpu().numpy()

        # Get the probability of the predicted class
        crop_probability = torch.max(crop_probabilities).item()
        state_probability = torch.max(state_probabilities).item()

        # Apply confidence threshold
        crop_mask = crop_probability > self.config['predict']['crop']['confidence_threshold'],
        state_mask = state_probability > self.config['predict']['state']['confidence_threshold'],

        # Get the crop class prediction and confidence
        crop_prediction = self.model_manager.model_meta.crops[
            predicted_crop[0]] if crop_mask else 'Unknown'
        state_prediction = self.model_manager.model_meta.states[
            predicted_state[0]] if state_mask else 'Unknown'

        # Get class names for all classes
        crop_class_names = [
            self.model_manager.model_meta.crops[i] for i in range(len(self.model_manager.model_meta.crops))
        ]
        state_class_names = [
            self.model_manager.model_meta.states[i] for i in range(len(self.model_manager.model_meta.states))
        ]

        # Map probabilities to class names
        crop_probabilities_dict = dict(
            zip(crop_class_names,
                crop_probabilities.squeeze().cpu().numpy()))
        state_probabilities_dict = dict(
            zip(state_class_names,
                state_probabilities.squeeze().cpu().numpy()))

        result = {
            'crop': {
                'prediction':
                crop_prediction,
                'confidence':
                crop_confidence[0],
                'probability':
                crop_probability,
                'confidence_threshold':
                self.config['predict']['crop']['confidence_threshold'],
                'class_probabilities':
                crop_probabilities_dict
            },
            'state': {
                'prediction':
                state_prediction,
                'confidence':
                state_confidence[0],
                'probability':
                state_probability,
                'confidence_threshold':
                self.config['predict']['state']['confidence_threshold'],
                'class_probabilities':
                state_probabilities_dict
            }
        }

        logger.info(f"Prediction for image {self.image_path}:\n{result}")

        return result
