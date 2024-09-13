from typing import Dict
import torchvision
from loguru import logger

class TransformerManager:
    def __init__(self):
        self.transformers = self._get_transformers()
        logger.info("Created a TransformerManager object with transformers")

    def __len__(self):
        return self._calculate_total_transforms()

    def _calculate_total_transforms(self):
        total_transforms = 0
        for split, transformer in self.transformers.items():
            total_transforms += len(transformer.transforms)
        return total_transforms

    def _get_transformers(self) -> Dict[str, torchvision.transforms.Compose]:
        logger.debug("Getting transformers")

        train_transformers = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(224),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            torchvision.transforms.RandomRotation(degrees=20),
            torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Created train transformers")

        val_transformers = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Created val transformers")

        test_transformers = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        logger.info("Created test transformers")

        transformer_dict = {
            'train': train_transformers,
            'val': val_transformers,
            'test': test_transformers
        }

        logger.info("Created a dict of transformers with keys 'train', 'val', and 'test' and corresponding composed transformers")

        return transformer_dict

    def get_transformer(self, split: str) -> torchvision.transforms.Compose:
        if split not in self.transformers:
            logger.error(f"Invalid split: {split}. Must be one of 'train', 'val', or 'test'.")
            return None
        return self.transformers[split]