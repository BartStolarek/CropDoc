from loguru import logger
import torchvision
from typing import List

class TransformerManager():

    def __init__(self, transformers: dict[str, list[dict]]):
        self.transformers = self._get_transformers(transformers)

    def __len__(self):
        return self._calculate_total_transforms()

    def _calculate_total_transforms(self):
        total_transforms = 0
        for split, transformer in self.transformers.items():
            total_transforms += len(transformer)
        return total_transforms

    def _get_transformers(
        self, transformers: dict[str, list[dict]]
    ) -> dict[str, torchvision.transforms.Compose]:
        logger.debug("Getting transformers")
        if not transformers:
            logger.warning("No transformers provided")
            return None

        if not isinstance(transformers, dict):
            logger.error(
                "Transformers must be a dictionary: provided a {type(transforms)} : {transforms}"
            )
            return None

        if not set(['train', 'val', 'test']).issubset(transformers.keys()):
            logger.error(
                "Transformers must contain keys 'train', 'val', and 'test'")
            return None

        # Get train transformers
        train_transformers = self._build_transformers(transformers['train'])

        # Get val transformers
        val_transformers = self._build_transformers(transformers['val'])

        # Get test transformers
        test_transformers = self._build_transformers(transformers['test'])

        transformer_dict = {
            'train': train_transformers,
            'val': val_transformers,
            'test': test_transformers
        }

        logger.info("Created transformers dict")

        return transformer_dict

    def _build_transformers(
            self,
            transformer_list: List[dict]) -> torchvision.transforms.Compose:
        logger.debug("Building transformers")
        if not transformer_list:
            logger.warning("No transformers provided")
            return None

        if not isinstance(transformer_list, list):
            logger.error("Transformers must be a list")
            return None

        transforms_list = []
        for transform_dict in transformer_list:
            try:
                transform = self._parse_transform(transform_dict)
                if transform:
                    transforms_list.append(transform)
            except Exception as e:
                logger.error(
                    f"Error parsing transform '{transform_dict}': {e}")

        logger.info(f"Built transformers list {transforms_list}")
        return torchvision.transforms.Compose(transforms_list)

    def _parse_transform(self, transform_dict: dict):
        logger.debug(f"Parsing transform {transform_dict}")
        transform_name = transform_dict['type']

        # Remove type and its value from dict
        transform_dict.pop('type')

        transform_fn = getattr(torchvision.transforms, transform_name)
        logger.info(f"Converted transformer dict to {transform_fn}")
        return transform_fn(**transform_dict)

