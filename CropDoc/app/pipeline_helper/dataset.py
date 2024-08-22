from loguru import logger
from PIL import Image
import torch
import torchvision
import numpy as np
import os
import re


class CropCCMTDataset(torch.utils.data.Dataset):

    def __init__(self, dataset_path, transformers=None, split='train', crop_index_map: dict = None, state_index_map: dict = None):
        if not isinstance(transformers, torchvision.transforms.Compose):
            logger.error(
                f"Invalid transforms: {transformers}. Must be of type {torchvision.transforms.Compose}"
            )
            raise ValueError(
                f"Invalid transforms: {transformers}. Must be of type {torchvision.transforms.Compose}"
            )

        self.root = dataset_path
        self.transforms = transformers
        self.split = split

        self.data_map = {'crop': {}, 'state': {}}

        self.images = np.empty(0, dtype=object)
        self.labels = np.empty(
            (0, 2), dtype=int)  # Two columns for crop and state labels

        splits = ['train', 'test']
        if split not in splits:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {splits}")

        self.walk_through_root_append_images_and_labels()

        self.crops = self._get_unique_crops(crop_index_map)
        self.states = self._get_unique_states(state_index_map)
        self.crop_index_map = {i: crop for i, crop in enumerate(self.crops)}
        self.state_index_map = {i: state for i, state in enumerate(self.states)}

    def get_unique_crop_count(self):
        return len(self.crops)

    def get_unique_state_count(self):
        return len(self.states)

    def equal_reduce(self, reduce_to: float):

        images = []
        labels = []
        for state, count in self.data_map['state'].items():
            reduction_amount = int(count * reduce_to)
            added = 0
            for i, img_path in enumerate(self.images):
                state_check = state
                if 'healthy' in state_check:
                    crop = state_check.split('-')[0]
                    if crop in img_path and 'healthy' in img_path:
                        images.append(self.images[i])
                        labels.append(self.labels[i])
                        added += 1
                elif state_check in img_path:
                    images.append(self.images[i])
                    labels.append(self.labels[i])
                    added += 1

                if added >= reduction_amount:
                    break
        self.images = images
        self.labels = labels
        self.crops = self._get_unique_crops()
        self.states = self._get_unique_states()

        logger.info(f'Reduced dataset to {len(self.images)} images')

    def walk_through_root_append_images_and_labels(self):
        logger.debug(
            f"Walking through root path: {self.root} to obtain images and labels"
        )
        # Walk through root path

        images = []
        labels = []

        for root, directory, files in os.walk(self.root):
            for file in files:
                file_name = os.path.basename(file)

                # Remove extension
                file_name = os.path.splitext(file_name)[0]

                # Remove file id from start of file name
                # file_id = re.match(r'^\d+', file_name)
                file_name = re.sub(r'^\d+', '', file_name)

                # Split file name into classes and split
                classes_and_split = file_name.split('_')

                # Obtain crop label and split directory
                crop_label = classes_and_split[0]
                split_directory = classes_and_split[1]

                # Change split directory to be consistent
                if split_directory == 'valid':
                    split_directory = 'test'

                # Check if required split matches
                if split_directory != self.split:
                    continue

                # Obtain state label
                state_label = classes_and_split[2]

                # Change healthy state labels to include crop label in them to differentiate
                if state_label == 'healthy':
                    state_label = crop_label + '-healthy'

                # Append image path and labels
                images.append(os.path.join(root, (os.path.basename(file))))
                labels.append((crop_label, state_label))

                if crop_label not in self.data_map['crop']:
                    self.data_map['crop'][crop_label] = 1
                else:
                    self.data_map['crop'][crop_label] += 1

                if state_label not in self.data_map['state']:
                    self.data_map['state'][state_label] = 1
                else:
                    self.data_map['state'][state_label] += 1

        self.images = np.array(images, dtype=object)
        self.labels = np.array(labels, dtype=object)

        logger.info(f"Crop Data Map: {self.data_map['crop']}")
        logger.info(f"State Data Map: {self.data_map['state']}")

    def _get_unique_crops(self, index_map=None):
        if not index_map:
            crops = []
            for crop, _ in self.labels:
                if crop not in crops:
                    crops.append(crop)
            crops = sorted(list(set(crops)))
            return crops
        else:
            states = []
            for i in range(len(index_map)):
                states.append(index_map[i])
            return states

    def _get_unique_states(self, index_map=None):
        if not index_map:
            states = []
            for _, state in self.labels:
                if state not in states:
                    states.append(state)
            states = sorted(list(set(states)))
            return states
        else:
            states = []
            for i in range(len(index_map)):
                states.append(index_map[i])
            return states

    def __getitem__(self, idx):
        img_path = self.images[idx]
        crop, state = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Convert crop and state to numeric labels
        crop_label = self.crops.index(crop)
        state_label = self.states.index(state)

        if self.transforms:
            img = self.transforms(img)

        return img, crop_label, state_label

    def __len__(self):
        return len(self.images)

