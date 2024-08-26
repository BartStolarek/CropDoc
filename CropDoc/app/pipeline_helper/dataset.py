import os
import re
from abc import ABC, abstractmethod

import numpy as np
import torch
from loguru import logger
from PIL import Image


class BaseDataset(torch.utils.data.Dataset, ABC):

    def __init__(self,
                 dataset_path,
                 split='train',
                 crop_index_map: dict = None,
                 state_index_map: dict = None):

        self.root = dataset_path
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
        self.crop_index_map = self._generate_index_map(self.crops)
        self.state_index_map = self._generate_index_map(self.states)

        self._fix_state_labels()
        logger.debug(f"Crops in dataset: {self.crops}")
        logger.debug(f"States in dataset: {self.states}")

    @abstractmethod
    def walk_through_root_append_images_and_labels(self):
        pass

    def _generate_index_map(self, class_list):
        index_map = {i: class_name for i, class_name in enumerate(class_list)}
        return index_map

    def _get_unique_crops(self, index_map=None):
        if not index_map:
            crops = []
            for crop, _ in self.labels:
                if crop not in crops:
                    crops.append(crop)
            crops = sorted(list(set(crops)))
            return crops
        else:
            crops = []
            for i in range(len(index_map)):
                crops.append(index_map[i])
            return crops

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

    def get_unique_crop_count(self):
        return len(self.crops)

    def get_unique_state_count(self):
        return len(self.states)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        crop, state = self.labels[idx]

        img = Image.open(img_path).convert('RGB')

        # Convert crop and state to numeric labels
        crop_label = self.crops.index(crop)
        state_label = self.states.index(state)

        return img, crop_label, state_label

    def __len__(self):
        return len(self.images)

    def __str__(self):
        string = "CCMT Augmented Dataset\n" + \
            f"Split: {self.split}\n" + \
            f"Image Count: {len(self.images)}\n" + \
            f"Crop Class Count: {len(self.crops)}\n" + \
            f"State Class Count: {len(self.states)}\n" + \
            f"Crop Data Map: \n {self.data_map['crop']}\n" + \
            f"State Data Map: \n {self.data_map['state']}\n"
        return string

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
        self.crop_index_map = self._generate_index_map(self.crops)
        self.state_index_map = self._generate_index_map(self.states)

        logger.info(f'Reduced dataset to {len(self.images)} images')

    def _fix_state_labels(self):
        for i, state in enumerate(self.states):
            if state == 'bb':
                self.states[i] = 'bacterial blight'
            elif state == 'bspot':
                self.states[i] = 'brown spot'
            elif state == 'farmyw':
                self.states[i] = 'fall armyworm'
            elif state == 'gmite':
                self.states[i] = 'green mite'

        for key, value in self.state_index_map.items():
            if value == 'bb':
                self.state_index_map[key] = 'bacterial blight'
            elif value == 'bspot':
                self.state_index_map[key] = 'brown spot'
            elif value == 'farmyw':
                self.state_index_map[key] = 'fall armyworm'
            elif value == 'gmite':
                self.state_index_map[key] = 'green mite'


class CropCCMTDataset(BaseDataset):

    def __init__(self, split='train', **kwargs):
        super().__init__(**kwargs)

        self.split = split

        splits = ['train', 'test']
        if split not in splits:
            raise ValueError(
                f"Invalid split: {split}. Must be one of {splits}")

    def walk_through_root_append_images_and_labels(self):
        logger.debug(
            f"Walking through root path: {self.root} to obtain images and labels"
        )
        # Walk through root path

        images = []
        labels = []

        for root, directories, files in os.walk(self.root):
            for file in files:

                try:
                    crop_label, state_label = self.handle_file(file)
                except Exception as e:
                    logger.error(
                        f"There was an issue with handling the file, make sure you have the right dataset class and dataset root directory set in the config file. {e}"
                    )
                    raise Exception(e)

                if crop_label is None or state_label is None:
                    continue

                # Append image path and labels
                images.append(os.path.join(root, (os.path.basename(file))))
                labels.append((crop_label.lower(), state_label.lower()))

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

        logger.info('Finished walking through root path')

    def handle_file(self, file):
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
            return None, None

        # Obtain state label
        state_label = classes_and_split[2]

        # Change healthy state labels to include crop label in them to differentiate
        if state_label == 'healthy':
            state_label = crop_label + '-healthy'
        elif state_label == 'bb':
            state_label = 'bacterial blight'
        elif state_label == 'bspot':
            state_label = 'brown spot'
        elif state_label == 'farmyw':
            state_label = 'fall armyworm'
        elif state_label == 'gmite':
            state_label = 'green mite'

        return crop_label, state_label


class PlantVillageDataset(BaseDataset):
    """A class to load the dataset from the data/dataset directory
    
    Dataset can be found: https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset?resource=download

    Args:
        BaseDataset (torch.utils.data.Dataset): Base class for all datasets in PyTorch
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        for i, (key, value) in enumerate(self.data_map['crop'].items()):
            if key not in self.crops:
                self.crops.append(key)

        for i, (key, value) in enumerate(self.data_map['state'].items()):
            if key not in self.states:
                self.states.append(key)

    def walk_through_root_append_images_and_labels(self):

        logger.debug(
            f'Walking through root path: {self.root} to obtain images and labels'
        )

        images = []
        labels = []

        for root, directories, files in os.walk(self.root):
            for directory in directories:

                directory_path = os.path.join(root, directory)

                sub_files = os.listdir(directory_path)

                classes = directory.split('___')
                crop_label = classes[0].lower()
                state_label = classes[1].lower()

                logger.debug(
                    f'Found directory with crop: {crop_label} and state: {state_label} and {len(sub_files)} images'
                )

                if state_label == 'healthy':
                    state_label = crop_label + '-healthy'

                for file in sub_files:

                    file_path = os.path.join(directory_path, file)

                    if not file_path.lower().endswith(
                            '.jpg') and not file_path.lower().endswith(
                                '.jpeg'):
                        logger.debug(
                            f'File {file_path} is not a jpg file, skipping')
                        continue
                    # Append image path and labels
                    images.append(file_path)
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

        logger.info(
            f'Finished walking through root path, found {len(self.images)} images, data map: {self.data_map}'
        )
