import torch.utils.data as data
import torch
from PIL import Image
import random
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np


def _dataset_info(txt_labels):
    """From txt_labels (file path) extracts the filename and the label
    associated to each image of the dataset"""

    # Opening the file "with" statement and then close it as soon as the command finishes
    # txt_labels is a file path
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []

    # Splits every row in two different parts, the file name
    # and the label associated to each element of the dataset
    # the split character is ' '
    for row in images_list:
        row = row.split(' ')
        file_names.append(row[0])
        labels.append(int(row[1]))

    return file_names, labels


class Dataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, is_multi, img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer
        self.is_multi = is_multi

    def __getitem__(self, index):
        """Get the image at the desired index, original and randomly rotated
        index_rot = 0 --> 0°
        index_rot = 1 --> 90°
        index_rot = 2 --> 180°
        index_rot = 3 --> 270°
        """
        # Open image, select the random rotation
        img = Image.open(self.data_path + self.names[index], mode='r')
        index_rot = random.randint(0, 3)

        # Apply the transformations on both original and rotated image
        img = self._image_transformer(img)
        img_rot = torch.rot90(img, k=index_rot, dims=[1, 2])

        # Select the proper index according to the rotational classifier
        if self.is_multi:
            index_rot = int(self.labels[index])*4 + index_rot
        else:
            index_rot = index_rot

        return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)


class TestDataset(data.Dataset):
    def __init__(self, names, labels, path_dataset, img_transformer=None):
        self.data_path = path_dataset
        self.names = names
        self.labels = labels
        self._image_transformer = img_transformer

    def __getitem__(self, index):
        """Get the image at the desired index, original and rotated
        index_rot = 0 --> 0°
        index_rot = 1 --> 90°
        index_rot = 2 --> 180°
        index_rot = 3 --> 270°
        """
        # Open image, select the random rotation
        img = Image.open(self.data_path + self.names[index], mode='r')
        index_rot = random.randint(0, 3)

        # Apply the transformations on both original and rotated image
        img = self._image_transformer(img)
        img_rot = torch.rot90(img, k=index_rot, dims=[1, 2])

        return img, int(self.labels[index]), img_rot, index_rot

    def __len__(self):
        return len(self.names)
