"""
Description: This script injects different types of drift to the data (and creates data with drift).

Author: Nisal Hemadasa
Date: 17-09-2024
Version: 1.0
"""
import numpy as np
import torch


def flip_mnist_labels_partial(data, flip_dict, flip_fraction=1.0):
    """
    Flip the labels of a part of the dataset according to the flip_dict.

    Parameters:
    data (torch.utils.data.Dataset): The dataset to modify.
    flip_dict (dict): A dictionary where the keys are the original labels and the values are the new labels.
    flip_fraction (float): The fraction of labels to flip for each label in flip_dict (default is 1.0, meaning all).

    Returns:
    torch.utils.data.Dataset: The modified dataset with partially flipped labels.
    """
    # Extract the labels from the dataset
    labels = np.array(data.targets)

    # Iterate through the flip_dict and apply the partial label flips
    for original_label, new_label in flip_dict.items():
        # Get indices where the label is the original label
        label_indices = np.where(labels == original_label)[0]

        # Calculate how many labels to flip based on flip_fraction
        num_to_flip = int(len(label_indices) * flip_fraction)

        # Randomly select a subset of these indices to flip
        flip_indices = np.random.choice(label_indices, num_to_flip, replace=False)

        # Apply the flip
        labels[flip_indices] = new_label

    # Update the dataset labels
    data.targets = torch.tensor(labels)

    return data