"""
Description: This script contains utility functions and classes regarding the dataset loading and processing.

Author: Nisal Hemadasa
Date: 19-09-2024
Version: 1.0
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch
from torch import Tensor
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torchvision.datasets.mnist import read_image_file, read_label_file
from torchvision import transforms
from collections import defaultdict

from typing import List, Tuple, Any

import constants


class CustomDataset(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[Tensor | Any, Tensor]:
        image = self.images[idx]
        label = self.labels[idx]

        if isinstance(image, torch.Tensor):
            # Skip applying transformations that expect PIL or ndarray, if image is already a tensor
            pass
        else:
            image = self.transform(image)

        return image, label


def convert_dataset_to_loader(_dataset: Dataset, _batch_size: int, _is_shuffle: bool = True) -> DataLoader:
    """
    Converts the Dataset object to DataLoader object.
    :param _dataset: Dataset (torch.utils.data objects) object that needs to be converted to DataLoader object.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :param _is_shuffle: whether to shuffle the data or not.
    :return: DataLoader object.
    """
    return DataLoader(_dataset, batch_size=_batch_size, shuffle=_is_shuffle)


def convert_custom_dataset_to_loader(_dataset: List[Tensor, Tensor], _batch_size: int,
                                     _is_shuffle: bool) -> DataLoader:
    """
    Converts the CustomDataset object to DataLoader object.
    :param _dataset: Input data and labels that needs to be converted to DataLoader object.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :param _is_shuffle: whether to shuffle the data or not.
    :return: DataLoader object.
    """
    # Assuming grayscale, update for RGB in 'transforms.Normalize', if necessary
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    # Wrap the image and label data in a Dataset and create a dataloader
    dataset = CustomDataset(images=_dataset[0], labels=_dataset[1], transform=transform_mnist)

    return DataLoader(dataset, batch_size=_batch_size, shuffle=_is_shuffle)


def read_datasets(_dataset_dir: str, _dataset_filename: constants.DatasetFileNames,
                  _dataset_name: str, _batch_size: int) -> List[Tuple[Tensor, Tensor]]:
    """
    Reads and returns datasets.
    :param _dataset_dir: directory where the dataset is stored.
    :param _dataset_filename: constants.MNISTFilesNames object that contains the filename of the dataset.
    :param _dataset_name: name of the dataset.
    :param _batch_size: batch size of loading data. i.e., not all the data available in the dataset is returned.
    :return: Dataset as a list of Tuples of Tensors.
    """

    trainset = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_train_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_train_labels(_dataset_name)[0])),
    )
    testset = (
        read_image_file(os.path.join(_dataset_dir, _dataset_filename.get_test_images(_dataset_name)[0])),
        read_label_file(os.path.join(_dataset_dir, _dataset_filename.get_test_labels(_dataset_name)[0]))
    )

    return [trainset, testset]


def split_iid_dataset(_dataset: Dataset, _num_partitions: int) -> List[Subset]:
    """
    Splits the dataset randomly into mutually exclusive partitions (Dataset -> Subset).
    :param _dataset: Dataset that needs to be split.
    :param _num_partitions: Number of partitions to split the dataset.
    :return: None
    """
    partition_size = len(_dataset) // _num_partitions  # Compute size of each partition
    partition_lengths = [partition_size] * _num_partitions  # Create a list; value=partition_size, length=num_partitions

    # If the dataset cannot be evenly split into partitions, add the remaining data to the last partition. This is to
    # avoid the possible runtime exception when calling random_split() function.
    if len(_dataset) % _num_partitions != 0:
        partition_lengths[-1] += len(_dataset) % _num_partitions

    # Randomly split the training dataset into partitions
    split_datasets = random_split(_dataset, partition_lengths)
    client_indices = [subset.indices for subset in split_datasets]

    # Create subsets based on the indices of the split datasets
    return [Subset(_dataset, indices) for indices in client_indices]



def split_noniid_dataset(dataset: Dataset, num_partitions: int, distribution: str) -> List[Subset]:
    """
    Split the dataset into a selected noon-IID distribution.
    :param dataset: The full dataset to split (e.g., FMNIST or MNIST training set)
    :param num_partitions: Number of clients to split the dataset into
    :param distribution: The type of non-IID distribution to create (e.g., "Pathological", "Dirichlet")
    :return: A list of torch.utils.data.Subset objects, one per client
    """
    if distribution == constants.DatasetPartitionDistribution.PATHOLOGICAL:
        return split_noniid_pathological(dataset, num_partitions)
    elif distribution == constants.DatasetPartitionDistribution.DIRICHLET:
        return split_noniid_dirichlet(dataset, num_partitions)
    else:
        raise ValueError(f"Unsupported non-IID distribution type: {distribution}")


def split_noniid_pathological(dataset: Dataset, num_partitions: int) -> List[Subset]:
    """
    Splits the dataset into Pathological non-IID partitions by first sorting by class label, then
    dividing the sorted data into disjoint chunks. Each client gets data from a limited
    number of classes.
    :param dataset: The full dataset to split (e.g., FMNIST or MNIST training set)
    :param num_partitions: Number of clients to split the dataset into
    :return: A list of torch.utils.data.Subset objects, one per client
    """
    # Extract labels
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    indices = np.arange(len(labels))

    # Sort indices by class label
    sorted_indices = indices[np.argsort(labels)]

    # Split sorted indices into equal parts
    partition_size = len(sorted_indices) // num_partitions
    partition_indices = [sorted_indices[i * partition_size: (i + 1) * partition_size] for i in range(num_partitions)]

    # If there's a remainder, add remaining samples to the last partition
    if len(sorted_indices) % num_partitions != 0:
        partition_indices[-1] = np.concatenate(
            [partition_indices[-1], sorted_indices[num_partitions * partition_size:]]
        )

    # Create subsets for each partition
    partitioned_datasets = [Subset(dataset, indices.tolist()) for indices in partition_indices]

    # for debug purposes
    _unique_labels = get_unique_labels_per_subset(dataset, partitioned_datasets)

    return partitioned_datasets


def split_noniid_dirichlet(dataset: Dataset, num_partitions: int, alpha: float = 1) -> List[Subset]:
    """
    Splits the dataset into non-IID partitions using a Dirichlet distribution. Each client gets a mixture of classes,
    with the degree of non-IID-ness controlled by the alpha parameter.
    :param dataset: The full dataset to split (e.g., FMNIST or MNIST training set)
    :param num_partitions: Number of clients to split the dataset into
    :param alpha: Concentration parameter for the Dirichlet distribution (smaller alpha -> more non-IID). Guo et al. 2024 in FedRC has used alpha=1 for their experiments.
    :return: A list of torch.utils.data.Subset objects, one per client
    """
    # Extract labels
    labels = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)
    indices = np.arange(len(labels))

    # Get unique class labels and their corresponding indices
    unique_labels = np.unique(labels)
    label_indices = {label: indices[labels == label] for label in unique_labels}

    # Create empty lists to hold the indices for each partition
    partition_indices = [[] for _ in range(num_partitions)]

    # For each class, distribute its samples among the partitions according to a Dirichlet distribution
    for label in unique_labels:
        class_indices = label_indices[label]
        np.random.shuffle(class_indices)  # Shuffle class indices

        # Sample proportions for this class from a Dirichlet distribution
        proportions = np.random.dirichlet(alpha * np.ones(num_partitions))

        # Calculate the number of samples for each partition based on the proportions
        num_samples_per_partition = (proportions * len(class_indices)).astype(int)

        # Distribute samples to partitions
        start_idx = 0
        for i in range(num_partitions):
            end_idx = start_idx + num_samples_per_partition[i]
            partition_indices[i].extend(class_indices[start_idx:end_idx])
            start_idx = end_idx

    # Create subsets for each partition
    partitioned_datasets = [Subset(dataset, indices) for indices in partition_indices]

    # for debug purposes
    _unique_labels = get_unique_labels_per_subset(dataset, partitioned_datasets)

    return partitioned_datasets


def get_unique_labels_per_subset(dataset, subsets: List[Subset]) -> List[np.ndarray]:
    """
    Returns a list of unique class labels for each subset.

    :param dataset: The full PyTorch dataset (e.g., MNIST)
    :param subsets: A list of torch.utils.data.Subset instances
    :return: A list of NumPy arrays, each containing the unique labels in the corresponding subset
    """
    # Get the full label array
    full_targets = dataset.targets.numpy() if hasattr(dataset.targets, 'numpy') else np.array(dataset.targets)

    unique_labels_list = []
    for subset in subsets:
        subset_indices = subset.indices
        subset_labels = full_targets[subset_indices]
        unique_labels = np.unique(subset_labels)
        unique_labels_list.append(unique_labels)

    return unique_labels_list


def get_num_classes_from_dataset(dataset: Dataset) -> int:
    """
    Returns the number of classes in a given dataset.
    :param dataset: The dataset object (e.g., torchvision dataset)
    :return: Number of classes in the dataset
    """
    # Case 1: torchvision datasets with `.classes`
    if hasattr(dataset, "classes") and dataset.classes is not None:
        return len(dataset.classes)

    # Case 2: labels already cached
    if hasattr(dataset, "_original_targets"):
        return int(dataset._original_targets.max().item() + 1)

    # Case 3: dataset.targets exists
    if hasattr(dataset, "targets"):
        if isinstance(dataset.targets, torch.Tensor):
            return int(dataset.targets.max().item() + 1)
        else:
            return max(dataset.targets) + 1

    raise ValueError("Dataset does not contain label information.")


def get_random_labels(labels: List[int], num_elements: int) -> Tensor:
    """
    Returns a list random labels of size num_elements (belonging to all available class labels in the dataset).
    Works with torchvision datasets like MNIST, CIFAR-10, Fashion-MNIST, etc.
    :param labels: Unique labels from the dataset
    :param num_elements: Size of the list of random labels to return
    :return: A tensor of random labels
    """
    # Generate a list of random labels
    random_labels = random.choices(labels, k=num_elements)
    # return as a tensor
    return torch.tensor(random_labels, dtype=torch.long)
