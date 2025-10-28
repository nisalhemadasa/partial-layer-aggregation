"""
Description: This script loads the downloaded the datasets for further processing. It downloads the dataset if needed.

Author: Nisal Hemadasa
Date: 06-08-2024
Version: 1.0
"""
import os
import sys
from typing import List

from torch.utils.data import Dataset
from torchvision import datasets, transforms

import constants


def load_datasets(_dataset_name: str) -> list[Dataset]:
    """
    Loads the dataset either by reading or downloading if the dataset does not exist.
    :param _dataset_name: Name of the dataset that needs to be downloaded.
    :return: List of datasets of type torchvision.datasets.
    """
    # Define transforms for the datasets
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    transform_cifar10 = transforms.Compose([transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if _dataset_name == constants.DatasetNames.MNIST:
        files_exist = os.path.exists(os.path.join(constants.Paths.DATASET, _dataset_name))
        # files_exist = os.path.exists(_dataset_name) in constants.Paths.DATASET

        if files_exist:
            # Load the dataset from the existing files
            trainset = datasets.MNIST(constants.Paths.DATASET, download=False, train=True, transform=transform_mnist)

            testset = datasets.MNIST(constants.Paths.DATASET, download=False, train=False, transform=transform_mnist)
        else:
            # Download the dataset and load it
            trainset = datasets.MNIST(constants.Paths.DATASET, download=True, train=True, transform=transform_mnist)

            testset = datasets.MNIST(constants.Paths.DATASET, download=True, train=False, transform=transform_mnist)

    elif _dataset_name == constants.DatasetNames.F_MNIST:
        files_exist = os.path.exists(os.path.join(constants.Paths.DATASET, _dataset_name))

        if files_exist:
            # Load the dataset from the existing files
            trainset = datasets.FashionMNIST(constants.Paths.DATASET, download=False, train=True,
                                             transform=transform_mnist)

            testset = datasets.FashionMNIST(constants.Paths.DATASET, download=False, train=False,
                                            transform=transform_mnist)
        else:
            # Download the dataset and load it
            trainset = datasets.FashionMNIST(constants.Paths.DATASET, download=True, train=True,
                                             transform=transform_mnist)

            testset = datasets.FashionMNIST(constants.Paths.DATASET, download=True, train=False,
                                            transform=transform_mnist)

    elif _dataset_name == constants.DatasetNames.CIFAR_10:
        files_exist = os.path.exists(os.path.join(constants.Paths.DATASET, _dataset_name))

        if files_exist:
            trainset = datasets.CIFAR10(constants.Paths.DATASET, download=False, train=True,
                                        transform=transform_cifar10)

            testset = datasets.CIFAR10(constants.Paths.DATASET, download=False, train=False,
                                       transform=transform_cifar10)
        else:
            # Download the dataset and load it
            trainset = datasets.CIFAR10(constants.Paths.DATASET, download=True, train=True, transform=transform_cifar10)

            testset = datasets.CIFAR10(constants.Paths.DATASET, download=True, train=False, transform=transform_cifar10)
    else:
        print("Invalid dataset name. Please enter a valid dataset name.")
        sys.exit()

    return [trainset, testset]


def get_dataset_classnames_from_indices(_dataset_name: str, _class_indices: List[int]) -> List[str]:
    """
    Returns the number of classes in the dataset.
    :param _dataset_name: Name of the dataset.
    :param _class_indices: List of class indices whose class name needs to be returned.
    :return: Class names corresponding to the _class_indices.
    """
    from torchvision import datasets

    # Dynamically get the dataset class from torchvision.datasets
    DatasetClass = getattr(datasets, _dataset_name)

    # Now you can access its class names if defined
    if hasattr(DatasetClass, "classes"):
        all_class_names = DatasetClass.classes
    else:
        print('WARNING: Class names of the dateset is not available. Returning class indices instead.')
        return [str(x) for x in _class_indices]

    # Filter the corresponding class names for the given class indices
    filtered_classes = [all_class_names[i] for i in _class_indices]

    return filtered_classes
