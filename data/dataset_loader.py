"""
Description: This script loads the downloaded the datasets for further processing. It downloads the dataset if needed.

Author: Nisal Hemadasa
Date: 06-08-2024
Version: 1.0
"""
import os
import sys
from typing import List

from sklearn.preprocessing import StandardScaler
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from ucimlrepo import fetch_ucirepo
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomResizedCrop
from torchvision.transforms import ToTensor, Normalize, InterpolationMode

import constants

import numpy as np
import pandas as pd

from data.data_classes import getAdult, getTinyImageNet


def load_datasets(_dataset_name: str, verbose: bool = False) -> list[Dataset]:
    """
    Loads the dataset either by reading or downloading if the dataset does not exist.
    :param _dataset_name: Name of the dataset that needs to be downloaded.
    :param verbose: If True, prints additional information during dataset loading.
    :return: List of datasets of type torchvision.datasets.
    """
    # Define transforms for the datasets
    transform_mnist = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5,), (0.5,))])

    transform_cifar = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # For TinyImageNet-200
    # transform_tiny_imagenet = transforms.Compose([transforms.ToTensor(),
    #                                               transforms.Normalize((0.5, 0.5, 0.5),
    #                                                                    (0.5, 0.5, 0.5))])
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform_train = Compose([
        RandomResizedCrop(64, scale=(0.8, 1.0), interpolation=InterpolationMode.BICUBIC),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std),
    ])

    transform_val = Compose([
        Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean, std),
    ])

    # deterministic transforms only for cached .data / .targets
    cache_transform_train = Compose([
        Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean, std),
    ])

    cache_transform_val = Compose([
        Resize((64, 64), interpolation=InterpolationMode.BICUBIC),
        ToTensor(),
        Normalize(mean, std),
    ])

    # =====================================================================
    # MNIST
    # =====================================================================
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

    # =====================================================================
    # Fashion-MNIST
    # =====================================================================
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

    # =====================================================================
    # CIFAR-10
    # =====================================================================
    elif _dataset_name == constants.DatasetNames.CIFAR_10:
        files_exist = os.path.exists(os.path.join(constants.Paths.DATASET, _dataset_name))

        if files_exist:
            trainset = datasets.CIFAR10(constants.Paths.DATASET, download=False, train=True,
                                        transform=transform_cifar)

            testset = datasets.CIFAR10(constants.Paths.DATASET, download=False, train=False,
                                       transform=transform_cifar)
        else:
            # Download the dataset and load it
            trainset = datasets.CIFAR10(constants.Paths.DATASET, download=True, train=True, transform=transform_cifar)

            testset = datasets.CIFAR10(constants.Paths.DATASET, download=True, train=False, transform=transform_cifar)


    # =====================================================================
    # CIFAR-100
    # =====================================================================
    elif _dataset_name == constants.DatasetNames.CIFAR_100:
        files_exist = os.path.exists(os.path.join(constants.Paths.DATASET, _dataset_name))

        if files_exist:
            trainset = datasets.CIFAR100(constants.Paths.DATASET, download=False, train=True,
                                         transform=transform_cifar)

            testset = datasets.CIFAR100(constants.Paths.DATASET, download=False, train=False,
                                        transform=transform_cifar)
        else:
            # Download the dataset and load it
            trainset = datasets.CIFAR100(constants.Paths.DATASET, download=True, train=True, transform=transform_cifar)

            testset = datasets.CIFAR100(constants.Paths.DATASET, download=True, train=False, transform=transform_cifar)

    # =====================================================================
    # Tiny-ImageNet-200 (custom loader)
    # =====================================================================
    elif _dataset_name == constants.DatasetNames.TINY_IMAGENET_200:
        trainset, testset = getTinyImageNet(_dataset_name, _transform_tiny_imagenet_train=transform_train,
                                            _transform_tiny_imagenet_val=transform_val,
                                            _cache_transform_train=cache_transform_train,
                                            _cache_transform_val=cache_transform_val, verbose=verbose)

    # =====================================================================
    # Adult (custom loader)
    # =====================================================================
    elif _dataset_name == constants.DatasetNames.ADULT:
        trainset, testset = getAdult(
            _dataset_name=_dataset_name,
            test_size=0.2,
            random_state=42,
            transform=None,
            verbose=verbose
        )

    # =====================================================================
    else:
        print("Invalid dataset name. Please enter a valid dataset name.")
        sys.exit()

    # Convert targets of CIFAR 10/100 and Tiny-ImageNet to tensors for consistency
    if _dataset_name in [constants.DatasetNames.CIFAR_10, constants.DatasetNames.CIFAR_100,
                         constants.DatasetNames.TINY_IMAGENET_200]:
        trainset.targets = torch.Tensor(trainset.targets).to(torch.int64)
        testset.targets = torch.Tensor(testset.targets).to(torch.int64)

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
