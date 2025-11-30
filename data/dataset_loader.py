"""
Description: This script loads the downloaded the datasets for further processing. It downloads the dataset if needed.

Author: Nisal Hemadasa
Date: 06-08-2024
Version: 1.0
"""
import os
import sys
import urllib
import zipfile
from typing import List

from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder

import constants


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

    transform_tiny_imagenet = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5, 0.5, 0.5),
                                                                       (0.5, 0.5, 0.5))])

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
        file_path = os.path.join(constants.Paths.DATASET, _dataset_name)
        files_exist = os.path.exists(file_path)

        if not files_exist:
            url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
            zip_path = os.path.join(constants.Paths.DATASET, "tiny-imagenet-200.zip")

            if verbose:
                print("Downloading Tiny ImageNet-200...")

            urllib.request.urlretrieve(url, zip_path)

            if verbose:
                print("Extracting...")

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(constants.Paths.DATASET)
            os.remove(zip_path)

        # Load using ImageFolder
        train_dir = os.path.join(file_path, "train")
        val_dir = os.path.join(file_path, "val")

        # Tiny ImageNet validation folder needs restructuring
        val_images_dir = os.path.join(val_dir, "images")
        val_annotations = os.path.join(val_dir, "val_annotations.txt")

        # Fix Tiny ImageNet validation folder structure if needed
        if os.path.exists(val_annotations):
            with open(val_annotations, "r") as f:
                for line in f:
                    parts = line.split()
                    img = parts[0]
                    cls = parts[1]
                    img_src = os.path.join(val_images_dir, img)
                    cls_dir = os.path.join(val_dir, cls)
                    os.makedirs(cls_dir, exist_ok=True)
                    img_dst = os.path.join(cls_dir, img)
                    if os.path.exists(img_src):
                        os.rename(img_src, img_dst)

            # Remove original val/images folder
            if os.path.exists(val_images_dir):
                os.rmdir(val_images_dir)

        trainset = ImageFolder(train_dir, transform=transform_tiny_imagenet)
        testset = ImageFolder(val_dir, transform=transform_tiny_imagenet)

        # --------- FAST bulk conversion to .data and .targets using DataLoader ----------
        # ---------- tainset -------------
        if verbose:
            print("Converting Tiny ImageNet trainset to tensors (data, targets)...")

        num_workers = min(8, os.cpu_count() or 1)

        train_loader = DataLoader(trainset, batch_size=512, shuffle=False, num_workers=num_workers, pin_memory=True)

        train_images = []
        train_labels = []
        for imgs, labels in train_loader:
            train_images.append(imgs)
            train_labels.append(labels)

        trainset.data = torch.cat(train_images, dim=0)          # [N_train, 3, 64, 64]
        trainset.targets = torch.cat(train_labels, dim=0)

        # ---------- testset -------------
        if verbose:
            print("Converting Tiny ImageNet testset to tensors (data, targets)...")

        test_loader = DataLoader(
            testset,
            batch_size=512,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_images = []
        test_labels = []
        for imgs, labels in test_loader:
            test_images.append(imgs)
            test_labels.append(labels)

        testset.data = torch.cat(test_images, dim=0)  # [N_test, 3, 64, 64]
        testset.targets = torch.cat(test_labels, dim=0)  # [N_test]

    # =====================================================================
    else:
        print("Invalid dataset name. Please enter a valid dataset name.")
        sys.exit()

    # Convert targets of CIFAR 10/100 and Tiny-ImageNet to tensors for consistency
    if _dataset_name in [constants.DatasetNames.CIFAR_10, constants.DatasetNames.CIFAR_100, constants.DatasetNames.TINY_IMAGENET_200]:
        trainset.targets = torch.Tensor(trainset.targets).to(torch.int64)
        testset.targets = torch.Tensor(testset.targets).to(torch.int64)

        # trainset.data = torch.from_numpy(trainset.data).to(torch.uint8)
        # trainset.data = trainset.data.permute(0, 3, 1, 2)
        #
        # testset.data = torch.from_numpy(testset.data).to(torch.uint8)
        # testset.data = testset.data.permute(0, 3, 1, 2)

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
