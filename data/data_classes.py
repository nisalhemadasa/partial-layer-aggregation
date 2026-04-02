import os
import urllib
import zipfile

import torch
import pandas as pd
import numpy as np

from urllib.request import urlretrieve
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.v2 import Compose
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from ucimlrepo import fetch_ucirepo

import constants


class AdultDataset(Dataset):
    """
    PyTorch Dataset for the UCI Adult dataset, exposing .data and .targets
    similar to torchvision datasets.
    """

    def __init__(self, data: torch.Tensor, targets: torch.Tensor, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform is not None:
            x = self.transform(x)

        return x, y


def _standardize_train_test(X_train: np.ndarray, X_test: np.ndarray, eps: float = 1e-8):
    """
    Standardize using only train statistics.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std < eps] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_test


def getAdult(
        _dataset_name: str,
        test_size: float = 0.2,
        random_state: int = 42,
        transform=None,
        verbose: bool = False,
) -> tuple[AdultDataset, AdultDataset]:
    """
    Loads the UCI Adult dataset using ucimlrepo and returns train/test datasets
    with .data and .targets attributes, similar to torchvision datasets.

    returns:
     - trainset: AdultDataset object containing training data
     - testset: AdultDataset object containing test data
    """

    if verbose:
        print(f"Loading {_dataset_name} from UCI repository...")

    # ---------------- Load dataset ----------------
    adult = fetch_ucirepo(id=2)

    X = adult.data.features.copy()
    y = adult.data.targets.copy()

    # ---------------- Combine and clean ----------------
    df = pd.concat([X, y], axis=1)

    # Replace missing values marked as "?"
    df = df.replace("?", np.nan)
    df = df.dropna()

    target_col = y.columns[0]

    X = df.drop(columns=[target_col])
    y = df[target_col].copy()

    # ---------------- Clean labels ----------------
    # Some versions may contain trailing dots
    y = y.astype(str).str.strip().str.replace(".", "", regex=False)
    y = y.map({"<=50K": 0, ">50K": 1})

    # Drop any rows that failed mapping, just in case
    valid_mask = y.notna()
    X = X.loc[valid_mask]
    y = y.loc[valid_mask].astype(int)

    # ---------------- One-hot encode categorical features ----------------
    X = pd.get_dummies(X)

    # Ensure numeric dtype
    X = X.astype(np.float32)

    # ---------------- Train/test split ----------------
    X_train, X_test, y_train, y_test = train_test_split(
        X.values,
        y.values,
        test_size=test_size,
        random_state=random_state,
        stratify=y.values,
    )

    # ---------------- Standardize ----------------
    X_train, X_test = _standardize_train_test(X_train, X_test)

    # ---------------- Convert to tensors ----------------
    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32)

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    # ---------------- Wrap in dataset objects ----------------
    trainset = AdultDataset(X_train, y_train, transform=transform)
    testset = AdultDataset(X_test, y_test, transform=transform)

    if verbose:
        print("Adult trainset.data shape   :", trainset.data.shape)
        print("Adult trainset.targets shape:", trainset.targets.shape)
        print("Adult testset.data shape    :", testset.data.shape)
        print("Adult testset.targets shape :", testset.targets.shape)

    return trainset, testset


def getTinyImageNet(_dataset_name: str, _transform_tiny_imagenet: Compose, verbose=False) -> tuple[
    ImageFolder, ImageFolder]:
    """
    Loads the Tiny ImageNet-200 dataset. If the dataset is not already downloaded, it downloads and extracts it.
    :return: trainset and testset of Tiny ImageNet-200 as torchvision.datasets.ImageFolder objects.
    :param _dataset_name: Name of the dataset (should be "tiny-imagenet-200")
    :param _transform_tiny_imagenet: Transformations to be applied to the Tiny ImageNet-200 dataset.
    :param verbose: If True, prints additional information during dataset loading.
    returns:
     - trainset: torchvision.datasets.ImageFolder object containing the training data of Tiny ImageNet-200
     - testset: torchvision.datasets.ImageFolder object containing the test data of Tiny ImageNet-200
    """
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

    trainset = ImageFolder(train_dir, transform=_transform_tiny_imagenet)
    testset = ImageFolder(val_dir, transform=_transform_tiny_imagenet)

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

    trainset.data = torch.cat(train_images, dim=0)  # [N_train, 3, 64, 64]
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

    return trainset, testset
