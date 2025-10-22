"""
Description: This module defines the attributes of the concept drift.

Author: Nisal Hemadasa
Date: 29-10-2024
Version: 1.0
"""
import bisect
import copy
import math
from typing import Dict, List

import numpy as np
import torch
from scipy.ndimage import rotate
import torchvision.transforms as transforms

import constants
from federated_network.client import Client


class Drift:
    def __init__(self, num_drifted_clients, drift_localization_factor, is_synchronous, async_drift_specs, drift_pattern,
                 drift_method, drift_start_round, drift_end_round, drifted_client_indices, max_rotation,
                 class_pairs_to_swap):
        # Number of clients to be applied with drifted data
        self.num_drifted_clients = num_drifted_clients

        # Factor to localize the drift to a certain concentrated group of clients. The value ranges from 0 to 1. E.g.,
        # 0.25 indicates that all drifted clients are concentrated to the first 0.25 indices of the clients.
        self.drift_localization_factor = drift_localization_factor

        # If the drift is synchronous or asynchronous
        self.is_synchronous = is_synchronous

        # Drift specifications for asynchronous drift
        self.async_drift_specs = async_drift_specs

        # Drift pattern, i.e., abrupt, gradual, incremental, reoccurring, incr-abrupt-reoc, incr-reoc, out-of-control.
        self.drift_pattern = drift_pattern

        # Label-swapping, rotations
        self.drift_method = drift_method

        # Defines the period in training rounds which drift starts appearing in clients
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round

        # List of clients that have drifted data
        self.drifted_client_indices = drifted_client_indices
        print("Drifted clients: ", self.drifted_client_indices)

        # Maximum rotation angle for the drift created by rotations
        self.max_rotation = max_rotation

        # Classes to be swapped in the label-swapping drift method
        self.class_pairs_to_swap = class_pairs_to_swap

        # Current round of the federated training
        self.current_round = 0

        # Flag ot indicate when the drift is being applied to the client data
        self.is_drift = False

        # Flag to indicate if the drift is applied on the client data at least once before
        self.is_already_applied = False

    def rotate_images(self, clients: List[Client]) -> List[Client]:
        """
        Apply rotation drift to the images of the client dataset. Both the rotation angle and the number of images to
        rotate increase linearly with the number of federated training rounds.
        :param clients: List of Client objects
        :return: List of Client objects with the rotated images in their datasets
        """

        def apply_rotation(dataset, _rotation_angle):
            """
            Apply rotation drift to a fraction of the images.
            :param dataset: Dataset to process
            :param _rotation_angle: Angle of rotation
            :return: Drifted images and original labels
            """
            _images = dataset.data  # Access dataset images
            _labels = dataset.targets  # Access dataset labels
            # num_images_to_rotate = int(_fraction_rotated * len(_images))
            _drifted_images = _images.clone()

            for i in range(len(_images)):
                rotated_image = rotate(_images[i].numpy(), _rotation_angle, reshape=False)
                _drifted_images[i] = torch.tensor(rotated_image)

            return _drifted_images, _labels

        # Calculate rotation parameters
        transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                self.drift_end_round - self.drift_start_round)
        rotation_angle = transition_progress * self.max_rotation
        # rotation_angle = self.max_rotation
        total_rounds = self.drift_end_round - self.drift_start_round + 1
        fraction_rotated = (self.current_round - self.drift_start_round + 1) / total_rounds

        # Check if there are drifted clients
        if self.drifted_client_indices:
            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = apply_rotation(first_drifted_client.local_trainset.dataset, rotation_angle)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = apply_rotation(first_drifted_client.testset.dataset, rotation_angle)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        return clients

    def swap_labels(self, clients: List[Client]) -> List[Client]:
        """
        Swap the labels of the specified classes in the training and testing sets for drifted clients.
        :param clients: List of Client objects
        :return: Updated list of Client objects with swapped labels in their datasets
        """

        def swap_labels_in_dataset(dataset):
            """
            Swap labels in a dataset based on the class pairs to swap.
            :param dataset: Dataset to process
            :return: Updated images and labels tensors
            """
            images = dataset.data  # Access dataset images
            labels = dataset.targets  # Access dataset labels

            for class_a, class_b in self.class_pairs_to_swap:
                indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
                indices_b = (labels == class_b).nonzero(as_tuple=True)[0]

                # Swap the labels
                labels[indices_a] = class_b
                labels[indices_b] = class_a

            return images, labels

        # Check if there are drifted clients
        if self.drifted_client_indices:
            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = swap_labels_in_dataset(first_drifted_client.local_trainset.dataset)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = swap_labels_in_dataset(first_drifted_client.testset.dataset)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        return clients

    # def rotate_images(self, clients: List[Client]) -> List[Client]:
    #     """
    #     Apply rotation drift to the images of the client dataset. Both the rotation angle and the number of images to
    #     rotate increase linearly with the number of federated training rounds.
    #     :param clients: List of Client objects
    #     :return: List of Client objects with the rotated images in their datasets
    #     """
    #
    #     def apply_rotation(dataset, _rotation_angle):
    #         """
    #         Apply rotation drift to a fraction of the images.
    #         :param dataset: Dataset to process
    #         :param _rotation_angle: Angle of rotation
    #         :return: Drifted images and original labels
    #         """
    #         drifted_images = []  # To store rotated images
    #         _labels = []  # To store labels
    #
    #         # Loop through the dataset directly
    #         for image, _label in dataset:
    #             # Convert to PIL Image, apply rotation, and convert back to Tensor
    #             pil_image = transforms.ToPILImage()(image)
    #             rotated_pil_image = rotate(pil_image, angle=_rotation_angle)
    #             rotated_image = transforms.ToTensor()(rotated_pil_image)
    #
    #             # rotation_transform = transforms.RandomRotation(degrees=_rotation_angle)
    #             # rotated_image = rotation_transform(image)
    #
    #             # Append to the drifted images and labels
    #             drifted_images.append(rotated_image)
    #             _labels.append(_label)
    #
    #         # Stack images and labels to create tensors
    #         drifted_images = torch.stack(drifted_images)
    #         _labels = torch.tensor(_labels)
    #
    #         return drifted_images, _labels
    #
    #     def update_dataset(dataset, _images, _labels):
    #         """
    #         Update the dataset's raw data and targets while handling both tensor and NumPy formats.
    #         :param dataset: Dataset to update
    #         :param _images: Rotated images
    #         :param _labels: Corresponding labels
    #         :return: None
    #         """
    #         # Update data
    #         if isinstance(dataset.data, torch.Tensor):
    #             dataset.data = _images  # Keep tensors directly for MNIST
    #         elif isinstance(dataset.data, np.ndarray):
    #             dataset.data = _images.numpy()  # Convert to NumPy for CIFAR-10
    #         else:
    #             raise TypeError("Unsupported data type for dataset.data")
    #
    #         # Update targets
    #         if isinstance(dataset.targets, torch.Tensor):
    #             dataset.targets = _labels  # Keep tensors directly for MNIST
    #         elif isinstance(dataset.targets, list):
    #             dataset.targets = _labels.tolist()  # Convert to list for CIFAR-10
    #         else:
    #             raise TypeError("Unsupported data type for dataset.targets")
    #
    #     # Calculate rotation parameters
    #     transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
    #             self.drift_end_round - self.drift_start_round)
    #     # rotation_angle = transition_progress * self.max_rotation
    #     rotation_angle = self.max_rotation
    #     total_rounds = self.drift_end_round - self.drift_start_round + 1
    #     fraction_rotated = (self.current_round - self.drift_start_round + 1) / total_rounds
    #
    #     # Check if there are drifted clients
    #     if self.drifted_client_indices:
    #         # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
    #         first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])
    #
    #         # Process training dataset
    #         train_images, train_labels = apply_rotation(first_drifted_client.local_trainset.dataset, rotation_angle)
    #         update_dataset(first_drifted_client.local_trainset.dataset, train_images, train_labels)
    #
    #         # Process testing dataset
    #         test_images, test_labels = apply_rotation(first_drifted_client.testset.dataset, rotation_angle)
    #         update_dataset(first_drifted_client.testset.dataset, test_images, test_labels)
    #
    #         for images, labels in first_drifted_client.testset.dataset:
    #             print(images.shape)  # Should be (batch_size, 3, 32, 32)
    #             l =0
    #
    #         img, label = first_drifted_client.testset.dataset[0]
    #         print(img.dtype)  # Ensure it is a valid type (e.g., uint8 or float32)
    #         print(img.shape)
    #
    #         # Assign the updated datasets to all drifted clients, since they share the same data
    #         for idx in self.drifted_client_indices:
    #             clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
    #             clients[idx].testset.dataset = first_drifted_client.testset.dataset
    #
    #     return clients
    #
    # def swap_cifar_labels(self, clients: List[Client]) -> List[Client]:
    #     """
    #     Swap the labels of the specified classes in the training and testing sets for drifted clients.
    #     :param clients: List of Client objects
    #     :return: Updated list of Client objects with swapped labels in their datasets
    #     """
    #
    #     def swap_labels_in_dataset(dataset):
    #         """
    #         Swap labels in a dataset based on the class pairs to swap.
    #         :param dataset: Dataset to process
    #         :return: Updated images and labels tensors
    #         """
    #         swapped_images = []  # To store images
    #         swapped_labels = []  # To store swapped labels
    #
    #         # Loop through the dataset directly
    #         for image, label in dataset:
    #             # Check and swap labels if they belong to specified class pairs
    #             for class_a, class_b in self.class_pairs_to_swap:
    #                 if label == class_a:
    #                     label = class_b  # Swap label to class_b
    #                 elif label == class_b:
    #                     label = class_a  # Swap label to class_a
    #
    #             # Append the image and the potentially swapped label
    #             swapped_images.append(image)
    #             swapped_labels.append(label)
    #
    #         # Stack images and labels into tensors
    #         swapped_images = torch.stack(swapped_images)  # Combine all images into a single tensor
    #         swapped_labels = torch.tensor(swapped_labels)  # Convert labels to a tensor
    #
    #         return swapped_images, swapped_labels
    #
    #         # images = dataset.data  # Access dataset images
    #         # labels = dataset.targets  # Access dataset labels
    #         #
    #         # for class_a, class_b in self.class_pairs_to_swap:
    #         #     indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
    #         #     indices_b = (labels == class_b).nonzero(as_tuple=True)[0]
    #         #
    #         #     # Swap the labels
    #         #     labels[indices_a] = class_b
    #         #     labels[indices_b] = class_a
    #         #
    #         # return images, labels
    #
    #     # Check if there are drifted clients
    #     if self.drifted_client_indices:
    #         # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
    #         first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])
    #
    #         # Process training dataset
    #         train_images, train_labels = swap_labels_in_dataset(first_drifted_client.local_trainset.dataset)
    #         first_drifted_client.local_trainset.dataset.data = train_images
    #         first_drifted_client.local_trainset.dataset.targets = train_labels
    #
    #         # Process testing dataset
    #         test_images, test_labels = swap_labels_in_dataset(first_drifted_client.testset.dataset)
    #         first_drifted_client.testset.dataset.data = test_images
    #         first_drifted_client.testset.dataset.targets = test_labels
    #
    #         # Assign the updated datasets to all drifted clients, since they share the same data
    #         for idx in self.drifted_client_indices:
    #             clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
    #             clients[idx].testset.dataset = first_drifted_client.testset.dataset
    #
    #     return clients
    #


def get_clients_with_drift(_num_client_instances: int, _clients_fraction_with_drift: float,
                           drift_localization_factor: float, is_synchronous: bool, async_drift_specs: Dict) -> list:
    """
    Get the list of clients that have drifted data.
    :param _num_client_instances: Total number of client instances in the federated network
    :param _clients_fraction_with_drift: Fraction of clients with drifted data
    :param drift_localization_factor: Factor to localize the drift to a certain concentrated group of clients
    :param is_synchronous: Boolean indicating if the drift is synchronous or asynchronous
    :param async_drift_specs: Dictionary containing the specifications for asynchronous drift
    :return: Indices of clients with drifted data
    """
    num_clients_with_drift = int(_clients_fraction_with_drift * _num_client_instances * drift_localization_factor)
    # The cohort of clients with the possibility of drift occurrence
    _num_client_cohort_with_drift = int(_num_client_instances * drift_localization_factor)
    client_indices = torch.randperm(_num_client_cohort_with_drift).tolist()
    drift_clients = client_indices[:num_clients_with_drift]

    if not is_synchronous:
        drifted_client_grouping_margin_indices = [
            math.ceil(i * (_num_client_instances / async_drift_specs['num_drift_groups']))
            for i in range(1, async_drift_specs['num_drift_groups'])
        ]

        # Initialize an empty list to store the grouped lists
        grouped_drifted_clients = []
        # Sort the drifted client indices to ensure proper ordering
        sorted_drift_clients = sorted(drift_clients)
        # Iterate over the drifted_client_grouping_margin_indices and progressively build the groups
        current_group = []

        # Each element in 'drift_groups' is a cumulative list of all previous groups, plus the new values up to the
        # current margin
        for margin in drifted_client_grouping_margin_indices:
            # Add all client indices that are less than or equal to the current margin
            current_group.extend(idx for idx in sorted_drift_clients if idx < margin and idx not in current_group)
            # Append a copy of the current_group to avoid modifying previous lists
            grouped_drifted_clients.append(current_group[:])
        # Add the final group containing all indices
        grouped_drifted_clients.append(sorted_drift_clients)

        # For testing purposes
        grouped_drifted_clients = [list(set(grouped_drifted_clients[1]) - set(grouped_drifted_clients[0])),
                                   grouped_drifted_clients[1]]

        async_drift_specs['drift_groups'] = grouped_drifted_clients

    return drift_clients


def modify_drifted_client_groups(drift: Drift, _round: int) -> None:
    """"
    Modify the drifted client groups based on the drift specifications.
    :param drift: Drift object
    :param _round: Current training round
    :return: None
    """
    # TODO: This function is only desinged for 2 groups. Generalize for more than 2 groups
    if _round < drift.async_drift_specs['drift_split_round']:
        drift.drifted_client_indices = drift.async_drift_specs['drift_groups'][0]
    else:
        if not drift.drifted_client_indices == drift.async_drift_specs['drift_groups'][1]:
            drift.is_already_applied = False
            drift.drifted_client_indices = drift.async_drift_specs['drift_groups'][1]


def drift_fn(num_client_instances: int, num_training_rounds: int, drift_specs: Dict) -> Drift:
    """
    Create a drift object using the specifications given as inputs.
    :param num_client_instances: Total number of client instances in the federated network
    :param num_training_rounds: Total number of training rounds
    :param drift_specs: Dictionary containing the drift specifications
    :return: Drift object
    """
    # Drift start and end rounds
    drift_start_round = math.ceil(drift_specs['drift_start_round'] * num_training_rounds)
    drift_end_round = math.ceil(drift_specs['drift_end_round'] * num_training_rounds)
    print("Drift start round: ", drift_start_round)
    print("Drift end round: ", drift_end_round)

    # The round which the drift starts to affect on the second group of drifted clients in asynchronous case
    if not drift_specs['is_synchronous']:
        drift_duration = drift_end_round - drift_start_round  # Duration of the drift in rounds
        drift_split_round = math.ceil(
            drift_start_round + drift_specs['async_drift_specs']['drift_split_round'] * drift_duration)
        print("Async drift split round: ", drift_split_round)
        drift_specs['async_drift_specs']['drift_split_round'] = drift_split_round

    return Drift(num_drifted_clients=drift_specs['clients_fraction'] * num_client_instances,
                 drift_localization_factor=drift_specs['drift_localization_factor'],
                 is_synchronous=drift_specs['is_synchronous'],
                 async_drift_specs=drift_specs['async_drift_specs'],
                 drift_pattern=drift_specs['drift_pattern'],
                 drift_method=drift_specs['drift_method'],
                 drift_start_round=math.ceil(drift_specs['drift_start_round'] * num_training_rounds),
                 drift_end_round=math.ceil(drift_specs['drift_end_round'] * num_training_rounds),
                 drifted_client_indices=get_clients_with_drift(num_client_instances, drift_specs['clients_fraction'],
                                                               drift_specs['drift_localization_factor'],
                                                               drift_specs['is_synchronous'],
                                                               drift_specs['async_drift_specs']),
                 max_rotation=drift_specs['max_rotation'],
                 class_pairs_to_swap=drift_specs['class_pairs_to_swap'])


def apply_drift(clients: List[Client], drift: Drift) -> List[Client]:
    """
    Apply drift to the training data of the clients.
    :param clients: List of Client objects
    :param drift: Drift object
    :return: List of Client objects with drifted data (dataloaders)
    """
    if drift.drift_method == constants.DriftCreationMethods.LABEL_SWAPPING:
        # For label swapping, application of drift once in the simulation is sufficient & speeds up the simulation
        if not drift.is_already_applied:
            drift.is_already_applied = True
            return drift.swap_labels(clients)
        else:
            return clients
    elif drift.drift_method == constants.DriftCreationMethods.ROTATION:
        # Since rotation is continuously applied, it is speed-wise optimum to apply drift for sampled data in each round
        for client in clients:
            # Each client samples data for local training from their mutually own (exclusively partitioned) datasets
            client.sample_data()

        return drift.rotate_images(clients)
    else:
        print("Drift method not recognized. No drift applied.")
        return clients
