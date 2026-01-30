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
from torch.utils.data import Dataset

import constants
from data.utils import get_num_classes_from_dataset, get_random_labels, convert_dataset_to_loader
from drift_concepts.drift_scenarios import transpose_drift_pattern, read_drift_pattern_from_csv
from drift_concepts.utils import cluster_client_indices_by_drift_patterns, get_clients_indices_with_drift
from federated_network.client import Client, get_client_by_id


# TODO: restructure the Drift class to multiple handler and util files, for improved readability


class Drift:
    def __init__(self, drifted_clients_fraction, drift_group_proportions, is_synchronous, is_random, async_drift_specs,
                 drift_mode, drift_start_round, drift_end_round, drift_step_rounds, num_client_instances, max_rotation,
                 class_pairs_to_swap, drift_pattern_id_map, drift_patterns_over_time, label_swap_percentage_steps,
                 current_drift_step):
        # Number of clients to be applied with drifted data
        self.num_drifted_clients = int(drifted_clients_fraction * num_client_instances)

        # Proportions are used to calculate the number of clients in each drift group, at each drift step
        self.drift_group_sizes = np.floor(
            np.asarray(drift_group_proportions) * self.num_drifted_clients + 0.5).astype(int).tolist()

        # If the drift is synchronous or asynchronous
        self.is_synchronous = is_synchronous

        # If the drifted clients are selected randomly at the beginning of the simulation
        self.is_random = is_random

        # Drift specifications for asynchronous drift
        self.async_drift_specs = async_drift_specs

        # Drift modes
        # 1. label_swap_one_time -> two classes are swapped on time
        # 2. label_swap_incremental_steps -> two classes are swapped during the first step (e.g., in MNIST: 1,2), and
        # 3. label_swap_incremental_steps -> four classes are swapped during the next step (e.g., in MNIST: 3,4 and 5,6)
        # 4. rotation_gradual -> rotation angles are gradually increased over time. (samples that are rotated are fixed)
        # 5. rotation_gradual_incremental -> rotation angles are gradually increased over time. Also, the number of
        # 6. rotation_gradual_incremental -> samples to be rotated is gradually increased
        # 7. rotation_random_incremental -> rotation angles are randomly changed over time. Also, the number of samples
        # to be rotated is gradually increased
        # 8. rotation_step_incremental -> rotation angles are changed abruptly for a given set of samples in a class
        # (e.g., rotation of the images by  +90 degrees). The number of samples to be rotated is gradually increased
        self.drift_mode = drift_mode

        # Defines the training rounds which drift starts appearing in clients as steps
        self.drift_step_rounds = drift_step_rounds

        # Defines the rounds where drift starts and ends affecting the clients
        self.drift_start_round = drift_start_round
        self.drift_end_round = drift_end_round

        # List of clients that have drifted data
        self.drifted_client_indices = get_clients_indices_with_drift(num_client_instances,
                                                                     self.num_drifted_clients,
                                                                     self.is_synchronous,
                                                                     self.is_random)
        print("Drifted clients: ", self.drifted_client_indices)

        # Maximum rotation angle for the drift created by rotations
        self.max_rotation = max_rotation

        # Classes to be swapped in the label-swapping drift method
        self.class_pairs_to_swap = class_pairs_to_swap

        # Current round of the federated training
        self.current_round = 0

        # Flag ot indicate when the drift is being applied to the client data
        self.is_drift = False

        # Flag ot indicate when the drift is over
        self.is_drift_end = False

        # Flag to indicate if the drift is applied on the client data at least once before
        self.is_already_applied = False

        # Percentage of data samples that needs to be label-swapped in a selected class
        self.label_swap_percentage_steps = label_swap_percentage_steps

        # Current drift step (used internally during simulation)
        self.current_drift_step = current_drift_step

        # -----------------------
        # for asynchronous-clustering based drift patterns
        # -----------------------
        # Map defining which swap pattern to be applied at each drift step
        self.drift_pattern_id_map = drift_pattern_id_map

        # List defining the swap patterns over time
        self.drift_patterns_over_time = drift_patterns_over_time

        # Get unique drift IDs including the no-drift ID (0)
        unique_drift_ids = {0} | {drift_id for timestep in drift_patterns_over_time for drift_id in timestep}
        self.unique_drift_ids = sorted(unique_drift_ids)  # convert to sorted list

        # Get clusters-wise client indices based on the drift patterns
        self.drift_clustered_client_indices = cluster_client_indices_by_drift_patterns(num_client_instances,
                                                                                       self.num_drifted_clients,
                                                                                       self.drift_group_sizes,
                                                                                       self.is_synchronous,
                                                                                       self.async_drift_specs)

        # initialize the drifted_client_indices to the first-timestep's drifted client indices
        self.drifted_client_indices = self.drift_clustered_client_indices[0]

    def rotate_images_gradually_incrementally(self, clients: List[Client]) -> List[Client]:
        """
        Apply rotation drift (gradually) to the images of the client dataset. Both the rotation angle and the number of
        images to rotate increase linearly with the number of federated training rounds (incremental).
        :param clients: List of Client objects
        :return: List of Client objects with the rotated images in their datasets
        """

        def apply_rotation(dataset, _rotation_angle: float, _fraction_rotated: float, seed: int = 42):
            """
            Apply rotation drift to a fraction of the images.
            :param dataset: Dataset to process
            :param _rotation_angle: Angle of rotation
            :param _fraction_rotated: Fraction of images to rotate
            :param seed: Random seed for reproducibility
            :return: Drifted images and original labels
            """
            _images = dataset.data  # Access dataset images
            _labels = dataset.targets  # Access dataset labels
            _drifted_images = copy.copy(_images)

            num_images = len(_images)
            num_images_to_rotate = int(_fraction_rotated * num_images)

            # pick random subset of indices to rotate
            rng = np.random.default_rng(seed)
            rotate_indices = rng.choice(num_images, size=num_images_to_rotate, replace=False)

            # apply rotation only to those indices
            for _idx in rotate_indices:
                # If tensor, convert to NumPy (For MNIST, FashionMNIST)
                if isinstance(_images[_idx], torch.Tensor):
                    img_np = _images[_idx].numpy()
                else:
                    img_np = _images[_idx]  # already NumPy (For CIFAR-10, CIFAR-100)

                rotated_image = rotate(img_np, _rotation_angle, reshape=False)
                _drifted_images[_idx] = torch.tensor(rotated_image)

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
            # TODO: uncomment the following part and modify the implementation to apply drift to each drifted client separately
            # for idx in self.drifted_client_indices:
            #     # get client with matching client_id (drifted client)
            #     client = get_client_by_id(clients, idx)
            #     # Duplicate a copy of the client (not the reference)
            #     drifted_client_copy = copy.deepcopy(client)

            # Identify the first drifted client to process the dataset and duplicate a copy (not the reference)
            first_drifted_client = copy.deepcopy(clients[self.drifted_client_indices[0]])

            # Process training dataset
            train_images, train_labels = apply_rotation(first_drifted_client.local_trainset.dataset, rotation_angle,
                                                        fraction_rotated)
            first_drifted_client.local_trainset.dataset.data = train_images
            first_drifted_client.local_trainset.dataset.targets = train_labels

            # Process testing dataset
            test_images, test_labels = apply_rotation(first_drifted_client.testset.dataset, rotation_angle,
                                                      fraction_rotated)
            first_drifted_client.testset.dataset.data = test_images
            first_drifted_client.testset.dataset.targets = test_labels

            # Assign the updated datasets to all drifted clients, since they share the same data
            for idx in self.drifted_client_indices:
                clients[idx].local_trainset.dataset = first_drifted_client.local_trainset.dataset
                clients[idx].testset.dataset = first_drifted_client.testset.dataset

        return clients

    def rotate_images_gradually(self, clients: List[Client]) -> List[Client]:
        """
        Apply rotation drift to the images of the client dataset. The rotation is applied to all the samples (images) of
         the drifted clients consistently
        constant with the number of federated training rounds.
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
            _drifted_images = copy.copy(_images)

            for _idx in range(len(_images)):
                # If tensor, convert to NumPy (For MNIST, FashionMNIST)
                if isinstance(_images[_idx], torch.Tensor):
                    img_np = _images[_idx].numpy()
                else:
                    img_np = _images[_idx]  # already NumPy (For CIFAR-10, CIFAR-100)

                rotated_image = rotate(img_np, _rotation_angle, reshape=False)
                _drifted_images[_idx] = torch.tensor(rotated_image)

            return _drifted_images, _labels

        # Calculate rotation parameters
        transition_progress = ((self.current_round + 1) - self.drift_start_round) / (
                self.drift_end_round - self.drift_start_round)
        rotation_angle = transition_progress * self.max_rotation

        # Check if there are drifted clients
        if self.drifted_client_indices:
            # TODO: uncomment the following part and modify the implementation to apply drift to each drifted client separately
            # for idx in self.drifted_client_indices:
            #     # get client with matching client_id (drifted client)
            #     client = get_client_by_id(clients, idx)
            #     # Duplicate a copy of the client (not the reference)
            #     drifted_client_copy = copy.deepcopy(client)

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

    def swap_labels(self, clients: List[Client], class_pair_to_swap: list[tuple[int, int]], verbose: bool = False) -> \
            List[Client]:
        """
        Swap the labels of the specified classes in the training and testing sets for drifted clients.
        :param clients: List of Client objects
        :param class_pair_to_swap: Tuple of the pair of classes whose labels should be swapped
        :param verbose: Flag to enable verbose logging
        :return: Updated list of Client objects with swapped labels in their datasets
        """

        # no_grad decorator to avoid tracking in autograd, thereby saving memory and computations
        @torch.no_grad()
        def swap_labels_in_dataset(_dataset, _class_pair_to_swap: list[tuple[int, int]]):
            """
            Swap labels in a dataset based on the class pairs to swap.
            :param _dataset: Dataset to process
            :param _class_pair_to_swap: Tuple of the pair of classes whose labels should be swapped
            :return: Updated images and labels tensors
            """
            labels = _dataset.targets  # Access dataset labels

            for class_a, class_b in _class_pair_to_swap:
                indices_a = (labels == class_a).nonzero(as_tuple=True)[0]
                indices_b = (labels == class_b).nonzero(as_tuple=True)[0]

                # Swap the labels
                labels[indices_a] = class_b
                labels[indices_b] = class_a

            _dataset.targets = labels
            return _dataset.data, _dataset.targets

        def create_auxiliary_dataset(_dataset, _class_pair_to_swap: list[tuple[int, int]]):
            """
            Create an auxiliary dataset with swapped samples assigned to random labels for training the auxiliary model.
            :param _dataset: Training dataset to process
            :param _class_pair_to_swap: Tuple of the pair of classes whose labels should be swapped
            :return: Auxiliary images and labels tensors
            """
            # Shallow copy: cheap, copies only the object shell, not the tensors
            _aux_dataset = copy.copy(_dataset)

            # # Make completely independent copies of the original images and labels and store it in new attributes
            # _aux_dataset.data = copy.copy(_dataset.data)    # compatible with both Tensor and NumPy formats
            # _aux_dataset.targets = _dataset.targets.clone() # performance optimum for tensor format

            # aux_images = _dataset.aux_data  # Access dataset images
            aux_labels = _aux_dataset.targets  # Access dataset labels
            unique_labels = torch.unique(aux_labels).tolist()  # get the list of unique labels in the dataset

            for class_a, class_b in _class_pair_to_swap:
                indices_a = (aux_labels == class_a).nonzero(as_tuple=True)[0]
                indices_b = (aux_labels == class_b).nonzero(as_tuple=True)[0]

                # Swap the labels-to-swap with random labels
                aux_labels[indices_a] = get_random_labels(unique_labels, num_elements=indices_a.size()[0])
                aux_labels[indices_b] = get_random_labels(unique_labels, num_elements=indices_b.size()[0])

            _aux_dataset.targets = aux_labels
            return _aux_dataset

        # Check if there are drifted clients
        if self.drifted_client_indices:
            for idx in self.drifted_client_indices:
                # get client with matching client_id (drifted client)
                client = get_client_by_id(clients, idx)
                # Duplicate a copy of the client (not the reference)
                drifted_client_copy = copy.deepcopy(client)

                # FedAU & FLUID: create dataset with the swapped samples assigning random labels for training the auxiliary model
                if (client.drift_recovery_method == constants.RecoveryAlgorithm.FEDAU or
                        client.drift_recovery_method == constants.RecoveryAlgorithm.FLUID):
                    aux_dataset = create_auxiliary_dataset(drifted_client_copy.local_trainset.dataset,
                                                           class_pair_to_swap)
                    client.aux_trainloader = convert_dataset_to_loader(aux_dataset, client.mini_batch_size)

                # Process training dataset
                train_images, train_labels = swap_labels_in_dataset(drifted_client_copy.local_trainset.dataset,
                                                                    class_pair_to_swap)
                drifted_client_copy.local_trainset.dataset.data = train_images
                drifted_client_copy.local_trainset.dataset.targets = train_labels

                # Process testing dataset
                test_images, test_labels = swap_labels_in_dataset(drifted_client_copy.testset.dataset,
                                                                  class_pair_to_swap)
                drifted_client_copy.testset.dataset.data = test_images
                drifted_client_copy.testset.dataset.targets = test_labels

                client.local_trainset.dataset = drifted_client_copy.local_trainset.dataset
                client.testset.dataset = drifted_client_copy.testset.dataset

                if verbose:
                    print(f"client {idx} train_dataset_id: {id(client.local_trainset.dataset)}")

        if verbose:
            for i, c in enumerate(clients):
                print(f"client {i} train_dataset_id: {id(c.local_trainset.dataset)}")

        return clients


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
    drift_start_round = math.ceil(drift_specs['drift_step_rounds'][0] * num_training_rounds)
    drift_end_round = math.ceil(drift_specs['drift_step_rounds'][-1] * num_training_rounds)
    print("Drift start round: ", drift_start_round)
    print("Drift end round: ", drift_end_round)

    return Drift(drifted_clients_fraction=drift_specs['clients_fraction'],
                 drift_group_proportions=drift_specs['drift_group_proportions'],
                 is_synchronous=drift_specs['is_synchronous'],
                 is_random=drift_specs['is_random'],
                 async_drift_specs=drift_specs['async_drift_specs'],
                 drift_mode=drift_specs['drift_mode'],
                 drift_start_round=drift_start_round,
                 drift_end_round=drift_end_round,
                 drift_step_rounds=[math.ceil(i * num_training_rounds) for i in drift_specs['drift_step_rounds']],
                 num_client_instances=num_client_instances,
                 max_rotation=drift_specs['max_rotation'],
                 class_pairs_to_swap=drift_specs['class_pairs_to_swap'],
                 drift_pattern_id_map=drift_specs['drift_pattern_id_map'],
                 drift_patterns_over_time=drift_specs['drift_patterns_over_time'],
                 label_swap_percentage_steps=drift_specs['label_swap_percentage_steps'],
                 current_drift_step=drift_specs['current_drift_step'])


def apply_drift(clients: List[Client], drift: Drift) -> List[Client]:
    """
    Apply drift to the training data of the clients.
    :param clients: List of Client objects
    :param drift: Drift object
    :return: List of Client objects with drifted data (dataloaders)
    """
    if drift.drift_mode == constants.DriftMode.LABEL_SWAP_ONCE:
        # For label swapping, application of drift once in the simulation is sufficient & speeds up the simulation
        if not drift.is_already_applied:
            print("labels swapped once")
            drift.is_already_applied = True
            class_pair_to_swap = drift.class_pairs_to_swap[0]  # There is only single pair to swap
            return drift.swap_labels(clients, class_pair_to_swap)
        else:
            return clients

    elif drift.drift_mode == constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS:
        if not drift.is_already_applied:
            drift.is_already_applied = True
            if drift.is_synchronous:
                # SYNCHRONOUS: apply drift to all drifted clients based on the current drift step
                class_pair_to_swap = drift.class_pairs_to_swap[drift.current_drift_step]
                drift.swap_labels(clients, class_pair_to_swap)
            else:
                # ORACLE: apply drift to clusters of clients based on the drift patterns defined over time
                original_drifted_client_indices = drift.drifted_client_indices  # save original drifted client indices

                for idx, (_drift_id, _drift_class_pairs_to_swap) in enumerate(drift.drift_pattern_id_map.items()):
                    # drift.drifted_client_indices = original_drifted_client_indices[idx]
                    drift_clustered_clients = [client for client in clients if client.drift_id == _drift_id]
                    drift.drifted_client_indices = [client.client_id for client in drift_clustered_clients]
                    drift.swap_labels(drift_clustered_clients, _drift_class_pairs_to_swap)

                drift.drifted_client_indices = original_drifted_client_indices  # restore original drifted client indices
        return clients

    elif drift.drift_mode == constants.DriftMode.ROTATION_GRADUAL_INCREMENTAL:
        # Incremental (angle & samples) drift happens at every round
        return drift.rotate_images_gradually_incrementally(clients)

    elif drift.drift_mode == constants.DriftMode.ROTATION_GRADUAL:
        # Gradual (angle) drift happens at every round
        return drift.rotate_images_gradually(clients)

    elif drift.drift_mode == constants.DriftMode.ROTATION_STEP_INCREMENTAL:
        # Incremental (angle & samples) drift happens at the defined drift steps
        if not drift.is_already_applied:
            drift.is_already_applied = True
            return drift.rotate_images_gradually_incrementally(clients)
        return clients

    else:
        print("Drift method not recognized. No drift applied.")
        return clients
