"""
Description: This module defines a client of the federated network.

Author: Nisal Hemadasa
Date: 18-10-2024
Version: 1.0
"""
import copy
import random
from collections import OrderedDict
from typing import List, Dict

import torch
from torch.utils.data import Dataset, Subset, DataLoader

import constants
from data.utils import convert_dataset_to_loader, get_num_classes_from_dataset
from models import ConvNeXtTinyImageNet
from models.CNNCIFAR100.model import ResNet18CIFAR100, ShallowResNetCIFAR100
from models.CNNTinyImageNet.model import ResNet18TinyImageNet, ConvNeXtTinyTinyImageNet
from models.utils import train, test, test_subset_classes, CNNModel, rapid_train, fedau_clientside_train, set_parameters, \
    CNNCIFAR10, CNNCIFAR100, TabularAdultModel
from strategy.FedRC import fedrc
from strategy.Ditto import resolve_ditto_parameters, train_ditto_personal_model

from device_utils import get_device


class Client:
    def __init__(self, client_id, if_iid, model, epochs, mini_batch_size, local_trainset, testset,
                 drift_recovery_method, fedrc_cluster_count, drift_recovery_parameters=None):
        self.client_id = client_id
        self.iid = if_iid  # whether the client has IID data or not
        model = model.to(get_device())
        self.model = model
        self.epochs = epochs
        self.mini_batch_size = mini_batch_size
        self.local_trainset = local_trainset
        self.testset = testset
        self.drift_recovery_method = drift_recovery_method
        self.trainloader = None  # initialized only when sample_data() is called
        self.testloader = None  # initialized only when sample_data() is called
        self.parent_server_id = None  # server ID in the server hierarchy to which the client is connected
        self.auxiliary_classifier_parameters = None  # distance of the client from the server in the server hierarchy
        self.aux_trainloader = None  # dateset with random labels for training the auxiliary classifier in FedAU
        self.drift_id = None  # drift pattern ID assigned to this client (for clustering-based methods, e.g., or Oracle)

        # ===== Ditto specific initializations =====
        self.ditto_parameters = None
        self.ditto_personal_model = None
        self.ditto_initialized = False
        self.ditto_validation_indices = None
        self.ditto_training_indices = None
        self.ditto_validation_loader = None
        self.ditto_selected_lambda = None
        self.ditto_lambda_decisions = []
        if self.drift_recovery_method == constants.RecoveryAlgorithm.DITTO:
            self.ditto_parameters = resolve_ditto_parameters(drift_recovery_parameters or {})
            self.ditto_selected_lambda = self.ditto_parameters['ditto_lambda']

        # ===== FedRC specific initializations =====
        if self.drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
            # Create a list of models of size 'fedrc_cluster_count' (equivalent to the number of models server) in each client for FedRC
            self.fedrc_models = [copy.deepcopy(model) for _ in range(fedrc_cluster_count)]

            self.fedrc_optimizers = [torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9) for model in
                                     self.fedrc_models]

            self.model = None

            # Cluster weights for client i and cluster K (initialized to 1/K, according to the original paper)
            self.omega_i_k = torch.full((fedrc_cluster_count,), 1.0 / fedrc_cluster_count, device=get_device())

            # Sample weights for client i, sample j, and cluster K (initialized to 1/K, according to the original paper)
            self.gamma_i_j_k = torch.full((fedrc_cluster_count,), 1.0 / fedrc_cluster_count, device=get_device())

            # Original paper does not mention the initialization of C_y_k.
            # Following, I(x,y;theta_k) = exp(-f(x,y;theta_k))/C_y_k, we approximate C_y_k to 1.0 initially, since,
            # 1. It causes the denominator in I to not distort exp(-loss) initially,
            # 2. It avoids NaNs or division by zero.
            # Therefore, we start with uniform C_y,k = 1.0 for all labels/clusters
            self.num_classes = get_num_classes_from_dataset(self.local_trainset.dataset)
            # How much of cluster k is made up of label y
            self.C_y_k = torch.ones(self.num_classes, fedrc_cluster_count, device=get_device())
        else:
            self.fedrc_models = None

    def get_model_weights(self):
        """ Get the model weights and biases """
        return self.model.state_dict()

    def sample_data(self):
        """ Sample data from the train and test datasets unique to each client and create DataLoaders"""
        # Create a DataLoader using a randomly sampled subset(fraction_of_data%) from the local training data
        fraction_of_data = 0.1
        subset_size = int(len(self.local_trainset) * fraction_of_data)
        if (self.drift_recovery_method == constants.RecoveryAlgorithm.DITTO and
                self.ditto_parameters['ditto_dynamic_lambda']):
            self._initialize_ditto_validation_partition()
            subset_size = max(1, min(subset_size, len(self.ditto_training_indices)))
            indices = random.sample(self.ditto_training_indices, subset_size)
            validation_subset = Subset(self.local_trainset, self.ditto_validation_indices)
            self.ditto_validation_loader = convert_dataset_to_loader(
                _dataset=validation_subset, _batch_size=self.mini_batch_size, _is_shuffle=False)
        else:
            indices = random.sample(range(len(self.local_trainset)), subset_size)
        subset = Subset(self.local_trainset, indices)

        self.trainloader = convert_dataset_to_loader(_dataset=subset,
                                                     _batch_size=self.mini_batch_size)
        self.testloader = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.mini_batch_size,
                                                    _is_shuffle=False)

    def _initialize_ditto_validation_partition(self) -> None:
        """
        Create one persistent, client-seeded Ditto train/validation partition.
        :return: None.
        """
        if self.ditto_validation_indices is not None:
            return
        local_size = len(self.local_trainset)
        if local_size < 2:
            raise ValueError("Dynamic Ditto lambda selection requires at least two local samples per client.")
        validation_size = max(1, int(local_size * self.ditto_parameters['ditto_validation_fraction']))
        validation_size = min(validation_size, local_size - 1)
        generator = torch.Generator().manual_seed(
            self.ditto_parameters['ditto_validation_seed'] + self.client_id)
        shuffled_indices = torch.randperm(local_size, generator=generator).tolist()
        self.ditto_validation_indices = shuffled_indices[:validation_size]
        self.ditto_training_indices = shuffled_indices[validation_size:]

    def _initialize_ditto_personal_model(self) -> None:
        """
        Initialize Ditto's persistent personalized model from the current upload model once.
        :return: None.
        """
        if self.ditto_personal_model is None:
            self.ditto_personal_model = copy.deepcopy(self.model).to(get_device())
            self.ditto_initialized = True

    @staticmethod
    def _snapshot_model_parameters(model_parameters: OrderedDict) -> OrderedDict:
        """
        Detach and copy a model state for use as an immutable Ditto reference.
        :param model_parameters: Model state dictionary to snapshot.
        :return: Detached cloned state dictionary.
        """
        return OrderedDict((key, value.detach().clone().to(get_device()))
                           for key, value in model_parameters.items())

    def _fit_ditto(self, server_model_parameters: OrderedDict) -> None:
        """
        Train Ditto's upload and persistent personalized models.
        :param server_model_parameters: Current assigned-server state, or None during local warm-up.
        :return: None.
        """
        if server_model_parameters is None:
            # The framework warm-up has no aggregated global reference. Train the upload model normally, then make
            # the one-time personalized copy without applying a synthetic proximal update.
            train(self.model, self.trainloader, _epochs=self.epochs)
            self._initialize_ditto_personal_model()
            return

        global_reference = self._snapshot_model_parameters(server_model_parameters)
        set_parameters(self.model, global_reference)
        self._initialize_ditto_personal_model()

        train(self.model, self.trainloader, _epochs=self.epochs)
        personal_learning_rate = self.ditto_parameters['ditto_learning_rate']
        if personal_learning_rate is None:
            personal_learning_rate = 0.01
        personal_epochs = self.ditto_parameters['ditto_personal_epochs']
        if personal_epochs is None:
            personal_epochs = self.epochs
        self.ditto_lambda_decisions = train_ditto_personal_model(
            self.ditto_personal_model,
            self.trainloader,
            global_reference,
            self.ditto_parameters['ditto_lambda'],
            personal_epochs,
            personal_learning_rate,
            dynamic_lambda=self.ditto_parameters['ditto_dynamic_lambda'],
            lambda_candidates=self.ditto_parameters['ditto_lambda_candidates'],
            validation_loader=self.ditto_validation_loader
        )
        self.ditto_selected_lambda = (self.ditto_lambda_decisions[-1]['selected_lambda']
                                      if self.ditto_lambda_decisions
                                      else self.ditto_parameters['ditto_lambda'])

    def fit(self, _is_drift: bool, _is_drift_end: bool, server_model_parameters: OrderedDict, _client_id: int,
            drift_recovery_method: str, _drifted_client_indices: List[int]) -> None:
        """
        Train the client model using new data and server parameters and return the updated model weights and
        biases
        :param _is_drift: Flag indicating whether drift has occurred or not
        :param _is_drift_end: Flag indicating whether the drift period has ended or not
        :param server_model_parameters: Aggregated server model parameters
        :param drift_recovery_method: Drift recovery method to be used
        :param _drifted_client_indices: List of client indices that have experienced drift
        :param _client_id: Client ID of the given client
        :return: None
        """
        if drift_recovery_method == constants.RecoveryAlgorithm.DITTO:
            self._fit_ditto(server_model_parameters)
            return

        if not _is_drift:
            if _is_drift_end and drift_recovery_method == constants.RecoveryAlgorithm.FLUID:  # after drift ends
                # Rapid retraining (2nd order) + reinitialization of client parameters from the global model from scratch
                rapid_train(self.model, self.trainloader, _epochs=self.epochs, _batch_size=self.mini_batch_size)
            elif drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
                # Train all models in client for the FedRC
                self.omega_i_k, self.C_y_k = fedrc.fit(self.fedrc_models, self.trainloader, self.fedrc_optimizers,
                                                       self.omega_i_k, self.C_y_k,
                                                       self.num_classes)
            else:  # before drift begins
                # Train the client model using new data and server parameters
                train(self.model, self.trainloader, _epochs=self.epochs)
        else:
            # Train the client model using new data and server parameters
            if drift_recovery_method in [constants.RecoveryAlgorithm.FEDAVG, constants.RecoveryAlgorithm.FEDEX,
                                         constants.RecoveryAlgorithm.ORACLE]:
                # Adam-based recovery (1st order) + reinitialization of client parameters from the global model from scratch
                train(self.model, self.trainloader, _epochs=self.epochs)
            elif drift_recovery_method == constants.RecoveryAlgorithm.RRT:
                # Rapid retraining (2nd order) + reinitialization of client parameters from the global model from scratch
                rapid_train(self.model, self.trainloader, _epochs=self.epochs, _batch_size=self.mini_batch_size)
            elif drift_recovery_method in [constants.RecoveryAlgorithm.FEDAU, constants.RecoveryAlgorithm.FLUID]:
                # FedAU client side operations
                self.auxiliary_classifier_parameters = fedau_clientside_train(self.model, self.trainloader,
                                                                              self.aux_trainloader,
                                                                              server_model_parameters,
                                                                              _drifted_client_indices, _client_id,
                                                                              _epochs=self.epochs,
                                                                              _mini_batch_size=self.mini_batch_size)
            elif drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
                # Train all models in client for the FedRC
                self.omega_i_k, self.C_y_k = fedrc.fit(self.fedrc_models, self.trainloader, self.fedrc_optimizers,
                                                       self.omega_i_k, self.C_y_k,
                                                       self.num_classes)

    def evaluate(self, model=None):
        """
        Evaluate a client or supplied server model on this client's validation data.
        :param model: Model to evaluate; defaults to the client's local model.
        :return: Loss and accuracy.
        """
        model = self.model if model is None else model
        loss, accuracy = test(model, self.testloader)
        return float(loss), float(accuracy)

    def evaluate_drifted_classes(self, target_classes: set[int], model=None) -> tuple[float | None, float | None, int]:
        """
        Evaluate a model on drifted classes in this client's validation data.
        :param target_classes: Labels affected by the active drift specification.
        :param model: Model to evaluate; defaults to the client's local model.
        :return: Loss, accuracy, and matching test-sample count.
        """
        model = self.model if model is None else model
        return test_subset_classes(model, self.testloader, target_classes)

    def evaluate_ditto_personalized(self) -> tuple[float, float]:
        """
        Evaluate the persistent Ditto personalized model.
        :return: Personalized-model loss and accuracy.
        """
        if self.ditto_personal_model is None:
            raise ValueError("Ditto personalized model has not been initialized.")
        loss, accuracy = test(self.ditto_personal_model, self.testloader)
        return float(loss), float(accuracy)

    def evaluate_ditto_personalized_drifted_classes(
            self, target_classes: set[int]) -> tuple[float | None, float | None, int]:
        """
        Evaluate the Ditto personalized model on drift-affected classes.
        :param target_classes: Labels affected by the active drift specification.
        :return: Loss, accuracy, and matching test-sample count.
        """
        if self.ditto_personal_model is None:
            raise ValueError("Ditto personalized model has not been initialized.")
        return test_subset_classes(self.ditto_personal_model, self.testloader, target_classes)

    def evaluate_fedrc_models(self):
        """ Evaluate all FedRC models in the client on the validation data and return the loss and accuracy """
        losses = []
        accuracies = []
        for model in self.fedrc_models:
            loss, accuracy = test(model, self.testloader)
            losses.append(float(loss))
            accuracies.append(float(accuracy))
        return losses, accuracies


def client_initial_training(_clients: List[Client], _is_drift: bool, _is_drift_end: bool,
                            _drift_recovery_method: str = None) -> List:
    """
    Train the clients initially using their local data.
    :param _clients: List of client instances
    :param _is_drift: Flag indicating whether drift has occurred or not
    :param _is_drift_end: Flag indicating whether the drift period has ended or not
    :param _drift_recovery_method: Drift recovery method to be used by the client
    :return:  List of loss and accuracy of each client after the initial training
        1. FedRC: [([loss1, loss2,...], [acc1, acc2,...]), ..., ]
        2. Others: [(loss, accuracy), ..., ]
    """
    initial_client_loss_and_accuracy = []
    # All the clients are trained individually using local data initially
    for client in _clients:
        client.sample_data()
        # We assume no drift during initial training. Hence, drift related parameters are set to None
        client.fit(_is_drift, _is_drift_end, None, client.client_id, _drift_recovery_method, None)

        if _drift_recovery_method == constants.RecoveryAlgorithm.FEDRC:
            # Evaluate all FedRC models in the client
            losses, accuracies = client.evaluate_fedrc_models()  # returns ([loss1, loss2,...], [acc1, acc2,...])
            initial_client_loss_and_accuracy.append((losses, accuracies))
        else:
            initial_client_loss_and_accuracy.append(client.evaluate())  # return (loss, accuracy)

    return initial_client_loss_and_accuracy


def change_client_drift_recovery_method(clients: List[Client], drift_recovery_method: str,
                                        drifted_client_indices: List[int]) -> None:
    """
    Change the drift recovery method of the clients.
    :param clients: List of client instances
    :param drift_recovery_method: New drift recovery method to be set
    :param drifted_client_indices: List of client indices that have experienced drift
    """
    if not drift_recovery_method == constants.RecoveryAlgorithm.FEDEX:
        for client in clients:
            if client.client_id in drifted_client_indices:
                client.drift_recovery_method = drift_recovery_method
    else:  # for FedEx      # TODO: here for the FedEx, it has to change for all 3 situations
        for client in clients:  # FedEx variant 1
            if client.client_id in drifted_client_indices:
                client.drift_recovery_method = drift_recovery_method


def set_client_drift_ids(clients: List[Client], drifted_client_indices: List[int], unique_drift_ids: List[int],
                         current_step_drift_patterns: List[int]) -> None:
    """
    Sets/changes the drift pattern IDs of the drifted clients.
    :param clients: List of client instances
    :param drifted_client_indices: List of drift-affected client indices arranged in clusters (each inner list represents a cluster)
    :param unique_drift_ids : List of unique drift IDs
    :param current_step_drift_patterns: List of drift pattern IDs for the current drift timestep
    :return: None
    """
    for client in clients:
        for idx, cluster in enumerate(drifted_client_indices):
            if client.client_id in cluster:
                client.drift_id = current_step_drift_patterns[idx]
                break
            else:
                client.drift_id = unique_drift_ids[0]  # non-drifted clients, drift_ID=0


def get_client_by_id(clients: List[Client], client_id: int) -> Client:
    """
    Get the client instance by its ID.
    :param clients: List of client instances
    :param client_id: Client ID
    :return: Client instance
    """
    for client in clients:
        if client.client_id == client_id:
            return client
    raise ValueError(f"Client with ID {client_id} not found.")


def client_fn(client_id: int, if_iid: bool, num_local_epochs: int, mini_batch_size: int,
              _dataset: List[Dataset], drift_recovery_method: str, fedrc_cluster_count: int,
              dataset_name: str, drift_recovery_parameters=None) -> Client:
    """
    Create a client instances on demand for the optimal use of resources.
    :param client_id: client id
    :param if_iid: whether the client has IID data or not
    :param num_local_epochs: number of local epochs, before being aggregation ready
    :param mini_batch_size: size of the batches for the clients to train on
    :param _dataset: train and test datasets
    :param drift_recovery_method: Drift recovery method to be used by the client
    :param fedrc_cluster_count: number of models (clusters) in the client (is equivalent to the number of multiple
    models (clusters) in the server. Used in FedRC)
    :param dataset_name: name of the dataset
    :param drift_recovery_parameters: Strategy-specific experiment parameters
    :returns Client: A Client instance.
    """
    # Load model
    if dataset_name == constants.DatasetNames.MNIST or dataset_name == constants.DatasetNames.F_MNIST:
        _model = CNNModel()
    elif dataset_name == constants.DatasetNames.CIFAR_10:
        _model = CNNCIFAR10()
    elif dataset_name == constants.DatasetNames.CIFAR_100:
        # _model = ShallowResNetCIFAR100()
        _model = ResNet18CIFAR100()
        # _model = CNNCIFAR100()
    elif dataset_name == constants.DatasetNames.TINY_IMAGENET_200:
        _model = ConvNeXtTinyImageNet()
    elif dataset_name == constants.DatasetNames.ADULT:
        _model = TabularAdultModel()
    else:
        raise ValueError("Unsupported dataset name")

    # Unpacking _dataset (which contains a subset of the complete training set (e.g., MNIST) and the global test set)
    local_trainset, testset = _dataset

    # Create a  single Flower client representing a single organization
    return Client(client_id=client_id, if_iid=if_iid, model=_model, epochs=num_local_epochs,
                  mini_batch_size=mini_batch_size, local_trainset=local_trainset, testset=testset,
                  drift_recovery_method=drift_recovery_method, fedrc_cluster_count=fedrc_cluster_count,
                  drift_recovery_parameters=drift_recovery_parameters)
