"""
Description: This module defines a federated network.

Author: Nisal Hemadasa
Date: 19-10-2024
Version: 1.0
"""
import random
import time
from typing import List

import constants
from data.dataset_loader import load_datasets
from data.utils import convert_dataset_to_loader, split_iid_dataset, split_noniid_dataset, get_unique_labels_per_subset
from federated_network.client import client_fn, Client, client_initial_training
from federated_network.server import server_fn, model_aggregation, model_distribution
from federated_network.utils import update_progress, link_server_hierarchy, train_client_models, link_clients_to_servers
from logs.analysis_functions import compute_client_average_metrics, compute_server_average_metrics, \
    split_clients_loss_and_accuracy
from logs.logging import write_logs
from plots.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds, \
    plot_client_avg_performance_vs_rounds, plot_server_lvl_avg_performance_vs_rounds, \
    plot_server_overall_avg_performance_vs_rounds, plot_client_layer_distance_vs_rounds, plot_client_distance_vs_rounds


class FederatedNetwork:
    def __init__(self, num_iid_client_instances, num_noniid_client_instances, server_tree_layout, num_training_rounds,
                 dataset_name, drift_specs, simulation_parameters, client_select_fraction=0.5, minibatch_size=32,
                 num_local_epochs=4):
        # Dataset name
        self.dataset_name = dataset_name

        # Fraction of clients to be selected for each round (represented by C in originally by McMahan et al. 2017)
        self.client_select_fraction = client_select_fraction

        # Minibatch size for each client (represented by B in originally by McMahan et al. 2017)
        self.minibatch_size = minibatch_size

        # Number of local epochs for each client (represented by E in originally by McMahan et al. 2017)
        self.num_local_epochs = num_local_epochs

        # Number of training rounds
        self.num_training_rounds = num_training_rounds

        # Number of client instances with IID data
        self.num_iid_client_instances = num_iid_client_instances

        # Number of client instances with non-IID data and IID data
        self.num_noniid_client_instances = num_noniid_client_instances
        self.num_iid_client_instances = num_iid_client_instances
        self.num_client_instances = num_iid_client_instances + num_noniid_client_instances

        # Load the dataset
        self.trainset, self.testset = load_datasets(dataset_name)

        # Partition the data set into subsets for each client
        partitioned_noniid_trainsets = split_noniid_dataset(self.trainset, self.num_client_instances)
        partitioned_noniid_testsets = split_noniid_dataset(self.testset, self.num_client_instances)
        partitioned_iid_trainsets = split_iid_dataset(self.trainset, self.num_client_instances)
        partitioned_iid_testsets = split_iid_dataset(self.testset, self.num_client_instances)

        # Pick num_noniid_client_instances number of random client instances from the partitioned non-IID datasets
        # Not implemented
        # Pick num_iid_client_instances number of random client instances from the partitioned IID datasets
        # Not implemented

        # For debugging purposes
        _labels_iid = get_unique_labels_per_subset(self.trainset, partitioned_iid_trainsets)
        _labels_noniid = get_unique_labels_per_subset(self.trainset, partitioned_noniid_trainsets)
        _labels_iid_test = get_unique_labels_per_subset(self.testset, partitioned_iid_testsets)
        _labels_noniid_test = get_unique_labels_per_subset(self.testset, partitioned_noniid_testsets)

        # Simulation parameters
        self.simulation_parameters = simulation_parameters

        # Create client instances
        self.noniid_clients = [
            client_fn(
                i,
                False,  # Non-IID clients
                self.num_local_epochs,
                self.minibatch_size,
                [partitioned_noniid_trainsets[i], partitioned_noniid_testsets[i]]
            )
            for i in range(num_noniid_client_instances)
        ]

        self.iid_clients = [
            client_fn(
                i + num_noniid_client_instances,
                True,  # IID clients
                self.num_local_epochs,
                self.minibatch_size,
                [partitioned_iid_trainsets[i], partitioned_iid_testsets[i]]
            )
            for i in range(num_iid_client_instances)
        ]

        # Create instances for servers at each level of the server tree
        server_hierarchy = []
        absolute_index = 0

        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances, by passing the absolute index
            servers_at_level = [server_fn(server_id, self.dataset_name, absolute_index + i)
                                for i, server_id in enumerate(range(server_tree_layout[depth_level]))]

            server_hierarchy.append(servers_at_level)
            absolute_index += server_tree_layout[depth_level]

        self.server_hierarchy = server_hierarchy

        # Link servers in the hierarchical structure
        link_server_hierarchy(self.server_hierarchy)

        # Distribute the clients to the leaf servers
        link_clients_to_servers(self.server_hierarchy[-1], [self.noniid_clients, self.iid_clients])

        # Combine all clients into a single list
        self.clients = self.noniid_clients + self.iid_clients

    def sample_clients(self) -> List[Client]:
        """ Sample clients from the client pool and returns a list of client instances """
        return random.sample(self.clients, int(self.client_select_fraction * len(self.clients)))

    def run_simulation(self, file_save_path=None, log_save_path=None) -> None:
        """
        Run the simulation for the specified number of rounds
        :param file_save_path: Path to save the logs
        :param log_save_path: Path to save the logs
        :return: None
        """
        clients_loss_and_accuracy = []  # Store the loss and accuracy of the all the clients at each round
        sampled_clients_in_each_round = []  # To keep track of the client IDs sampled in each round
        server_loss_and_accuracy = []  # Store the loss and accuracy at each level of the server hierarchy
        client_model_distance = []  # Store the model distances of the clients at each round
        client_layer_distance = []  # Store the model distances of the clients at each round

        # Start the timer
        start_time = time.time()

        # Train the clients initially using their local data
        initial_client_loss_and_accuracy = client_initial_training(self.clients)
        # clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Load the test set for server evaluation
        server_test_set = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.minibatch_size)

        for _round in range(self.num_training_rounds):
            # Clients sampled for a single round. In this simulation, all clients are sampled, in order (not randomly)
            sampled_clients = self.clients

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # As an example, only one server is considered
            sampled_clients_model_parameters = [sampled_client.model.state_dict() for sampled_client in self.clients]

            # Aggregation (upwards): Aggregate client model parameters to the edge model and edge model parameters to
            # the global model (returns the round_server_loss_and_accuracy, global_avg_loss_and_accuracy after
            # aggregating upwards, before the distribution stage)
            _ = model_aggregation(self.server_hierarchy, sampled_clients_model_parameters, server_test_set)

            # Updating (downwards): update the edge models using the global model parameters. (returns the
            # round_server_loss_and_accuracy, global_avg_loss_and_accuracy after both aggregating upwards and
            # distribution stage)
            round_server_loss_and_accuracy = model_distribution(self.server_hierarchy, server_test_set)

            server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # If the clients download the model from the leaf servers of the hierarchy
            server_depth = len(self.server_hierarchy) - 1

            round_client_loss_and_accuracy = train_client_models(self.clients,
                                                                 sampled_client_ids,
                                                                 self.server_hierarchy[server_depth],
                                                                 self.simulation_parameters)
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)

            # Update the progress of the simulation
            update_progress(_round=_round + 1, num_training_rounds=self.num_training_rounds)

        # Stop the timer
        end_time = time.time()

        print(f"Runtime: {end_time - start_time} seconds")

        # Plot layer-distance (between the layers of the client model and the corresponding edge server model)
        plot_client_layer_distance_vs_rounds(client_layer_distance, file_save_path=file_save_path)

        # Plot client-edgeserver-distance (overall L2 distance between client model weights and the edge model weights)
        plot_client_distance_vs_rounds(client_model_distance, file_save_path=file_save_path)

        # Plot the performance of the clients
        plot_client_performance_vs_rounds(clients_loss_and_accuracy, file_save_path=file_save_path)

        # Plot the performance of the server hierarchy
        plot_server_performance_vs_rounds(server_loss_and_accuracy, file_save_path=file_save_path)

        # Get average performance of the clients
        client_averages = compute_client_average_metrics(clients_loss_and_accuracy)

        if log_save_path is None:
            log_save_path = constants.Paths.LOG_SAVE_PATH

        # Log the performance of the clients
        write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
        # Log the performance of the clients separated by drifted and non-drifted
        write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
        # Average performance of the clients
        write_logs(client_averages, file_name=log_save_path + constants.Logs.CLIENT_AVG_LOG)

        # Get average performance of the servers
        server_level_averages, server_overall_averages = compute_server_average_metrics(server_loss_and_accuracy)

        # Log the performance of the server hierarchy
        write_logs(server_loss_and_accuracy, file_name=log_save_path + constants.Logs.SERVER_LOG)
        write_logs(server_level_averages, file_name=log_save_path + constants.Logs.SERVER_LVL_AVG_LOG)
        write_logs(server_overall_averages, file_name=log_save_path + constants.Logs.SERVER_OVERALL_AVG_LOG)

        # Plot average performances
        # plot_client_avg_performance_vs_rounds([client_averages], self.drift.is_synchronous,
        #                                       file_save_path=file_save_path)
        plot_server_lvl_avg_performance_vs_rounds(server_level_averages, file_save_path=file_save_path)
        plot_server_overall_avg_performance_vs_rounds(server_overall_averages, file_save_path=file_save_path)
