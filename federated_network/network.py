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
from drift_concepts.drift import drift_fn
from federated_network.client import client_fn, Client, client_initial_training
from federated_network.server import server_fn, model_aggregation, model_distribution_fedex, \
    server_hierarchy_evaluate, model_distribution_fedrc, model_distribution_hierarchy
from federated_network.utils import update_progress, link_server_hierarchy, train_client_models, \
    link_clients_to_servers, handle_drift_for_round, apply_drift_to_clients
from log_utils.analysis_functions import compute_client_average_metrics, compute_server_average_metrics, \
    split_clients_loss_and_accuracy, convert_fedrc_metrics_to_pairs
from log_utils.logging import write_logs
from plot_utils.plotting import plot_client_performance_vs_rounds, plot_server_performance_vs_rounds, \
    plot_dataset_distribution, \
    plot_client_avg_performance_vs_rounds


class FederatedNetwork:
    def __init__(self, num_iid_client_instances, num_noniid_client_instances, server_tree_layout, num_training_rounds,
                 dataset_name, noniid_partitioning_strategy, drift_specs, simulation_parameters,
                 drift_recovery_parameters, client_select_fraction=0.5, minibatch_size=128, num_local_epochs=5):
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

        # Partition the data set into subsets for each client, following a distribution strategy
        # For Personalized FL, we want to know how well does each client's model perform on data it would actually see.
        # Giving each client a local test split that mirrors its training distribution achieves this (Hussaini et al. 2025, Jiang & Lin 2022, Lee et al. 2025)
        # For performance measures on single global model-global ground truth performance measures:  want to know how
        # good is this model overall. Keep the test set IID/global achieves this (Zhao et al. 2028, Li at al. 2021).
        partitioned_noniid_trainsets = split_noniid_dataset(self.trainset, self.num_client_instances,
                                                            noniid_partitioning_strategy)
        partitioned_noniid_testsets = split_noniid_dataset(self.testset, self.num_client_instances,
                                                           noniid_partitioning_strategy)
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

        # Concept drift properties
        self.drift = drift_fn(self.num_client_instances, num_training_rounds, drift_specs)

        # Simulation parameters
        self.simulation_parameters = simulation_parameters

        # Drift recovery parameters
        self.drift_recovery_parameters = drift_recovery_parameters

        # Determine the cluster count based on the recovery method
        if self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.FEDRC:
            _cluster_count = self.drift_recovery_parameters['fedrc_cluster_count']
        elif self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
            _cluster_count = self.drift_recovery_parameters['cluster_count']
            server_tree_layout = [
                _cluster_count]  # Oracle method uses a flat server structure, with each server representing a cluster base
        else:
            # For all the other strategies, cluster size is 1, if not specified otherwise
            _cluster_count = 1

        # Create client instances
        self.noniid_clients = [
            client_fn(
                i,
                False,  # Non-IID clients
                self.num_local_epochs,
                self.minibatch_size,
                [partitioned_noniid_trainsets[i], partitioned_noniid_testsets[i]],
                self.drift_recovery_parameters['recovery_method'],
                _cluster_count,
                # for FedRC, each client maintains the same number of local models as the multiple global models the server maintains
                dataset_name
            )
            for i in range(num_noniid_client_instances)
        ]

        self.iid_clients = [
            client_fn(
                i + num_noniid_client_instances,
                True,  # IID clients
                self.num_local_epochs,
                self.minibatch_size,
                [partitioned_iid_trainsets[i], partitioned_iid_testsets[i]],
                self.drift_recovery_parameters['recovery_method'],
                _cluster_count,
                dataset_name
            )
            for i in range(num_iid_client_instances)
        ]

        # Create instances for servers (Create multiple servers)
        server_hierarchy = []
        absolute_index = 0

        for depth_level in range(len(server_tree_layout)):
            # For each level in the tree, create a list of server instances, by passing the absolute index
            servers_at_level = [
                server_fn(
                    server_id,
                    self.dataset_name,
                    absolute_index + i,
                    self.drift_recovery_parameters['recovery_method'],
                    self.drift_recovery_parameters['fedex_alpha'],
                    _cluster_count
                )
                for i, server_id in enumerate(range(_cluster_count))]

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
        initial_client_loss_and_accuracy = client_initial_training(self.clients, self.drift.is_drift,
                                                                   self.drift.is_drift_end,
                                                                   self.drift_recovery_parameters['recovery_method'])
        # clients_loss_and_accuracy.append(initial_client_loss_and_accuracy)

        # Load the test set for server evaluation
        server_test_set = convert_dataset_to_loader(_dataset=self.testset, _batch_size=self.minibatch_size)

        for _round in range(self.num_training_rounds + 1):
            # 1. Add drift to the clients
            handle_drift_for_round(_round, self.drift, self.server_hierarchy, self.drift_recovery_parameters,
                                   self.clients)

            # 2. Sample clients participating in the current round TODO: transfer the following part ot a separate module
            # Clients sampled for a single round. In this simulation, all clients are sampled, in order (not randomly)
            sampled_clients = self.clients

            # Extract the sampled client IDs and store them
            sampled_client_ids = [client.client_id for client in sampled_clients]
            sampled_clients_in_each_round.append(sampled_client_ids)

            # 2. Server aggregation (upwards): Aggregate client model parameters to the edge model and edge model
            # parameters to the global model (returns the round_server_loss_and_accuracy, global_avg_loss_and_accuracy
            # after aggregating upwards, before the distribution stage)
            _ = model_aggregation(self.server_hierarchy, server_test_set, sampled_clients, self.drift,
                                  self.drift_recovery_parameters['fedau_alpha'],
                                  self.simulation_parameters['servers_have_test_data'])

            # If the clients download the model from the leaf servers of the hierarchy
            server_depth = len(self.server_hierarchy) - 1

            # 3. Updating (downwards) & evaluation: update the edge models using the global model parameters. (returns the
            # round_server_loss_and_accuracy, global_avg_loss_and_accuracy after both aggregating upwards and
            # distribution stage)
            global_server = self.server_hierarchy[0][
                0]  # TODO: needs to change in case of a hierarchical server structure
            if global_server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDEX:
                model_distribution_fedex(self.server_hierarchy[server_depth], self.clients)
                # model_distribution_hierarchy(self.server_hierarchy)
            elif global_server.strategy.strategy_name == constants.RecoveryAlgorithm.FEDRC:
                model_distribution_fedrc(self.server_hierarchy[server_depth], sampled_clients)
            else:
                model_distribution_hierarchy(self.server_hierarchy)

            # 4. Evaluate server loss and accuracy after aggregation and distribution
            round_server_loss_and_accuracy = server_hierarchy_evaluate(self.server_hierarchy, server_test_set,
                                                                       self.clients,
                                                                       self.simulation_parameters[
                                                                           'servers_have_test_data'],
                                                                       self.drift_recovery_parameters[
                                                                           'recovery_method'])
            server_loss_and_accuracy.append(round_server_loss_and_accuracy)

            # Additional: Update the progress of the simulation
            update_progress(_round=_round, num_training_rounds=self.num_training_rounds)

            # Break the federated learning training round.
            # A training round in FL is defined as follows:
            # 1. Server → Clients
            # 2. Clients (local training) & evaluation
            # 3. Clients → Server
            # 4. Server aggregation & evaluation
            if _round == self.num_training_rounds:
                break

            # 6. Apply drift to clients before local training
            apply_drift_to_clients(self.drift, self.clients)

            # 7. set server parameters to clients before local training
            # TODO: implement this by transferring the part of the code from train_client_models function to here

            # 8. Client local training and evaluation
            round_client_loss_and_accuracy = train_client_models(self.clients,
                                                                 sampled_client_ids,
                                                                 self.server_hierarchy[server_depth],
                                                                 self.drift,
                                                                 self.simulation_parameters,
                                                                 self.drift_recovery_parameters['recovery_method'])
            clients_loss_and_accuracy.append(round_client_loss_and_accuracy)

        # Stop the timer
        end_time = time.time()
        minutes, secs = divmod(end_time - start_time, 60)
        print(f"Runtime: {minutes} minutes {secs} seconds")

        if self.drift_recovery_parameters['recovery_method'] in [constants.RecoveryAlgorithm.FEDRC]:
            # ============================
            #          CLIENTS
            # ============================
            # Concert round_client_loss_and_accuracy of each cluster into a format compatible with logging functions
            cluster_clients_loss_and_accuracy = convert_fedrc_metrics_to_pairs(clients_loss_and_accuracy,
                                                                               self.drift_recovery_parameters[
                                                                                   'cluster_count'])
            file_save_path_stem = file_save_path
            log_save_path_stem = log_save_path
            for cluster_idx, clients_loss_and_accuracy in enumerate(cluster_clients_loss_and_accuracy):
                # TODO: the following part also needs refactoring
                file_save_path = file_save_path_stem + f"cluster_{cluster_idx}/" if file_save_path is not None else None

                # =========PLOTTING FUNCTION CALLS==============
                # plot data distribution of a given list of clients
                if self.simulation_parameters['is_plot_client_data_distributions']:
                    client_ids = self.simulation_parameters['client_ids_to_plot_data_distributions']
                    clients_to_plot = [self.clients[i] for i in client_ids]
                    plot_dataset_distribution(clients_to_plot, self.dataset_name, file_save_path=file_save_path)

                # # Plot layer-distance (between the layers of the client model and the corresponding edge server model)
                # plot_client_layer_distance_vs_rounds(client_layer_distance, file_save_path=file_save_path)

                # # Plot client-edgeserver-distance (overall L2 distance between client model weights and the edge model weights)
                # plot_client_distance_vs_rounds(client_model_distance, file_save_path=file_save_path)

                # Plot the performance of the clients
                plot_client_performance_vs_rounds(clients_loss_and_accuracy, file_save_path=file_save_path)

                # Plot the performance of the server hierarchy
                plot_server_performance_vs_rounds(server_loss_and_accuracy, file_save_path=file_save_path)

                # Get average performance of the clients    #TODO: fix for fedrc and uncomment
                client_averages = compute_client_average_metrics(clients_loss_and_accuracy)

                # ==========LOGGING FUNCTION CALLS==============
                # Split the client performance to drifted and non-drifted clients
                if self.drift.is_synchronous:
                    non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                        clients_loss_and_accuracy, self.drift.drifted_client_indices, None)
                else:
                    non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                        clients_loss_and_accuracy, self.drift.drifted_client_indices,
                        self.drift.async_drift_specs['drift_groups'])

                # Get average performance of the clients
                non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
                if self.drift.is_synchronous:
                    drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)
                else:
                    drifted_client_averages = []
                    for drited_groups in drifted_clients_loss_and_accuracy:
                        drifted_client_averages.append(compute_client_average_metrics(drited_groups))

                log_save_path = log_save_path_stem + f"cluster_{cluster_idx}/"

                # Log the performance of the clients
                write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
                # Log the performance of the clients separated by drifted and non-drifted
                write_logs(non_drifted_clients_loss_and_accuracy,
                           file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_LOG)
                write_logs(drifted_clients_loss_and_accuracy,
                           file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_LOG)
                # Average performance of the clients
                write_logs(non_drifted_client_averages,
                           file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_AVG_LOG)
                write_logs(drifted_client_averages,
                           file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_AVG_LOG)
                # write_logs(client_averages, file_name=log_save_path + constants.Logs.CLIENT_AVG_LOG)

                # =========PLOTTING FUNCTION CALLS==============
                # Split the average performance of the clients to drifted and non-drifted clients
                non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                    clients_loss_and_accuracy, self.drift.drifted_client_indices)
                non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
                drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)

                # Plot average performances
                plot_client_avg_performance_vs_rounds([non_drifted_client_averages, drifted_client_averages],
                                                      self.drift.is_synchronous,
                                                      file_save_path=file_save_path)
                # plot_server_lvl_avg_performance_vs_rounds(server_level_averages, file_save_path=file_save_path)
                # plot_server_overall_avg_performance_vs_rounds(server_overall_averages, file_save_path=file_save_path)

            # ============================
            #          SERVERS
            # ============================
            # Concert round_client_loss_and_accuracy of each cluster into a format compatible with logging functions
            cluster_server_loss_and_accuracy = convert_fedrc_metrics_to_pairs(server_loss_and_accuracy,
                                                                              self.drift_recovery_parameters[
                                                                                  'cluster_count'])

            for cluster_idx, server_loss_and_accuracy in enumerate(cluster_server_loss_and_accuracy):
                # data formatting modification for server metrics
                server_loss_and_accuracy = [[t] for t in server_loss_and_accuracy]

                # ==========logging==================
                # Get average performance of the servers
                server_level_averages, server_overall_averages = compute_server_average_metrics(
                    server_loss_and_accuracy)

                # Log the performance of the server hierarchy
                write_logs(server_loss_and_accuracy, file_name=log_save_path + constants.Logs.SERVER_LOG)
                write_logs(server_level_averages, file_name=log_save_path + constants.Logs.SERVER_LVL_AVG_LOG)
                write_logs(server_overall_averages, file_name=log_save_path + constants.Logs.SERVER_OVERALL_AVG_LOG)

        else:
            # =========PLOTTING FUNCTION CALLS==============
            # plot data distribution of a given list of clients
            if self.simulation_parameters['is_plot_client_data_distributions']:
                client_ids = self.simulation_parameters['client_ids_to_plot_data_distributions']
                clients_to_plot = [self.clients[i] for i in client_ids]
                plot_dataset_distribution(clients_to_plot, self.dataset_name, file_save_path=file_save_path)

            # # Plot layer-distance (between the layers of the client model and the corresponding edge server model)
            # plot_client_layer_distance_vs_rounds(client_layer_distance, file_save_path=file_save_path)

            # # Plot client-edgeserver-distance (overall L2 distance between client model weights and the edge model weights)
            # plot_client_distance_vs_rounds(client_model_distance, file_save_path=file_save_path)

            # Plot the performance of the clients
            plot_client_performance_vs_rounds(clients_loss_and_accuracy, file_save_path=file_save_path)

            # Plot the performance of the server hierarchy
            plot_server_performance_vs_rounds(server_loss_and_accuracy, file_save_path=file_save_path)

            # Get average performance of the clients    #TODO: fix for fedrc and uncomment
            client_averages = compute_client_average_metrics(clients_loss_and_accuracy)

            # ==========LOGGING FUNCTION CALLS==============
            # Split the client performance to drifted and non-drifted clients
            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                    clients_loss_and_accuracy, self.drift.drifted_client_indices, None)
            else:
                # TODO: do not implement yet. Implement only if needed
                pass
                # non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                #     clients_loss_and_accuracy, self.drift.drifted_client_indices,
                #     self.drift.async_drift_specs['drift_groups'])

            # Get average performance of the clients
            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
                if self.drift.is_synchronous:
                    drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)
                else:
                    drifted_client_averages = []
                    for drited_groups in drifted_clients_loss_and_accuracy:
                        drifted_client_averages.append(compute_client_average_metrics(drited_groups))
            else:
                # TODO: do not implement yet. Implement only if needed
                # In Oracle (cluster based), we do not separate the drifted from the non-drifted clients. And the
                # average accuracies of each server is the average accuracies of the clients under that server. So
                # additional client performance averaging is not needed.
                pass

            if log_save_path is None:
                log_save_path = constants.Paths.LOG_SAVE_PATH

            # Log the performance of the clients
            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)
                # Log the performance of the clients separated by drifted and non-drifted
                write_logs(non_drifted_clients_loss_and_accuracy,
                           file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_LOG)
                write_logs(drifted_clients_loss_and_accuracy,
                           file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_LOG)
                # Average performance of the clients
                write_logs(non_drifted_client_averages,
                           file_name=log_save_path + constants.Logs.NON_DRIFTED_CLIENT_AVG_LOG)
                write_logs(drifted_client_averages,
                           file_name=log_save_path + constants.Logs.DRIFTED_CLIENT_AVG_LOG)
                # write_logs(client_averages, file_name=log_save_path + constants.Logs.CLIENT_AVG_LOG)
            else:
                write_logs(clients_loss_and_accuracy, file_name=log_save_path + constants.Logs.CLIENT_LOG)

            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                # Get average performance of the servers
                server_level_averages, server_overall_averages = compute_server_average_metrics(
                    server_loss_and_accuracy)

            # Log the performance of the server hierarchy
            write_logs(server_loss_and_accuracy, file_name=log_save_path + constants.Logs.SERVER_LOG)
            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                write_logs(server_level_averages, file_name=log_save_path + constants.Logs.SERVER_LVL_AVG_LOG)
                write_logs(server_overall_averages, file_name=log_save_path + constants.Logs.SERVER_OVERALL_AVG_LOG)

            # Log drift specifications
            drift_specs = {
                "drift_step_rounds": self.drift.drift_step_rounds,
                "drift_clustered_client_indices": self.drift.drift_clustered_client_indices
            }
            write_logs(drift_specs, file_name=log_save_path + constants.Logs.DRIFT_SPECS_LOG)

            # =========PLOTTING FUNCTION CALLS==============
            if not self.drift_recovery_parameters['recovery_method'] == constants.RecoveryAlgorithm.ORACLE:
                # Split the average performance of the clients to drifted and non-drifted clients
                non_drifted_clients_loss_and_accuracy, drifted_clients_loss_and_accuracy = split_clients_loss_and_accuracy(
                    clients_loss_and_accuracy, self.drift.drifted_client_indices)
                non_drifted_client_averages = compute_client_average_metrics(non_drifted_clients_loss_and_accuracy)
                drifted_client_averages = compute_client_average_metrics(drifted_clients_loss_and_accuracy)

                # Plot average performances
                plot_client_avg_performance_vs_rounds([non_drifted_client_averages, drifted_client_averages],
                                                      self.drift.is_synchronous,
                                                      file_save_path=file_save_path)
                # plot_server_lvl_avg_performance_vs_rounds(server_level_averages, file_save_path=file_save_path)
                # plot_server_overall_avg_performance_vs_rounds(server_overall_averages, file_save_path=file_save_path)
