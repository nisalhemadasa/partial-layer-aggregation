"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 02-04-2025
Version: 2.0
"""
import constants
from federated_network.network import FederatedNetwork


def main():
    async_drift_specs = dict(
        num_drift_groups=2,  # Number of groups of clients that are affected by the drift asynchronously
        drift_groups=None,  # Groups of clients that are affected by the drift asynchronously
        drift_split_round=0.8,  # Times at which the drift is split into multiple asynchronous drifts,
        # Whether to read the asynchronous drift patterns from the ./drift_concepts/scenarios directory
        is_read_scenarios=True,
        # Scenario number for the asynchronous drift patterns defined in the ./drift_concepts/scenarios directory
        scenario_num=1
    )

    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9  # EMA weight (alpha) parameter for the FedAU algorithm
    )

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.6,  # Fraction of clients that are drift affected(literature also uses a list of fractions)
        # Proportions of the size of the drift affected client groups at each drift_step_rounds.
        # outer list - timesteps
        # inner list: group size proportions
        drift_group_proportions=[[0.2, 0.3, 0.5],  # At the first drift step - drift_step_rounds[0]
                                 [0.3, 0.5, 0.2],  # drift_step_rounds[1]
                                 [0.3, 0.2, 0.5]],
        # drift_step_rounds[2], the last drift_step_round indicates the end of the drift period.
        is_synchronous=False,  # If the drift is synchronous or asynchronous
        is_random=True, # Whether to randomly select the clients to be affected by the drift
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        # --------------------------------------------------------------------------------
        drift_mode=constants.DriftMode.LABEL_SWAP_ONCE,  # Drift creation method
        drift_step_rounds=[0.4, 0.6, 0.7, 0.9],
        # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # #--------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS, # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1],  # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # #--------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_GRADUAL,  # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1],    # In Rotation gradual case, this indicates only the start and end of drift period.
        # # # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_GRADUAL_INCREMENTAL,  # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1],    # In Rotation gradual case, this indicates only the start and end of drift period.
        # # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_STEP_INCREMENTAL,  # Drift creation method
        # drift_step_rounds=[0.2, 0.6, 1], # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # --------------------------------------------------------------------------------
        # Therefore, it must have at least two entries (start and end of drift).
        max_rotation=45,  # Maximum rotation angle for the drift created by rotations
        class_pairs_to_swap=[[(1, 2), (3, 4)], [(5, 7)]],  # label indices (not the class names)
        async_class_pairs_to_swap=[[[(1, 2), (3, 4)], [(5, 7)], [(1, 3), (2, 5), (6, 9)]],  # drift_step_rounds[0] (2-swaps, 1-swap, 3-swaps)
                                   [[(5, 7)], [(1, 2), (3, 4)], [(1, 3), (2, 5), (6, 9)]],  # drift_step_rounds[1] (1-swap, 2-swaps, 3-swaps)
                                   [[(5, 7)], [(1, 3), (2, 5), (6, 9)], [(1, 2), (3, 4)]]], # drift_step_rounds[2] (1-swap, 3-swaps, 2-swaps)
        # class_pairs_to_swap=[[(1, 2),(3, 4)]],   # label indices (not the class names)
        # Classes to be swapped in the label-swapping drift method
        # class_pairs_to_swap=[[(1, 2), (5, 7)], [(1, 2), (5, 7)]],  # Classes to be swapped in the label-swapping drift method
        # class_pairs_to_swap=[[('T-shirt/top', 'Pullover'), ('Sandal', 'Sneaker')]],  # Classes to be swapped in F_MNIST
        # class_pairs_to_swap=[[('airplane', 'automobile'), ('bird', 'cat'), ('frog', 'horse')]],  # Classes to be swapped in CIFAR-10
        # class_pairs_to_swap=[[('bicycle', 'bus'), ('camel', 'fox'), ('dolphin', 'roses'), ('clock', 'bed'), ('bridge', 'cloud'), ('rocket', 'oak'), ('snake','crab')]],  # Classes to be swapped in CIFAR-100
        label_swap_percentage_steps=[1, 1],  # Percentages to swap per step (label-swapping)
        current_drift_step=-1  # Current drift step (used internally during simulation. -1 represents no drift yet)
    )

    # Define simulation parameters
    simulation_parameters = dict(
        is_server_adaptability=False,  # Evaluate the adaptability of servers/clients to the data/drift distribution
        is_plot_client_data_distributions=False,  # Whether to plot the client data distributions
        client_ids_to_plot_data_distributions=[0, 1],  # Client IDs whose internal data distributions to be plotted.
        servers_have_test_data=True
        # Whether servers have test data for evaluation or the server accuracy/loss is got by averaging the client test accuracy/losses
    )

    # 000000000000000000000000000000000000000000000
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        cluster_count=3  # Number of clusters (K) for the multi-global-model-based algorithms (FedRC, Oracle)
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        # num_iid_client_instances=100,  # Suggested at FLTA
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
        num_training_rounds=10,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/saved_plots_fedrc/',
        log_save_path='logs/swap/saved_logs_fedrc/')

    # # 000000000000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDAU,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3 # Number of clusters (K) for the FedRC algorithm
    # )
    #
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_iid_client_instances=10,  # Number of IID clients in the federated network
    #     # num_iid_client_instances=100,  # Suggested at FLTA
    #     num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
    #     num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/rot_gradual/saved_plots_fedau/',
    #     log_save_path='logs/rot_gradual/saved_logs_fedau/')
    #
    # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.RRT,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3 # Number of clusters (K) for the FedRC algorithm
    # )
    #
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_iid_client_instances=10,  # Number of IID clients in the federated network
    #     # num_iid_client_instances=100,  # Suggested at FLTA
    #     num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
    #     num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/rot_gradual/saved_plots_rrt/',
    #     log_save_path='logs/rot_gradual/saved_logs_rrt/')
    #
    # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FLUID,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3 # Number of clusters (K) for the FedRC algorithm
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_iid_client_instances=10,  # Number of IID clients in the federated network
    #     # num_iid_client_instances=100,  # Suggested at FLTA
    #     num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
    #     num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/rot_gradual/saved_plots_fluid/',
    #     log_save_path='logs/rot_gradual/saved_logs_fluid/')
    #
    # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3 # Number of clusters (K) for the FedRC algorithm
    # )
    #
    # # Create a federated network
    # fed_net = FederatedNetwork(
    #     num_iid_client_instances=10,  # Number of IID clients in the federated network
    #     # num_iid_client_instances=100,  # Suggested at FLTA
    #     num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
    #     server_tree_layout=[1],
    #     # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #     # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
    #     num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/rot_gradual/saved_plots_fedex/',
    #     log_save_path='logs/rot_gradual/saved_logs_fedex/')


if __name__ == "__main__":
    main()
