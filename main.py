"""
This is the main entry point for the simulation. You can run this script

Author: Nisal Hemadasa
Date: 02-04-2025
Version: 2.0
"""
import constants
from federated_network.network import FederatedNetwork

# Prevents the error on CUDA device-side assertion failure, which are likely triggered by invalid tensor operations
# (e.g., NaN, Inf, or out-of-bounds values) during loss computation in the training loop.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

    # Define the drift specifications
    drift_specifications = dict(
        clients_fraction=0.8,  # Fraction of clients that are drift affected(literature also uses a list of fractions)
        # Proportions of the size of the drift affected client groups at each drift_step_rounds.
        # outer list - timesteps
        # inner list: group size proportions
        drift_group_proportions=[[0.5, 0.5],  # At the first drift step - drift_step_rounds[0]
                                 [0.3, 0.7],  # drift_step_rounds[1]
                                 [0.8, 0.2]],
        # drift_group_proportions=[[0.5, 0.5],  # At the first drift step - drift_step_rounds[0]
        #                          [0.5, 0.5],  # drift_step_rounds[1]
        #                          [0.5, 0.5]],
        # drift_group_proportions=[[0.5, 0.5],  # Scenario A
        #                          [0.2, 0.8]],
        # drift_group_proportions=[[0.1, 0.9],  # Scenario B
        #                          [0.8, 0.2]],
        # drift_group_proportions=[[1],  # At the first drift step - drift_step_rounds[0]
        #                          [1],  # drift_step_rounds[1]
        #                          [1]],
        # drift_step_rounds[2], the last drift_step_round indicates the end of the drift period.
        is_synchronous=False,  # If the drift is synchronous or asynchronous
        is_random=True, # Whether to randomly select the clients to be affected by the drift
        async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
        # --------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.LABEL_SWAP_ONCE,  # Drift creation method
        # drift_step_rounds=[0.4, 0.6, 0.7, 0.9],
        # # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # #--------------------------------------------------------------------------------
        drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS, # Drift creation method
        drift_step_rounds=[0.4, 0.65, 0.7, 1],  # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
        # drift_step_rounds=[0.4, 0.65, 1],
        # #--------------------------------------------------------------------------------
        # drift_mode=constants.DriftMode.ROTATION_GRADUAL,  # Drift creation method
        # drift_step_rounds=[0.4, 0.65, 0.7, 0.9],   # In Rotation gradual case, this indicates only the start and end of drift period.
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
        # class_pairs_to_swap=[[(1, 2),(3, 4)]],   # label indices (not the class names)
        #-----------------------------------------
        # MNIST, CIFAR-10: for asynchronous drift in clustering algorithms
        drift_pattern_id_map = {
            1: [(1, 2), (3, 4)],
            # 1: [(1, 2)],
            2: [(5, 7)]
        },  # 0 - no drift
        # #-----------------------------------------
        # # CIFAR-100: for asynchronous drift in clustering algorithms
        # drift_pattern_id_map = {
        #     1: [(1, 2), (3, 4), (10, 11), (20, 12), (30, 13), (40, 14)],
        #     2: [(5, 7), (50, 70), (60, 80)]
        # },  # 0 - no drift
        # #-----------------------------------------
        # # Tiny ImageNet200: for asynchronous drift in clustering algorithms
        # drift_pattern_id_map = {
        #     1: [(1, 2), (3, 4), (10, 11), (20, 12), (30, 13), (40, 14), (101, 102), (103, 104), (110, 111), (120, 112), (130, 113), (140, 114)],
        #     2: [(5, 7), (50, 70), (60, 80), (105, 107), (150, 170), (160, 180)]
        # },  # 0 - no drift
        #-----------------------------------------

        # async_class_pairs_to_swap=[[[(1, 2), (3, 4)], [(5, 7)]],  # drift_step_rounds[0] (2-swaps, 1-swap)
        #                            [[(1, 2), (3, 4)], [(1, 2), (3, 4)]],  # drift_step_rounds[1] (2-swap, 2-swaps)
        #                            [[(5, 7)], [(1, 2), (3, 4)]]], # drift_step_rounds[2] (1-swap, 2-swaps)
        drift_patterns_over_time=[[1, 2],
                                 [1, 2],
                                 [2, 1]],   # TODO: test [1,1] case
        # drift_patterns_over_time=[[1, 2],
        #                          [1, 2],
        #                          [1, 2]],   # TODO: test [1,1] case
        # drift_patterns_over_time=[[1, 1],
        #                           [1, 2]],   # Scenario A
        # drift_patterns_over_time=[[1, 2],
        #                           [1, 2]],   # Scenario B

        #--------------------
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
        # Whether servers have test data for evaluation or the server accuracy/loss is got by averaging the client test accuracy/losses
        servers_have_test_data=False
    )

    # 000000000000000000000000000000000000000000000
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,
        # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9,  # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3,  # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=0,  # Number of IID clients in the federated network
        # num_iid_client_instances=100,  # Suggested at FLTA
        num_noniid_client_instances=10,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
        num_training_rounds=20,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters,  # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/test/saved_plots_fedex/',
        log_save_path='logs/swap/test/saved_logs_fedex/')

    print('break')

    # # # # 000000000000000000000000000000000000000000000
    # # # Paper 1 - layer removal experiments
    # # For these experiments, please refer to the modifications done inside the fedex.py amd model.py files
    # # Only CIFAR-10 and MNIST dataset is used for these experiments
    # # # 00000000000000000000000000000000000000000000000
    # #00000000000000 MNIST 0000000000000000000000
    # # case 1: dropping the layer 'fc2' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_1_rm_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_1_rm_fc_2/saved_logs_fedex/')
    # print('Break!!!!!!!')
    #--------------------------------------------------------
    # # case 2: dropping the layer 'fc2' and 'fc1' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_2_rm_fc_1_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_2_rm_fc_1_2/saved_logs_fedex/')

    #--------------------------------------------------------
    # # case 3: dropping all layer except layer 'fc2' (last layer) in FedEx (aggregating only the last layer)
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_3_agg_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_3_agg_fc_2/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 4: add an additional fully connected layer 'fc3' after 'fc2' in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_4_add_fc_3/saved_logs_fedex/')
    #
    # #--------------------------------------------------------
    # # case 5: dropping the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_5_rm_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_5_rm_fc_3/saved_logs_fedex/')
    # print('Break!!!!!!!')
    # #--------------------------------------------------------
    # # case 6: keep only the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_6_agg_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_6_agg_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    #     # # case 7: keep only the layers 'fc2' and 'fc3' (last 2 layers) in FedEx
    #     # fed_net.run_simulation(
    #     #     file_save_path='plots/swap/layer_removal/MNIST/case_7_agg_fc_2_3/saved_plots_fedex/',
    #     #     log_save_path='logs/swap/layer_removal/MNIST/case_7_agg_fc_2_3/saved_logs_fedex/')
    #     # print('Break!!!!!!!')
    #--------------------------------------------------------
    # # case 8: Drop layers 'fc2' (middle layer) with fc3 available in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/MNIST/case_8_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/MNIST/case_8_agg_fc_2_3/saved_logs_fedex/')
    # print('Break!!!!!!!')

    #00000000000000 CIFAR-10 0000000000000000000000
    # # case 1: dropping the layer 'fc2' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_1_rm_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_1_rm_fc_2/saved_logs_fedex/')

    #--------------------------------------------------------
    # # case 2: dropping the layer 'fc2' and 'fc1' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_2_rm_fc_1_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_2_rm_fc_1_2/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 3: dropping all layer except layer 'fc2' (last layer) in FedEx (aggregating only the last layer)
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_3_agg_fc_2/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_3_agg_fc_2/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 4: add an additional fully connected layer 'fc3' after 'fc2' in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 5: dropping the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_4_add_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 6: keep only the layer 'fc3' (last layer) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_6_agg_fc_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_6_agg_fc_3/saved_logs_fedex/')

    # #--------------------------------------------------------
    # # case 7: keep only the layers 'fc2' and 'fc3' (last 2 layers) in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_7_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_7_agg_fc_2_3/saved_logs_fedex/')

    #--------------------------------------------------------
    # # case 8: Drop layers 'fc2' (middle layer) with fc3 available in FedEx
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/layer_removal/CIFAR-10/case_8_agg_fc_2_3/saved_plots_fedex/',
    #     log_save_path='logs/swap/layer_removal/CIFAR-10/case_8_agg_fc_2_3/saved_logs_fedex/')


    # #0000000000000000000000000000000000000
    # # Paper 1  - performance experiments
    # #0000000000000000000000000000000000000
    # # #00000000000000000 MNIST 00000000000000000000
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        # num_iid_client_instances=100,  # Suggested at FLTA
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/MNIST/saved_plots_fedavg/',
        log_save_path='logs/swap/MNIST/saved_logs_fedavg/')


    # #00000000000000000 FedEx 00000000000000000000
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        # num_iid_client_instances=100,  # Suggested at FLTA
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/MNIST/saved_plots_fedex/',
        log_save_path='logs/swap/MNIST/saved_logs_fedex/')

    # # # #00000000000000000 Oracle 00000000000000000000
    # Define drift recovery algorithm related parameters
    drift_recovery_parameters = dict(
        recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
        base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation algorithm used outside the drift period
        fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
        fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
        # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
        #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
        #   - '+1' -> for the non-drift affected client group
        cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
        fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
    )

    # Create a federated network
    fed_net = FederatedNetwork(
        num_iid_client_instances=10,  # Number of IID clients in the federated network
        # num_iid_client_instances=100,  # Suggested at FLTA
        num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
        server_tree_layout=[1],
        # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
        # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
        num_training_rounds=50,  # Number of training rounds (in literature, over 50 rounds are trained.)
        dataset_name=constants.DatasetNames.MNIST,  # Name of the dataset
        noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
        drift_specs=drift_specifications,  # Drift specifications
        simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
        client_select_fraction=1,  # Fraction of clients to be selected for each round
        drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    )

    # Running the simulation
    fed_net.run_simulation(
        file_save_path='plots/swap/MNIST/saved_plots_oracle/',
        log_save_path='logs/swap/MNIST/saved_logs_oracle/')
    print('break!!!!')

    # # # 00000000000000000000000000000000000000000000000000000000000000
    # # # # 0000000000000000000000000 F_MNIST 000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
    #       noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/F_MNIST/saved_plots_fedavg/',
    #     log_save_path='logs/swap/F_MNIST/saved_logs_fedavg/')
    #
    #
    # # #00000000000000000 FedEx 00000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/F_MNIST/saved_plots_fedex/',
    #     log_save_path='logs/swap/F_MNIST/saved_logs_fedex/')
    #
    # # # # #00000000000000000 Oracle 00000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     dataset_name=constants.DatasetNames.F_MNIST,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/F_MNIST/saved_plots_oracle/',
    #     log_save_path='logs/swap/F_MNIST/saved_logs_oracle/')

    # # # 00000000000000000000000000000000000000000000000000000000000000
    # # # 0000000000000000000000000 CIFAR-10 000000000000000000000000000
    # fedex_alpha_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # for idx, _fedex_alpha in enumerate(fedex_alpha_list):
    #     # Define drift recovery algorithm related parameters
    #     drift_recovery_parameters = dict(
    #         recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
    #         base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation algorithm used outside the drift period
    #         fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #         fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #         # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #         #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #         #   - '+1' -> for the non-drift affected client group
    #         cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    #         fedex_alpha=_fedex_alpha,  # EMA weight (alpha) parameter for the FedEx algorithm.
    #     )
    #
    #     # Create a federated network
    #     fed_net = FederatedNetwork(
    #         num_iid_client_instances=10,  # Number of IID clients in the federated network
    #         # num_iid_client_instances=100,  # Suggested at FLTA
    #         num_noniid_client_instances=0,  # Number of non-IID clients in the federated network
    #         server_tree_layout=[1],
    #         # Number of servers at each level of the server tree of depth n = [n, n-1,..., 1]
    #         # num_training_rounds=100,  # In literature, over 50 rounds are trained. FLUID trains 100 rounds
    #         num_training_rounds=100,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #         dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #         drift_specs=drift_specifications,  # Drift specifications
    #         simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #         client_select_fraction=1,  # Fraction of clients to be selected for each round
    #         drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    #     )
    #
    #     # Running the simulation
    #     fed_net.run_simulation(
    #         file_save_path='plots/swap/CIFAR-10/saved_plots_fedex/alpha_' + str(idx+1) + '/',
    #         log_save_path='logs/swap/CIFAR-10/saved_logs_fedex/alpha_' + str(idx+1) + '/')

    # # 0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=100,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/CIFAR-10/saved_plots_oracle/',
    #     log_save_path='logs/swap/CIFAR-10/saved_logs_oracle/')
    #
    # # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=100,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.CIFAR_10,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/CIFAR-10/saved_plots_fedavg/',
    #     log_save_path='logs/swap/CIFAR-10/saved_logs_fedavg/')
    #

    # # # # 00000000000000000000000000000000000000000000000000000000000000
    # # # 0000000000000000000000000 CIFAR-100 000000000000000000000000000
    # # Define the drift specifications
    # drift_specifications = dict(
    #     clients_fraction=0.8,  # Fraction of clients that are drift affected(literature also uses a list of fractions)
    #     # Proportions of the size of the drift affected client groups at each drift_step_rounds.
    #     # outer list - timesteps
    #     # inner list: group size proportions
    #     drift_group_proportions=[[0.1, 0.9],  # Scenario B
    #                              [0.8, 0.2]],
    #     # drift_step_rounds[2], the last drift_step_round indicates the end of the drift period.
    #     is_synchronous=False,  # If the drift is synchronous or asynchronous
    #     is_random=True, # Whether to randomly select the clients to be affected by the drift
    #     async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
    #     # #--------------------------------------------------------------------------------
    #     drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS, # Drift creation method
    #     drift_step_rounds=[0.4, 0.65, 1],
    #     # #-----------------------------------------
    #     # Therefore, it must have at least two entries (start and end of drift).
    #     max_rotation=45,  # Maximum rotation angle for the drift created by rotations
    #     class_pairs_to_swap=[[(1, 2), (3, 4)], [(5, 7)]],  # label indices (not the class names)
    #     # class_pairs_to_swap=[[(1, 2),(3, 4)]],   # label indices (not the class names)
    #     #-----------------------------------------
    #     # # CIFAR-100: for asynchronous drift in clustering algorithms
    #     # drift_pattern_id_map = {
    #     #     1: [(1, 2), (3, 4), (10, 11), (20, 12), (30, 13), (40, 14)],
    #     #     2: [(5, 7), (50, 70), (60, 80)]
    #     # },  # 0 - no drift
    #     # CIFAR-100: for asynchronous drift in clustering algorithms
    #     drift_pattern_id_map = {
    #         1: [(1, 2), (3, 4), (10, 11), (20, 12), (30, 13), (40, 14), (50, 15), (60,16), (70,17), (80,18), (25, 35), (45, 55), (65,75), (85,95)],
    #         2: [(5, 7), (50, 70), (60, 80), (90, 81), (91,82), (92,83), (93,84), (94,85), (96,86), (97,87), (98,88), (99,89)]
    #     },
    #     # #-----------------------------------------
    #     drift_patterns_over_time=[[1, 2],
    #                               [1, 2]],   # Scenario B
    #     #--------------------
    #     label_swap_percentage_steps=[1, 1],  # Percentages to swap per step (label-swapping)
    #     current_drift_step=-1  # Current drift step (used internally during simulation. -1 represents no drift yet)
    # )
    #
    #     # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.CIFAR_100,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/CIFAR-100/saved_plots_fedex/',
    #     log_save_path='logs/swap/CIFAR-100/saved_logs_fedex/')
    #
    # # 0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.CIFAR_100,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/CIFAR-100/saved_plots_oracle/',
    #     log_save_path='logs/swap/CIFAR-100/saved_logs_oracle/')
    #
    # # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.CIFAR_100,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/CIFAR-100/saved_plots_fedavg/',
    #     log_save_path='logs/swap/CIFAR-100/saved_logs_fedavg/')

    # # # 00000000000000000000000000000000000000000000000000000000000000
    # # # 0000000000000000000000000 Tiny-ImageNet 000000000000000000000000000
    # # Define the drift specifications
    # drift_specifications = dict(
    #     clients_fraction=0.8,  # Fraction of clients that are drift affected(literature also uses a list of fractions)
    #     # Proportions of the size of the drift affected client groups at each drift_step_rounds.
    #     # outer list - timesteps
    #     # inner list: group size proportions
    #     drift_group_proportions=[[0.1, 0.9],  # Scenario B
    #                              [0.8, 0.2]],
    #     # drift_step_rounds[2], the last drift_step_round indicates the end of the drift period.
    #     is_synchronous=False,  # If the drift is synchronous or asynchronous
    #     is_random=True, # Whether to randomly select the clients to be affected by the drift
    #     async_drift_specs=async_drift_specs,  # Specifications for the asynchronous case
    #     # --------------------------------------------------------------------------------
    #     # #--------------------------------------------------------------------------------
    #     drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS, # Drift creation method
    #     drift_step_rounds=[0.4, 0.65, 1],  # Rounds at which the drift steps occurs. Also indicates the start and end of drift period.
    #     # #--------------------------------------------------------------------------------
    #     # Therefore, it must have at least two entries (start and end of drift).
    #     max_rotation=45,  # Maximum rotation angle for the drift created by rotations
    #     class_pairs_to_swap=[[(1, 2), (3, 4)], [(5, 7)]],  # label indices (not the class names)
    #     # class_pairs_to_swap=[[(1, 2),(3, 4)]],   # label indices (not the class names)
    #     #-----------------------------------------
    #     # Tiny ImageNet200: for asynchronous drift in clustering algorithms
    #     drift_pattern_id_map = {
    #         1: [(1, 2), (3, 4), (10, 11), (20, 12), (30, 13), (40, 14), (101, 102), (103, 104), (110, 111), (120, 112),
    #             (130, 113), (140, 114), (150, 115), (160, 116), (170, 117), (180, 118), (25, 35), (45, 55), (65,75),
    #             (85,95),
    #             (48, 49), (51, 52), (53, 54), (56, 57), (58, 59),
    #             (61, 62), (63, 64), (66, 67), (68, 69), (71, 72),
    #             (73, 74), (76, 77), (78, 79), (81, 82), (83, 84),
    #             (86, 87), (88, 89), (91, 92), (93, 94), (96, 97),
    #             (161, 162), (163, 164), (165, 166), (167, 168), (169, 171),
    #             (172, 173), (174, 175), (176, 177), (178, 179), (181, 182)],
    #
    #         2: [(5, 7), (50, 70), (60, 80), (105, 107), (8, 9), (15, 16), (17, 18), (19, 21), (22, 23), (24, 26),
    #             (27, 28), (29, 31), (106,108), (6, 32), (33, 34), (36, 37), (38, 39), (41, 42), (43, 44), (46, 47),
    #             (98, 99), (100, 109), (119, 121), (122, 124), (125, 126),
    #             (127, 128), (129, 131), (132, 133), (134, 135), (136, 137),
    #             (138, 139), (141, 142), (143, 144), (145, 146), (147, 148),
    #             (149, 151), (152, 153), (154, 155), (156, 157), (158, 159),
    #             (183, 184), (185, 186), (187, 188), (189, 190), (191, 192),
    #             (193, 194), (195, 196), (197, 198), (199, 160), (173, 174)]
    #     },  # 0 - no drift
    #     #-----------------------------------------
    #     drift_patterns_over_time=[[1, 2],
    #                               [1, 2]],   # Scenario B
    #     #--------------------
    #     label_swap_percentage_steps=[1, 1],  # Percentages to swap per step (label-swapping)
    #     current_drift_step=-1  # Current drift step (used internally during simulation. -1 represents no drift yet)
    # )
    #
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDEX,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.TINY_IMAGENET_200,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/Tiny/saved_plots_fedex/',
    #     log_save_path='logs/swap/Tiny/saved_logs_fedex/')
    #
    # # 0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.ORACLE,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.TINY_IMAGENET_200,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/Tiny/saved_plots_oracle/',
    #     log_save_path='logs/swap/Tiny/saved_logs_oracle/')
    #
    # # # #0000000000000000000000000000000000000
    # # Define drift recovery algorithm related parameters
    # drift_recovery_parameters = dict(
    #     recovery_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation method used during the drift period
    #     base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG,  # Aggregation algorithm used outside the drift period
    #     fedau_alpha=0.9, # EMA weight (alpha) parameter for the FedAU algorithm
    #     fedrc_cluster_count=3, # Number of clusters (K) for the FedRC algorithm
    #     # Number of clusters (K) for the Oracle (multi-global-model-based) algorithm
    #     #   - drift_specifications['drift_group_proportions'][0] -> number of drift affected client groups
    #     #   - '+1' -> for the non-drift affected client group
    #     cluster_count=len(drift_specifications['drift_group_proportions'][0]) + 1,
    # fedex_alpha=0.9,  # EMA weight (alpha) parameter for the FedEx algorithm
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
    #     num_training_rounds=200,  # Number of training rounds (in literature, over 50 rounds are trained.)
    #     dataset_name=constants.DatasetNames.TINY_IMAGENET_200,  # Name of the dataset
    #     noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
    #     drift_specs=drift_specifications,  # Drift specifications
    #     simulation_parameters=simulation_parameters,  # Parameters specifying the simulation scenarios
    #     client_select_fraction=1,  # Fraction of clients to be selected for each round
    #     drift_recovery_parameters=drift_recovery_parameters, # Drift recovery algorithm related parameters
    # )
    #
    # # Running the simulation
    # fed_net.run_simulation(
    #     file_save_path='plots/swap/Tiny/saved_plots_fedavg/',
    #     log_save_path='logs/swap/Tiny/saved_logs_fedavg/')



if __name__ == "__main__":
    main()
