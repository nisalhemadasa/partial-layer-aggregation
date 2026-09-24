"""Run small real-MNIST fixed Ditto, dynamic Ditto, and FedAvg validation experiments."""

import copy
import json
import os
import pickle
import random
from unittest.mock import patch

import numpy as np
import torch

import constants
from data.dataset_loader import load_datasets
from device_utils import configure_device
from federated_network.network import FederatedNetwork


def seed_everything(seed):
    """Seed the validation experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def small_mnist_copy(dataset, sample_count):
    """Create an independent MNIST object backed by real downloaded samples."""
    result = copy.copy(dataset)
    result.data = dataset.data[:sample_count].clone()
    result.targets = dataset.targets[:sample_count].clone()
    return result


def drift_parameters():
    """Return a supported one-group asynchronous label-swap transition."""
    return {
        'clients_fraction': 0.34,
        'drift_group_proportions': [[1.0]],
        'is_synchronous': False,
        'is_random': False,
        'async_drift_specs': {
            'num_drift_groups': 1,
            'drift_groups': None,
            'drift_split_round': 1.0,
            'is_read_scenarios': False,
            'scenario_num': 1,
        },
        'drift_mode': constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS,
        'drift_step_rounds': [0.5, 1.0],
        'max_rotation': 0,
        'class_pairs_to_swap': [[(1, 2)]],
        'drift_pattern_id_map': {1: [(1, 2)]},
        'drift_patterns_over_time': [[1]],
        'label_swap_percentage_steps': [1.0],
        'current_drift_step': -1,
    }


def run_scenario(name, recovery_method, dynamic_lambda, source_trainset, source_testset, output_root):
    """Run one isolated real-MNIST scenario and return final metrics."""
    seed_everything(42)
    trainset = small_mnist_copy(source_trainset, 600)
    testset = small_mnist_copy(source_testset, 300)
    simulation_parameters = {
        'is_server_adaptability': False,
        'is_plot_client_data_distributions': False,
        'client_ids_to_plot_data_distributions': [],
        'servers_have_test_data': False,
        'client_evaluation_stage': 'local_after_training',
        'drifted_class_metrics_enabled': True,
        'server_metric_weighting': 'uniform',
        'model_distance_logging_enabled': True,
        'model_distance_interval': 1,
    }
    recovery_parameters = {
        'recovery_method': recovery_method,
        'base_aggregation_method': constants.RecoveryAlgorithm.FEDAVG,
        'fedau_alpha': 0.9,
        'fedrc_cluster_count': 1,
        'cluster_count': 1,
        'fedex_alpha': 0.9,
        'ditto_lambda': 0.05,
        'ditto_learning_rate': None,
        'ditto_personal_epochs': None,
        'ditto_eval_personalized': True,
        'ditto_dynamic_lambda': dynamic_lambda,
        'ditto_lambda_candidates': [0.01, 0.05, 0.1],
        'ditto_validation_fraction': 0.1,
        'ditto_validation_seed': 42,
    }
    output_dir = os.path.join(output_root, name)
    os.makedirs(output_dir, exist_ok=True)
    output_path = output_dir + os.sep

    with patch('federated_network.network.load_datasets', return_value=(trainset, testset)), \
            patch('federated_network.network.plot_client_performance_vs_rounds'), \
            patch('federated_network.network.plot_server_performance_vs_rounds'), \
            patch('federated_network.network.plot_client_avg_performance_vs_rounds'):
        network = FederatedNetwork(
            num_iid_client_instances=3,
            num_noniid_client_instances=0,
            server_tree_layout=[1],
            num_training_rounds=2,
            dataset_name=constants.DatasetNames.MNIST,
            noniid_partitioning_strategy=constants.DatasetPartitionDistribution.DIRICHLET,
            drift_specs=drift_parameters(),
            simulation_parameters=simulation_parameters,
            drift_recovery_parameters=recovery_parameters,
            client_select_fraction=1,
            minibatch_size=16,
            num_local_epochs=1,
        )
        network.run_simulation(file_save_path=output_path, log_save_path=output_path)

    with open(os.path.join(output_dir, constants.Logs.CLIENT_LOG + '.pkl'), 'rb') as file:
        client_log = pickle.load(file)
    final_upload_accuracy = float(np.mean([metrics[1] for metrics in client_log[-1]]))
    summary = {
        'scenario': name,
        'recovery_method': recovery_method,
        'dynamic_lambda': dynamic_lambda,
        'rounds_logged': len(client_log),
        'final_upload_accuracy': final_upload_accuracy,
        'drift_start_round': 1,
        'drift_end_round': 2,
    }

    if recovery_method == constants.RecoveryAlgorithm.DITTO:
        with open(os.path.join(output_dir, constants.Logs.DITTO_PERSONALIZED_CLIENT_LOG + '.pkl'), 'rb') as file:
            personal_log = pickle.load(file)
        final_personal_records = personal_log['records'][-1]['clients']
        summary['final_personalized_accuracy'] = float(
            np.mean([record['accuracy'] for record in final_personal_records]))
        summary['personalized_rounds_logged'] = len(personal_log['records'])
        if dynamic_lambda:
            with open(os.path.join(output_dir, constants.Logs.DITTO_SELECTED_LAMBDA_LOG + '.pkl'), 'rb') as file:
                lambda_log = pickle.load(file)
            summary['selected_lambdas'] = [
                client['selected_lambda'] for client in lambda_log['records'][-1]['clients']
            ]

    return summary


def main():
    """Run and summarize all real-MNIST validation scenarios."""
    configure_device('cpu')
    source_trainset, source_testset = load_datasets(constants.DatasetNames.MNIST)
    output_root = os.path.join('plots', '.cpu_validation', 'real_mnist_ditto')
    scenarios = [
        ('ditto_fixed', constants.RecoveryAlgorithm.DITTO, False),
        ('ditto_dynamic', constants.RecoveryAlgorithm.DITTO, True),
        ('fedavg_baseline', constants.RecoveryAlgorithm.FEDAVG, False),
    ]
    summaries = [
        run_scenario(name, method, dynamic, source_trainset, source_testset, output_root)
        for name, method, dynamic in scenarios
    ]
    with open(os.path.join(output_root, 'summary.json'), 'w', encoding='utf-8') as file:
        json.dump(summaries, file, indent=2)
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
