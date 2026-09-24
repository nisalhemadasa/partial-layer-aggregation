"""Small end-to-end CPU validation for fixed and dynamic Ditto logging."""

import os
import pickle
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Subset, TensorDataset

import constants
import device_utils
from federated_network.network import FederatedNetwork


class NoDriftDuringSmokeRun:
    """Minimal drift state scheduled beyond the smoke-run rounds."""
    is_drift = False
    is_drift_end = False
    is_already_applied = False
    is_synchronous = True
    drift_start_round = 10
    drift_end_round = 11
    drift_step_rounds = [10, 11]
    current_drift_step = -1
    current_round = 0
    drifted_client_indices = [0]
    drift_clustered_client_indices = [[0]]
    async_drift_specs = {'drift_groups': None}
    drift_mode = constants.DriftMode.LABEL_SWAP_ONCE
    drift_pattern_id_map = {1: [(0, 1)]}


def split_two(dataset, _count, _distribution=None):
    """Split a synthetic dataset into two stable client partitions."""
    midpoint = len(dataset) // 2
    return [Subset(dataset, range(0, midpoint)), Subset(dataset, range(midpoint, len(dataset)))]


class DittoSimulationTests(unittest.TestCase):
    def setUp(self):
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        device_utils._device = self.old_device

    def _run_simulation(self, dynamic_lambda):
        """Run a small Ditto simulation and validate its persistent outputs."""
        generator = torch.Generator().manual_seed(7)
        trainset = TensorDataset(torch.randn(40, 1, 28, 28, generator=generator),
                                 torch.arange(40) % 10)
        testset = TensorDataset(torch.randn(20, 1, 28, 28, generator=generator),
                                torch.arange(20) % 10)
        simulation_parameters = {
            'is_server_adaptability': False,
            'is_plot_client_data_distributions': False,
            'client_ids_to_plot_data_distributions': [],
            'servers_have_test_data': False,
            'client_evaluation_stage': 'local_after_training',
            'drifted_class_metrics_enabled': True,
            'server_metric_weighting': 'uniform',
            'model_distance_logging_enabled': False,
        }
        recovery_parameters = {
            'recovery_method': constants.RecoveryAlgorithm.DITTO,
            'base_aggregation_method': constants.RecoveryAlgorithm.FEDAVG,
            'fedau_alpha': 0.9,
            'fedrc_cluster_count': 1,
            'cluster_count': 1,
            'fedex_alpha': 0.9,
            'ditto_lambda': 0.05,
            'ditto_eval_personalized': True,
            'ditto_dynamic_lambda': dynamic_lambda,
            'ditto_lambda_candidates': [0.0, 0.1],
            'ditto_validation_fraction': 0.2,
            'ditto_validation_seed': 17,
        }

        with tempfile.TemporaryDirectory() as output_dir, \
                patch('federated_network.network.load_datasets', return_value=(trainset, testset)), \
                patch('federated_network.network.split_noniid_dataset', side_effect=split_two), \
                patch('federated_network.network.split_iid_dataset', side_effect=split_two), \
                patch('federated_network.network.get_unique_labels_per_subset', return_value=[]), \
                patch('federated_network.network.drift_fn', return_value=NoDriftDuringSmokeRun()), \
                patch('federated_network.network.plot_client_performance_vs_rounds'), \
                patch('federated_network.network.plot_server_performance_vs_rounds'), \
                patch('federated_network.network.plot_client_avg_performance_vs_rounds'):
            network = FederatedNetwork(
                num_iid_client_instances=0,
                num_noniid_client_instances=2,
                server_tree_layout=[1],
                num_training_rounds=1,
                dataset_name=constants.DatasetNames.MNIST,
                noniid_partitioning_strategy=constants.DatasetPartitionDistribution.IID,
                drift_specs={},
                simulation_parameters=simulation_parameters,
                drift_recovery_parameters=recovery_parameters,
                client_select_fraction=1,
                minibatch_size=2,
                num_local_epochs=1
            )
            log_path = output_dir + os.sep
            network.run_simulation(file_save_path=log_path, log_save_path=log_path)

            expected_logs = [
                constants.Logs.DITTO_STATE_LOG,
                constants.Logs.DITTO_PERSONALIZED_CLIENT_LOG,
                constants.Logs.DITTO_PERSONALIZED_DRIFTED_CLASS_LOG,
                constants.Logs.CLIENT_LOG,
            ]
            if dynamic_lambda:
                expected_logs.append(constants.Logs.DITTO_SELECTED_LAMBDA_LOG)
            for log_name in expected_logs:
                self.assertTrue(os.path.exists(os.path.join(output_dir, log_name + '.pkl')))

            with open(os.path.join(output_dir, constants.Logs.DITTO_PERSONALIZED_CLIENT_LOG + '.pkl'), 'rb') as file:
                personalized_log = pickle.load(file)
            self.assertEqual(len(personalized_log['records']), 1)
            self.assertEqual(len(personalized_log['records'][0]['clients']), 2)
            self.assertTrue(all(record['model_role'] == 'ditto_personalized'
                                for record in personalized_log['records'][0]['clients']))

            with open(os.path.join(output_dir, constants.Logs.DITTO_STATE_LOG + '.pkl'), 'rb') as file:
                state_log = pickle.load(file)
            self.assertTrue(all(client['ditto_initialized'] for client in state_log['clients']))
            expected_validation_count = 4 if dynamic_lambda else 0
            self.assertTrue(all(client['validation_sample_count'] == expected_validation_count
                                for client in state_log['clients']))

            if dynamic_lambda:
                with open(os.path.join(output_dir, constants.Logs.DITTO_SELECTED_LAMBDA_LOG + '.pkl'), 'rb') as file:
                    lambda_log = pickle.load(file)
                self.assertEqual(len(lambda_log['records']), 1)
                for client_record in lambda_log['records'][0]['clients']:
                    self.assertTrue(client_record['decisions'])
                    self.assertIn(client_record['selected_lambda'], [0.0, 0.1])

    def test_fixed_lambda_simulation_writes_personalized_logs(self):
        """Run fixed-lambda warm-up, training, evaluation, and log persistence."""
        self._run_simulation(dynamic_lambda=False)

    def test_dynamic_lambda_simulation_writes_selection_log(self):
        """Run dynamic candidate selection and persist per-client decision details."""
        self._run_simulation(dynamic_lambda=True)


if __name__ == '__main__':
    unittest.main()
