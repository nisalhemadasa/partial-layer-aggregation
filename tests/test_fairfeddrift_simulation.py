"""Small end-to-end CPU check for FairFedDrift scheduling and persisted logs."""
import os
import pickle
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Subset, TensorDataset

import constants
import device_utils

# The simulation receives synthetic datasets below; UCI dataset retrieval is unused.
try:
    import ucimlrepo  # noqa: F401
except ModuleNotFoundError:
    _ucimlrepo_stub = types.ModuleType('ucimlrepo')
    _ucimlrepo_stub.fetch_ucirepo = lambda *args, **kwargs: None
    sys.modules['ucimlrepo'] = _ucimlrepo_stub
    from federated_network.network import FederatedNetwork
else:
    from federated_network.network import FederatedNetwork


class NoDriftDuringSmokeRun:
    """A stable drift clock; the detector test forces one learned split."""
    is_drift = False
    is_drift_end = False
    is_already_applied = False
    is_synchronous = True
    drift_start_round = 100
    drift_end_round = 101
    drift_step_rounds = [100, 101]
    current_drift_step = -1
    current_round = 0
    drifted_client_indices = []
    drift_clustered_client_indices = [[]]
    async_drift_specs = {'drift_groups': None}
    drift_mode = constants.DriftMode.LABEL_SWAP_ONCE
    drift_pattern_id_map = {1: [(0, 1)]}


def split_two(dataset, _count, _distribution=None):
    """Create two stable client partitions from ordered synthetic data."""
    midpoint = len(dataset) // 2
    return [Subset(dataset, range(midpoint)), Subset(dataset, range(midpoint, len(dataset)))]


class FairFedDriftSimulationTests(unittest.TestCase):
    def setUp(self):
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        device_utils._device = self.old_device

    def test_split_merge_history_cadence_evaluation_and_logs(self):
        """Exercise real network training around deterministic split and merge decisions."""
        trainset = TensorDataset(
            torch.cat((torch.zeros(20, 1, 28, 28), torch.ones(20, 1, 28, 28))),
            torch.cat((torch.zeros(20, dtype=torch.long), torch.ones(20, dtype=torch.long)))
        )
        testset = TensorDataset(
            torch.cat((torch.zeros(4, 1, 28, 28), torch.ones(4, 1, 28, 28))),
            torch.cat((torch.zeros(4, dtype=torch.long), torch.ones(4, dtype=torch.long)))
        )
        simulation_parameters = {
            'is_server_adaptability': False,
            'is_plot_client_data_distributions': False,
            'client_ids_to_plot_data_distributions': [],
            'servers_have_test_data': False,
            'client_evaluation_stage': 'local_after_training',
            'drifted_class_metrics_enabled': False,
            'server_metric_weighting': 'uniform',
            'model_distance_logging_enabled': False,
        }
        recovery_parameters = {
            'recovery_method': constants.RecoveryAlgorithm.FAIRFEDDRIFT,
            'base_aggregation_method': constants.RecoveryAlgorithm.FAIRFEDDRIFT,
            'fedau_alpha': 0.9,
            'fedex_alpha': 0.9,
            'fairfeddrift_loss_threshold': 0.1,
            'fairfeddrift_window': 3,
            'fairfeddrift_rounds_per_timestep': 2,
            'fairfeddrift_seed': 19,
        }
        drift_state = {}

        def make_drift(*_args):
            drift_state['drift'] = NoDriftDuringSmokeRun()
            return drift_state['drift']

        def candidate_loss(_model, loader):
            client_marker = float(loader.dataset[0][0].reshape(-1)[0])
            if drift_state['drift'].current_round >= 2 and client_marker > 0.5:
                return 1.0
            return 0.0

        with tempfile.TemporaryDirectory() as output_dir, \
                patch('federated_network.network.load_datasets', return_value=(trainset, testset)), \
                patch('federated_network.network.split_noniid_dataset', side_effect=split_two), \
                patch('federated_network.network.split_iid_dataset', side_effect=split_two), \
                patch('federated_network.network.get_unique_labels_per_subset', return_value=[]), \
                patch('federated_network.network.drift_fn', side_effect=make_drift), \
                patch('federated_network.utils.evaluate_fairfeddrift_loss', side_effect=candidate_loss), \
                patch('strategy.FairFedDrift.utils.evaluate_fairfeddrift_loss', return_value=0.0), \
                patch('federated_network.network.plot_client_performance_vs_rounds'), \
                patch('federated_network.network.plot_server_performance_vs_rounds'), \
                patch('federated_network.network.plot_client_avg_performance_vs_rounds'):
            network = FederatedNetwork(
                num_iid_client_instances=0,
                num_noniid_client_instances=2,
                server_tree_layout=[1],
                num_training_rounds=6,
                dataset_name=constants.DatasetNames.MNIST,
                noniid_partitioning_strategy=constants.DatasetPartitionDistribution.IID,
                drift_specs={},
                simulation_parameters=simulation_parameters,
                drift_recovery_parameters=recovery_parameters,
                client_select_fraction=1.0,
                minibatch_size=2,
                num_local_epochs=1
            )
            path = output_dir + os.sep
            network.run_simulation(file_save_path=path, log_save_path=path)
            strategy_log_path = os.path.join(output_dir, constants.Logs.FAIRFEDDRIFT_STATE_LOG + '.pkl')
            evaluation_log_path = os.path.join(output_dir, constants.Logs.EVALUATION_LOG + '.pkl')
            client_log_path = os.path.join(output_dir, constants.Logs.CLIENT_LOG + '.pkl')
            server_log_path = os.path.join(output_dir, constants.Logs.SERVER_LOG + '.pkl')
            for log_path in (strategy_log_path, evaluation_log_path, client_log_path, server_log_path):
                self.assertTrue(os.path.isfile(log_path), log_path)

            with open(strategy_log_path, 'rb') as file:
                state_log = pickle.load(file)
            events = state_log['events']
            self.assertEqual([event['round'] for event in events], list(range(6)))
            self.assertEqual([event['new_timestep'] for event in events], [True, False, True, False, True, False])
            self.assertFalse(any(event['decisions'][0]['created_cluster'] for event in events[:1]))
            self.assertTrue(any(decision['created_cluster'] and decision['client_id'] == 1
                                for decision in events[2]['decisions']))
            self.assertTrue(events[4]['merges'])
            self.assertEqual(events[4]['active_cluster_ids'], [2])
            self.assertEqual(state_log['final_state']['retained_assignment_history'],
                             {0: {4: 2}, 1: {4: 2}})
            self.assertEqual([round_idx for round_idx, _ in
                              network.fairfeddrift_strategy.client_data_histories[0].records()], [4])
            self.assertTrue(all(not event['decisions'] for event in (events[1], events[3], events[5])))

            with open(evaluation_log_path, 'rb') as file:
                evaluation_log = pickle.load(file)
            self.assertEqual([record['round'] for record in evaluation_log['records']], list(range(6)))
            self.assertEqual([client['fairfeddrift_cluster_id']
                              for client in evaluation_log['records'][-1]['clients']], [2, 2])
            with open(client_log_path, 'rb') as file:
                self.assertEqual(len(pickle.load(file)), 6)
            with open(server_log_path, 'rb') as file:
                self.assertEqual(len(pickle.load(file)), 7)


if __name__ == '__main__':
    unittest.main()
