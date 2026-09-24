"""Round/timestep scheduling and bounded FairFedDrift history checks."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import constants
from federated_network.utils import prepare_fairfeddrift_timestep
from strategy.FairFedDrift import aggregator_fn


class FairFedDriftTimestepTests(unittest.TestCase):
    def make_runtime(self, rounds_per_timestep=2, window=100, num_clients=2):
        strategy = aggregator_fn({
            'fairfeddrift_loss_threshold': 0.5,
            'fairfeddrift_rounds_per_timestep': rounds_per_timestep,
            'fairfeddrift_window': window
        })
        server = SimpleNamespace(
            server_id=0, abs_id=0, strategy=strategy, model=nn.Linear(1, 2),
            client_ids=[], child_server_ids=[], parent_server_id=None, drift_id=None,
            multi_models=None, fedex_alpha=1.0, fairfeddrift_strategy=strategy,
            fairfeddrift_cluster_id=0)
        clients = []
        for client_id in range(num_clients):
            dataset = TensorDataset(torch.tensor([[float(client_id)], [float(client_id + 1)]]),
                                    torch.tensor([0, 1]))
            clients.append(SimpleNamespace(
                client_id=client_id, parent_server_id=0, mini_batch_size=2,
                trainloader=DataLoader(dataset, batch_size=2, shuffle=True)))
        return strategy, [server], clients

    def test_assign_once_per_timestep_and_train_only_prior_history(self):
        strategy, servers, clients = self.make_runtime()
        with patch('federated_network.utils.evaluate_fairfeddrift_loss', return_value=0.25) as evaluate:
            first = prepare_fairfeddrift_timestep(servers, clients, 0)
            repeated = prepare_fairfeddrift_timestep(servers, clients, 1)
            second = prepare_fairfeddrift_timestep(servers, clients, 2)

        self.assertTrue(first['new_timestep'])
        self.assertFalse(repeated['new_timestep'])
        self.assertTrue(second['new_timestep'])
        self.assertEqual(evaluate.call_count, 4)
        self.assertEqual([len(first['history_loaders'][client.client_id]) for client in clients], [0, 0])
        self.assertEqual([len(repeated['history_loaders'][client.client_id]) for client in clients], [0, 0])
        self.assertEqual([second['history_sample_counts'][client.client_id][0] for client in clients], [2, 2])
        self.assertEqual(strategy.assignment_history, {0: {0: 0, 2: 0}, 1: {0: 0, 2: 0}})
        self.assertEqual([round_idx for round_idx, _ in strategy.client_data_histories[0].records()], [0, 2])
        self.assertEqual([event['round'] for event in strategy.runtime_history], [0, 1, 2])
        self.assertEqual(strategy.runtime_history[0]['decisions'][0]['data_sample_count'], 2)
        self.assertEqual(strategy.runtime_history[1]['decisions'], [])
        self.assertEqual(strategy.runtime_history[2]['history_upload_sample_counts'][0], {0: 2})

    def test_drift_state_does_not_change_configured_timestep_cadence(self):
        strategy, servers, clients = self.make_runtime(num_clients=1)
        with patch('federated_network.utils.evaluate_fairfeddrift_loss', return_value=0.25) as evaluate:
            initial = prepare_fairfeddrift_timestep(servers, clients, 0)
            # Drift state is deliberately not an input to this scheduler.
            within_timestep = prepare_fairfeddrift_timestep(servers, clients, 1)
            next_timestep = prepare_fairfeddrift_timestep(servers, clients, 2)

        self.assertTrue(initial['new_timestep'])
        self.assertFalse(within_timestep['new_timestep'])
        self.assertTrue(next_timestep['new_timestep'])
        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(next_timestep['timestep_start_round'], 2)
        self.assertEqual(next_timestep['history_sample_counts'][0][0], 2)

    def test_round_window_expiry_removes_assignment_references(self):
        strategy, servers, clients = self.make_runtime(rounds_per_timestep=1, window=2, num_clients=1)
        with patch('federated_network.utils.evaluate_fairfeddrift_loss', return_value=0.25):
            prepare_fairfeddrift_timestep(servers, clients, 0)
            prepare_fairfeddrift_timestep(servers, clients, 1)
            current = prepare_fairfeddrift_timestep(servers, clients, 2)

        self.assertEqual([round_idx for round_idx, _ in strategy.client_data_histories[0].records()], [1, 2])
        self.assertEqual(strategy.assignment_history[0], {1: 0, 2: 0})
        self.assertEqual(current['history_sample_counts'][0][0], 2)


if __name__ == '__main__':
    unittest.main()
