"""Verify configured drift application without driving FairFedDrift strategy state."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import Dataset, Subset

import constants
from data.utils import convert_dataset_to_loader
from drift_concepts.drift import drift_fn
from federated_network.server import change_server_aggregation_strategy
from federated_network.utils import apply_drift_to_clients, handle_drift_for_round
from strategy.FairFedDrift import aggregator_fn


class DriftDataset(Dataset):
    def __init__(self):
        self.data = torch.eye(10)
        self.targets = torch.arange(10)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        return self.data[index], self.targets[index]


class DriftClient:
    def __init__(self, client_id):
        self.client_id = client_id
        self.drift_id = None
        self.parent_server_id = 0
        self.drift_recovery_method = constants.RecoveryAlgorithm.FAIRFEDDRIFT
        self.local_trainset = Subset(DriftDataset(), list(range(10)))
        self.testset = Subset(DriftDataset(), list(range(10)))
        self.refresh_count = 0

    def sample_data(self):
        """Refresh loaders after the real drift code changes dataset references."""
        self.refresh_count += 1
        self.trainloader = convert_dataset_to_loader(self.local_trainset, 10, False)
        self.testloader = convert_dataset_to_loader(self.testset, 10, False)


class FairFedDriftDriftTests(unittest.TestCase):
    def test_fairfeddrift_state_survives_base_and_recovery_switches(self):
        """Restore the same FairFedDrift instance after an alternate phase method."""
        fairfeddrift = aggregator_fn({'fairfeddrift_loss_threshold': 0.1})
        server = SimpleNamespace(strategy=fairfeddrift, fairfeddrift_strategy=fairfeddrift,
                                 fairfeddrift_cluster_id=0)
        hierarchy = [[server]]
        drift = SimpleNamespace(unique_drift_ids=[])
        fairfeddrift.previous_losses[17] = 0.42

        change_server_aggregation_strategy(hierarchy, constants.RecoveryAlgorithm.FEDAVG, drift)
        self.assertIs(server.fairfeddrift_strategy, fairfeddrift)
        change_server_aggregation_strategy(
            hierarchy, constants.RecoveryAlgorithm.FAIRFEDDRIFT, drift,
            {'fairfeddrift_loss_threshold': 0.1})

        self.assertIs(server.strategy, fairfeddrift)
        self.assertIs(server.strategy, server.fairfeddrift_strategy)
        self.assertEqual(server.strategy.previous_losses, {17: 0.42})

    def test_configured_swaps_and_strategy_persistence(self):
        """Apply drift timing and switch from FairFedDrift recovery back to its base."""
        specifications = dict(
            clients_fraction=2 / 3, drift_group_proportions=[[0.5, 0.5]],
            is_synchronous=False, is_random=False, async_drift_specs={'drift_groups': None},
            drift_mode=constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS,
            drift_step_rounds=[0.5, 1], max_rotation=0,
            class_pairs_to_swap=[[(0, 9)], [(2, 6)]],
            drift_pattern_id_map={1: [(0, 9)], 2: [(2, 6)]},
            drift_patterns_over_time=[[1, 2]], label_swap_percentage_steps=[1], current_drift_step=-1)
        # Deliberately different from the synthetic fixture: runtime must honor specs.
        drift = drift_fn(6, 4, specifications)
        self.assertEqual(drift.drift_start_round, 2)
        self.assertEqual(drift.drift_end_round, 4)
        clients = [DriftClient(index) for index in range(6)]
        strategy = aggregator_fn()
        strategy.saved_state = {'sentinel': 7}
        server = SimpleNamespace(strategy=strategy, client_ids=list(range(6)), server_id=0)
        recovery = dict(recovery_method=constants.RecoveryAlgorithm.FAIRFEDDRIFT,
                        base_aggregation_method=constants.RecoveryAlgorithm.FEDAVG)
        with patch('federated_network.utils.change_server_aggregation_strategy') as switch_server, \
                patch('federated_network.utils.change_client_drift_recovery_method') as switch_client, \
                patch('federated_network.utils.link_clients_to_servers_by_drift_id') as oracle_link:
            for round_index in range(5):
                handle_drift_for_round(round_index, drift, [[server]], recovery, clients)
                apply_drift_to_clients(drift, clients)
                for client in clients:
                    expected = list(range(10))
                    if round_index >= 2 and client.client_id < 4:
                        left, right = (0, 9) if client.client_id < 2 else (2, 6)
                        expected[left], expected[right] = expected[right], expected[left]
                    self.assertEqual(client.local_trainset.dataset.targets.tolist(), expected)
                    self.assertEqual(client.testset.dataset.targets.tolist(), expected)
                    self.assertEqual(next(iter(client.trainloader))[1].tolist(), expected)
                    self.assertEqual(client.parent_server_id, 0)
                    self.assertEqual(client.drift_recovery_method, constants.RecoveryAlgorithm.FAIRFEDDRIFT)
                    self.assertTrue(torch.equal(client.local_trainset.dataset.data, torch.eye(10)))
            self.assertEqual([call.args[1] for call in switch_server.call_args_list], [
                constants.RecoveryAlgorithm.FAIRFEDDRIFT, constants.RecoveryAlgorithm.FEDAVG])
            self.assertEqual([call.args[1] for call in switch_client.call_args_list], [
                constants.RecoveryAlgorithm.FAIRFEDDRIFT, constants.RecoveryAlgorithm.FEDAVG])
            oracle_link.assert_not_called()
        self.assertIs(server.strategy, strategy)
        self.assertEqual(strategy.saved_state, {'sentinel': 7})
        self.assertEqual(server.client_ids, list(range(6)))
        self.assertTrue(drift.is_drift_end)
        self.assertFalse(drift.is_drift)
        self.assertEqual([client.refresh_count for client in clients], [5] * 6)

    def test_oracle_base_servers_are_restored_after_fairfeddrift(self):
        """Return to Oracle's original server instances after learned clustering."""
        oracle_servers = [SimpleNamespace(server_id=0), SimpleNamespace(server_id=1)]
        active_fairfeddrift_server = SimpleNamespace(server_id=0)
        server_hierarchy = [[active_fairfeddrift_server]]
        drift = SimpleNamespace(is_drift=True, is_synchronous=False,
                                fairfeddrift_parked_base_servers=oracle_servers,
                                drifted_client_indices=[0])
        clients = [SimpleNamespace(client_id=0)]
        recovery = dict(recovery_method=constants.RecoveryAlgorithm.FAIRFEDDRIFT,
                        base_aggregation_method=constants.RecoveryAlgorithm.ORACLE)

        with patch('federated_network.utils.change_server_aggregation_strategy') as switch_server, \
                patch('federated_network.utils.change_client_drift_recovery_method'), \
                patch('federated_network.utils.link_clients_to_servers_by_drift_id') as relink:
            from federated_network.utils import handle_after_drift_configurations
            handle_after_drift_configurations(drift, server_hierarchy, recovery, clients)

        self.assertEqual(server_hierarchy[-1], oracle_servers)
        self.assertIs(server_hierarchy[-1][0], oracle_servers[0])
        self.assertIs(server_hierarchy[-1][1], oracle_servers[1])
        self.assertFalse(hasattr(drift, 'fairfeddrift_parked_base_servers'))
        self.assertEqual(switch_server.call_args.args[1], constants.RecoveryAlgorithm.ORACLE)
        relink.assert_called_once_with(clients, oracle_servers)


if __name__ == '__main__':
    unittest.main()
