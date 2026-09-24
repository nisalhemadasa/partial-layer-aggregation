"""FairFedDrift server evaluation follows its learned active cluster layout."""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import constants
from federated_network.server import server_hierarchy_evaluate
from federated_network.utils import evaluate_clients_for_stage


def make_server(client_ids, loss, accuracy):
    server = SimpleNamespace(
        strategy=SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT),
        client_ids=client_ids,
        model_evaluate=Mock(return_value=(loss, accuracy)),
        average_client_evaluation_results=Mock(return_value=(loss, accuracy)))
    return server


class FairFedDriftEvaluationTests(unittest.TestCase):
    def test_staged_evaluation_resolves_flat_slot_and_records_learned_cluster_id(self):
        other_server = make_server([0], 0.2, 0.8)
        other_server.server_id = 0
        other_server.abs_id = 10
        server = make_server([9], 0.3, 0.7)
        server.server_id = 1
        server.abs_id = 14
        server.fairfeddrift_cluster_id = 7
        server.model = object()
        client = SimpleNamespace(
            client_id=9, parent_server_id=1, fedrc_models=None,
            model=object(), evaluate=Mock(return_value=(0.4, 0.6)))

        result = evaluate_clients_for_stage(
            [client], [other_server, server], 5, [9],
            'local_after_training', drift=None)

        record = result['clients'][0]
        self.assertEqual(record['parent_server_id'], 1)
        self.assertEqual(record['server_abs_id'], 14)
        self.assertEqual(record['fairfeddrift_cluster_id'], 7)
        self.assertEqual(record['model_id'], 'primary')
        client.evaluate.assert_called_once_with(client.model)

    def test_evaluates_every_active_cluster_server_on_server_test_data(self):
        servers = [make_server([0, 1], 0.2, 0.8), make_server([2], 0.5, 0.6)]
        test_loader = object()

        result = server_hierarchy_evaluate(
            [servers], test_loader, [], True, constants.RecoveryAlgorithm.FEDAVG)

        self.assertEqual(result, [[(0.2, 0.8)], [(0.5, 0.6)]])
        for server in servers:
            server.model_evaluate.assert_called_once_with(test_loader)
            server.average_client_evaluation_results.assert_not_called()

    def test_server_performance_record_shape_matches_oracle(self):
        fairfed_servers = [make_server([0], 0.2, 0.8), make_server([1], 0.4, 0.6)]
        oracle_servers = [make_server([0], 0.2, 0.8), make_server([1], 0.4, 0.6)]
        for server in oracle_servers:
            server.strategy = SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.ORACLE)

        fairfed_record = server_hierarchy_evaluate([fairfed_servers], object(), [], True,
                                                   constants.RecoveryAlgorithm.FAIRFEDDRIFT)
        oracle_record = server_hierarchy_evaluate([oracle_servers], object(), [], True,
                                                  constants.RecoveryAlgorithm.ORACLE)

        self.assertEqual(fairfed_record, oracle_record)
        self.assertEqual(fairfed_record, [[(0.2, 0.8)], [(0.4, 0.6)]])

    def test_uses_assigned_client_metrics_and_preserves_empty_server_placeholder(self):
        servers = [make_server([0], 0.3, 0.7), make_server([], 0.0, 0.0)]
        clients = [object()]

        result = server_hierarchy_evaluate(
            [servers], object(), clients, False, constants.RecoveryAlgorithm.FEDAVG,
            server_metric_weighting='train_samples')

        self.assertEqual(result, [[(0.3, 0.7)], [(0.0, 0.0)]])
        servers[0].average_client_evaluation_results.assert_called_once_with(clients, 'train_samples')
        servers[0].model_evaluate.assert_not_called()
        servers[1].average_client_evaluation_results.assert_not_called()

    def test_inactive_cached_fairfeddrift_strategy_does_not_force_cluster_evaluation(self):
        server = make_server([0], 0.3, 0.7)
        server.strategy = SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.FEDAVG)
        server.fairfeddrift_strategy = SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT)

        result = server_hierarchy_evaluate(
            [[server]], object(), [], True, constants.RecoveryAlgorithm.FEDAVG)

        self.assertEqual(result, [[(0.3, 0.7)]])
        server.model_evaluate.assert_called_once()


if __name__ == '__main__':
    unittest.main()
