"""Behavioral checks for reusable model-distance diagnostics."""

from collections import OrderedDict
import copy
import unittest
from types import SimpleNamespace

import torch
import constants

from distance_metrics.distance_metrics import ModelDistanceHistory, compute_model_l2_distance, \
    collect_model_distance_diagnostics


class FakeModel:
    def __init__(self, values, counter=0):
        self.values = torch.tensor(values, dtype=torch.float32, requires_grad=True)
        self.counter = torch.tensor(counter, dtype=torch.int64)

    def state_dict(self):
        return OrderedDict(weight=self.values, counter=self.counter)


class FakeStrategy:
    strategy_name = 'fake'


class FakeServer:
    def __init__(self, server_id, abs_id, clients, model=None, multi_models=None):
        self.server_id = server_id
        self.abs_id = abs_id
        self.client_ids = clients
        self.model = model
        self.multi_models = multi_models
        self.strategy = FakeStrategy()


class FakeClient:
    def __init__(self, client_id, parent_server_id, model=None, fedrc_models=None):
        self.client_id = client_id
        self.parent_server_id = parent_server_id
        self.model = model
        self.fedrc_models = fedrc_models


class ModelDistanceTests(unittest.TestCase):
    def test_true_l2_and_integer_buffer(self):
        """Combine tensor differences quadratically and safely include integer buffers."""
        server = OrderedDict(a=torch.tensor([0, 0]), b=torch.tensor([0.0]))
        client = OrderedDict(a=torch.tensor([3, 0]), b=torch.tensor([4.0]))
        result = compute_model_l2_distance(server, client)
        self.assertEqual(result['whole_model_l2'], 5.0)
        self.assertEqual(result['layers'], {'a': 3.0, 'b': 4.0})

    def test_parent_server_mapping_and_no_mutation(self):
        """Compare clients with assigned servers while leaving tensors, gradients, and RNG unchanged."""
        server0 = FakeServer(0, 10, [7], FakeModel([0.0, 0.0]))
        server1 = FakeServer(1, 11, [3], FakeModel([10.0, 10.0]))
        client7 = FakeClient(7, 0, FakeModel([3.0, 4.0]))
        client3 = FakeClient(3, 1, FakeModel([13.0, 14.0]))
        client7.model.values.grad = torch.tensor([1.0, 2.0])
        original_values = client7.model.values.detach().clone()
        original_gradient = client7.model.values.grad.clone()
        rng_state = torch.random.get_rng_state().clone()
        record = collect_model_distance_diagnostics([server0, server1], [client3, client7], 2, [7])
        first = record['servers'][0]['models'][0]['clients'][7]
        second = record['servers'][1]['models'][0]['clients'][3]
        self.assertEqual(first['whole_model_l2'], 5.0)
        self.assertEqual(second['whole_model_l2'], 5.0)
        self.assertTrue(first['participated'])
        self.assertFalse(second['participated'])
        torch.testing.assert_close(client7.model.values.detach(), original_values)
        torch.testing.assert_close(client7.model.values.grad, original_gradient)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)

    def test_fairfeddrift_cluster_identity_survives_flat_slot_reindexing(self):
        """Distance records identify a learned cluster after its server slot changes."""
        server = FakeServer(1, 14, [9], FakeModel([10.0, 10.0]))
        server.strategy = SimpleNamespace(strategy_name=constants.RecoveryAlgorithm.FAIRFEDDRIFT)
        server.fairfeddrift_cluster_id = 7
        client = FakeClient(9, 1, FakeModel([13.0, 14.0]))

        record = collect_model_distance_diagnostics([server], [client], 5, [9])
        server_record = record['servers'][0]
        client_record = server_record['models'][0]['clients'][9]

        self.assertEqual(server_record['server_id'], 1)
        self.assertEqual(server_record['server_abs_id'], 14)
        self.assertEqual(server_record['fairfeddrift_cluster_id'], 7)
        self.assertEqual(client_record['whole_model_l2'], 5.0)

    def test_fedrc_and_history_views(self):
        """Record corresponding multi-model distances and expose compact reusable views."""
        server = FakeServer(0, 4, [5], model=None,
                            multi_models=[FakeModel([0.0]), FakeModel([10.0])])
        client = FakeClient(5, 0, model=None,
                            fedrc_models=[FakeModel([2.0]), FakeModel([13.0])])
        history = ModelDistanceHistory()
        history.append(collect_model_distance_diagnostics([server], [client], 0, [5]))
        self.assertEqual([model['model_id'] for model in history.latest()['servers'][0]['models']], [0, 1])
        self.assertEqual(history.latest()['servers'][0]['models'][1]['clients'][5]['whole_model_l2'], 3.0)
        self.assertEqual(history.client_distances(), {})
        self.assertNotIn('layers', history.model_log()['records'][0]['servers'][0]['models'][0]['clients'][5])
        self.assertNotIn('whole_model_l2', history.layer_log()['records'][0]['servers'][0]['models'][0]['clients'][5])

    def test_mismatch_errors(self):
        """Surface incompatible state dictionaries rather than producing partial distances."""
        with self.assertRaisesRegex(ValueError, 'keys do not match'):
            compute_model_l2_distance(OrderedDict(a=torch.ones(1)), OrderedDict(b=torch.ones(1)))
        with self.assertRaisesRegex(ValueError, 'shape mismatch'):
            compute_model_l2_distance(OrderedDict(a=torch.ones(1)), OrderedDict(a=torch.ones(2)))


if __name__ == '__main__':
    unittest.main()
