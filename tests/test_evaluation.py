"""Focused behavioral checks for staged evaluation and class-subset metrics."""

import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import device_utils
import constants
from federated_network.server import Server
from federated_network.utils import evaluate_clients_for_stage
from models.utils import test_subset_classes


class LookupModel(nn.Module):
    def forward(self, inputs):
        return inputs


class FakeServer:
    def __init__(self, server_id, abs_id, model):
        self.server_id = server_id
        self.abs_id = abs_id
        self.model = model
        self.multi_models = None


class FakeClient:
    def __init__(self, client_id, parent_server_id, local_model, train_size):
        self.client_id = client_id
        self.parent_server_id = parent_server_id
        self.model = local_model
        self.fedrc_models = None
        self.local_trainset = list(range(train_size))

    def evaluate(self, model=None):
        model = self.model if model is None else model
        return model['loss'], model['accuracy']

    def evaluate_drifted_classes(self, classes, model=None):
        if not classes:
            return None, None, 0
        model = self.model if model is None else model
        return model['class_loss'], model['class_accuracy'], model['class_count']


class FakeDrift:
    drift_pattern_id_map = {1: [(1, 2)]}
    drift_mode = constants.DriftMode.LABEL_SWAP_INCREMENTAL_STEPS


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        """Run tensor evaluation on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the prior shared device selection."""
        device_utils._device = self.old_device

    def test_class_subset_uses_only_requested_labels(self):
        """Loss, accuracy, and count exclude every non-target class."""
        logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0],
                               [0.0, 0.0, 10.0]])
        labels = torch.tensor([0, 1, 2, 1])
        loss, accuracy, count = test_subset_classes(LookupModel(), DataLoader(TensorDataset(logits, labels), 2), {1})
        self.assertEqual(count, 2)
        self.assertEqual(accuracy, 0.5)
        self.assertGreater(loss, 4.0)
        self.assertEqual(test_subset_classes(LookupModel(), DataLoader(TensorDataset(logits, labels), 2), {9}),
                         (None, None, 0))

    def test_stage_selects_server_or_local_without_mutation(self):
        """Global stage reads the assigned server model while local stages retain client model identity."""
        local = {'loss': 3.0, 'accuracy': 0.3, 'class_loss': 4.0, 'class_accuracy': 0.4, 'class_count': 5}
        server_model = {'loss': 1.0, 'accuracy': 0.8, 'class_loss': 2.0, 'class_accuracy': 0.9, 'class_count': 6}
        client = FakeClient(7, 4, local, 3)
        server = FakeServer(4, 14, server_model)
        global_record = evaluate_clients_for_stage([client], [server], 2, [7], 'global_after_download',
                                                   FakeDrift(), True)['clients'][0]
        local_record = evaluate_clients_for_stage([client], [server], 2, [], 'local_after_training',
                                                  FakeDrift(), True)['clients'][0]
        self.assertEqual(global_record['accuracy'], 0.8)
        self.assertEqual(global_record['model_role'], 'assigned_server')
        self.assertTrue(global_record['participated'])
        self.assertEqual(global_record['drifted_class_metrics']['classes'], [1, 2])
        self.assertEqual(local_record['accuracy'], 0.3)
        self.assertFalse(local_record['participated'])
        self.assertIs(client.model, local)

    def test_rotation_drift_has_no_label_subset(self):
        """Rotation drift does not present label-pair metadata as a class-subset metric."""
        local = {'loss': 3.0, 'accuracy': 0.3, 'class_loss': 4.0, 'class_accuracy': 0.4, 'class_count': 5}
        client = FakeClient(7, 4, local, 3)
        server = FakeServer(4, 14, local)
        drift = FakeDrift()
        drift.drift_mode = constants.DriftMode.ROTATION_GRADUAL
        record = evaluate_clients_for_stage([client], [server], 2, [7], 'local_after_training', drift, True)
        self.assertEqual(record['clients'][0]['drifted_class_metrics']['classes'], [])
        self.assertEqual(record['clients'][0]['drifted_class_metrics']['sample_count'], 0)

    def test_server_metric_weighting(self):
        """Uniform and train-sample weighting produce their independently expected results."""
        client_a = FakeClient(0, 0, {'loss': 2.0, 'accuracy': 0.2}, 1)
        client_b = FakeClient(1, 0, {'loss': 10.0, 'accuracy': 0.8}, 3)
        server = type('ServerLike', (), {'client_ids': [0, 1]})()
        uniform = Server.average_client_evaluation_results(server, [client_a, client_b], 'uniform')
        weighted = Server.average_client_evaluation_results(server, [client_a, client_b], 'train_samples')
        self.assertEqual(uniform, (6.0, 0.5))
        self.assertEqual(weighted[0], 8.0)
        self.assertAlmostEqual(weighted[1], 0.65)
        self.assertEqual(Server.average_client_evaluation_results(server, [], 'uniform'), (0.0, 0.0))


if __name__ == '__main__':
    unittest.main()
