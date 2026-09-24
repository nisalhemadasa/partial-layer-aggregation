"""Focused checks for Ditto strategy selection and global aggregation."""

from collections import OrderedDict
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import constants
import device_utils
from federated_network.utils import evaluate_ditto_personalized_clients, build_ditto_state_log, \
    build_ditto_lambda_record
from federated_network.network import FederatedNetwork
from federated_network.client import Client
from federated_network.server import Server, model_aggregation_fedavg, server_fn
from models.utils import train
from strategy.Ditto import aggregator_fn as ditto_aggregator_fn, compute_ditto_proximal_loss, \
    resolve_ditto_parameters, train_ditto_personal_model
from strategy.FedAvg import aggregator_fn as fedavg_aggregator_fn


class ScalarModel(nn.Module):
    def __init__(self, value=0.0, counter=0):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([value], dtype=torch.float32))
        self.register_buffer('counter', torch.tensor(counter, dtype=torch.int64))


class FakeClient:
    def __init__(self, client_id, value, sample_count, personal_value):
        self.client_id = client_id
        self.model = ScalarModel(value, counter=client_id)
        self.ditto_personal_model = ScalarModel(personal_value, counter=99)
        self.local_trainset = list(range(sample_count))


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.register_buffer('counter', torch.tensor(0, dtype=torch.int64))

    def forward(self, inputs):
        return torch.log_softmax(self.linear(inputs), dim=1)


class LabelDrift:
    drift_mode = constants.DriftMode.LABEL_SWAP_ONCE
    drift_pattern_id_map = {1: [(0, 1)]}


class DittoStrategyTests(unittest.TestCase):
    def setUp(self):
        """Run strategy checks on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the prior shared device selection."""
        device_utils._device = self.old_device

    def test_strategy_selection(self):
        """Create a Ditto server through the standard server factory."""
        server = server_fn(0, constants.DatasetNames.MNIST, 0, constants.RecoveryAlgorithm.DITTO,
                           cluster_count=1, fedex_alpha=0.9)
        self.assertEqual(server.strategy.strategy_name, constants.RecoveryAlgorithm.DITTO)

    def test_weighted_upload_aggregation_ignores_personal_models(self):
        """Use upload states and exact one-to-three sample weights with arbitrary client IDs."""
        client_a = FakeClient(4, value=0.0, sample_count=1, personal_value=1000.0)
        client_b = FakeClient(9, value=4.0, sample_count=3, personal_value=-1000.0)
        server = Server(0, 0, ditto_aggregator_fn(), ScalarModel(), 1, 0.9)
        server.client_ids = [4, 9]

        model_aggregation_fedavg(server, [client_b, client_a], None, None)

        self.assertEqual(server.model.weight.item(), 3.0)
        self.assertEqual(server.model.counter.item(), 9)

    def test_invalid_inputs_fail_clearly(self):
        """Reject missing uploads, mismatched IDs, invalid counts, keys, and shapes."""
        strategy = ditto_aggregator_fn()
        model = ScalarModel()
        valid_state = ScalarModel(1.0).state_dict()
        with self.assertRaisesRegex(ValueError, 'at least one'):
            strategy.aggregate_models(model, {}, {})
        with self.assertRaisesRegex(ValueError, 'identical client IDs'):
            strategy.aggregate_models(model, {2: valid_state}, {3: 1})
        with self.assertRaisesRegex(ValueError, 'positive integers'):
            strategy.aggregate_models(model, {2: valid_state}, {2: 0})
        with self.assertRaisesRegex(ValueError, 'keys'):
            strategy.aggregate_models(model, {2: OrderedDict(weight=torch.tensor([1.0]))}, {2: 1})
        wrong_shape = OrderedDict(weight=torch.tensor([1.0, 2.0]), counter=torch.tensor(0))
        with self.assertRaisesRegex(ValueError, 'shape'):
            strategy.aggregate_models(model, {2: wrong_shape}, {2: 1})

    def test_existing_fedavg_remains_uniform(self):
        """Keep ordinary FedAvg independent of Ditto sample weighting."""
        model = ScalarModel()
        uploads = {0: ScalarModel(0.0).state_dict(), 1: ScalarModel(4.0).state_dict()}
        fedavg_aggregator_fn().aggregate_models(model, uploads)
        self.assertEqual(model.weight.item(), 2.0)

    def test_deep_hierarchy_is_rejected_before_dataset_loading(self):
        """Fail early until descendant sample-count propagation is implemented."""
        with self.assertRaisesRegex(ValueError, 'single-level'):
            FederatedNetwork(1, 0, [1, 1], 1, constants.DatasetNames.MNIST, None, {}, {},
                             {'recovery_method': constants.RecoveryAlgorithm.DITTO})

    def test_configuration_and_proximal_geometry(self):
        """Validate configuration and compute the exact named-parameter proximal penalty."""
        resolved = resolve_ditto_parameters({})
        self.assertEqual(resolved['ditto_lambda'], 0.05)
        self.assertEqual(resolved['ditto_lambda_candidates'], [0.1, 1.0, 2.0])
        with self.assertRaisesRegex(ValueError, 'ditto_lambda'):
            resolve_ditto_parameters({'ditto_lambda': -1})
        with self.assertRaisesRegex(ValueError, 'personal_epochs'):
            resolve_ditto_parameters({'ditto_personal_epochs': 0})

        model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[3.0, 4.0]]))
        reference = OrderedDict(weight=torch.zeros((1, 2)))
        self.assertEqual(compute_ditto_proximal_loss(model, reference, 2.0).item(), 25.0)
        self.assertEqual(compute_ditto_proximal_loss(model, model.state_dict(), 2.0).item(), 0.0)

    def test_lambda_zero_matches_ordinary_training(self):
        """Match the framework SGD/NLL update when the proximal coefficient is zero."""
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1, 1, 0])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        ordinary_model = TinyClassifier()
        personal_model = TinyClassifier()
        personal_model.load_state_dict(ordinary_model.state_dict())
        reference = OrderedDict((key, value.detach().clone())
                                for key, value in ordinary_model.state_dict().items())

        train(ordinary_model, loader, _epochs=2)
        train_ditto_personal_model(personal_model, loader, reference, ditto_lambda=0.0,
                                   epochs=2, learning_rate=0.01)

        for ordinary_parameter, personal_parameter in zip(ordinary_model.parameters(), personal_model.parameters()):
            torch.testing.assert_close(ordinary_parameter, personal_parameter, rtol=0, atol=0)

    def test_warmup_initialization_and_round_persistence(self):
        """Initialize once after warm-up, retain personal state, and reset only the upload model."""
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]])
        labels = torch.tensor([0, 1, 1, 0])
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        client = Client(3, True, TinyClassifier(), 1, 2, dataset, dataset,
                        constants.RecoveryAlgorithm.DITTO, 1,
                        {'ditto_lambda': 0.2, 'ditto_personal_epochs': 1})
        client.trainloader = loader
        client.testloader = loader

        client.fit(False, False, None, client.client_id, constants.RecoveryAlgorithm.DITTO, None)
        self.assertTrue(client.ditto_initialized)
        self.assertIsNot(client.model, client.ditto_personal_model)
        for upload_parameter, personal_parameter in zip(client.model.parameters(),
                                                         client.ditto_personal_model.parameters()):
            torch.testing.assert_close(upload_parameter, personal_parameter)

        personal_model_identity = id(client.ditto_personal_model)
        personal_before_round = [parameter.detach().clone() for parameter in client.ditto_personal_model.parameters()]
        server_model = TinyClassifier()
        with torch.no_grad():
            for parameter in server_model.parameters():
                parameter.zero_()
        server_snapshot = OrderedDict((key, value.detach().clone())
                                      for key, value in server_model.state_dict().items())
        server_snapshot_before = OrderedDict((key, value.detach().clone())
                                             for key, value in server_snapshot.items())
        expected_upload = TinyClassifier()
        expected_upload.load_state_dict(server_snapshot)
        train(expected_upload, loader, _epochs=1)

        client.fit(True, False, server_snapshot, client.client_id, constants.RecoveryAlgorithm.DITTO, [3])

        self.assertEqual(id(client.ditto_personal_model), personal_model_identity)
        for expected_parameter, upload_parameter in zip(expected_upload.parameters(), client.model.parameters()):
            torch.testing.assert_close(expected_parameter, upload_parameter, rtol=0, atol=0)
        self.assertTrue(any(not torch.equal(before, after)
                            for before, after in zip(personal_before_round,
                                                     client.ditto_personal_model.parameters())))
        for key, value in server_snapshot.items():
            torch.testing.assert_close(value, server_snapshot_before[key])

    def test_personalized_evaluation_identity_metadata_and_nonmutation(self):
        """Evaluate only the personal model and retain model parameters and training mode."""
        inputs = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
        labels = torch.tensor([0, 0])
        dataset = TensorDataset(inputs, labels)
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        client = Client(5, True, TinyClassifier(), 1, 2, dataset, dataset,
                        constants.RecoveryAlgorithm.DITTO, 1, {})
        client.testloader = loader
        client.ditto_personal_model = TinyClassifier()
        client.ditto_initialized = True
        client.parent_server_id = 0
        with torch.no_grad():
            client.model.linear.weight.copy_(torch.tensor([[3.0, 0.0], [-3.0, 0.0]]))
            client.model.linear.bias.zero_()
            client.ditto_personal_model.linear.weight.copy_(torch.tensor([[-3.0, 0.0], [3.0, 0.0]]))
            client.ditto_personal_model.linear.bias.zero_()
        client.model.train()
        client.ditto_personal_model.train()
        personal_before = OrderedDict((key, value.detach().clone())
                                      for key, value in client.ditto_personal_model.state_dict().items())

        upload_loss, upload_accuracy = client.evaluate()
        personal_loss, personal_accuracy = client.evaluate_ditto_personalized()
        class_loss, class_accuracy, class_count = \
            client.evaluate_ditto_personalized_drifted_classes({0})

        self.assertEqual(upload_accuracy, 1.0)
        self.assertEqual(personal_accuracy, 0.0)
        self.assertGreater(personal_loss, upload_loss)
        self.assertEqual(class_accuracy, 0.0)
        self.assertEqual(class_count, 2)
        self.assertGreater(class_loss, 0.0)
        self.assertTrue(client.model.training)
        self.assertTrue(client.ditto_personal_model.training)
        for key, value in client.ditto_personal_model.state_dict().items():
            torch.testing.assert_close(value, personal_before[key])

        server = Server(0, 12, ditto_aggregator_fn(), TinyClassifier(), 1, 0.9)
        record = evaluate_ditto_personalized_clients([client], [server], 3, [5], LabelDrift(), True)
        client_record = record['clients'][0]
        self.assertEqual(record['stage'], 'local_after_training')
        self.assertEqual(client_record['model_role'], 'ditto_personalized')
        self.assertEqual(client_record['server_abs_id'], 12)
        self.assertTrue(client_record['participated'])
        self.assertEqual(client_record['drifted_class_metrics']['sample_count'], 2)

        state = build_ditto_state_log([client], client.ditto_parameters)
        self.assertEqual(state['personal_optimizer_state_policy'], 'recreated_per_round')
        self.assertTrue(state['clients'][0]['has_personalized_model'])

    def test_dynamic_validation_partition_is_stable_disjoint_and_rng_independent(self):
        """Build the same client-local split without advancing the global torch RNG."""
        inputs = torch.arange(40, dtype=torch.float32).reshape(20, 2)
        labels = torch.arange(20) % 2
        dataset = TensorDataset(inputs, labels)
        parameters = {
            'ditto_dynamic_lambda': True,
            'ditto_validation_fraction': 0.2,
            'ditto_validation_seed': 91,
        }
        client_a = Client(7, True, TinyClassifier(), 1, 2, dataset, dataset,
                          constants.RecoveryAlgorithm.DITTO, 1, parameters)
        client_b = Client(7, True, TinyClassifier(), 1, 2, dataset, dataset,
                          constants.RecoveryAlgorithm.DITTO, 1, parameters)
        rng_before = torch.random.get_rng_state().clone()
        client_a._initialize_ditto_validation_partition()
        torch.testing.assert_close(torch.random.get_rng_state(), rng_before)
        client_b._initialize_ditto_validation_partition()

        self.assertEqual(client_a.ditto_validation_indices, client_b.ditto_validation_indices)
        self.assertEqual(client_a.ditto_training_indices, client_b.ditto_training_indices)
        self.assertFalse(set(client_a.ditto_validation_indices) & set(client_a.ditto_training_indices))
        self.assertEqual(len(client_a.ditto_validation_indices), 4)
        self.assertEqual(len(client_a.ditto_training_indices), 16)
        original_validation_indices = list(client_a.ditto_validation_indices)
        client_a.sample_data()
        self.assertEqual(client_a.ditto_validation_indices, original_validation_indices)
        self.assertFalse(set(client_a.trainloader.dataset.indices) & set(client_a.ditto_validation_indices))
        changed_index = client_a.ditto_validation_indices[0]
        changed_label = int((dataset.tensors[1][changed_index] + 1) % 2)
        dataset.tensors[1][changed_index] = changed_label
        validation_labels = [int(label) for _, label in client_a.ditto_validation_loader.dataset]
        self.assertEqual(validation_labels[0], changed_label)

    def test_dynamic_lambda_uses_comparable_candidates_and_ordered_tie_breaking(self):
        """Try candidates from one base state and retain the first candidate on equal validation loss."""
        inputs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        loader = DataLoader(TensorDataset(inputs, labels), batch_size=2, shuffle=False)
        model = TinyClassifier()
        reference = OrderedDict((key, value.detach().clone()) for key, value in model.state_dict().items())
        decisions = train_ditto_personal_model(
            model, loader, reference, ditto_lambda=0.5, epochs=1, learning_rate=0.01,
            dynamic_lambda=True, lambda_candidates=[0.0, 1.0], validation_loader=loader)

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['selected_candidate_index'], 0)
        self.assertEqual(decisions[0]['selected_lambda'], 0.0)
        self.assertEqual([candidate['lambda'] for candidate in decisions[0]['candidates']], [0.0, 1.0])
        self.assertAlmostEqual(decisions[0]['candidates'][0]['validation_loss'],
                               decisions[0]['candidates'][1]['validation_loss'])

        client = type('DynamicClient', (), {
            'client_id': 2,
            'parent_server_id': 0,
            'ditto_selected_lambda': 0.0,
            'ditto_lambda_decisions': decisions,
        })()
        record = build_ditto_lambda_record([client], 4)
        self.assertEqual(record['clients'][0]['decisions'][0]['selected_lambda'], 0.0)


if __name__ == '__main__':
    unittest.main()
