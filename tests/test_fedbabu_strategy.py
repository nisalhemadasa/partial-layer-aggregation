"""Focused FedBABU strategy registration checks."""

from collections import OrderedDict
import copy
import ast
import os
import pickle
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import constants
import device_utils
from federated_network.server import (Server, model_aggregation, model_distribution_hierarchy,
                                      change_server_aggregation_strategy, server_fn)
from federated_network.client import Client
from federated_network.utils import (build_fedbabu_personalized_log,
                                     evaluate_fedbabu_personalized_clients,
                                     finalize_fedbabu_personalization,
                                     train_client_models)
from log_utils.logging import write_structured_log
from strategy.FedAvg import aggregator_fn as fedavg_aggregator_fn
from strategy.FedBABU import (FedBABU, aggregate_fedbabu_body_parameters,
                              aggregate_fedbabu_tensor_values, aggregator_fn,
                              initialize_fedbabu_personal_models,
                              initialize_fedbabu_shared_head, resolve_fedbabu_parameters,
                              train_fedbabu_head)
from strategy.FedBABU.utils import split_fedbabu_body_and_head
from strategy.FedEx import aggregator_fn as fedex_aggregator_fn


class FedBABUStrategyRegistrationTests(unittest.TestCase):
    def setUp(self):
        """Run server construction checks on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the shared device selection."""
        device_utils._device = self.old_device

    def test_recovery_algorithm_constant(self):
        """Expose the stable strategy name used by experiment configuration."""
        self.assertEqual(constants.RecoveryAlgorithm.FEDBABU, 'fedbabu')

    def test_factory_returns_fedbabu_strategy(self):
        """Create the strategy through its package factory."""
        strategy = aggregator_fn()
        self.assertIsInstance(strategy, FedBABU)
        self.assertEqual(strategy.strategy_name, constants.RecoveryAlgorithm.FEDBABU)

    def test_server_factory_selects_fedbabu(self):
        """Create a FedBABU server through the regular dataset model factory."""
        server = server_fn(
            server_id=0,
            dataset_name=constants.DatasetNames.MNIST,
            server_abs_id=0,
            drift_recovery_method=constants.RecoveryAlgorithm.FEDBABU,
            cluster_count=1,
            fedex_alpha=0.0,
            drift_recovery_parameters={},
        )
        self.assertIsInstance(server.strategy, FedBABU)
        self.assertEqual(server.strategy.strategy_name, constants.RecoveryAlgorithm.FEDBABU)

    def test_server_train_dispatches_to_fedbabu_aggregation(self):
        """Reach FedBABU aggregation instead of falling through to FedAvg."""
        server = Server(0, 0, aggregator_fn(), nn.Linear(1, 1), 1, 0.0)
        with self.assertRaisesRegex(ValueError, 'at least one client upload'):
            server.train({})

    def test_drift_strategy_switch_selects_fedbabu(self):
        """Keep FedBABU active when the framework enters or exits its drift phase."""
        server = Server(0, 0, fedavg_aggregator_fn(), nn.Linear(1, 1), 1, 0.0)
        drift = SimpleNamespace(drifted_client_indices=[0])

        change_server_aggregation_strategy(
            [[server]], constants.RecoveryAlgorithm.FEDBABU, drift, {})

        self.assertEqual(server.strategy.strategy_name, constants.RecoveryAlgorithm.FEDBABU)


class PartitionModel(nn.Module):
    def __init__(self, model_type, head_name='fc2', linear_head=True):
        super().__init__()
        self.body = nn.Linear(3, 4)
        if linear_head:
            setattr(self, head_name, nn.Linear(4, 2))

        self._model_type = model_type

    def get_model_type(self):
        return self._model_type

    def forward(self, inputs):
        return F.log_softmax(self.fc2(self.body(inputs)), dim=1)


class BufferedPartitionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4), nn.ReLU())
        self.fc2 = nn.Linear(4, 2)

    def get_model_type(self):
        return constants.ModelTypes.CNN_MODEL

    def forward(self, inputs):
        return F.log_softmax(self.fc2(self.body(inputs)), dim=1)


class FedBABUModelPartitionTests(unittest.TestCase):
    def test_supported_final_classifier_mappings(self):
        """Use the final linear layer named by each active model type."""
        supported_models = (
            (constants.ModelTypes.CNN_MODEL, 'fc2'),
            (constants.ModelTypes.CNN_CIFAR_10, 'fc2'),
            (constants.ModelTypes.CNN_CIFAR_100, 'fc2'),
            (constants.ModelTypes.CONVNET_TINY_IMAGENET, 'head'),
            ('TabularAdultModel', 'fc2'),
        )
        for model_type, head_name in supported_models:
            with self.subTest(model_type=model_type):
                model = PartitionModel(model_type, head_name)
                model_state = model.state_dict()
                body, head = split_fedbabu_body_and_head(model)
                self.assertEqual(set(head), {head_name + '.weight', head_name + '.bias'})
                self.assertEqual(set(body), {'body.weight', 'body.bias'})
                self.assertTrue(set(body).isdisjoint(head))
                self.assertEqual(set(body) | set(head), set(model_state))
                for key, tensor in model_state.items():
                    partition = body if key in body else head
                    self.assertEqual(partition[key].shape, tensor.shape)

    def test_supplied_state_is_split_without_mutation(self):
        """Use a supplied upload state while preserving key order and values."""
        model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        supplied_state = OrderedDict((key, tensor.clone()) for key, tensor in model.state_dict().items())
        supplied_state['body.weight'].fill_(2.0)
        supplied_state['fc2.weight'].fill_(3.0)
        before = OrderedDict((key, tensor.clone()) for key, tensor in supplied_state.items())

        body, head = split_fedbabu_body_and_head(model, supplied_state)

        self.assertEqual(list(body), ['body.weight', 'body.bias'])
        self.assertEqual(list(head), ['fc2.weight', 'fc2.bias'])
        for key, tensor in before.items():
            partition = body if key in body else head
            self.assertTrue(torch.equal(partition[key], tensor))
            self.assertTrue(torch.equal(supplied_state[key], tensor))

    def test_state_key_and_shape_mismatches_are_rejected(self):
        """Reject incomplete, unexpected, or incorrectly shaped state dicts."""
        model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        model_state = model.state_dict()

        missing_key_state = OrderedDict(model_state)
        del missing_key_state['body.weight']
        with self.assertRaisesRegex(ValueError, 'state keys do not match'):
            split_fedbabu_body_and_head(model, missing_key_state)

        unexpected_key_state = OrderedDict(model_state)
        unexpected_key_state['unexpected.weight'] = torch.zeros((1,))
        with self.assertRaisesRegex(ValueError, 'state keys do not match'):
            split_fedbabu_body_and_head(model, unexpected_key_state)

        wrong_shape_state = OrderedDict(model_state)
        wrong_shape_state['body.weight'] = torch.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'tensor shape'):
            split_fedbabu_body_and_head(model, wrong_shape_state)

    def test_unsupported_model_type_is_rejected(self):
        """Do not infer a classifier split for unknown architectures."""
        with self.assertRaisesRegex(ValueError, 'does not support model type'):
            split_fedbabu_body_and_head(PartitionModel('UnregisteredModel'))

    def test_non_linear_classifier_module_is_rejected(self):
        """Require the configured final classifier to be a linear module."""
        model = PartitionModel(constants.ModelTypes.CNN_MODEL, linear_head=False)
        with self.assertRaisesRegex(ValueError, 'expected the final classifier module'):
            split_fedbabu_body_and_head(model)


class FedBABUDistributionTests(unittest.TestCase):
    def test_existing_client_download_sends_body_and_fixed_head(self):
        """Reuse the ordinary full-state client download for flat FedBABU."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        with torch.no_grad():
            server_model.body.weight.fill_(7.0)
            server_model.body.bias.fill_(8.0)
            server_model.fc2.weight.fill_(9.0)
            server_model.fc2.bias.fill_(10.0)
            client_model.body.weight.fill_(-7.0)
            client_model.body.bias.fill_(-8.0)
            client_model.fc2.weight.fill_(-9.0)
            client_model.fc2.bias.fill_(-10.0)

        server = Server(0, 0, aggregator_fn(), server_model, 1, 0.0)
        server.client_ids = [11]
        client = type('TestClient', (), {})()
        client.client_id = 11
        client.parent_server_id = 0
        client.model = client_model
        client.drift_recovery_method = constants.RecoveryAlgorithm.FEDBABU
        client.fit = lambda *args: None
        client.evaluate = lambda: (1.0, 0.5)
        drift = type('TestDrift', (), {
            'is_drift': False,
            'is_drift_end': False,
            'drifted_client_indices': [],
        })()

        # For the supported one-server layout this helper is a no-op; the
        # ordinary client orchestration performs the complete model download.
        model_distribution_hierarchy([[server]])
        train_client_models([client], [11], [server], drift,
                            {'is_server_adaptability': False}, constants.RecoveryAlgorithm.FEDBABU)

        self.assertTrue(torch.equal(client.model.body.weight, server.model.body.weight))
        self.assertTrue(torch.equal(client.model.body.bias, server.model.body.bias))
        self.assertTrue(torch.equal(client.model.fc2.weight, server.model.fc2.weight))
        self.assertTrue(torch.equal(client.model.fc2.bias, server.model.fc2.bias))


class FedBABUClientTrainingTests(unittest.TestCase):
    def setUp(self):
        """Run client-training checks on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the shared device selection."""
        device_utils._device = self.old_device

    def _make_client(self):
        inputs = torch.tensor([
            [1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0], [0.5, -1.0, 1.0], [-0.5, 1.0, -1.0],
        ])
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        dataset = TensorDataset(inputs, labels)
        model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client = Client(0, True, model, 2, 3, dataset, dataset,
                        constants.RecoveryAlgorithm.FEDBABU, 1)
        client.trainloader = DataLoader(dataset, batch_size=3, shuffle=False)
        return client

    def _assert_only_body_changes(self, client, before):
        """Assert body updates, head stays fixed, and original grad flags return."""
        after = client.model.state_dict()
        self.assertTrue(any(not torch.equal(before[key], after[key])
                            for key in ('body.weight', 'body.bias')))
        for key in ('fc2.weight', 'fc2.bias'):
            self.assertTrue(torch.equal(before[key], after[key]))
        self.assertTrue(all(parameter.requires_grad for parameter in client.model.parameters()))

    def test_fedbabu_warmup_trains_body_only(self):
        """Route no-server-parameter warm-up through the body-only trainer."""
        client = self._make_client()
        before = OrderedDict((key, value.clone()) for key, value in client.model.state_dict().items())

        client.fit(False, False, None, client.client_id,
                   constants.RecoveryAlgorithm.FEDBABU, [])

        self._assert_only_body_changes(client, before)

    def test_body_training_reuses_framework_loader_epochs_and_trainer(self):
        """Delegate with the existing loader and epoch count while head is frozen."""
        client = self._make_client()
        gradient_state_during_training = {}

        def observe_training(model, loader, epochs, verbose, **kwargs):
            gradient_state_during_training.update(
                (name, parameter.requires_grad) for name, parameter in model.named_parameters())
            self.assertIs(loader, client.trainloader)
            self.assertEqual(epochs, client.epochs)
            self.assertFalse(verbose)
            self.assertEqual(len(kwargs['_optimizer_parameters']), 2)

        with patch('strategy.FedBABU.utils.train', side_effect=observe_training) as train_mock:
            client.fit(False, False, None, client.client_id,
                       constants.RecoveryAlgorithm.FEDBABU, [])

        train_mock.assert_called_once()
        call_args, call_kwargs = train_mock.call_args
        self.assertEqual(call_args, (client.model, client.trainloader, client.epochs, False))
        optimizer_parameters = call_kwargs['_optimizer_parameters']
        self.assertEqual(len(optimizer_parameters), 2)
        self.assertTrue(all(parameter is client.model.body.weight or
                            parameter is client.model.body.bias
                            for parameter in optimizer_parameters))
        self.assertTrue(gradient_state_during_training['body.weight'])
        self.assertTrue(gradient_state_during_training['body.bias'])
        self.assertFalse(gradient_state_during_training['fc2.weight'])
        self.assertFalse(gradient_state_during_training['fc2.bias'])
        self.assertTrue(all(parameter.requires_grad for parameter in client.model.parameters()))

    def test_fedbabu_round_and_drift_training_keep_head_fixed(self):
        """Use the same body-only route for normal rounds and drift onset."""
        for is_drift in (False, True):
            with self.subTest(is_drift=is_drift):
                client = self._make_client()
                server_state = OrderedDict((key, value.clone())
                                           for key, value in client.model.state_dict().items())
                server_state['body.weight'].add_(0.25)
                before_head = {key: client.model.state_dict()[key].clone()
                               for key in ('fc2.weight', 'fc2.bias')}
                client.fit(is_drift, False, server_state, client.client_id,
                           constants.RecoveryAlgorithm.FEDBABU, [])
                after = client.model.state_dict()
                self.assertTrue(any(not torch.equal(server_state[key], after[key])
                                    for key in ('body.weight', 'body.bias')))
                for key, value in before_head.items():
                    self.assertTrue(torch.equal(after[key], value))
                self.assertTrue(all(parameter.requires_grad for parameter in client.model.parameters()))

    def test_non_fedbabu_client_training_keeps_standard_full_model_route(self):
        """Keep ordinary FedAvg training on the existing unrestricted trainer."""
        client = self._make_client()
        with patch('federated_network.client.train') as train_mock:
            client.fit(False, False, None, client.client_id,
                       constants.RecoveryAlgorithm.FEDAVG, [])

        train_mock.assert_called_once_with(client.model, client.trainloader, _epochs=client.epochs)
        self.assertTrue(all(parameter.requires_grad for parameter in client.model.parameters()))


class FedBABUSharedHeadInitializationTests(unittest.TestCase):
    def test_shared_initial_head_is_copied_without_changing_any_body(self):
        """Copy the first server head to every client while preserving bodies."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        clients = [SimpleNamespace(model=PartitionModel(constants.ModelTypes.CNN_MODEL))
                   for _ in range(3)]
        bodies_before = [
            {key: value.clone() for key, value in model.state_dict().items()
             if key.startswith('body.')}
            for model in [server_model] + [client.model for client in clients]
        ]
        canonical_head = {key: value.clone() for key, value in server_model.state_dict().items()
                          if key.startswith('fc2.')}
        with torch.no_grad():
            for client in clients:
                client.model.fc2.weight.add_(100.0)
                client.model.fc2.bias.sub_(100.0)

        initialize_fedbabu_shared_head(clients, [SimpleNamespace(model=server_model)])

        for model_index, model in enumerate([server_model] + [client.model for client in clients]):
            state = model.state_dict()
            for key, value in canonical_head.items():
                self.assertTrue(torch.equal(state[key], value))
            for key, value in bodies_before[model_index].items():
                self.assertTrue(torch.equal(state[key], value))

    def test_shared_head_initialization_rejects_empty_collections(self):
        """Fail clearly when setup has no model source or no clients."""
        with self.assertRaisesRegex(ValueError, 'requires clients and servers'):
            initialize_fedbabu_shared_head([], [SimpleNamespace(model=PartitionModel(
                constants.ModelTypes.CNN_MODEL))])


class FedBABUPersonalModelInitializationTests(unittest.TestCase):
    def setUp(self):
        """Run personalized-model construction checks on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the shared device selection."""
        device_utils._device = self.old_device

    def test_create_isolated_copies_of_final_shared_model_for_clients(self):
        """Start every client evaluation model from final shared state."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        with torch.no_grad():
            server_model.body.weight.fill_(7.0)
            server_model.body.bias.fill_(8.0)
            server_model.fc2.weight.fill_(9.0)
            server_model.fc2.bias.fill_(10.0)
        clients = [SimpleNamespace(client_id=client_id,
                                   model=PartitionModel(constants.ModelTypes.CNN_MODEL),
                                   fedbabu_personal_model=None)
                   for client_id in (3, 8)]
        original_client_states = [
            OrderedDict((key, value.clone()) for key, value in client.model.state_dict().items())
            for client in clients
        ]

        initialize_fedbabu_personal_models(clients, [SimpleNamespace(model=server_model)])

        for client in clients:
            self.assertIsNot(client.fedbabu_personal_model, server_model)
            self.assertTrue(all(torch.equal(value, server_model.state_dict()[key])
                                for key, value in client.fedbabu_personal_model.state_dict().items()))
        self.assertIsNot(clients[0].fedbabu_personal_model, clients[1].fedbabu_personal_model)
        with torch.no_grad():
            clients[0].fedbabu_personal_model.body.weight.add_(1.0)
        self.assertFalse(torch.equal(clients[0].fedbabu_personal_model.body.weight,
                                    clients[1].fedbabu_personal_model.body.weight))
        self.assertTrue(torch.equal(server_model.body.weight, torch.full_like(server_model.body.weight, 7.0)))
        for index, client in enumerate(clients):
            for key, value in original_client_states[index].items():
                self.assertTrue(torch.equal(client.model.state_dict()[key], value))

    def test_personalized_evaluation_uses_existing_test_path_and_model_role(self):
        """Create a separate structured result without changing model states."""
        inputs = torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5],
                               [-1.0, 0.5, 1.0], [0.5, -1.0, 1.0]])
        labels = torch.tensor([0, 1, 1, 0])
        dataset = TensorDataset(inputs, labels)
        client = Client(5, True, PartitionModel(constants.ModelTypes.CNN_MODEL), 1, 2,
                        dataset, dataset, constants.RecoveryAlgorithm.FEDBABU, 1)
        client.parent_server_id = 0
        client.testloader = DataLoader(dataset, batch_size=2, shuffle=False)
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client.fedbabu_personal_model = copy.deepcopy(server_model)
        client_state_before = OrderedDict((key, value.clone()) for key, value in client.model.state_dict().items())
        server_state_before = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
        personal_state_before = OrderedDict(
            (key, value.clone()) for key, value in client.fedbabu_personal_model.state_dict().items())

        record = evaluate_fedbabu_personalized_clients(
            [client], [SimpleNamespace(server_id=0, abs_id=13)], 20)

        self.assertEqual(record['round'], 20)
        self.assertEqual(record['stage'], 'post_training_personalization')
        self.assertEqual(len(record['clients']), 1)
        client_record = record['clients'][0]
        self.assertEqual(client_record['client_id'], 5)
        self.assertEqual(client_record['parent_server_id'], 0)
        self.assertEqual(client_record['server_abs_id'], 13)
        self.assertEqual(client_record['model_id'], 'personalized')
        self.assertEqual(client_record['model_role'], 'fedbabu_personalized')
        self.assertGreaterEqual(client_record['accuracy'], 0.0)
        self.assertLessEqual(client_record['accuracy'], 1.0)
        for key, value in client_state_before.items():
            self.assertTrue(torch.equal(client.model.state_dict()[key], value))
        for key, value in server_state_before.items():
            self.assertTrue(torch.equal(server_model.state_dict()[key], value))
        for key, value in personal_state_before.items():
            self.assertTrue(torch.equal(client.fedbabu_personal_model.state_dict()[key], value))

    def test_personalized_results_use_separate_versioned_log(self):
        """Persist final personalized results under their dedicated log name."""
        evaluation_record = {
            'round': 20,
            'stage': 'post_training_personalization',
            'clients': [{
                'client_id': 5,
                'parent_server_id': 0,
                'server_abs_id': 13,
                'model_id': 'personalized',
                'model_role': 'fedbabu_personalized',
                'loss': 0.4,
                'accuracy': 0.75,
            }]
        }
        payload = build_fedbabu_personalized_log(evaluation_record)
        self.assertEqual(payload['schema_version'], 1)
        self.assertEqual(payload['records'], [evaluation_record])
        self.assertNotEqual(constants.Logs.FEDBABU_PERSONALIZED_CLIENT_LOG, constants.Logs.CLIENT_LOG)

        with tempfile.TemporaryDirectory() as output_dir:
            output_path = os.path.join(output_dir, constants.Logs.FEDBABU_PERSONALIZED_CLIENT_LOG)
            write_structured_log(payload, output_path)
            with open(output_path + constants.FileExtentions.PKL, 'rb') as log_file:
                saved_payload = pickle.load(log_file)
        self.assertEqual(saved_payload, payload)


class FedBABUPersonalHeadTrainingTests(unittest.TestCase):
    def setUp(self):
        """Run personalized-head checks on CPU."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')

    def tearDown(self):
        """Restore the shared device selection."""
        device_utils._device = self.old_device

    @staticmethod
    def _make_client_with_parameters(parameters):
        inputs = torch.tensor([
            [1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0], [0.5, -1.0, 1.0], [-0.5, 1.0, -1.0],
        ])
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        dataset = TensorDataset(inputs, labels)
        return Client(0, True, PartitionModel(constants.ModelTypes.CNN_MODEL), 2, 3,
                      dataset, dataset, constants.RecoveryAlgorithm.FEDBABU, 1,
                      drift_recovery_parameters=parameters)

    def test_config_defaults_and_values_are_validated(self):
        """Expose named defaults and reject invalid optimizer settings."""
        defaults = resolve_fedbabu_parameters({})
        self.assertEqual(defaults['fedbabu_head_finetune_epochs'], 5)
        self.assertEqual(defaults['fedbabu_head_finetune_learning_rate'], 0.01)
        self.assertEqual(defaults['fedbabu_head_finetune_momentum'], 0.5)
        self.assertEqual(defaults['fedbabu_head_finetune_weight_decay'], 0.0)
        with self.assertRaisesRegex(ValueError, 'epochs must be a positive integer'):
            resolve_fedbabu_parameters({'fedbabu_head_finetune_epochs': 0})
        with self.assertRaisesRegex(ValueError, 'learning_rate must be finite and positive'):
            resolve_fedbabu_parameters({'fedbabu_head_finetune_learning_rate': float('nan')})
        with self.assertRaisesRegex(ValueError, 'momentum must be finite and in'):
            resolve_fedbabu_parameters({'fedbabu_head_finetune_momentum': 1.0})
        with self.assertRaisesRegex(ValueError, 'weight_decay must be finite and non-negative'):
            resolve_fedbabu_parameters({'fedbabu_head_finetune_weight_decay': -0.1})

    def test_main_exposes_resolver_defaults_as_shared_experiment_parameters(self):
        """Keep main.py's explicit FedBABU settings aligned with its resolver."""
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')
        with open(main_path, encoding='utf-8') as source_file:
            source = source_file.read()
        module = ast.parse(source, filename=main_path)

        assignment = next(
            node for node in ast.walk(module)
            if isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == 'fedbabu_parameters'
                for target in node.targets))
        values = {keyword.arg: ast.literal_eval(keyword.value)
                  for keyword in assignment.value.keywords}
        self.assertEqual(values, resolve_fedbabu_parameters({}))

    def test_main_keeps_supported_fedbabu_handles_disabled(self):
        """Document per-dataset handles without constructing extra networks."""
        main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main.py')
        with open(main_path, encoding='utf-8') as source_file:
            source = source_file.read()
        module = ast.parse(source, filename=main_path)
        constructed_handles = [node for node in ast.walk(module)
                               if isinstance(node, ast.Assign) and
                               any(isinstance(target, ast.Name) and
                                   target.id == 'fedbabu_fed_net'
                                   for target in node.targets)]

        self.assertEqual(constructed_handles, [])
        self.assertEqual(source.count('# fedbabu_fed_net = FederatedNetwork('), 5)
        for dataset_name in ('MNIST', 'F_MNIST', 'CIFAR_10', 'CIFAR_100',
                             'TINY_IMAGENET_200'):
            self.assertIn('dataset_name=constants.DatasetNames.' + dataset_name,
                          source)

    def test_client_finetunes_head_on_full_local_trainset_and_preserves_shared_body(self):
        """Use client train data, update only the head, and honor optimizer config."""
        parameters = {
            'fedbabu_head_finetune_epochs': 2,
            'fedbabu_head_finetune_learning_rate': 0.025,
            'fedbabu_head_finetune_momentum': 0.6,
            'fedbabu_head_finetune_weight_decay': 0.002,
        }
        client = self._make_client_with_parameters(parameters)
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client.fedbabu_personal_model = copy.deepcopy(server_model)
        personal_before = OrderedDict((key, value.clone())
                                      for key, value in client.fedbabu_personal_model.state_dict().items())
        client_before = OrderedDict((key, value.clone()) for key, value in client.model.state_dict().items())
        server_before = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())

        import models.utils as model_utils
        with patch('strategy.FedBABU.utils.train', wraps=model_utils.train) as train_mock:
            client.fine_tune_fedbabu_personal_head()

        train_mock.assert_called_once()
        call_args, call_kwargs = train_mock.call_args
        self.assertIs(call_args[1].dataset, client.local_trainset)
        self.assertEqual(call_args[2], 2)
        self.assertEqual(call_kwargs['_learning_rate'], 0.025)
        self.assertEqual(call_kwargs['_momentum'], 0.6)
        self.assertEqual(call_kwargs['_weight_decay'], 0.002)
        optimizer_parameters = call_kwargs['_optimizer_parameters']
        self.assertEqual(len(optimizer_parameters), 2)
        self.assertTrue(all(parameter is client.fedbabu_personal_model.fc2.weight or
                            parameter is client.fedbabu_personal_model.fc2.bias
                            for parameter in optimizer_parameters))

        personal_after = client.fedbabu_personal_model.state_dict()
        self.assertTrue(any(not torch.equal(personal_before[key], personal_after[key])
                            for key in ('fc2.weight', 'fc2.bias')))
        for key in ('body.weight', 'body.bias'):
            self.assertTrue(torch.equal(personal_before[key], personal_after[key]))
        for key, value in client_before.items():
            self.assertTrue(torch.equal(client.model.state_dict()[key], value))
        for key, value in server_before.items():
            self.assertTrue(torch.equal(server_model.state_dict()[key], value))

    def test_finetuning_keeps_batchnorm_buffers_in_frozen_body_unchanged(self):
        """Prevent train-mode BatchNorm statistics from modifying the body."""
        model = BufferedPartitionModel()
        before = OrderedDict((key, value.clone()) for key, value in model.state_dict().items())
        dataset = TensorDataset(torch.tensor([
            [1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0], [0.5, -1.0, 1.0], [-0.5, 1.0, -1.0],
        ]), torch.tensor([0, 1, 0, 1, 0, 1]))
        loader = DataLoader(dataset, batch_size=3, shuffle=False)

        train_fedbabu_head(model, loader, {'fedbabu_head_finetune_epochs': 2})

        after = model.state_dict()
        for key in before:
            if not key.startswith('fc2.'):
                self.assertTrue(torch.equal(before[key], after[key]), key)
        self.assertTrue(any(not torch.equal(before[key], after[key])
                            for key in ('fc2.weight', 'fc2.bias')))

    def test_finalization_runs_copy_finetune_and_evaluate_sequence(self):
        """Finalize after federation without mutating upload or server models."""
        torch.manual_seed(19)
        inputs = torch.tensor([
            [1.0, 0.0, 0.5], [0.0, 1.0, -0.5], [1.0, 1.0, 0.0],
            [-1.0, 0.5, 1.0], [0.5, -1.0, 1.0], [-0.5, 1.0, -1.0],
        ])
        labels = torch.tensor([0, 1, 0, 1, 0, 1])
        dataset = TensorDataset(inputs, labels)
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        server = Server(0, 0, aggregator_fn(), server_model, 1, 0.0)
        client = Client(2, True, PartitionModel(constants.ModelTypes.CNN_MODEL), 1, 3,
                        dataset, dataset, constants.RecoveryAlgorithm.FEDBABU, 1,
                        drift_recovery_parameters={'fedbabu_head_finetune_epochs': 2})
        client.parent_server_id = 0
        client.testloader = DataLoader(dataset, batch_size=3, shuffle=False)
        client_state_before = OrderedDict((key, value.clone()) for key, value in client.model.state_dict().items())
        server_state_before = OrderedDict((key, value.clone()) for key, value in server.model.state_dict().items())

        record = finalize_fedbabu_personalization([client], [server], 6)

        self.assertEqual(record['round'], 6)
        self.assertEqual(record['stage'], 'post_training_personalization')
        self.assertEqual(record['clients'][0]['model_role'], 'fedbabu_personalized')
        personal_state = client.fedbabu_personal_model.state_dict()
        self.assertTrue(any(not torch.equal(server_state_before[key], personal_state[key])
                            for key in ('fc2.weight', 'fc2.bias')))
        for key in ('body.weight', 'body.bias'):
            self.assertTrue(torch.equal(server_state_before[key], personal_state[key]))
        for key, value in client_state_before.items():
            self.assertTrue(torch.equal(client.model.state_dict()[key], value))
        for key, value in server_state_before.items():
            self.assertTrue(torch.equal(server.model.state_dict()[key], value))


class FedBABUTensorAggregationTests(unittest.TestCase):
    def test_strategy_updates_server_body_but_preserves_server_head(self):
        """Apply the averaged body through the existing non-strict setter only."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client_a = PartitionModel(constants.ModelTypes.CNN_MODEL).state_dict()
        client_b = PartitionModel(constants.ModelTypes.CNN_MODEL).state_dict()
        with torch.no_grad():
            for key in ('body.weight', 'body.bias'):
                client_a[key].fill_(2.0)
                client_b[key].fill_(6.0)
            client_a['fc2.weight'].fill_(10.0)
            client_b['fc2.weight'].fill_(-10.0)
        server_head_before = {key: server_model.state_dict()[key].clone()
                              for key in ('fc2.weight', 'fc2.bias')}

        aggregator_fn().aggregate_models(server_model, {4: client_a, 7: client_b})

        updated_state = server_model.state_dict()
        for key in ('body.weight', 'body.bias'):
            self.assertTrue(torch.equal(updated_state[key], torch.full_like(updated_state[key], 4.0)))
        for key, value in server_head_before.items():
            self.assertTrue(torch.equal(updated_state[key], value))

    def test_client_body_states_are_averaged_and_heads_excluded(self):
        """Average only body state, with no mutation of clients or server."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client_a = PartitionModel(constants.ModelTypes.CNN_MODEL).state_dict()
        client_b = PartitionModel(constants.ModelTypes.CNN_MODEL).state_dict()
        with torch.no_grad():
            for key in ('body.weight', 'body.bias'):
                client_a[key].fill_(1.0)
                client_b[key].fill_(3.0)
            client_a['fc2.weight'].fill_(100.0)
            client_b['fc2.weight'].fill_(-100.0)
        client_a_before = OrderedDict((key, value.clone()) for key, value in client_a.items())
        client_b_before = OrderedDict((key, value.clone()) for key, value in client_b.items())
        server_before = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())

        aggregated = aggregate_fedbabu_body_parameters(server_model, {8: client_a, 2: client_b})

        self.assertEqual(list(aggregated), ['body.weight', 'body.bias'])
        self.assertTrue(torch.equal(aggregated['body.weight'], torch.full_like(aggregated['body.weight'], 2.0)))
        self.assertTrue(torch.equal(aggregated['body.bias'], torch.full_like(aggregated['body.bias'], 2.0)))
        self.assertNotIn('fc2.weight', aggregated)
        self.assertNotIn('fc2.bias', aggregated)
        for key, value in client_a_before.items():
            self.assertTrue(torch.equal(client_a[key], value))
        for key, value in client_b_before.items():
            self.assertTrue(torch.equal(client_b[key], value))
        for key, value in server_before.items():
            self.assertTrue(torch.equal(server_model.state_dict()[key], value))

    def test_body_aggregation_validates_uploads_and_schema(self):
        """Reject empty uploads, malformed states, and unsupported model types."""
        model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        with self.assertRaisesRegex(ValueError, 'at least one client upload'):
            aggregate_fedbabu_body_parameters(model, {})

        state = model.state_dict()
        wrong_state = OrderedDict(state)
        wrong_state['body.weight'] = torch.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'tensor shape'):
            aggregate_fedbabu_body_parameters(model, {3: wrong_state})

        unsupported = PartitionModel('Unknown')
        with self.assertRaisesRegex(ValueError, 'does not support model type'):
            aggregate_fedbabu_body_parameters(unsupported, {3: unsupported.state_dict()})

    def test_server_aggregation_then_client_download_preserves_fixed_head(self):
        """Exercise normal server aggregation and download with a fixed head."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        client_models = [PartitionModel(constants.ModelTypes.CNN_MODEL) for _ in range(2)]
        with torch.no_grad():
            for key in ('fc2.weight', 'fc2.bias'):
                source_head = server_model.state_dict()[key]
                for client_model in client_models:
                    target = client_model.state_dict()[key]
                    target.copy_(source_head)
            for value, client_model in zip((2.0, 6.0), client_models):
                client_model.body.weight.fill_(value)
                client_model.body.bias.fill_(value)

        fixed_head = {key: server_model.state_dict()[key].clone()
                      for key in ('fc2.weight', 'fc2.bias')}
        clients = []
        for client_id, model in zip((4, 7), client_models):
            clients.append(SimpleNamespace(
                client_id=client_id,
                parent_server_id=0,
                model=model,
                drift_recovery_method=constants.RecoveryAlgorithm.FEDBABU,
                fit=lambda *args: None,
                evaluate=lambda: (1.0, 0.5),
            ))
        server = Server(0, 0, aggregator_fn(), server_model, 1, 0.0)
        server.client_ids = [4, 7]

        model_aggregation([[server]], None, clients, None, 0.0, False)

        for key in ('body.weight', 'body.bias'):
            self.assertTrue(torch.equal(server.model.state_dict()[key],
                                        torch.full_like(server.model.state_dict()[key], 4.0)))
        for key, value in fixed_head.items():
            self.assertTrue(torch.equal(server.model.state_dict()[key], value))

        model_distribution_hierarchy([[server]])
        train_client_models(clients, [4, 7], [server],
                            SimpleNamespace(is_drift=False, is_drift_end=False,
                                            drifted_client_indices=[]),
                            {'is_server_adaptability': False},
                            constants.RecoveryAlgorithm.FEDBABU)
        for client in clients:
            for key in ('body.weight', 'body.bias'):
                self.assertTrue(torch.equal(client.model.state_dict()[key],
                                            torch.full_like(client.model.state_dict()[key], 4.0)))
            for key, value in fixed_head.items():
                self.assertTrue(torch.equal(client.model.state_dict()[key], value))

    def test_failed_upload_validation_does_not_mutate_server(self):
        """Ensure invalid uploads are rejected before any server update."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        before = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
        strategy = aggregator_fn()
        with self.assertRaisesRegex(ValueError, 'at least one client upload'):
            strategy.aggregate_models(server_model, {})

        wrong_shape = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
        wrong_shape['body.weight'] = torch.zeros((2, 2))
        with self.assertRaisesRegex(ValueError, 'tensor shape'):
            strategy.aggregate_models(server_model, {4: wrong_shape})

        missing = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
        del missing['body.bias']
        with self.assertRaisesRegex(ValueError, 'state keys do not match'):
            strategy.aggregate_models(server_model, {4: missing})

        for key, value in before.items():
            self.assertTrue(torch.equal(server_model.state_dict()[key], value))

    def test_fedavg_still_averages_body_and_head(self):
        """Guard FedAvg's existing full-state equal-average behavior."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        uploads = {}
        for client_id, fill_value in ((4, 2.0), (7, 6.0)):
            state = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
            for value in state.values():
                value.fill_(fill_value)
            uploads[client_id] = state

        fedavg_aggregator_fn().aggregate_models(server_model, uploads)

        for value in server_model.state_dict().values():
            self.assertTrue(torch.equal(value, torch.full_like(value, 4.0)))

    def test_fedex_still_averages_only_its_existing_extractor_partition(self):
        """Guard FedEx's extractor-only update and unchanged classifier."""
        server_model = PartitionModel(constants.ModelTypes.CNN_MODEL)
        server_head = {key: server_model.state_dict()[key].clone()
                       for key in ('fc2.weight', 'fc2.bias')}
        uploads = {}
        for client_id, body_value, head_value in ((4, 1.0, 100.0), (7, 3.0, -100.0)):
            state = OrderedDict((key, value.clone()) for key, value in server_model.state_dict().items())
            state['body.weight'].fill_(body_value)
            state['body.bias'].fill_(body_value)
            state['fc2.weight'].fill_(head_value)
            state['fc2.bias'].fill_(head_value)
            uploads[client_id] = state

        fedex_aggregator_fn().aggregate_models(server_model, uploads)

        for key in ('body.weight', 'body.bias'):
            self.assertTrue(torch.equal(server_model.state_dict()[key],
                                        torch.full_like(server_model.state_dict()[key], 2.0)))
        for key, value in server_head.items():
            self.assertTrue(torch.equal(server_model.state_dict()[key], value))

    def test_floating_tensor_values_use_equal_mean_and_keep_dtype(self):
        """Average real-valued body state in the server dtype and device."""
        reference = torch.tensor([0.0, 0.0], dtype=torch.float64)
        result = aggregate_fedbabu_tensor_values(
            [torch.tensor([1.0, 5.0]), torch.tensor([3.0, 7.0])], reference)
        self.assertTrue(torch.equal(result, torch.tensor([2.0, 6.0], dtype=torch.float64)))
        self.assertEqual(result.dtype, reference.dtype)
        self.assertEqual(result.device, reference.device)

    def test_integer_counter_uses_max_and_keeps_integer_dtype(self):
        """Preserve the largest BatchNorm batch counter without fractional values."""
        reference = torch.tensor(0, dtype=torch.int64)
        result = aggregate_fedbabu_tensor_values(
            [torch.tensor(3, dtype=torch.int64), torch.tensor(8, dtype=torch.int64)], reference)
        self.assertEqual(result.item(), 8)
        self.assertEqual(result.dtype, torch.int64)

    def test_boolean_buffer_uses_elementwise_max(self):
        """Reduce boolean buffers using their discrete elementwise maximum."""
        reference = torch.tensor([False, False], dtype=torch.bool)
        result = aggregate_fedbabu_tensor_values(
            [torch.tensor([False, True]), torch.tensor([True, False])], reference)
        self.assertTrue(torch.equal(result, torch.tensor([True, True])))
        self.assertEqual(result.dtype, torch.bool)

    def test_invalid_tensor_inputs_fail_clearly(self):
        """Reject empty client sets, non-tensors, and shape mismatches."""
        reference = torch.zeros((2,))
        with self.assertRaisesRegex(ValueError, 'at least one'):
            aggregate_fedbabu_tensor_values([], reference)
        with self.assertRaisesRegex(ValueError, 'must be tensors'):
            aggregate_fedbabu_tensor_values([torch.ones(2), 'invalid'], reference)
        with self.assertRaisesRegex(ValueError, 'shape'):
            aggregate_fedbabu_tensor_values([torch.ones(3)], reference)


if __name__ == '__main__':
    unittest.main()
