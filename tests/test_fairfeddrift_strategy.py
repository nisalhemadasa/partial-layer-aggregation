"""Focused FairFedDrift configuration checks."""
import unittest
from unittest.mock import patch

import torch

from strategy.FairFedDrift import aggregator_fn, resolve_fairfeddrift_parameters, validate_fairfeddrift_setup


class FairFedDriftConfigurationTests(unittest.TestCase):
    def parameters(self):
        """
        Build an explicit local-loss threshold for configuration checks.
        :return: Fresh recovery settings dictionary.
        """
        return dict(fairfeddrift_loss_threshold=0.25)

    def test_defaults_and_input_preservation(self):
        """Resolve defaults without changing shared experiment settings."""
        parameters = self.parameters()
        parameters['fedex_alpha'] = 0.9
        before = dict(parameters)
        resolved = resolve_fairfeddrift_parameters(parameters)
        self.assertEqual(parameters, before)
        self.assertEqual(resolved['fairfeddrift_window'], 100)
        self.assertEqual(resolved['fairfeddrift_rounds_per_timestep'], 1)
        self.assertEqual(resolved['fairfeddrift_seed'], 42)
        self.assertEqual(resolved['fairfeddrift_loss_threshold'], 0.25)
        self.assertNotIn('fedex_alpha', resolved)

    def test_explicit_settings(self):
        """Accept finite windows, multiple rounds and integer seeds."""
        parameters = self.parameters()
        parameters.update(fairfeddrift_window=2, fairfeddrift_rounds_per_timestep=10, fairfeddrift_seed=7)
        self.assertEqual(resolve_fairfeddrift_parameters(parameters), parameters)

    def test_round_window_independent_of_timestep_length(self):
        """Keep a 100-round default when timesteps contain multiple rounds."""
        parameters = self.parameters()
        parameters['fairfeddrift_rounds_per_timestep'] = 10
        self.assertEqual(resolve_fairfeddrift_parameters(parameters)['fairfeddrift_window'], 100)
        parameters['fairfeddrift_window'] = None
        self.assertIsNone(resolve_fairfeddrift_parameters(parameters)['fairfeddrift_window'])

    def test_invalid_thresholds(self):
        """Reject missing, negative, nonnumeric and nonfinite thresholds."""
        for key in self.parameters():
            for value in (None, True, False, -1, float('nan'), float('inf'), float('-inf'), '0.1'):
                with self.subTest(key=key, value=value):
                    parameters = self.parameters()
                    parameters[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        resolve_fairfeddrift_parameters(parameters)
            parameters = self.parameters()
            del parameters[key]
            with self.assertRaisesRegex(ValueError, key):
                resolve_fairfeddrift_parameters(parameters)

    def test_zero_and_positive_thresholds(self):
        """Accept explicit thresholds without any sensitive-group settings."""
        for threshold in (0, 0.1, 2):
            with self.subTest(threshold=threshold):
                resolved = resolve_fairfeddrift_parameters(dict(fairfeddrift_loss_threshold=threshold))
                self.assertEqual(resolved['fairfeddrift_loss_threshold'], float(threshold))
                self.assertIsInstance(resolved['fairfeddrift_loss_threshold'], float)

    def test_obsolete_threshold_keys(self):
        """Reject old keys even when the replacement threshold is also provided."""
        old_keys = ('fairfeddrift_threshold_privileged', 'fairfeddrift_threshold_unprivileged')
        for keys in ((old_keys[0],), (old_keys[1],), old_keys):
            for include_new in (False, True):
                with self.subTest(keys=keys, include_new=include_new):
                    parameters = self.parameters() if include_new else {}
                    parameters.update({key: None for key in keys})
                    before = dict(parameters)
                    with self.assertRaisesRegex(ValueError, 'Remove these keys and set fairfeddrift_loss_threshold') as error:
                        resolve_fairfeddrift_parameters(parameters)
                    for key in keys:
                        self.assertIn(key, str(error.exception))
                    self.assertEqual(parameters, before)

    def test_invalid_window_rounds_and_seed(self):
        """Reject ambiguous window values and invalid integer settings."""
        cases = {
            'fairfeddrift_window': (True, 0, -1, 1.5, float('inf'), 'inf'),
            'fairfeddrift_rounds_per_timestep': (True, 0, -1, 1.5, None),
            'fairfeddrift_seed': (True, 1.5, '42', None),
        }
        for key, values in cases.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    parameters = self.parameters()
                    parameters[key] = value
                    with self.assertRaisesRegex(ValueError, key):
                        resolve_fairfeddrift_parameters(parameters)
        with self.assertRaises(ValueError):
            resolve_fairfeddrift_parameters(None)

    def test_supported_setup(self):
        """Allow full-participation CPU or CUDA execution with flat cluster servers."""
        for layout in ([1], [3], (2,)):
            for device in (torch.device('cpu'), torch.device('cuda')):
                validate_fairfeddrift_setup(layout, 1.0, device)

    def test_unsupported_setup(self):
        """Reject deep layouts, partial participation and unresolved devices."""
        for layout in ([], [1, 2], [0], [-1], [True], [1.5], None):
            with self.subTest(layout=layout), self.assertRaisesRegex(ValueError, 'single-level'):
                validate_fairfeddrift_setup(layout, 1, torch.device('cpu'))
        for fraction in (True, 0, 0.5, 2, float('nan'), '1'):
            with self.subTest(fraction=fraction), self.assertRaisesRegex(ValueError, 'full participation'):
                validate_fairfeddrift_setup([1], fraction, torch.device('cpu'))
        for device in (torch.device('mps'), 'auto', None):
            with self.subTest(device=device), self.assertRaisesRegex(ValueError, 'CPU or CUDA'):
                validate_fairfeddrift_setup([1], 1, device)


class FairFedDriftSelectionTests(unittest.TestCase):
    def test_initial_reference_and_lowest_candidate(self):
        strategy = aggregator_fn()
        self.assertEqual(strategy.select_model(0, {7: 1000.0}, 0), (7, 1000.0))
        self.assertEqual(strategy.select_model(1, {7: 1000.1}, 0), (None, 1000.1))
        self.assertEqual(strategy.select_model(2, {7: .75, 3: .25, 9: .5}, 0), (3, .25))

    def test_absolute_threshold_boundary_and_zero(self):
        for loss, expected in ((.75, 4), (.750001, None), (.749999, 4)):
            strategy = aggregator_fn()
            strategy.select_model(0, {4: .5}, 0)
            self.assertEqual(strategy.select_model(0, {4: loss}, .25), (expected, loss))
        strategy = aggregator_fn()
        strategy.select_model(0, {4: .5}, 0)
        self.assertEqual(strategy.select_model(0, {4: .5}, 0), (4, .5))

    def test_failure_reference_and_clients_remain_independent(self):
        strategy = aggregator_fn()
        strategy.select_model(0, {4: .25}, .125)
        strategy.select_model(1, {4: 1.0}, .125)
        losses = {4: .75, 8: .5}
        self.assertEqual(strategy.select_model(0, losses, .125), (None, .5))
        self.assertEqual(strategy.previous_losses[1], 1.0)
        self.assertEqual(strategy.select_model(1, losses, .125), (8, .5))
        self.assertEqual(strategy.select_model(0, {8: .625}, .125), (8, .625))
        self.assertEqual(losses, {4: .75, 8: .5})

    def test_ties_follow_active_model_order(self):
        strategy = aggregator_fn()
        self.assertEqual(strategy.select_model(0, {9: .5, 2: .5}, .25), (9, .5))
        self.assertEqual(strategy.select_model(0, {2: .5, 9: .5}, .25), (2, .5))

    def test_invalid_input_does_not_update_reference(self):
        strategy = aggregator_fn()
        strategy.select_model(0, {4: .5}, .25)
        invalid = [{}, None, {None: .5}, {-1: .5}, {True: .5}]
        invalid += [{4: .5, 8: loss} for loss in
                    (-1, float('nan'), float('inf'), None, True, '0.5')]
        for candidates in invalid:
            with self.subTest(candidates=candidates), self.assertRaises(ValueError):
                strategy.select_model(0, candidates, .25)
            self.assertEqual(strategy.previous_losses, {0: .5})
        for threshold in (-1, float('nan'), float('inf'), True, None):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                strategy.select_model(1, {4: .25}, threshold)
            self.assertEqual(strategy.previous_losses, {0: .5})


class FairFedDriftClusterTests(unittest.TestCase):
    def test_initialization_and_split_copy_parameters_and_buffers(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.BatchNorm1d(2))
        strategy = aggregator_fn()
        self.assertEqual(strategy.initialize_clusters(model), 0)
        self.assertEqual(strategy.assign_client(4, {0: .25}, .125), (0, False))
        self.assertEqual(strategy.assign_client(4, {0: .75}, .125), (1, True))
        self.assertEqual(strategy.client_assignments, {4: 1})
        self.assertEqual(strategy.previous_losses, {4: .75})
        copies = [model, strategy.cluster_models[0], strategy.cluster_models[1]]
        for key, original in model.state_dict().items():
            tensors = [copy.state_dict()[key] for copy in copies]
            self.assertEqual(len({tensor.data_ptr() for tensor in tensors}), 3)
            self.assertTrue(all(torch.equal(original, tensor) for tensor in tensors))
        with torch.no_grad():
            strategy.cluster_models[1][0].weight.add_(10)
        self.assertTrue(torch.equal(model[0].weight, strategy.cluster_models[0][0].weight))
        self.assertFalse(torch.equal(model[0].weight, strategy.cluster_models[1][0].weight))

    def test_reassignment_and_frozen_candidates(self):
        strategy = aggregator_fn()
        strategy.initialize_clusters(torch.nn.Linear(2, 2))
        strategy.assign_client(0, {0: .25}, 0)
        strategy.assign_client(1, {0: .25}, 0)
        self.assertEqual(strategy.assign_client(0, {0: .5}, 0), (1, True))
        # Another client uses the same pre-creation candidate set.
        self.assertEqual(strategy.assign_client(1, {0: .5}, 0), (2, True))
        self.assertEqual(strategy.assign_client(0, {0: .5, 1: .5, 2: .25}, 0), (2, False))
        self.assertEqual(strategy.client_assignments, {0: 2, 1: 2})
        self.assertEqual(len(strategy.cluster_models), 3)

    def test_first_active_copy_and_monotonic_ids(self):
        strategy = aggregator_fn()
        strategy.initialize_clusters(torch.nn.Linear(1, 1, bias=False))
        strategy.assign_client(0, {0: 1001.0}, 0)
        strategy.assign_client(1, {0: 1001.0}, 0)
        # Emulate retirement of an unassigned model; merging is a later step.
        del strategy.cluster_models[0]
        with torch.no_grad():
            strategy.cluster_models[1].weight.fill_(3)
            strategy.cluster_models[2].weight.fill_(9)
        self.assertEqual(strategy.assign_client(2, {1: 1002.0, 2: 1001.0}, 0), (3, True))
        self.assertEqual(strategy.cluster_models[3].weight.item(), 3)
        self.assertEqual(list(strategy.cluster_models), [1, 2, 3])

    def test_invalid_operations_preserve_state(self):
        strategy = aggregator_fn()
        with self.assertRaises(ValueError):
            strategy.assign_client(0, {0: .5}, 0)
        with self.assertRaises(ValueError):
            strategy.initialize_clusters(None)
        strategy.initialize_clusters(torch.nn.Linear(1, 1))
        strategy.assign_client(0, {0: .5}, 0)
        with self.assertRaises(ValueError):
            strategy.initialize_clusters(torch.nn.Linear(1, 1))
        for client, losses in ((0, {5: .25}), (0, {}), (-1, {0: .25}), (0, {0: float('nan')})):
            with self.subTest(client=client, losses=losses), self.assertRaises(ValueError):
                strategy.assign_client(client, losses, 0)
            self.assertEqual(strategy.previous_losses, {0: .5})
            self.assertEqual(strategy.client_assignments, {0: 0})
            self.assertEqual(list(strategy.cluster_models), [0])

    def test_copy_failure_rolls_back_reference_and_id(self):
        strategy = aggregator_fn()
        strategy.initialize_clusters(torch.nn.Linear(1, 1))
        strategy.assign_client(0, {0: .5}, 0)
        with patch('strategy.FairFedDrift.fairfeddrift.deepcopy', side_effect=RuntimeError('copy failed')):
            for client in (0, 1):
                with self.assertRaisesRegex(RuntimeError, 'copy failed'):
                    strategy.assign_client(client, {0: 1001.0}, 0)
        self.assertEqual(strategy.previous_losses, {0: .5})
        self.assertEqual(strategy.client_assignments, {0: 0})
        self.assertEqual(list(strategy.cluster_models), [0])
        self.assertEqual(strategy.assign_client(0, {0: .75}, 0), (1, True))


if __name__ == '__main__':
    unittest.main()
