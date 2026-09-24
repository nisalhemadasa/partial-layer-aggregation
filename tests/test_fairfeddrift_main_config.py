"""Validate shared FairFedDrift experiment settings without importing main.py."""
import ast
from pathlib import Path
import unittest

import torch

from strategy.FairFedDrift import resolve_fairfeddrift_parameters, validate_fairfeddrift_setup


def find_assignment(tree, variable_name):
    return next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == variable_name for target in node.targets))


class FairFedDriftMainConfigTests(unittest.TestCase):
    def test_shared_strategy_and_phase_configuration_resolve(self):
        tree = ast.parse(Path('main.py').read_text(encoding='utf-8'))
        shared_assignment = find_assignment(tree, 'fairfeddrift_parameters')
        shared_parameters = {
            keyword.arg: ast.literal_eval(keyword.value)
            for keyword in shared_assignment.value.keywords
        }
        self.assertEqual(shared_parameters, {
            'fairfeddrift_loss_threshold': 0.1,
            'fairfeddrift_window': 100,
            'fairfeddrift_rounds_per_timestep': 1,
            'fairfeddrift_seed': 42
        })

        recovery_assignment = find_assignment(tree, 'fairfeddrift_recovery_parameters')
        recovery_keywords = {keyword.arg: keyword.value for keyword in recovery_assignment.value.keywords}
        for phase_key in ('recovery_method', 'base_aggregation_method'):
            self.assertIsInstance(recovery_keywords[phase_key], ast.Attribute)
            self.assertEqual(recovery_keywords[phase_key].attr, 'FAIRFEDDRIFT')
        self.assertTrue(any(keyword.arg is None and isinstance(keyword.value, ast.Name) and
                            keyword.value.id == 'fairfeddrift_parameters'
                            for keyword in recovery_assignment.value.keywords))
        common_parameters = {
            key: ast.literal_eval(recovery_keywords[key])
            for key in ('fedau_alpha', 'fedex_alpha')
        }
        self.assertNotIn('fedrc_cluster_count', recovery_keywords)
        self.assertNotIn('cluster_count', recovery_keywords)
        resolved = resolve_fairfeddrift_parameters(shared_parameters | common_parameters)
        self.assertEqual(resolved['fairfeddrift_window'], 100)
        validate_fairfeddrift_setup([1], 1.0, torch.device('cpu'))
        # Device validation is independent of local hardware availability.
        validate_fairfeddrift_setup([1], 1.0, torch.device('cuda'))

    def test_mnist_handle_has_disabled_constructor_and_run_call(self):
        source = Path('main.py').read_text(encoding='utf-8')
        fairfed_block = source.split('# #00000000000000000 FairFedDrift', 1)[1].split(
            '# # # #00000000000000000 Oracle', 1)[0]
        fairfed_block = '\n'.join(fairfed_block.splitlines()[1:])
        self.assertIn('fairfeddrift_fed_net = FederatedNetwork(', fairfed_block)
        self.assertIn('fairfeddrift_fed_net.run_simulation(', fairfed_block)
        self.assertIn("file_save_path='plots/swap/MNIST/saved_plots_fairfeddrift/'", fairfed_block)
        self.assertIn("log_save_path='logs/swap/MNIST/saved_logs_fairfeddrift/'", fairfed_block)
        self.assertIn('server_tree_layout=[1]', fairfed_block)
        self.assertIn('client_select_fraction=1', fairfed_block)
        executable_lines = [line for line in fairfed_block.splitlines()
                            if line.strip() and not line.lstrip().startswith('#')]
        self.assertEqual(executable_lines, [])

    def test_all_dataset_handles_are_disabled_and_have_separate_paths(self):
        lines = Path('main.py').read_text(encoding='utf-8').splitlines()
        marker = '# #00000000000000000 FairFedDrift'
        marker_indices = [index for index, line in enumerate(lines) if marker in line]
        expected = {
            'MNIST': ('DatasetNames.MNIST', 'plots/swap/MNIST/saved_plots_fairfeddrift/',
                      'logs/swap/MNIST/saved_logs_fairfeddrift/'),
            'F_MNIST': ('DatasetNames.F_MNIST', 'plots/swap/F_MNIST/saved_plots_fairfeddrift/',
                        'logs/swap/F_MNIST/saved_logs_fairfeddrift/'),
            'CIFAR_10': ('DatasetNames.CIFAR_10', 'plots/swap/CIFAR-10/saved_plots_fairfeddrift/',
                         'logs/swap/CIFAR-10/saved_logs_fairfeddrift/'),
            'CIFAR_100': ('DatasetNames.CIFAR_100', 'plots/swap/CIFAR-100/saved_plots_fairfeddrift/',
                          'logs/swap/CIFAR-100/saved_logs_fairfeddrift/'),
            'TINY_IMAGENET_200': ('DatasetNames.TINY_IMAGENET_200', 'plots/swap/Tiny/saved_plots_fairfeddrift/',
                                  'logs/swap/Tiny/saved_logs_fairfeddrift/')
        }
        self.assertEqual(len(marker_indices), len(expected))
        observed_datasets = set()
        for marker_index in marker_indices:
            block_lines = []
            for line in lines[marker_index + 1:]:
                if line.strip() and not line.lstrip().startswith('#'):
                    break
                if not line.strip():
                    if block_lines:
                        break
                    continue
                block_lines.append(line)
            block = '\n'.join(block_lines)
            self.assertTrue(all(line.lstrip().startswith('#') for line in block_lines))
            self.assertIn('fairfeddrift_fed_net = FederatedNetwork(', block)
            self.assertIn('fairfeddrift_fed_net.run_simulation(', block)
            self.assertIn('server_tree_layout=[1]', block)
            self.assertIn('client_select_fraction=1', block)
            self.assertIn('drift_recovery_parameters=fairfeddrift_recovery_parameters', block)
            matching = [key for key, values in expected.items()
                        if f'dataset_name=constants.{values[0]},' in block]
            self.assertEqual(len(matching), 1)
            dataset_key = matching[0]
            observed_datasets.add(dataset_key)
            self.assertIn(expected[dataset_key][1], block)
            self.assertIn(expected[dataset_key][2], block)
        self.assertEqual(observed_datasets, set(expected))


if __name__ == '__main__':
    unittest.main()
