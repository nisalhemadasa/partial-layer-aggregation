"""Numerical checks for historical scalar-loss merge eligibility."""
import math
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import TensorDataset

from device_utils import configure_device
from strategy.FairFedDrift import aggregator_fn
from strategy.FairFedDrift.utils import (
    ClientDataHistory, calculate_fairfeddrift_merge_distances,
    update_fairfeddrift_complete_linkage, average_fairfeddrift_parameters,
)


class LossTableModel(nn.Module):
    def __init__(self, losses):
        super().__init__()
        probabilities = torch.exp(-torch.tensor(losses, dtype=torch.float64))
        self.register_buffer('logits', torch.stack((probabilities.log(), torch.log1p(-probabilities)), 1))

    def forward(self, inputs):
        return self.logits[inputs.long().flatten()]


def samples(marker, count=1):
    return TensorDataset(torch.full((count, 1), marker, dtype=torch.float32),
                         torch.zeros(count, dtype=torch.long))


class MustNotEvaluate(nn.Module):
    def forward(self, inputs):
        raise AssertionError('A cluster without retained history must not be evaluated')


class FairFedDriftMergeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device('cpu')

    def history(self, marker, count=1):
        history = ClientDataHistory()
        history.add(0, samples(marker, count))
        history.advance(1)
        return history

    def test_all_client_pairs_block_merge_that_cluster_averages_would_accept(self):
        models = {3: LossTableModel([.25, .75, .75]), 8: LossTableModel([.5, .5, .5])}
        histories = {0: self.history(0), 1: self.history(1), 2: self.history(2)}
        assignments = {0: {0: 3}, 1: {0: 3}, 2: {0: 8}}
        distances, counts = calculate_fairfeddrift_merge_distances(models, histories, assignments, 1, .3)
        self.assertTrue(math.isinf(distances[3][8]))
        self.assertEqual(counts, {3: 2, 8: 1})
        distances, _ = calculate_fairfeddrift_merge_distances(models, histories, assignments, 1, .6)
        self.assertAlmostEqual(distances[3][8], .5)
        self.assertEqual(distances[3][8], distances[8][3])
        self.assertTrue(math.isinf(distances[3][3]))
        # Use the computed distance to test the inclusive boundary exactly.
        boundary, _ = calculate_fairfeddrift_merge_distances(models, histories, assignments, 1, distances[3][8])
        self.assertEqual(boundary[3][8], distances[3][8])

    def test_client_arrivals_are_combined_by_sample_count(self):
        models = {3: LossTableModel([.25, .75, .75]), 8: LossTableModel([.5, .5, .5])}
        history = self.history(0)
        history.add(1, samples(1, 3))
        history.advance(2)
        other = self.history(2)
        other.advance(2)
        distances, counts = calculate_fairfeddrift_merge_distances(
            models, {0: history, 1: other}, {0: {0: 3, 1: 3}, 1: {0: 8}}, 2, .2, batch_size=3)
        self.assertAlmostEqual(distances[3][8], .125)
        self.assertEqual(counts, {3: 4, 8: 1})

    def test_excludes_entire_current_timestep_and_expired_arrivals(self):
        models = {3: LossTableModel([.5]), 8: LossTableModel([.5]), 12: MustNotEvaluate()}
        histories = {}
        assignments = {}
        for client, cluster in ((0, 3), (1, 8)):
            history = ClientDataHistory(window=3)
            history.add(0, samples(99))  # Invalid marker would fail if evaluated.
            history.add(1, samples(0))
            history.add(2, samples(99))
            history.add(3, samples(99))
            histories[client] = history
            assignments[client] = {0: 999, 1: cluster}
        distances, counts = calculate_fairfeddrift_merge_distances(models, histories, assignments, 2, 0)
        self.assertEqual(distances[3][8], 0)
        self.assertTrue(math.isinf(distances[3][12]))
        self.assertEqual(counts, {3: 1, 8: 1, 12: 0})
        self.assertEqual([r for r, _ in histories[0].records()], [1, 2, 3])

    def test_no_history_and_invalid_history_fail_clearly(self):
        models = {3: MustNotEvaluate(), 8: MustNotEvaluate()}
        distances, counts = calculate_fairfeddrift_merge_distances(models, {}, {}, 0, .25)
        self.assertTrue(math.isinf(distances[3][8]))
        self.assertEqual(counts, {3: 0, 8: 0})
        history = self.history(0)
        for assignments in ({}, {0: {0: 99}}):
            with self.assertRaises(ValueError):
                calculate_fairfeddrift_merge_distances(models, {0: history}, assignments, 1, .25)
        with self.assertRaises(ValueError):
            calculate_fairfeddrift_merge_distances(models, {0: history}, {0: {0: 3}}, 2, .25)
        for threshold in (-1, True, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                calculate_fairfeddrift_merge_distances(models, {}, {}, 0, threshold)

    def test_history_free_models_are_not_evaluated(self):
        models = {3: LossTableModel([.5, .5]), 8: LossTableModel([.5, .5]), 12: MustNotEvaluate()}
        histories = {0: self.history(0), 1: self.history(0)}
        assignments = {0: {0: 3}, 1: {0: 8}}
        distances, counts = calculate_fairfeddrift_merge_distances(
            models, histories, assignments, 1, .25)
        self.assertEqual(distances[3][8], 0)
        self.assertTrue(math.isinf(distances[3][12]))
        self.assertTrue(math.isinf(distances[8][12]))
        self.assertEqual(counts, {3: 1, 8: 1, 12: 0})

    def test_one_history_bearing_cluster_skips_all_model_evaluation(self):
        models = {3: LossTableModel([.5]), 8: MustNotEvaluate()}
        history = ClientDataHistory()
        history.add(1, samples(0))
        history.advance(2)
        distances, counts = calculate_fairfeddrift_merge_distances(
            models, {0: history}, {0: {1: 3}}, 2, .25)
        self.assertTrue(math.isinf(distances[3][8]))
        self.assertEqual(counts, {3: 1, 8: 0})

    def test_complete_linkage_retains_worst_distance_and_does_not_mutate(self):
        distances = {3: {3: math.inf, 8: .1, 12: .2},
                     8: {3: .1, 8: math.inf, 12: math.inf},
                     12: {3: .2, 8: math.inf, 12: math.inf}}
        updated = update_fairfeddrift_complete_linkage(distances, 3, 8, 15)
        self.assertEqual(list(updated), [12, 15])
        self.assertTrue(math.isinf(updated[15][12]))
        self.assertEqual(list(distances), [3, 8, 12])
        distances[8][12] = distances[12][8] = .3
        updated = update_fairfeddrift_complete_linkage(distances, 3, 8, 15)
        self.assertEqual(updated[15][12], .3)
        self.assertEqual(updated[12][15], .3)
        with self.assertRaises(ValueError):
            update_fairfeddrift_complete_linkage(distances, 3, 8, 12)


class FairFedDriftMergeExecutionTests(unittest.TestCase):
    def test_weighting_rejects_invalid_counts_and_incompatible_models(self):
        parameters = {'weight': torch.tensor([0.])}
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                average_fairfeddrift_parameters([parameters, parameters], [1, count])
        for other in ({'other': torch.tensor([4.])}, {'weight': torch.tensor([4., 4.])},
                      {'weight': torch.tensor([4.], dtype=torch.float64)}):
            with self.assertRaises(ValueError):
                average_fairfeddrift_parameters([parameters, other], [1, 3])
        self.assertEqual(parameters['weight'].item(), 0.)

    def setup_clusters(self, values=(0., 4., 8.), counts=(1, 3, 4)):
        strategy = aggregator_fn()
        model = nn.Linear(1, 1, bias=False)
        model.register_buffer('counter', torch.tensor(0))
        strategy.initialize_clusters(model)
        histories = {}
        for cluster, (value, count) in enumerate(zip(values, counts)):
            if cluster:
                strategy.assign_client(cluster, {0: 1001.}, 0)
            else:
                strategy.assign_client(cluster, {0: .5}, 0)
            with torch.no_grad():
                strategy.cluster_models[cluster].weight.fill_(value)
                strategy.cluster_models[cluster].counter.fill_(cluster + 1)
            history = ClientDataHistory()
            history.add(0, samples(0, count))
            history.add(1, samples(0, 20))
            histories[cluster] = history
            strategy.record_assignment(cluster, 0)
            strategy.record_assignment(cluster, 1)
        return strategy, histories

    def test_repeated_merges_weights_counts_and_all_assignments(self):
        strategy, histories = self.setup_clusters()
        original_models = dict(strategy.cluster_models)
        references = dict(strategy.previous_losses)
        distances = {0: {0: math.inf, 1: .1, 2: .3},
                     1: {0: .1, 1: math.inf, 2: .2},
                     2: {0: .3, 1: .2, 2: math.inf}}
        with patch('strategy.FairFedDrift.fairfeddrift.calculate_fairfeddrift_merge_distances',
                   return_value=(distances, {0: 1, 1: 3, 2: 4})) as calculate:
            records = strategy.merge_clusters(histories, 1, .5)
        self.assertEqual(calculate.call_count, 1)
        self.assertEqual([r['source_ids'] for r in records], [(0, 1), (2, 3)])
        self.assertEqual([r['sample_count'] for r in records], [4, 8])
        self.assertEqual(records[1]['distance'], .3)
        self.assertEqual(list(strategy.cluster_models), [4])
        self.assertEqual(strategy.cluster_models[4].weight.item(), 5.5)
        self.assertEqual(strategy.cluster_models[4].counter.item(), 3)  # First contributor on tie.
        self.assertEqual(strategy.client_assignments, {0: 4, 1: 4, 2: 4})
        self.assertEqual(strategy.assignment_history, {i: {0: 4, 1: 4} for i in range(3)})
        self.assertEqual(strategy.previous_losses, references)
        for cluster, model in original_models.items():
            self.assertEqual(model.weight.item(), (0., 4., 8.)[cluster])
            self.assertNotEqual(model.weight.data_ptr(), strategy.cluster_models[4].weight.data_ptr())
        self.assertEqual(strategy.assign_client(5, {4: 1001.}, 0), (5, True))

    def test_real_history_evaluation_excludes_current_counts_and_keeps_new_cluster(self):
        strategy, histories = self.setup_clusters(values=(0., 0.), counts=(1, 3))
        # One-class logits have zero cross-entropy, allowing an exact zero-threshold merge.
        strategy.assign_client(2, {0: 1001.}, 0)
        histories[2] = ClientDataHistory()
        histories[2].add(1, samples(0, 30))
        strategy.record_assignment(2, 1)
        records = strategy.merge_clusters(histories, 1, 0)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]['sample_counts'], (1, 3))
        self.assertEqual(list(strategy.cluster_models), [2, 3])
        self.assertEqual(strategy.cluster_models[3].counter.item(), 2)
        self.assertEqual(strategy.client_assignments, {0: 3, 1: 3, 2: 2})
        self.assertEqual(strategy.assignment_history[2], {1: 2})

    def test_complete_linkage_blocks_further_merge(self):
        strategy, histories = self.setup_clusters()
        distances = {0: {0: math.inf, 1: .1, 2: .2},
                     1: {0: .1, 1: math.inf, 2: math.inf},
                     2: {0: .2, 1: math.inf, 2: math.inf}}
        with patch('strategy.FairFedDrift.fairfeddrift.calculate_fairfeddrift_merge_distances',
                   return_value=(distances, {0: 1, 1: 3, 2: 4})):
            records = strategy.merge_clusters(histories, 1, .5)
        self.assertEqual(len(records), 1)
        self.assertEqual(list(strategy.cluster_models), [2, 3])
        self.assertEqual(strategy.cluster_models[3].weight.item(), 3.)

    def test_failed_second_merge_does_not_commit_first_merge(self):
        strategy, histories = self.setup_clusters()
        original_models = dict(strategy.cluster_models)
        original_assignments = {i: dict(a) for i, a in strategy.assignment_history.items()}
        from models.utils import set_parameters
        calls = []

        def fail_second(model, parameters):
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError('load failed')
            set_parameters(model, parameters)

        with patch('strategy.FairFedDrift.fairfeddrift.set_parameters', side_effect=fail_second):
            with self.assertRaisesRegex(RuntimeError, 'load failed'):
                strategy.merge_clusters(histories, 1, 0)
        self.assertEqual(strategy.cluster_models, original_models)
        self.assertEqual(strategy.assignment_history, original_assignments)
        self.assertEqual(strategy.client_assignments, {i: i for i in range(3)})
        self.assertEqual(strategy._next_cluster_id, 3)

    def test_expiry_and_duplicate_arrival_preserve_correct_assignments(self):
        strategy, histories = self.setup_clusters(values=(0., 0.), counts=(1, 3))
        histories[0].window = histories[1].window = 1
        for history in histories.values():
            history.advance(1)
        strategy.client_assignments[0] = 1
        strategy.record_assignment(0, 0)
        self.assertEqual(strategy.assignment_history[0][0], 0)
        self.assertEqual(strategy.merge_clusters(histories, 1, 0), [])
        self.assertEqual(strategy.assignment_history, {0: {1: 0}, 1: {1: 1}})
        with self.assertRaises(ValueError):
            strategy.merge_clusters({0: histories[0]}, 1, 0)


if __name__ == '__main__':
    unittest.main()
