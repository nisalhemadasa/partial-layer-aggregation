"""Checks for bounded history, snapshot isolation and communication-round expiry."""
import gc
import unittest
import weakref

import torch
from torch.utils.data import Subset, TensorDataset

from data.utils import DatasetSnapshot, convert_dataset_to_loader
from strategy.FairFedDrift.utils import ClientDataHistory, build_fairfeddrift_history_loaders


class FairFedDriftHistoryTests(unittest.TestCase):
    def dataset(self):
        """
        Build ordinary local samples with visible identities.
        :return: A tensor dataset with six samples.
        """
        return TensorDataset(torch.arange(12).reshape(6, 2).float(), torch.arange(6))

    def test_snapshot_isolates_selected_samples(self):
        """Copy nested subset values without retaining live tensors or mutable outputs."""
        source = self.dataset()
        snapshot = DatasetSnapshot(Subset(Subset(source, [5, 2, 0]), [1, 0]))
        source.tensors[0].fill_(-1)
        source.tensors[1].fill_(9)
        inputs, label = snapshot[0]
        self.assertEqual(inputs.tolist(), [4, 5])
        self.assertEqual(label.item(), 2)
        inputs.fill_(99)
        label.fill_(99)
        batch, labels = next(iter(convert_dataset_to_loader(snapshot, 2, False)))
        self.assertEqual(batch.tolist(), [[4, 5], [10, 11]])
        self.assertEqual(labels.tolist(), [2, 5])
        self.assertEqual(batch.device.type, 'cpu')
        self.assertFalse(batch.requires_grad)

    def test_default_window_and_release(self):
        """The 101st arrival evicts round zero and releases the historical object."""
        history = ClientDataHistory()
        source = self.dataset()
        history.add(0, source)
        oldest = weakref.ref(history.records()[0][1])
        for round_idx in range(1, 101):
            unique_version = torch.utils.data.TensorDataset(
                source.tensors[0] + round_idx, source.tensors[1])
            history.add(round_idx, unique_version)
        self.assertEqual([r for r, _ in history.records()], list(range(1, 101)))
        gc.collect()
        self.assertIsNone(oldest())
        self.assertEqual(history.advance(200), list(range(1, 101)))
        self.assertEqual(history.records(), [])

    def test_reuse_does_not_refresh_arrival(self):
        """Inner rounds and duplicate calls do not extend the lifetime of a batch."""
        history = ClientDataHistory(window=2)
        source = self.dataset()
        self.assertTrue(history.add(0, source))
        history.advance(1)
        self.assertFalse(history.add(0, source))
        self.assertEqual(history.advance(2), [0])
        self.assertFalse(history.add(0, source))
        self.assertEqual(history.records(), [])
        self.assertTrue(history.add(2, source))
        self.assertEqual([r for r, _ in history.records()], [2])

    def test_identical_versions_share_one_snapshot_until_all_arrivals_expire(self):
        """Retain round references without storing repeated input/label values twice."""
        history = ClientDataHistory(window=2)
        first = self.dataset()
        same_values = self.dataset()
        changed_labels = torch.utils.data.TensorDataset(
            first.tensors[0].clone(), torch.tensor([1, 0, 2, 3, 4, 5]))
        history.add(0, first)
        first_snapshot = history.records()[0][1]
        history.add(1, same_values)
        self.assertIs(history.records()[0][1], history.records()[1][1])
        self.assertEqual(history._snapshot_reference_counts[history._arrival_fingerprints[0]], 2)
        history.add(2, changed_labels)
        self.assertEqual([round_idx for round_idx, _ in history.records()], [1, 2])
        self.assertIs(history.records()[0][1], first_snapshot)
        self.assertIsNot(history.records()[1][1], first_snapshot)
        history.advance(3)
        self.assertEqual([round_idx for round_idx, _ in history.records()], [2])
        self.assertEqual(len(history._snapshot_cache), 1)

    def test_unbounded_and_large_clock_jump(self):
        """Explicit unbounded history retains all arrivals across a large round gap."""
        history = ClientDataHistory(window=None)
        history.add(0, self.dataset())
        history.add(10, self.dataset())
        self.assertEqual(history.advance(1000), [])
        self.assertEqual([r for r, _ in history.records()], [0, 10])

    def test_history_loaders_group_one_clients_data_by_historical_cluster(self):
        """A client contributes distinct retained versions to two learned clusters."""
        history = ClientDataHistory(window=4)
        original = self.dataset()
        repeated_version = self.dataset()
        changed_labels = torch.utils.data.TensorDataset(
            original.tensors[0].clone(), torch.tensor([9, 8, 7, 6, 5, 4]))
        history.add(0, original)
        history.add(1, repeated_version)
        history.add(2, changed_labels)

        loaders, sample_counts = build_fairfeddrift_history_loaders(
            history, {0: 4, 1: 4, 2: 9}, batch_size=4)

        self.assertEqual(set(loaders), {4, 9})
        self.assertEqual(sample_counts, {4: 6, 9: 6})
        self.assertEqual(len(loaders[4].dataset.datasets), 1)
        self.assertEqual(len(loaders[9].dataset.datasets), 1)
        labels_by_cluster = {cluster_id: [] for cluster_id in loaders}
        for cluster_id, loader in loaders.items():
            for _, labels in loader:
                labels_by_cluster[cluster_id].extend(labels.tolist())
        self.assertEqual(sorted(labels_by_cluster[4]), [0, 1, 2, 3, 4, 5])
        self.assertEqual(sorted(labels_by_cluster[9]), [4, 5, 6, 7, 8, 9])

    def test_history_loader_rejects_missing_assignments_and_bad_batch_size(self):
        history = ClientDataHistory()
        history.add(0, self.dataset())
        with self.assertRaisesRegex(ValueError, 'missing'):
            build_fairfeddrift_history_loaders(history, {}, 2)
        with self.assertRaisesRegex(ValueError, 'batch size'):
            build_fairfeddrift_history_loaders(history, {0: 3}, 0)

    def test_invalid_capture_preserves_history(self):
        """Reject invalid clocks/windows and empty batches without partial mutation."""
        for window in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                ClientDataHistory(window)
        history = ClientDataHistory(window=1)
        history.add(0, self.dataset())
        with self.assertRaisesRegex(ValueError, 'empty'):
            history.add(1, Subset(self.dataset(), []))
        self.assertEqual(history.current_round, 0)
        self.assertEqual([r for r, _ in history.records()], [0])
        history.advance(3)
        for round_idx in (-1, True, 1.5, 2):
            with self.assertRaises(ValueError):
                history.advance(round_idx)
        with self.assertRaises(ValueError):
            history.add(1, self.dataset())


if __name__ == '__main__':
    unittest.main()
