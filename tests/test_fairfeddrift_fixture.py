"""Check the test-only fixture without running a simulation or detector."""
import unittest

import torch
from torch.utils.data import DataLoader

from fairfeddrift_fixtures import make_client_drift_fixture


class FairFedDriftFixtureTests(unittest.TestCase):
    def test_labels_and_counts(self):
        """Verify stationary clients and both simultaneous label-swap patterns."""
        before, after, drift_ids = make_client_drift_fixture()
        expected = {
            0: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
            1: [0, 2, 1, 4, 3, 5, 6, 7, 8, 9],
            2: [0, 1, 2, 3, 4, 7, 6, 5, 8, 9],
        }
        self.assertEqual(len(before), 6)
        self.assertEqual(set(before), set(after))
        for client_id, drift_id in drift_ids.items():
            with self.subTest(client_id=client_id):
                self.assertEqual(len(before[client_id]), 20)
                self.assertEqual(len(after[client_id]), 20)
                self.assertEqual(before[client_id].tensors[1].tolist(), expected[0] * 2)
                self.assertEqual(after[client_id].tensors[1].tolist(), expected[drift_id] * 2)
                self.assertTrue(torch.equal(before[client_id].tensors[0], after[client_id].tensors[0]))
                self.assertEqual(torch.bincount(after[client_id].tensors[1]).tolist(), [2] * 10)

    def test_no_metadata_and_two_field_batches(self):
        """Expose only inputs and labels, keeping oracle identities in the harness."""
        before, after, _ = make_client_drift_fixture()
        for datasets in (before, after):
            for dataset in datasets.values():
                self.assertFalse(hasattr(dataset, 'group_ids'))
                self.assertFalse(hasattr(dataset, 'drift_id'))
                self.assertEqual(len(dataset[0]), 2)
                inputs, labels = next(iter(DataLoader(dataset, batch_size=4)))
                self.assertEqual(inputs.shape, (4, 10))
                self.assertEqual(labels.dtype, torch.long)

    def test_independent_clients_and_snapshots(self):
        """Mutating one client must not alter other clients or prior snapshots."""
        before, after, _ = make_client_drift_fixture()
        expected_before = {key: tuple(t.clone() for t in dataset.tensors) for key, dataset in before.items()}
        expected_other = {key: tuple(t.clone() for t in dataset.tensors) for key, dataset in after.items() if key != 2}
        after[2].tensors[0].fill_(-1)
        after[2].tensors[1].fill_(9)
        for datasets, expected in ((before, expected_before), (after, expected_other)):
            for key, tensors in expected.items():
                for actual, original in zip(datasets[key].tensors, tensors):
                    self.assertTrue(torch.equal(actual, original))

    def test_repeatability_without_global_rng_changes(self):
        """Rebuilding the fixture gives fresh identical data without reseeding experiments."""
        state = torch.random.get_rng_state()
        first = make_client_drift_fixture()
        second = make_client_drift_fixture()
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        for first_stage, second_stage in zip(first[:2], second[:2]):
            for key in first_stage:
                for left, right in zip(first_stage[key].tensors, second_stage[key].tensors):
                    self.assertTrue(torch.equal(left, right))
                    self.assertNotEqual(left.data_ptr(), right.data_ptr())


if __name__ == '__main__':
    unittest.main()
