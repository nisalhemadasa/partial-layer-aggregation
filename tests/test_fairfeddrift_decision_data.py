"""Check current decision data isolation from evaluation data and retained history."""
from types import SimpleNamespace
import unittest

import torch
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from strategy.FairFedDrift.utils import ClientDataHistory, prepare_fairfeddrift_decision_loader


class ChangingInputs(Dataset):
    """Simulate an input transform that changes on every sample read."""

    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 6

    def __getitem__(self, index):
        self.reads += 1
        return torch.tensor([index, self.reads], dtype=torch.float32), index


class TrainingOnlyClient:
    def __init__(self, trainloader):
        self.trainloader = trainloader

    @property
    def testloader(self):
        raise AssertionError('Decision preparation must not access testloader')

    @property
    def testset(self):
        raise AssertionError('Decision preparation must not access testset')

    def sample_data(self):
        """Guard against candidate preparation resampling the current client batch."""
        raise AssertionError('Decision preparation must reuse the already selected samples')


class FairFedDriftDecisionDataTests(unittest.TestCase):
    def test_candidates_receive_identical_selected_samples(self):
        """Freeze transforms once and retain the final partial batch for all candidates."""
        source = ChangingInputs()
        selected = Subset(Subset(source, [5, 1, 3, 0]), [2, 0, 1])
        client = TrainingOnlyClient(DataLoader(selected, batch_size=2, shuffle=True, drop_last=True))
        decision_loader = prepare_fairfeddrift_decision_loader(client)
        self.assertIs(client.trainloader.dataset, selected)
        self.assertEqual(source.reads, 3)
        first = list(decision_loader)
        second = list(decision_loader)
        self.assertEqual(source.reads, 3)
        self.assertEqual(torch.cat([labels for _, labels in first]).tolist(), [3, 5, 1])
        for left, right in zip(first, second):
            self.assertTrue(torch.equal(left[0], right[0]))
            self.assertTrue(torch.equal(left[1], right[1]))
        self.assertEqual([len(labels) for _, labels in first], [2, 1])

    def test_current_labels_are_separate_from_history(self):
        """Read post-drift labels while retained pre-drift labels stay intact."""
        source = TensorDataset(torch.arange(6).float().reshape(3, 2), torch.tensor([0, 1, 2]))
        history = ClientDataHistory()
        history.add(0, source)
        source.tensors[1].copy_(torch.tensor([1, 0, 2]))
        client = TrainingOnlyClient(DataLoader(source, batch_size=3))
        decisions = prepare_fairfeddrift_decision_loader(client)
        self.assertEqual(next(iter(decisions))[1].tolist(), [1, 0, 2])
        self.assertEqual([round_idx for round_idx, _ in history.records()], [0])
        historical = history.records()[0][1]
        self.assertEqual([historical[i][1].item() for i in range(3)], [0, 1, 2])
        source.tensors[0].fill_(99)
        source.tensors[1].fill_(9)
        inputs, labels = next(iter(decisions))
        self.assertEqual(inputs.tolist(), [[0, 1], [2, 3], [4, 5]])
        self.assertEqual(labels.tolist(), [1, 0, 2])
        inputs.fill_(-1)
        labels.fill_(-1)
        self.assertEqual(next(iter(decisions))[1].tolist(), [1, 0, 2])

    def test_missing_empty_or_unbatched_training_data(self):
        """Fail without falling back to test data or historical samples."""
        for client in (SimpleNamespace(), TrainingOnlyClient(None)):
            with self.assertRaisesRegex(ValueError, 'trainloader'):
                prepare_fairfeddrift_decision_loader(client)
        empty = TensorDataset(torch.empty(0, 2), torch.empty(0, dtype=torch.long))
        with self.assertRaisesRegex(ValueError, 'empty'):
            prepare_fairfeddrift_decision_loader(TrainingOnlyClient(DataLoader(empty, batch_size=2)))
        with self.assertRaisesRegex(ValueError, 'batch size'):
            prepare_fairfeddrift_decision_loader(TrainingOnlyClient(DataLoader(ChangingInputs(), batch_size=None)))


if __name__ == '__main__':
    unittest.main()
