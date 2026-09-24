"""Check whole-client candidate loss and non-training evaluation behavior."""
import math
import unittest
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from device_utils import configure_device
from strategy.FairFedDrift.utils import evaluate_fairfeddrift_loss, prepare_fairfeddrift_decision_loader
from fairfeddrift_fixtures import make_client_drift_fixture


class FairFedDriftLossTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        configure_device('cpu')

    def test_sample_mean_includes_short_final_batch(self):
        logits = torch.tensor([[.8, .2], [.5, .5], [.25, .75]]).log()
        dataset = TensorDataset(logits, torch.zeros(3, dtype=torch.long))
        expected = -sum(math.log(p) for p in (.8, .5, .25)) / 3
        for batch_size in (1, 2, 3):
            with self.subTest(batch_size=batch_size):
                actual = evaluate_fairfeddrift_loss(nn.Identity(), DataLoader(dataset, batch_size))
                self.assertAlmostEqual(actual, expected, places=6)

    def test_preserves_parameters_buffers_gradients_and_mixed_modes(self):
        model = nn.Sequential(nn.BatchNorm1d(2), nn.Dropout(.9), nn.Linear(2, 2))
        model.train()
        model[2].eval()
        for parameter in model.parameters():
            parameter.grad = torch.ones_like(parameter)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        gradients = [p.grad.clone() for p in model.parameters()]
        modes = [module.training for module in model.modules()]
        observed = []
        handle = model.register_forward_pre_hook(
            lambda module, args: observed.append((torch.is_grad_enabled(), module.training)))
        loader = DataLoader(TensorDataset(torch.ones(3, 2), torch.zeros(3, dtype=torch.long)), 2)
        try:
            first = evaluate_fairfeddrift_loss(model, loader)
            self.assertEqual(first, evaluate_fairfeddrift_loss(model, loader))
        finally:
            handle.remove()
        self.assertTrue(observed)
        self.assertTrue(all(flags == (False, False) for flags in observed))
        self.assertEqual(modes, [module.training for module in model.modules()])
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[key]))
        for parameter, gradient in zip(model.parameters(), gradients):
            self.assertTrue(torch.equal(parameter.grad, gradient))

    def test_restores_modes_when_forward_fails(self):
        model = nn.Sequential(nn.Linear(2, 2), nn.Dropout())
        model.train()
        model[1].eval()
        modes = [module.training for module in model.modules()]
        loader = DataLoader(TensorDataset(torch.ones(1, 3), torch.zeros(1, dtype=torch.long)))
        with self.assertRaises(RuntimeError):
            evaluate_fairfeddrift_loss(model, loader)
        self.assertEqual(modes, [module.training for module in model.modules()])

    def test_rejects_empty_and_nonfinite_losses(self):
        model = nn.Identity().eval()
        for inputs in (torch.empty(0, 2), torch.full((1, 2), float('nan'))):
            loader = DataLoader(TensorDataset(inputs, torch.zeros(len(inputs), dtype=torch.long)))
            with self.assertRaises(ValueError):
                evaluate_fairfeddrift_loss(model, loader)
            self.assertFalse(model.training)

    def test_fixture_uses_ordinary_labels_without_sensitive_metadata(self):
        before, after, _ = make_client_drift_fixture()
        for client_id in before:
            losses = []
            for dataset in (before[client_id], after[client_id]):
                client = SimpleNamespace(trainloader=DataLoader(dataset, batch_size=7))
                loader = prepare_fairfeddrift_decision_loader(client)
                losses.append(evaluate_fairfeddrift_loss(nn.Identity(), loader))
            expected_increase = {0: 0, 1: 0, 2: .4, 3: .4, 4: .2, 5: .2}[client_id]
            self.assertAlmostEqual(losses[1] - losses[0], expected_increase, places=6)


if __name__ == '__main__':
    unittest.main()
