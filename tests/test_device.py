"""CPU device and training regression checks using synthetic data."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset, Subset

import constants
import device_utils
from federated_network.client import Client
from federated_network.server import server_fn
from models.CNNModel.model import CNNModel
from models.utils import train, test as evaluate, auxiliary_model_train, rapid_train
from strategy.FedRC import fedrc


class DeviceTests(unittest.TestCase):
    def setUp(self):
        """Select CPU before constructing each test's models."""
        self.old_device = device_utils._device
        device_utils.configure_device('cpu')
        torch.manual_seed(7)
        self.dataset = TensorDataset(torch.randn(20, 1, 28, 28), torch.arange(20) % 10)
        self.dataset.targets = self.dataset.tensors[1]
        self.loader = DataLoader(self.dataset, batch_size=10)

    def tearDown(self):
        """Restore the device configuration after each test."""
        device_utils._device = self.old_device

    def test_selection(self):
        """Resolve auto, forced CPU, and unavailable CUDA without GPU allocation."""
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(device_utils.configure_device('auto').type, 'cpu')
            with self.assertRaisesRegex(RuntimeError, 'CUDA was requested'):
                device_utils.configure_device('cuda')
        with patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(device_utils.configure_device('cpu').type, 'cpu')
            self.assertEqual(device_utils.configure_device('auto').type, 'cuda')
        with self.assertRaises(ValueError):
            device_utils.configure_device('invalid')

    def test_full_precision_training_unchanged(self):
        """Match the previous disabled-scaler training update on identical data and RNG."""
        model = CNNModel()
        reference = copy.deepcopy(model)
        torch.manual_seed(17)
        train(model, self.loader, 1)
        optimizer = torch.optim.SGD(reference.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        torch.manual_seed(17)
        reference.train()
        for inputs, labels in self.loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=False):
                loss = torch.nn.functional.nll_loss(reference(inputs.float()), labels.long())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        loss, accuracy = evaluate(model, self.loader)
        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertTrue(0 <= accuracy <= 1)

    def test_client_server_and_auxiliary_devices(self):
        """Construct clients/servers after imports and exercise auxiliary training on CPU."""
        subset = Subset(self.dataset, list(range(20)))
        client = Client(0, True, CNNModel(), 1, 10, subset, subset, constants.RecoveryAlgorithm.FEDAVG, 1)
        server = server_fn(0, constants.DatasetNames.MNIST, 0, constants.RecoveryAlgorithm.FEDAVG, 1, 0.9)
        self.assertEqual(next(client.model.parameters()).device.type, 'cpu')
        self.assertEqual(next(server.model.parameters()).device.type, 'cpu')
        auxiliary = auxiliary_model_train(client.model, self.loader, server.model.state_dict(), 1, 10)
        self.assertTrue(auxiliary)
        self.assertTrue(all(tensor.device.type == 'cpu' for tensor in auxiliary.values()))
        rapid_train(client.model, self.loader, 1, 10)
        self.assertTrue(all(torch.isfinite(p).all() for p in client.model.parameters()))

    def test_fedrc_state_and_fit(self):
        """Keep cluster models, optimizer parameters, and cluster statistics on CPU."""
        subset = Subset(self.dataset, list(range(20)))
        client = Client(0, True, CNNModel(), 1, 10, subset, subset, constants.RecoveryAlgorithm.FEDRC, 2)
        for value in (client.omega_i_k, client.gamma_i_j_k, client.C_y_k):
            self.assertEqual(value.device.type, 'cpu')
        omega, counts = fedrc.fit(client.fedrc_models, self.loader, client.fedrc_optimizers,
                                 client.omega_i_k, client.C_y_k, client.num_classes)
        self.assertTrue(torch.isfinite(omega).all() and torch.isfinite(counts).all())
        for model, optimizer in zip(client.fedrc_models, client.fedrc_optimizers):
            self.assertEqual(next(model.parameters()).device.type, 'cpu')
            self.assertIs(next(model.parameters()), optimizer.param_groups[0]['params'][0])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
