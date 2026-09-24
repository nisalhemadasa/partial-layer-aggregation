"""Contract checks for sample-count-weighted FairFedDrift uploads."""
from collections import OrderedDict
import unittest

import torch
from torch import nn

from strategy.FairFedDrift import aggregator_fn


class AggregationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)
        self.register_buffer('counter', torch.tensor(0, dtype=torch.long))


class FairFedDriftAggregationTests(unittest.TestCase):
    def setUp(self):
        self.strategy = aggregator_fn()
        self.server = AggregationModel()
        with torch.no_grad():
            self.server.linear.weight.zero_()
        self.clients = {}

    def state(self, weight, counter):
        """Build client state in the target model's ordered key layout."""
        return OrderedDict((('counter', torch.tensor(counter, dtype=torch.long)),
                            ('linear.weight', torch.tensor([[weight]], dtype=torch.float32)))
                           if list(self.server.state_dict())[0] == 'counter' else
                           (('linear.weight', torch.tensor([[weight]], dtype=torch.float32)),
                            ('counter', torch.tensor(counter, dtype=torch.long))))

    def valid_clients(self):
        return {0: self.state(0, 7), 1: self.state(4, 9)}

    def test_weighted_average_uses_local_training_counts(self):
        clients = self.valid_clients()
        server_before = {key: value.clone() for key, value in self.server.state_dict().items()}
        self.strategy.aggregate_models(self.server, clients, {0: 1, 1: 3})
        self.assertAlmostEqual(self.server.linear.weight.item(), 3.0)
        self.assertEqual(self.server.counter.item(), 9)
        for client_id, state in clients.items():
            self.assertEqual(state['linear.weight'].item(), (0., 4.)[client_id])
        self.assertEqual(server_before['linear.weight'].item(), 0.0)

    def test_current_and_historical_uploads_share_their_exact_weighted_aggregate(self):
        self.strategy.aggregate_models(
            self.server,
            {1: self.state(2, 7)}, {1: 1},
            {1: self.state(8, 7), 2: self.state(6, 7)}, {1: 3, 2: 2})
        self.assertAlmostEqual(self.server.linear.weight.item(), 38.0 / 6.0, places=6)

    def test_historical_uploads_can_update_cluster_without_current_upload(self):
        self.strategy.aggregate_models(self.server, {}, {}, {7: self.state(9, 7)}, {7: 4})
        self.assertAlmostEqual(self.server.linear.weight.item(), 9.0)

    def test_invalid_history_counts_fail_before_server_mutation(self):
        with torch.no_grad():
            self.server.linear.weight.fill_(5.0)
        with self.assertRaisesRegex(ValueError, 'identical client IDs'):
            self.strategy.aggregate_models(self.server, {1: self.state(2, 7)}, {1: 1},
                                           {2: self.state(8, 7)}, {3: 2})
        self.assertAlmostEqual(self.server.linear.weight.item(), 5.0)

    def test_equal_counts_and_integer_buffer_tie_use_first_upload(self):
        self.strategy.aggregate_models(self.server, self.valid_clients(), {0: 2, 1: 2})
        self.assertAlmostEqual(self.server.linear.weight.item(), 2.0)
        self.assertEqual(self.server.counter.item(), 7)

    def test_empty_uploads_counts_and_id_mismatches_fail_before_mutation(self):
        original = {key: value.clone() for key, value in self.server.state_dict().items()}
        invalid = (({}, {}), (self.valid_clients(), {}),
                   (self.valid_clients(), {0: 1}),
                   (self.valid_clients(), {0: 1, 1: 3, 2: 1}))
        for uploads, counts in invalid:
            with self.subTest(uploads=list(uploads), counts=counts), self.assertRaises(ValueError):
                self.strategy.aggregate_models(self.server, uploads, counts)
            for key, value in self.server.state_dict().items():
                self.assertTrue(torch.equal(value, original[key]))

    def test_invalid_counts_ids_keys_shapes_and_dtypes_fail_before_mutation(self):
        original = {key: value.clone() for key, value in self.server.state_dict().items()}
        for count in (0, -1, True, 1.5, None):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.strategy.aggregate_models(self.server, {0: self.state(0, 7)}, {0: count})
        with self.assertRaises(ValueError):
            self.strategy.aggregate_models(self.server, {-1: self.state(0, 7)}, {-1: 1})
        malformed = [OrderedDict((('wrong', torch.tensor([1.])),)),
                     OrderedDict((('linear.weight', torch.zeros(2, 1)), ('counter', torch.tensor(7)))),
                     OrderedDict((('linear.weight', torch.zeros(1, 1, dtype=torch.float64)),
                                  ('counter', torch.tensor(7))))]
        for state in malformed:
            with self.subTest(state=state), self.assertRaises(ValueError):
                self.strategy.aggregate_models(self.server, {0: state}, {0: 1})
        for key, value in self.server.state_dict().items():
            self.assertTrue(torch.equal(value, original[key]))

    def test_single_upload_copies_values_without_aliasing(self):
        client = self.state(5, 13)
        self.strategy.aggregate_models(self.server, {4: client}, {4: 6})
        self.assertEqual(self.server.linear.weight.item(), 5.)
        self.assertEqual(self.server.counter.item(), 13)
        self.assertNotEqual(self.server.linear.weight.data_ptr(), client['linear.weight'].data_ptr())


if __name__ == '__main__':
    unittest.main()
