"""FairFedDrift structured logs retain scalar decisions without dataset snapshots."""
import os
import pickle
import tempfile
import unittest

from log_utils.logging import write_structured_log
from federated_network.utils import build_fairfeddrift_state_log
from strategy.FairFedDrift import aggregator_fn


class FairFedDriftLoggingTests(unittest.TestCase):
    def test_state_record_is_serializable_and_preserves_strategy_events(self):
        strategy = aggregator_fn({'fairfeddrift_loss_threshold': 0.25})
        strategy.cluster_models = {0: object(), 4: object()}
        strategy.client_assignments = {2: 4}
        strategy.previous_losses = {2: 0.35}
        strategy.assignment_history = {2: {3: 4}}
        strategy.runtime_history = [{
            'round': 3,
            'timestep_id': 3,
            'decisions': [{
                'client_id': 2,
                'candidate_losses': {0: 0.7, 4: 0.35},
                'assigned_cluster_id': 4,
                'created_cluster': False,
                'data_sample_count': 12
            }],
            'merges': [{'source_ids': (1, 2), 'cluster_id': 4, 'sample_counts': (5, 7)}],
            'history_upload_sample_counts': {2: {4: 12}}
        }]

        record = build_fairfeddrift_state_log(strategy)
        self.assertEqual(record['adaptation'], 'single_group_client_drift')
        self.assertEqual(record['parameters']['fairfeddrift_loss_threshold'], 0.25)
        self.assertEqual(record['events'][0]['decisions'][0]['candidate_losses'], {0: 0.7, 4: 0.35})
        self.assertEqual(record['final_state']['active_cluster_ids'], [0, 4])
        self.assertEqual(record['final_state']['retained_assignment_history'], {2: {3: 4}})
        self.assertNotIn('model', repr(record))
        pickle.dumps(record)

    def test_structured_log_round_trips_and_rejects_unresolved_strategy(self):
        strategy = aggregator_fn({'fairfeddrift_loss_threshold': 0.1})
        record = build_fairfeddrift_state_log(strategy)
        with tempfile.TemporaryDirectory() as temp_dir:
            log_path = os.path.join(temp_dir, 'fairfeddrift_state_log')
            write_structured_log(record, log_path)
            with open(log_path + '.pkl', 'rb') as file:
                loaded = pickle.load(file)

        self.assertEqual(loaded, record)
        with self.assertRaisesRegex(ValueError, 'resolved strategy parameters'):
            build_fairfeddrift_state_log(aggregator_fn())


if __name__ == '__main__':
    unittest.main()
