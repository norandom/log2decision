import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from transfer218 import partition, metrics, preflight_source, paired_support


class TransferTests(unittest.TestCase):
    def test_redaction_never_rescues_unsupported_original(self):
        self.assertFalse(paired_support('x'*1025, 'short'))
        self.assertFalse(paired_support('é'*513, 'short'))
        self.assertTrue(paired_support('é'*512, 'short'))
        self.assertFalse(paired_support('short', 'x'*1025))
        self.assertFalse(paired_support('', 'short'))

    def test_preflight_refuses_wrong_identity_before_reader(self):
        with self.assertRaises(ValueError):
            preflight_source(Path(__file__), 'wrong', 64 * 1024)

    def test_preflight_refuses_oversize_before_hash(self):
        with self.assertRaises(ValueError):
            preflight_source(Path(__file__), 'wrong', 1)

    def test_partition_is_duplicate_stable_and_disjoint(self):
        self.assertEqual(partition('same', 1, 0), partition('same', 1, 0))
        self.assertEqual(partition('held', 0, 0), 'held')
        self.assertNotEqual(partition('same', 1, 0), 'held')

    def test_unsupported_positive_is_missed(self):
        m = metrics([1, 0, 1], [0.9, 0.1, None], ['a', 'b', 'c'], 1)
        self.assertEqual(m['recall_at_budget'], 0.5)
        self.assertEqual(m['unsupported_positives'], 1)
        self.assertEqual(m['missed_positives'], 1)
        self.assertEqual(m['roc_auc'], 1.0)

    def test_undefined_single_class_and_tie_break(self):
        m = metrics([1, 1], [0.5, 0.5], ['b', 'a'], 1)
        self.assertIsNone(m['roc_auc'])
        self.assertEqual(m['selected_ids'], ['a'])


if __name__ == '__main__':
    unittest.main()
