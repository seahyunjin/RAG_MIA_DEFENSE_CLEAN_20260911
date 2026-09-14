import unittest

from mba_scorer_paper_aligned import format_compliance, parse_predictions, reconstruction_accuracy


GT = {"Mask_1": ["Alpha"], "Mask_2": ["Beta"], "Mask_3": ["Gamma"]}


class MBAPaperAlignedScorerTests(unittest.TestCase):
    def test_complete_output(self):
        answer = "[Mask_1]: alpha\n[Mask_2]: beta\n[Mask_3]: gamma"
        self.assertEqual(reconstruction_accuracy(answer, GT), 1.0)
        self.assertTrue(format_compliance(answer, 3)[0])

    def test_missing_masks_are_incorrect_not_excluded(self):
        self.assertEqual(reconstruction_accuracy("[Mask_1]: Alpha", GT), 1 / 3)

    def test_idk_is_zero_not_fuzzy_repaired(self):
        self.assertEqual(reconstruction_accuracy("I don't know.", GT), 0.0)
        self.assertFalse(format_compliance("I don't know.", 3)[0])

    def test_malformed_line_is_not_parsed(self):
        self.assertEqual(parse_predictions("Mask_1: Alpha", 3), {})

    def test_out_of_range_mask_is_ignored(self):
        self.assertEqual(reconstruction_accuracy("[Mask_9]: Alpha", GT), 0.0)

    def test_non_contiguous_ground_truth_fails(self):
        with self.assertRaises(ValueError):
            reconstruction_accuracy("[Mask_1]: Alpha", {"Mask_2": ["Alpha"]})


if __name__ == "__main__":
    unittest.main()
