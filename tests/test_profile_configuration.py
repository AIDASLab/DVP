"""Profile assembly options on synthetic summaries; no model dependencies."""
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from value_profiler import distributional_dataset as profiles


class ProfileConfigurationTests(unittest.TestCase):
    def test_state_support_applies_to_both_modalities(self):
        discrete = {
            "N_a": "5", "N_b": "7", "K_bins": "2",
            "state_a": "low", "state_b": "high",
            "pA_bin1_mean": "0.8", "pA_bin2_mean": "0.2",
            "pB_bin1_mean": "0.3", "pB_bin2_mean": "0.7",
        }
        continuous = {"N_a": "5", "N_b": "7", "state_a": "low", "state_b": "high"}
        kwargs = dict(
            attribute_raw="user_group", value_item="care", factor="pressure",
            state_a="low", state_b="high", valid_value_items={"care": {}},
            factor_lookup={}, discrete_ot_row=discrete, continuous_ot_row=continuous,
            source_discrete_ot_csv=Path("discrete.csv"),
            source_continuous_ot_csv=Path("continuous.csv"),
        )
        self.assertIsNone(profiles.normalize_pairwise_ot_row(**kwargs))
        row = profiles.normalize_pairwise_ot_row(**kwargs, min_ot_state_count=5)
        self.assertEqual(row["discrete_ot"]["N_a"], 5)
        self.assertEqual(row["continuous_ot"]["N_b"], 7)
        self.assertEqual(row["attribute_label"], "user_group")
        self.assertIsNone(profiles.normalize_pairwise_ot_row(**kwargs, min_ot_state_count=6))
        # A custom call must not mutate defaults used by later calls.
        self.assertIsNone(profiles.normalize_pairwise_ot_row(**kwargs))

    def test_negative_support_rejected_before_reading_inputs(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            profiles.DistributionalProfileDataset(attributes=[], min_ot_state_count=-1)

    def test_attribute_labels_are_not_remapped(self):
        for label in ("my_cohort", "culturax_de_cleaned", "Conservative_doc"):
            self.assertEqual(profiles._safe_get_attribute_label(label), label)


if __name__ == "__main__":
    unittest.main()
