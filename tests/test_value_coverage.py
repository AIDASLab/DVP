import json
from pathlib import Path
import tempfile
import unittest

from value_item_parser.check_value_coverage import check_coverage


class ValueCoverageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paths = {key: self.root / name for key, name in (
            ("input_jsonl", "input.jsonl"), ("value_items_json", "values.json"),
            ("value_defs_json", "definitions.json"), ("factors_json", "factors.json"),
            ("regimes_jsonl", "regimes.jsonl"))}
        self.paths["value_items_json"].write_text(json.dumps(["care", "fairness"]))
        self.paths["value_defs_json"].write_text(json.dumps({"care": "Synthetic definition"}))
        self.paths["factors_json"].write_text(json.dumps({"items": {"care": {}, "fairness": {}}}))
        self.paths["regimes_jsonl"].write_text('\n'.join(json.dumps({"value_item": v}) for v in ["care", "fairness"]))

    def check_rows(self, rows):
        self.paths["input_jsonl"].write_text('\n'.join(json.dumps(row) for row in rows))
        before = self.paths["input_jsonl"].read_bytes()
        result = check_coverage(**self.paths)
        self.assertEqual(before, self.paths["input_jsonl"].read_bytes())
        return result

    def test_known_without_definition_is_reported_not_remapped(self):
        result = self.check_rows([{"target_sentence": "Synthetic", "value_items": [{"value_item": "fairness"}]}])
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["without_continuous_definitions"], 1)

    def test_unknown_and_bad_items_fail(self):
        result = self.check_rows([{"target_sentence": "Synthetic", "value_items": [{"value_item": "unknown"}, 3]}])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["unknown_values"], 1)
        self.assertEqual(result["invalid_value_items"], 1)

    def test_empty_and_malformed_fail(self):
        self.assertEqual(self.check_rows([])["status"], "FAIL")
        self.assertEqual(self.check_rows([{"target_sentence": "Synthetic", "value_items": []}])["status"], "FAIL")
        self.assertEqual(self.check_rows([[]])["status"], "FAIL")
        self.paths["input_jsonl"].write_text("{broken\n")
        self.assertEqual(check_coverage(**self.paths)["invalid_rows"], 1)

    def test_missing_catalog_coverage_fails(self):
        self.paths["factors_json"].write_text(json.dumps({"items": {"care": {}}}))
        result = self.check_rows([{"target_sentence": "Synthetic", "value_items": [{"value_item": "fairness"}]}])
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(result["missing_factors"], 1)


if __name__ == "__main__":
    unittest.main()
