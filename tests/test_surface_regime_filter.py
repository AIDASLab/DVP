"""Base-profile filtering drops the non-invocation category whatever its label."""

import json
from pathlib import Path
import tempfile
import unittest

from profile_constructor.filter_surface_regimes import process_file


def counted_row(value_item, labels, counts):
    return {"value_item": value_item,
            "categories": [{"name": name, "count": count} for name, count in zip(labels, counts)]}


class SurfaceRegimeFilterTests(unittest.TestCase):
    def filter_rows(self, rows, threshold):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.jsonl", Path(tmp) / "out.jsonl"
            source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            process_file(str(source), str(target), threshold=threshold, round_digits=4)
            return {row["value_item"]: row for row in map(json.loads, target.read_text(encoding="utf-8").splitlines())}

    def test_value_specific_non_invocation_labels_are_excluded(self):
        counts = [90, 30, 20, 10, 100]
        rows = [
            counted_row("fairness", ["A", "B", "C", "D", "Not Invoked"], counts),
            counted_row("progress", ["A", "B", "C", "D", "Progress Not Invoked"], counts),
            counted_row("opposition to war", ["A", "B", "C", "D", "Не упоминается антивоенность"], counts),
        ]
        kept = self.filter_rows(rows, threshold=100)
        self.assertEqual(set(kept), {"fairness", "progress", "opposition to war"})
        for row in kept.values():
            self.assertEqual(row["total_count"], 150)
            self.assertEqual([c["name"] for c in row["categories"]], ["A", "B", "C", "D"])
            self.assertEqual([c["proportion"] for c in row["categories"]], [0.6, 0.2, 0.1333, 0.0667])

        # Support counts invoked observations only, for every label.
        self.assertEqual(self.filter_rows(rows, threshold=150), {})


if __name__ == "__main__":
    unittest.main()
