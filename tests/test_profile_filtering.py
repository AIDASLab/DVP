"""Support-protocol checks on small, complete aggregate bundles."""

from contextlib import redirect_stderr, redirect_stdout
import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from profile_data import organize


class ProfileFilteringTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.manifest = {"schema_version": "profile-bundle-v1", "files": [],
                         "groups": {"sample": {"source": "synthetic"}}, "families": {},
                         "selection": {"scope": "synthetic full bundle"}}
        self.rows = {}

        def base(name, n):
            counts = [n // 2, n - n // 2]
            return {"value_item": name, "total_count": n,
                    "categories": [{"name": label, "count": count, "proportion": count / n,
                                    "description": "Synthetic category", "signals": ["example"]}
                                   for label, count in zip(("Affirmed", "Traded off"), counts)],
                    "meta": {"n_categories": 5}}

        self.base = [base("alpha", 200), base("beta", 199), base("gamma", 220)]
        for name, label in (("progress", "Progress Not Invoked"),
                            ("opposition to war", "Не упоминается антивоенность")):
            row = base(name, 500)
            row["categories"][-1]["name"] = label
            self.base.append(row)
        self.add_rows("base", self.base)
        self.add_rows("continuous_base", [
            {"value_item": name, "group_mode": "attribute_value", "mean": 0.1,
             "std": 0.2, "support_min": -0.5, "support_max": 0.6,
             "q05": -0.3, "q50": 0.1, "q95": 0.5,
             "n_unique_sentences": n, "n_samples": 10000,
             "gaussian_like_flag": 0}
            for name, n in (("alpha", 200), ("beta", 199), ("gamma", 300))])

        cases = [("twenty", "none", "high", 20, 25),
                 ("thirty", "medium", "high", 30, 40),
                 ("sparse", "high", "none", 19, 100),
                 ("unassigned", "NONE", "high", 100, 100),
                 ("unknown", "unknown", "high", 100, 100)]
        categorical, continuous, ot, deltas = [], [], [], []
        for case, a, b, na, nb in cases:
            common = {"factor": "pressure", "state_a": a, "state_b": b,
                      "N_a": na, "N_b": nb, "case": case}
            effect = {"ot_mean": 0, "ot_q025": 0, "ot_q975": 0,
                      "ot_ci_width": 0, "mean_shift": 0,
                      "mean_shift_q025": -0.05, "mean_shift_q975": 0.05}
            cat = {**common, **effect, "value_item": "alpha", "K_bins": 2}
            for i in (1, 2):
                cat.update({f"pA_bin{i}_mean": 0.5, f"pB_bin{i}_mean": 0.5,
                            f"delta_bin{i}_mean": 0, f"delta_bin{i}_q025": -0.05,
                            f"delta_bin{i}_q975": 0.05})
            categorical.append(cat)
            cont = {**common, **effect, "value_item": "alpha",
                    "ot_empirical": 0, "mean_shift_empirical": 0,
                    "mean_a": 0.1, "mean_b": 0.1, "std_a": 0.2, "std_b": 0.2}
            for side in ("a", "b"):
                cont.update({f"q05_{side}": -0.3, f"q50_{side}": 0.1, f"q95_{side}": 0.5})
            continuous.append(cont)
            joint = {**common, "anchor_value": "alpha", "support_size_pair": 2}
            joint.update({key: 0 for key in (
                "joint_ot_fast_mean", "joint_ot_fast_within_relation",
                "joint_ot_fast_across_partner", "joint_jsd_mean", "joint_jsd_q025",
                "joint_jsd_q975", "value_jsd_mean", "relation_jsd_mean")})
            ot.append(joint)
            for partner, probability in (("rare", 0.001), ("frequent", 0.999)):
                deltas.append({**common, "anchor_value": "alpha", "partner_value": partner,
                               "relation": "JUSTIFIES", "pA_mean": probability,
                               "pB_mean": probability, "delta_mean": 0,
                               "delta_q025": -0.05, "delta_q975": 0.05,
                               "support_size_pair": 2})
        self.add_rows("context", categorical)
        self.add_rows("continuous_context", continuous)
        self.add_rows("context_relation_ot", ot)
        self.add_rows("context_relation_deltas", deltas)
        counts = []
        for state, n in (("none", 20), ("high", 30), ("medium", 19),
                         ("NONE", 200), ("unknown", 200)):
            for partner, count in (("rare", 1), ("frequent", n - 1)):
                counts.append({"anchor_value": "alpha", "factor": "pressure", "state": state,
                               "partner_value": partner, "relation": "JUSTIFIES",
                               "count": count, "empirical_prob": count / n,
                               "support_N_state": n})
        self.add_rows("context_relation_counts", counts)
        relation_profile = {
            "counts": {"cooccurrence": {"rare": 1, "frequent": 99, "unlabeled": 10},
                       "relations": {"rare": {"JUSTIFIES": 1},
                                     "frequent": {"CONFLICT": 60, "SYNERGY": 39}}},
            "distribution": {
                "cooccurrence": {"rare": 1 / 110, "frequent": 99 / 110, "unlabeled": 10 / 110},
                "relations": {"rare": {"JUSTIFIES": 1},
                              "frequent": {"CONFLICT": 60 / 99, "SYNERGY": 39 / 99}}}}
        self.relations = {"attribute": "native sample label", "value_relations": {
            name: copy.deepcopy(relation_profile)
            for name in ("alpha", "beta", "gamma", "progress", "epsilon")}}
        self.add_rows("relations", self.relations)
        self.catalog = {"meta": {"note": "synthetic"}, "items": {
            "alpha": {"factors": [{"factor": "pressure", "states": ["none", "medium", "high"]}]}}}
        self.add_payload("catalog", "json", json.dumps(self.catalog).encode(), 2,
                         "resources/context_factors_revised.json.gz")
        self.save_manifest()

    def add_rows(self, family, rows):
        self.rows[family] = copy.deepcopy(rows)
        fmt = organize.FAMILIES[family][0]
        if fmt == "jsonl":
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()
        elif fmt == "json":
            payload = json.dumps(rows, ensure_ascii=False).encode()
        else:
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
            payload = stream.getvalue().encode()
        self.add_payload(family, fmt, payload,
                         len(rows["value_relations"]) if fmt == "json" else len(rows))

    def add_payload(self, family, fmt, payload, rows, relative=None):
        value_field, supports = (None, ()) if family == "catalog" else organize.FAMILIES[family][1:]
        relative = relative or f"data/{family}/sample.{fmt}.gz"
        path = self.source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        compressed = gzip.compress(payload, mtime=0)
        path.write_bytes(compressed)
        self.manifest["files"].append({
            "family": family, "group": None if family == "catalog" else "sample",
            "path": relative, "format": fmt, "value_field": value_field,
            "support_fields": list(supports), "rows": rows, "bytes": len(compressed),
            "sha256": hashlib.sha256(compressed).hexdigest()})
        if family != "catalog":
            self.manifest["families"][family] = {
                "files": 1, "rows": rows, "groups": ["sample"], "support_fields": list(supports)}

    def save_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest))

    def replace_family_rows(self, family, rows):
        self.manifest["files"] = [entry for entry in self.manifest["files"]
                                  if entry["family"] != family]
        self.add_rows(family, rows)
        self.save_manifest()

    def invoke(self, *arguments):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            try:
                status = organize.main(["--source-dir", str(self.source), *map(str, arguments)])
            except SystemExit as exc:
                status = exc.code
        return status, output.getvalue(), error.getvalue()

    def run_protocol(self, name, *arguments):
        output = self.root / name
        status, _, error = self.invoke("--output-dir", output, "--protocol", "support", *arguments)
        self.assertEqual(status, 0, error)
        return output

    @staticmethod
    def read(output, family):
        fmt = organize.FAMILIES[family][0]
        path = output / f"data/{family}/sample.{fmt}"
        if fmt == "csv":
            with path.open(newline="") as stream:
                return list(csv.DictReader(stream))
        if fmt == "jsonl":
            return [json.loads(line) for line in path.read_text().splitlines()]
        return json.loads(path.read_text())

    def test_protocol_thresholds_preserve_zero_effects_and_complete_state_cells(self):
        twenty = self.run_protocol("twenty")
        thirty = self.run_protocol("thirty", "--context-support", 30)
        for family in ("context", "continuous_context", "context_relation_ot"):
            self.assertEqual([row["case"] for row in self.read(twenty, family)], ["twenty", "thirty"])
            self.assertEqual([row["case"] for row in self.read(thirty, family)], ["thirty"])
        deltas = self.read(twenty, "context_relation_deltas")
        self.assertEqual(len(deltas), 4)
        self.assertEqual({row["partner_value"] for row in deltas}, {"rare", "frequent"})
        self.assertTrue(all(float(row["delta_mean"]) == 0 for row in deltas))
        counts = self.read(twenty, "context_relation_counts")
        self.assertEqual(len(counts), 4)
        self.assertEqual({row["state"] for row in counts}, {"none", "high"})
        self.assertTrue(any(row["count"] == "1" for row in counts))
        self.assertEqual(len(self.read(thirty, "context_relation_counts")), 2)

    def test_base_uses_invoked_counts_and_continuous_uses_unique_sentences(self):
        output = self.run_protocol("base_protocol")
        self.assertEqual([row["value_item"] for row in self.read(output, "base")], ["alpha", "gamma"])
        continuous = self.read(output, "continuous_base")
        self.assertEqual([row["value_item"] for row in continuous], ["alpha", "gamma"])
        self.assertEqual(continuous[0]["gaussian_like_flag"], 0)
        self.assertEqual(continuous[0], self.rows["continuous_base"][0])

    def test_relation_only_selection_loads_base_dependency_and_preserves_partners(self):
        output = self.run_protocol("relations", "--family", "relations")
        relations = self.read(output, "relations")
        self.assertEqual(list(relations["value_relations"]), ["alpha", "gamma"])
        self.assertEqual(relations["value_relations"]["alpha"], self.relations["value_relations"]["alpha"])
        self.assertFalse((output / "data/base").exists())
        self.assertIn("unlabeled", relations["value_relations"]["alpha"]["distribution"]["cooccurrence"])

    def test_threshold_controls_and_per_family_override(self):
        output = self.run_protocol("overrides", "--context-support", 30,
                                   "--min-support", "context=20", "--base-support", 210)
        self.assertEqual([row["value_item"] for row in self.read(output, "base")], ["gamma"])
        self.assertEqual([row["case"] for row in self.read(output, "context")], ["twenty", "thirty"])
        self.assertEqual([row["case"] for row in self.read(output, "continuous_context")], ["thirty"])
        self.assertEqual(list(self.read(output, "relations")["value_relations"]), ["gamma"])

    def test_no_protocol_preserves_all_rows_and_does_not_apply_hidden_filters(self):
        output = self.root / "unfiltered"
        status, _, error = self.invoke("--output-dir", output)
        self.assertEqual(status, 0, error)
        self.assertEqual(self.read(output, "base"), self.base)
        self.assertEqual(self.read(output, "relations"), self.relations)
        self.assertEqual(len(self.read(output, "context")), 5)
        self.assertEqual(len(self.read(output, "context_relation_counts")), 10)
        self.assertEqual(len(self.read(output, "context_relation_deltas")), 10)

    def test_selected_manifest_and_catalogs_remain_valid_for_another_selection(self):
        output = self.run_protocol("manifest")
        manifest = json.loads((output / "manifest.json").read_text())
        protocol = manifest["selection"]["filtering_protocol"]
        self.assertEqual(protocol["base_support"], 200)
        self.assertEqual(protocol["context_support"], 20)
        self.assertEqual(json.loads((output / "resources/context_factors_revised.json").read_text()), self.catalog)
        for entry in manifest["files"]:
            payload = (output / entry["path"]).read_bytes()
            self.assertEqual(entry["bytes"], len(payload))
            self.assertEqual(entry["sha256"], hashlib.sha256(payload).hexdigest())
        second = self.root / "second"
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = organize.main(["--source-dir", str(output), "--output-dir", str(second),
                                    "--protocol", "support"])
        self.assertEqual(status, 0)
        self.assertEqual(self.read(second, "base"), self.read(output, "base"))
        self.assertEqual(self.read(second, "relations"), self.read(output, "relations"))

    def test_protocol_requires_payload_mode_and_rejects_unused_threshold_options(self):
        for index, args in enumerate((("--list", "--protocol", "support"),
                                      ("--base-support", 200), ("--context-support", 20))):
            output = self.root / f"invalid{index}"
            action = () if "--list" in args else ("--output-dir", output)
            status, _, _ = self.invoke(*action, *args)
            self.assertNotEqual(status, 0)
            self.assertFalse(output.exists())

    def test_missing_base_dependency_or_context_catalog_fails_without_partial_output(self):
        original = copy.deepcopy(self.manifest)
        for family, requested, label in (("base", "relations", "base"),
                                          ("catalog", "context", "catalog")):
            self.manifest = copy.deepcopy(original)
            self.manifest["files"] = [entry for entry in self.manifest["files"] if entry["family"] != family]
            self.save_manifest()
            output = self.root / f"missing_{family}"
            status, _, error = self.invoke("--output-dir", output, "--family", requested,
                                           "--protocol", "support")
            self.assertNotEqual(status, 0, error)
            self.assertIn(label, error.lower())
            self.assertFalse(output.exists())
            self.assertFalse(list(self.root.glob(f".{output.name}.staging-*")))

    def test_base_dependency_hash_is_checked_even_when_only_relations_are_selected(self):
        entry = next(e for e in self.manifest["files"] if e["family"] == "base")
        path = self.source / entry["path"]
        path.write_bytes(path.read_bytes() + b"corrupt")
        output = self.root / "bad_dependency"
        status, _, error = self.invoke("--output-dir", output, "--family", "relations", "--protocol", "support")
        self.assertNotEqual(status, 0)
        self.assertIn("SHA-256", error)
        self.assertFalse(output.exists())
        self.assertFalse(list(self.root.glob(f".{output.name}.staging-*")))

    def test_invalid_relation_cell_aborts_without_emitting_a_partial_distribution(self):
        for family, field, invalid in (("context_relation_counts", "count", -1),
                                       ("context_relation_deltas", "delta_mean", "NaN")):
            with self.subTest(family=family):
                rows = copy.deepcopy(self.rows[family])
                # The first sibling is valid, so cleanup must also remove any
                # partially written output when the second cell is encountered.
                rows[1][field] = invalid
                self.replace_family_rows(family, rows)
                output = self.root / f"invalid_cell_{family}"
                status, _, error = self.invoke("--output-dir", output, "--family", family,
                                               "--protocol", "support")
                self.assertEqual(status, 1, error)
                self.assertFalse(output.exists())
                self.assertFalse(list(self.root.glob(f".{output.name}.staging-*")))

    def test_inconsistent_parent_support_aborts_even_when_one_cell_is_below_threshold(self):
        original = copy.deepcopy(self.rows)
        for family, field in (("context_relation_counts", "support_N_state"),
                              ("context_relation_deltas", "N_a"),
                              ("context_relation_deltas", "N_b")):
            with self.subTest(family=family, field=field):
                rows = copy.deepcopy(original[family])
                # Both rows belong to the same state/comparison. Make the second
                # denominator individually valid but inconsistent with the first.
                rows[1][field] = 19
                if family == "context_relation_counts":
                    rows[1]["empirical_prob"] = rows[1]["count"] / 19
                self.replace_family_rows(family, rows)
                output = self.root / f"inconsistent_{family}_{field}"
                status, _, error = self.invoke("--output-dir", output, "--family", family,
                                               "--protocol", "support")
                self.assertEqual(status, 1, error)
                self.assertFalse(output.exists())
                self.assertFalse(list(self.root.glob(f".{output.name}.staging-*")))


if __name__ == "__main__":
    unittest.main()
