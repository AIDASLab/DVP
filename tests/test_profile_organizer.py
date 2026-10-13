"""CPU-only bundle selection, provenance, and fail-closed boundary tests."""

import copy
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr, redirect_stdout

from profile_data import organize


class ProfileOrganizerTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.manifest = {
            "schema_version": "profile-bundle-v1",
            "files": [],
            "groups": {"sample_group": {"source": "synthetic"}, "second_group": {"source": "synthetic"}},
            "families": {},
            "selection": {"scope": "synthetic source bundle"},
        }
        self.native = {}
        self.base = [
            {"value_item": "alpha", "total_count": 600, "distribution": [0.33333, 0.66667], "flag": 0},
            {"value_item": "beta", "total_count": 250, "distribution": [0.2, 0.8], "flag": 1},
        ]
        for family, (fmt, value_field, supports) in organize.FAMILIES.items():
            if family == "base":
                payload = self.jsonl(self.base)
            elif family == "continuous_base":
                payload = self.jsonl([
                    {"value_item": "alpha", "n_unique_sentences": 300, "n_samples": 900, "mean": -0.125, "variance": 0.3125, "gaussian_flag": 0},
                    {"value_item": "beta", "n_unique_sentences": 200, "n_samples": 250, "mean": 0.625, "variance": 0.125, "gaussian_flag": 1},
                ])
            elif family == "relations":
                payload = json.dumps({
                    "attribute": "native_label",
                    "value_relations": {
                        "alpha": {"beta": {"count": 31, "distribution": {"cooccur": 0.3, "supports": 0.7}}},
                        "beta": {"alpha": {"count": 12, "distribution": {"cooccur": 1.0}}},
                    },
                }).encode()
            else:
                stream = io.StringIO(newline="")
                writer = csv.DictWriter(stream, fieldnames=[value_field, *supports, "statistic", "quality_flag"])
                writer.writeheader()
                for name, counts, stat, flag in (("alpha", [20, 40], "-0.125000", "low_n"), ("beta", [40, 50], "0.250000", "ok")):
                    row = {value_field: name, "statistic": stat, "quality_flag": flag}
                    row.update(zip(supports, counts))
                    writer.writerow(row)
                payload = stream.getvalue().encode()
            self.add_entry(family, "sample_group", fmt, value_field, list(supports), payload, 2)
            self.manifest["families"][family] = {
                "files": 1, "rows": 2, "groups": ["sample_group"],
                "support_fields": list(supports), "note": "Saved statistics, not recomputed",
            }
        self.add_entry("base", "second_group", "jsonl", "value_item", ["total_count"], self.jsonl(self.base[:1]), 1)
        self.manifest["families"]["base"].update(files=2, rows=3, groups=["sample_group", "second_group"])
        self.add_entry("catalog", None, "json", None, [], b'{"alpha": "A definition", "beta": "B definition"}\n', 2, "resources/definitions.json.gz")
        self.add_entry("catalog", None, "jsonl", None, [], b'{"value_item": "alpha"}\n{"value_item": "beta"}\n', 2, "resources/categories.jsonl.gz")
        self.save_manifest()

    @staticmethod
    def jsonl(rows):
        return b"".join((json.dumps(row) + "\n").encode() for row in rows)

    def add_entry(self, family, group, fmt, value_field, supports, payload, rows, relative=None):
        entry = {
            "path": relative or f"data/{family}/{group}.{fmt}.gz",
            "family": family, "group": group, "format": fmt,
            "value_field": value_field, "support_fields": supports, "rows": rows,
        }
        self.replace_payload(entry, payload)
        self.manifest["files"].append(entry)

    def replace_payload(self, entry, payload):
        target = self.source / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        compressed = gzip.compress(payload, mtime=0)
        target.write_bytes(compressed)
        entry.update(bytes=len(compressed), sha256=hashlib.sha256(compressed).hexdigest(), source_sha256=hashlib.sha256(payload).hexdigest())
        self.native[entry["path"]] = payload

    def save_manifest(self):
        (self.source / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")

    def invoke(self, *arguments, remote=False):
        output, error = io.StringIO(), io.StringIO()
        prefix = [] if remote else ["--source-dir", str(self.source)]
        with redirect_stdout(output), redirect_stderr(error):
            result = organize.main([*prefix, *map(str, arguments)])
        return result, output.getvalue(), error.getvalue()

    def assert_failed_cleanly(self, output, *arguments):
        status, _, error = self.invoke("--output-dir", output, *arguments)
        self.assertEqual(status, 1, error)
        self.assertFalse(output.exists())
        self.assertFalse(list(output.parent.glob(f".{output.name}.staging-*")))
        return error

    @staticmethod
    def read_rows(path):
        if path.suffix == ".jsonl":
            return [json.loads(line) for line in path.read_text().splitlines()]
        if path.suffix == ".csv":
            with path.open(newline="") as stream:
                return list(csv.DictReader(stream))
        return json.loads(path.read_text())

    def test_all_eight_families_preserve_native_statistics_and_catalogs(self):
        output = self.root / "all"
        status, _, error = self.invoke("--output-dir", output)
        self.assertEqual(status, 0, error)
        result = json.loads((output / "manifest.json").read_text())
        self.assertEqual(len(result["files"]), len(self.manifest["files"]))
        self.assertEqual(result["source_bundle"]["selection"], self.manifest["selection"])
        self.assertEqual(result["source_bundle"]["manifest_sha256"], hashlib.sha256((self.source / "manifest.json").read_bytes()).hexdigest())
        for entry in result["files"]:
            path = output / entry["path"]
            self.assertEqual(entry["bytes"], path.stat().st_size)
            self.assertEqual(entry["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
            source_entry = next(e for e in self.manifest["files"] if e["path"].removesuffix(".gz") == entry["path"])
            self.assertEqual(entry["source_sha256"], source_entry["source_sha256"])
            if entry["format"] in {"jsonl", "csv"}:
                self.assertEqual(path.read_bytes(), self.native[source_entry["path"]])
            else:
                self.assertEqual(json.loads(path.read_bytes()), json.loads(self.native[source_entry["path"]]))

    def test_filters_recompute_output_coverage_and_require_both_state_supports(self):
        output = self.root / "filtered"
        status, _, error = self.invoke("--output-dir", output, "--family", "context", "--group", "sample_group", "--min-support", "context=30")
        self.assertEqual(status, 0, error)
        rows = self.read_rows(output / "data/context/sample_group.csv")
        self.assertEqual([row["value_item"] for row in rows], ["beta"])
        self.assertEqual(rows[0]["statistic"], "0.250000")
        manifest = json.loads((output / "manifest.json").read_text())
        self.assertEqual(set(manifest["groups"]), {"sample_group"})
        self.assertEqual(set(manifest["families"]), {"context"})
        self.assertEqual(manifest["families"]["context"]["rows"], 1)
        self.assertEqual(manifest["families"]["context"]["files"], 1)
        self.assertEqual(manifest["families"]["context"]["support_fields"], ["N_a", "N_b"])
        self.assertEqual(manifest["families"]["context"]["note"], "Saved statistics, not recomputed")
        self.assertEqual(manifest["selection"]["min_support"], {"context": 30})
        self.assertEqual(sum(e["family"] == "catalog" for e in manifest["files"]), 2)

    def test_relation_value_filter_keeps_partner_statistics_without_renormalizing(self):
        output = self.root / "relation"
        status, _, error = self.invoke("--output-dir", output, "--family", "relations", "--value", "alpha")
        self.assertEqual(status, 0, error)
        row = self.read_rows(output / "data/relations/sample_group.json")
        self.assertEqual(row["attribute"], "native_label")
        self.assertEqual(list(row["value_relations"]), ["alpha"])
        self.assertEqual(row["value_relations"]["alpha"]["beta"], {"count": 31, "distribution": {"cooccur": 0.3, "supports": 0.7}})
        self.assert_failed_cleanly(self.root / "bad_support", "--family", "relations", "--min-support", "relations=10")

    def test_reorganizing_plain_output_preserves_provenance_chain(self):
        first, second = self.root / "first", self.root / "second"
        self.assertEqual(self.invoke("--output-dir", first, "--family", "base")[0], 0)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            status = organize.main(["--source-dir", str(first), "--output-dir", str(second), "--value", "alpha", "--group", "sample_group"])
        self.assertEqual(status, 0)
        result = json.loads((second / "manifest.json").read_text())
        self.assertEqual(self.read_rows(second / "data/base/sample_group.jsonl"), self.base[:1])
        self.assertIn("upstream", result["source_bundle"])
        self.assertEqual(result["families"]["base"]["rows"], 1)

    def test_corrupt_payload_and_wrong_row_count_do_not_leave_partial_outputs(self):
        entry = self.manifest["files"][0]
        path = self.source / entry["path"]
        original = path.read_bytes()
        path.write_bytes(original + b"corrupted")
        self.assertIn("SHA-256", self.assert_failed_cleanly(self.root / "corrupt", "--family", "base"))
        path.write_bytes(original)
        entry["rows"] += 1
        self.save_manifest()
        self.assertIn("row count", self.assert_failed_cleanly(self.root / "count", "--family", "base"))

    def test_unknown_or_empty_selections_fail_without_output(self):
        for index, args in enumerate((
            ("--group", "absent"), ("--value", "absent"),
            ("--family", "base", "--min-support", "base=10000"),
            ("--family", "base", "--min-support", "context=10"),
            ("--family", "base", "--min-support", "base=1.5"),
            ("--value", ""),
        )):
            with self.subTest(args=args):
                self.assert_failed_cleanly(self.root / f"missing{index}", *args)

    def test_existing_output_is_never_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        sentinel = output / "user_file"
        sentinel.write_bytes(b"preserve me")
        status, _, error = self.invoke("--output-dir", output)
        self.assertEqual(status, 1, error)
        self.assertEqual(sentinel.read_bytes(), b"preserve me")
        self.assertEqual(list(output.iterdir()), [sentinel])

    def test_manifest_rejects_traversal_duplicates_and_contract_mismatches(self):
        valid = copy.deepcopy(self.manifest)
        for index, mutation in enumerate((
            lambda m: m["files"][0].update(path="../escape.jsonl.gz"),
            lambda m: m["files"][0].update(path="/tmp/escape.jsonl.gz"),
            lambda m: m["files"][0].update(path="data\\base\\sample_group.jsonl.gz"),
            lambda m: m["files"].append(copy.deepcopy(m["files"][0])),
            lambda m: m["files"][0].update(rows=True),
            lambda m: m["files"][0].update(support_fields=["n_samples"]),
            lambda m: m["files"][0].update(family="unknown"),
            lambda m: m["files"][0].update(unexpected="field"),
            lambda m: m.update(schema_version="other"),
        )):
            with self.subTest(index=index):
                self.manifest = copy.deepcopy(valid)
                mutation(self.manifest)
                self.save_manifest()
                self.assert_failed_cleanly(self.root / f"unsafe{index}")

    def test_local_payload_symlink_cannot_escape_bundle(self):
        entry = self.manifest["files"][0]
        path = self.source / entry["path"]
        outside = self.root / "outside_payload"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)
        self.assertIn("escapes", self.assert_failed_cleanly(self.root / "escape", "--family", "base"))

    def test_invalid_support_counts_and_nonfinite_json_fail_closed(self):
        entry = self.manifest["files"][0]
        for index, value in enumerate((-1, 1.5, True, None, "NaN", "Infinity")):
            with self.subTest(value=value):
                self.replace_payload(entry, self.jsonl([{"value_item": "alpha", "total_count": value}]))
                entry["rows"] = 1
                self.save_manifest()
                self.assert_failed_cleanly(self.root / f"support{index}", "--family", "base")
        self.replace_payload(entry, b'{"value_item":"alpha","total_count":10,"mean":NaN}\n')
        self.save_manifest()
        self.assertIn("Non-finite", self.assert_failed_cleanly(self.root / "nan", "--family", "base"))

    def test_nonfinite_relation_or_catalog_json_is_rejected(self):
        for index, family in enumerate(("relations", "catalog")):
            entry = next(e for e in self.manifest["files"] if e["family"] == family)
            original = self.native[entry["path"]]
            bad = b'{"value_relations":{"alpha":{"mean":Infinity}}}' if family == "relations" else b'{"alpha":NaN}'
            self.replace_payload(entry, bad)
            self.save_manifest()
            self.assertIn("Non-finite", self.assert_failed_cleanly(self.root / f"nonfinite{index}", "--family", "relations"))
            self.replace_payload(entry, original)
            self.save_manifest()

    def remote_module(self):
        """Model an HF cache with suffix-free blob names; no network is used."""
        module = ModuleType("huggingface_hub")
        api = Mock()
        api.dataset_info.return_value = SimpleNamespace(sha="a" * 40)
        module.HfApi = Mock(return_value=api)

        def download(**kwargs):
            blob = Path(kwargs["cache_dir"]) / "blobs" / hashlib.sha256(kwargs["filename"].encode()).hexdigest()
            blob.parent.mkdir(parents=True, exist_ok=True)
            blob.write_bytes((self.source / kwargs["filename"]).read_bytes())
            return str(blob)

        module.hf_hub_download = Mock(side_effect=download)
        return module, api

    def test_remote_listing_downloads_only_manifest_and_pins_revision(self):
        module, api = self.remote_module()
        with patch.dict("sys.modules", {"huggingface_hub": module}):
            status, listing, error = self.invoke("--repo-id", "owner/profiles", "--revision", "release", "--cache-dir", self.root / "cache", "--list", remote=True)
        self.assertEqual(status, 0, error)
        self.assertIn("context_relation_deltas", listing)
        api.dataset_info.assert_called_once_with(repo_id="owner/profiles", revision="release")
        calls = module.hf_hub_download.call_args_list
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0].kwargs["filename"], "manifest.json")
        self.assertEqual(calls[0].kwargs["revision"], "a" * 40)
        self.assertEqual(calls[0].kwargs["repo_type"], "dataset")

    def test_remote_selection_fetches_only_selected_profiles_and_catalogs(self):
        module, _ = self.remote_module()
        output = self.root / "remote"
        with patch.dict("sys.modules", {"huggingface_hub": module}):
            status, _, error = self.invoke("--repo-id", "owner/profiles", "--cache-dir", self.root / "cache", "--output-dir", output, "--family", "base", "--group", "sample_group", "--value", "alpha", remote=True)
        self.assertEqual(status, 0, error)
        self.assertEqual(self.read_rows(output / "data/base/sample_group.jsonl"), self.base[:1])
        calls = module.hf_hub_download.call_args_list
        self.assertEqual({call.kwargs["filename"] for call in calls}, {"manifest.json", "data/base/sample_group.jsonl.gz", "resources/definitions.json.gz", "resources/categories.jsonl.gz"})
        self.assertTrue(all(call.kwargs["revision"] == "a" * 40 for call in calls))
        result = json.loads((output / "manifest.json").read_text())
        self.assertEqual(result["source_bundle"]["resolved_commit"], "a" * 40)
        self.assertEqual(result["source_bundle"]["repo_id"], "owner/profiles")

    def test_remote_unresolved_revision_and_outside_cache_are_rejected(self):
        module, api = self.remote_module()
        api.dataset_info.return_value.sha = "main"
        with patch.dict("sys.modules", {"huggingface_hub": module}):
            status, _, error = self.invoke("--repo-id", "owner/profiles", "--cache-dir", self.root / "cache", "--list", remote=True)
        self.assertEqual(status, 1)
        self.assertIn("resolved commit", error)
        module.hf_hub_download.assert_not_called()
        api.dataset_info.return_value.sha = "a" * 40
        module.hf_hub_download.side_effect = lambda **kwargs: str(self.source / "manifest.json")
        with patch.dict("sys.modules", {"huggingface_hub": module}):
            status, _, error = self.invoke("--repo-id", "owner/profiles", "--cache-dir", self.root / "cache", "--list", remote=True)
        self.assertEqual(status, 1)
        self.assertIn("outside", error)

    def test_list_does_not_accept_filters_requiring_payload_reads(self):
        for arguments in (("--value", "alpha"), ("--min-support", "base=10")):
            status, _, error = self.invoke("--list", *arguments)
            self.assertEqual(status, 1)
            self.assertIn("manifest-only", error)


if __name__ == "__main__":
    unittest.main()
