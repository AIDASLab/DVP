"""CPU-only launcher contracts; no model download or inference is invoked."""

import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="value-pipeline-launchers-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        shutil.copytree(ROOT / "scripts", self.root / "scripts")
        self.env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHON": "/nonexistent-value-pipeline-python",
            "ANNOTATION_MODEL": "example/custom-chat",
            "EMBEDDING_MODEL": "example/custom-embed",
            "MODEL_NAME": "example/custom-decoder",
            "LAYERS": "1,2",
        }

    def launch(self, name, *args, env=None, success=True):
        result = subprocess.run(
            ["bash", str(self.root / "scripts" / name), *args],
            cwd=self.root,
            env=self.env if env is None else env,
            text=True,
            capture_output=True,
            timeout=15,
        )
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    @staticmethod
    def commands(result):
        return [shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")]

    def module(self, result, name):
        return next(cmd for cmd in self.commands(result) if "-m" in cmd and cmd[cmd.index("-m") + 1] == name)

    def option(self, command, name):
        self.assertIn(name, command)
        return command[command.index(name) + 1]

    def test_help_needs_no_model_configuration(self):
        env = {"PATH": self.env["PATH"]}
        for name in ("annotate.sh", "value_vectors.sh", "score.sh", "profiles.sh"):
            with self.subTest(script=name):
                result = self.launch(name, "--help", env=env)
                self.assertIn("Usage:", result.stdout)
        self.assertFalse((self.root / "outputs").exists())

    def test_dry_runs_allow_custom_models_and_sources_without_writes(self):
        invocations = (
            ("annotate.sh", "sample", "missing.csv"),
            ("value_vectors.sh", "generate"),
            ("value_vectors.sh", "custom_source"),
            ("score.sh", "sample", "custom_source"),
            ("profiles.sh", "sample", "custom_source"),
            ("profiles.sh", "sample", "custom_source", "second-source"),
        )
        before = sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*"))
        for invocation in invocations:
            with self.subTest(invocation=invocation):
                result = self.launch(*invocation, "--dry-run")
                self.assertIn(self.env["PYTHON"], result.stdout)
        self.assertEqual(before, sorted(str(p.relative_to(self.root)) for p in self.root.rglob("*")))

    def test_model_configuration_is_required_and_names_are_safe(self):
        cases = (
            ("annotate.sh", ("sample", "in.csv"), "ANNOTATION_MODEL"),
            ("annotate.sh", ("sample", "in.csv"), "EMBEDDING_MODEL"),
            ("value_vectors.sh", ("generate",), "ANNOTATION_MODEL"),
            ("value_vectors.sh", ("custom_source",), "MODEL_NAME"),
            ("score.sh", ("sample", "custom_source"), "LAYERS"),
        )
        for name, args, key in cases:
            with self.subTest(script=name, variable=key):
                env = dict(self.env)
                env.pop(key)
                result = self.launch(name, *args, "--dry-run", env=env, success=False)
                self.assertIn(key, result.stderr)
        self.launch("profiles.sh", "sample", "same", "same", "--dry-run", success=False)
        self.launch("profiles.sh", "sample", "../outside", "--dry-run", success=False)
        self.launch("score.sh", "../outside", "custom_source", "--dry-run", success=False)
        self.launch("value_vectors.sh", "custom_source", "--dry-run", env={**self.env, "LAYERS": "1;2"}, success=False)
        for name, args in (("value_vectors.sh", ("custom_source",)), ("score.sh", ("sample", "custom_source"))):
            result = self.launch(name, *args, "--dry-run", env={**self.env, "POOLING": "all_tokens"}, success=False)
            self.assertIn("one vector per sentence", result.stderr)
        self.assertFalse((self.root / "outputs").exists())

    def test_mapping_is_optional_and_resource_overrides_are_connected(self):
        env = {**self.env, "FACTORS_JSON": "custom/factors.json", "VALUE_ITEMS_JSON": "custom/items.json", "VALUE_DEFS_JSON": "custom/defs.json", "REGIMES_JSONL": "custom/states.jsonl", "EMBEDDING_DEVICE": "cpu"}
        result = self.launch("annotate.sh", "sample", "input.csv", "--dry-run", env=env)
        mapping = self.module(result, "value_item_parser.deduplicate_value_items")
        self.assertEqual(self.option(mapping, "--mapping_jsonl"), "outputs/mapping/sample.jsonl")
        self.assertEqual(self.option(mapping, "--rep_json"), "custom/defs.json")
        self.assertEqual(self.option(mapping, "--device"), "cpu")
        coverage = self.module(result, "value_item_parser.check_value_coverage")
        self.assertEqual(self.option(coverage, "--value_items_json"), "custom/items.json")
        self.assertEqual(self.option(coverage, "--factors_json"), "custom/factors.json")
        self.assertEqual(self.option(coverage, "--regimes_jsonl"), "custom/states.jsonl")
        customized = self.launch("annotate.sh", "sample", "input.csv", "--dry-run", env={**env, "REPRESENTATIVES_JSON": "custom/full-list.json", "TOP_K_VALUE_ITEMS": "6"})
        self.assertEqual(self.option(self.module(customized, "value_item_parser.deduplicate_value_items"), "--rep_json"), "custom/full-list.json")
        for name in ("relation_parser.extract_value_relationships", "intensity_parser.classify_surface_value_regimes"):
            self.assertEqual(self.option(self.module(customized, name), "--top_k_value_items"), "6")
        existing = self.root / "map.jsonl"
        existing.write_text('{"unchanged": true}\n', encoding="utf-8")
        result = self.launch("annotate.sh", "sample", "input.csv", "map.jsonl", "body", "comment", "--dry-run")
        self.assertEqual(self.option(self.module(result, "processing.extract_valued_text"), "--text_col"), "body")
        self.assertEqual(existing.read_text(encoding="utf-8"), '{"unchanged": true}\n')
        (self.root / "not-a-file").mkdir()
        self.launch("annotate.sh", "sample", "input.csv", "not-a-file", "--dry-run", success=False)

    def test_new_mapping_does_not_block_actual_preflight(self):
        # Stop before inference with the standard false executable. This checks
        # real (non-dry-run) file guards without installing an inference backend.
        fixture = self.root / "fixture"
        fixture.write_text("placeholder\n", encoding="utf-8")
        env = {**self.env, "PYTHON": shutil.which("false"), "FACTORS_JSON": str(fixture), "VALUE_ITEMS_JSON": str(fixture), "VALUE_DEFS_JSON": str(fixture), "REGIMES_JSONL": str(fixture)}
        result = self.launch("annotate.sh", "sample", str(fixture), env=env, success=False)
        self.assertIn("processing.extract_valued_text", result.stdout)
        self.assertNotIn("Missing input", result.stderr)
        self.assertFalse((self.root / "outputs/mapping/sample.jsonl").exists())

    def test_vector_score_settings_and_alignment_guards(self):
        env = {**self.env, "POOLING": "mean_last_k_tokens", "LAST_K": "5", "DTYPE": "float32", "TRUST_REMOTE_CODE": "1", "EXTRACTION_MAX_LENGTH": "128", "ALLOW_MISSING_NEG_OR_POS": "0", "SCORE_MAX_LENGTH": "128", "SCORE_BATCH_SIZE": "3", "SCORE_QUANT": "float16"}
        vectors = self.launch("value_vectors.sh", "custom_source", "--dry-run", env=env)
        extraction = self.module(vectors, "activation_parser.save_activations_from_jsonl")
        self.assertEqual(self.option(extraction, "--model_name"), "example/custom-decoder")
        self.assertEqual(self.option(extraction, "--layers"), "1,2")
        self.assertEqual(self.option(extraction, "--last_k"), "5")
        self.assertIn("--require_fresh_output", extraction)
        self.assertIn("--trust_remote_code", extraction)
        self.assertNotIn("--allow_missing_neg_or_pos", self.module(vectors, "activation_parser.save_value_vectors"))
        scores = self.launch("score.sh", "sample", "custom_source", "--dry-run", env=env)
        scorer = self.module(scores, "activation_parser.calculate_activation_intensity")
        self.assertEqual(self.option(scorer, "--activation_config"), "outputs/value_vectors/custom_source/activations/example__custom-decoder_activations_config.json")
        self.assertEqual(self.option(scorer, "--max_length"), "128")
        self.assertEqual(self.option(scorer, "--batch_size"), "3")
        self.assertEqual(self.option(scorer, "--quant"), "float16")
        self.assertNotIn("--drop_text", scorer)
        attachment = self.module(scores, "profile_constructor.attach_valueitem_projection_scores")
        self.assertIn("--strict_meta_check", attachment)
        self.assertIn("--fail_fast", attachment)
        self.assertEqual(self.option(attachment, "--scores_npz"), "outputs/scores/custom_source/sample..scores.npz")
        no_trust = self.launch("score.sh", "sample", "custom_source", "--dry-run")
        self.assertNotIn("--trust_remote_code", self.module(no_trust, "activation_parser.calculate_activation_intensity"))

    def test_profile_settings_propagate_to_producers_and_assembly(self):
        env = {**self.env, "MIN_VALUE_ITEM_COUNT": "12", "MIN_N_PER_STATE": "3", "MIN_GROUP_N": "4", "DISCRETE_DRAWS": "7", "CONTINUOUS_DRAWS": "8", "JOINT_DRAWS": "9", "PROFILE_SEED": "42", "NORMALIZATION_MODE": "minmax", "PREFER_LAYER": "2"}
        result = self.launch("profiles.sh", "sample", "first", "second", "--dry-run", env=env)
        discrete = self.module(result, "context_parser.context_effect_ot")
        continuous = self.module(result, "context_parser.context_effect_ot_raw_score")
        base = self.module(result, "profile_constructor.construct_distributional_raw_intensity")
        joint = self.module(result, "context_parser.context_effect_joint_ot_relations")
        assembled = self.module(result, "value_profiler.distributional_dataset")
        self.assertEqual(self.option(discrete, "--min_value_item_count"), "12")
        self.assertEqual(self.option(discrete, "--draws"), "7")
        self.assertEqual(self.option(continuous, "--draws"), "8")
        self.assertEqual(self.option(joint, "--draws"), "9")
        for command in (discrete, continuous, joint):
            self.assertEqual(self.option(command, "--min_n_per_state"), "3")
        self.assertEqual(self.option(joint, "--relation_jsonl"), "outputs/relations/sample.with_graph.jsonl")
        self.assertEqual(self.option(base, "--model_names"), "first,second")
        self.assertEqual(self.option(base, "--prefer_layer"), "2")
        self.assertEqual(self.option(continuous, "--prefer_layer"), "2")
        self.assertEqual(self.option(assembled, "--min_ot_state_count"), "3")
        for producer, flag, directory_flag, suffix_flag in (
            (base, "--out_fit_jsonl", "--continuous_base_dir", "--continuous_base_suffix"),
            (discrete, "--out_csv", "--ot_dir", "--ot_suffix"),
            (continuous, "--out_csv", "--continuous_ot_dir", "--continuous_ot_suffix"),
        ):
            expected = Path(self.option(assembled, directory_flag))
            if producer is base:
                expected /= "sample"
            expected /= "sample" + self.option(assembled, suffix_flag)
            self.assertEqual(self.option(producer, flag), str(expected))
        self.assertEqual(self.option(assembled, "--distribution_suffix"), ".surface_regimes.filtered_noNotInv_gt12.jsonl")

    def test_existing_outputs_are_refused_and_preserved(self):
        path = self.root / "outputs/activations/custom_source/processed/sample.jsonl"
        path.parent.mkdir(parents=True)
        path.write_text("keep me\n", encoding="utf-8")
        result = self.launch("score.sh", "sample", "custom_source", "--dry-run", success=False)
        self.assertIn("Output already exists", result.stderr)
        self.assertEqual(path.read_text(encoding="utf-8"), "keep me\n")
        existing = self.root / "outputs/distributional_profiles/continuous_distribution/sample"
        existing.mkdir(parents=True)
        self.launch("profiles.sh", "sample", "custom_source", "--dry-run", success=False)


if __name__ == "__main__":
    unittest.main()
