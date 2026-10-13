"""Synthetic native-format handoff checks; no downloads, APIs or GPU execution.

Run: python -B -m unittest discover -s tests -v
The tokenizer/model stand-in exercises real extraction, pooling, shards, vectors,
projection and attachment; only pretrained model loading is replaced.
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from activation_parser import calculate_activation_intensity as scorer
from activation_parser import save_activations_from_jsonl as activation
from activation_parser import save_value_vectors as vectors
from profile_constructor import attach_valueitem_projection_scores as attach

try:
    import torch
except ImportError:
    torch = None


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_main(module, args):
    with patch.object(sys, "argv", [module.__file__, *map(str, args)]):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            module.main()


class AttachmentChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="native_attach_test_")
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.source = self.work / "input.jsonl"
        self.meta = self.work / "meta.jsonl"
        self.columns = self.work / "columns.json"
        self.scores = self.work / "scores.npz"
        self.output = self.work / "out.jsonl"
        self.rows = [
            {"idx": 7, "text_type": "plain", "target_sentence": " a ", "value_items": [{"value_item": "care"}]},
            {"idx": 7, "text_type": "plain", "target_sentence": "", "value_items": []},
            {"idx": 7, "text_type": "plain", "target_sentence": None, "value_items": []},
            {"idx": 7, "text_type": "plain", "target_sentence": "b", "value_items": [{"value_item": "care"}]},
        ]
        self.meta_rows = [
            {"row_id": 0, "idx": 7, "text_type": "plain", "target_sentence": "a"},
            {"row_id": 1, "idx": 7, "text_type": "plain", "target_sentence": "b"},
        ]
        self.column_data = {"selected_value_items": ["care"], "metric": "cosine", "quant": "int16", "layers": [1, 2]}
        write_jsonl(self.source, self.rows)
        write_jsonl(self.meta, self.meta_rows)
        self.columns.write_text(json.dumps(self.column_data), encoding="utf-8")
        np.savez_compressed(self.scores, layer_1=np.array([[8944], [-4472]], dtype=np.int16),
                            layer_2=np.array([[8944], [-4472]], dtype=np.int16))

    def args(self):
        return ["--input_jsonl", self.source, "--columns_json", self.columns,
                "--scores_npz", self.scores, "--meta_jsonl", self.meta,
                "--out_jsonl", self.output, "--strict_meta_check", "--fail_fast"]

    def assert_rejected_before_write(self, message, args=None):
        # An existing destination must remain byte-identical after a failed preflight.
        self.output.write_text("existing output\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, message):
            run_main(attach, self.args() if args is None else args)
        self.assertEqual(self.output.read_text(encoding="utf-8"), "existing output\n")
        self.assertFalse(Path(str(self.output) + ".errors.jsonl").exists())

    def test_valid_native_rows_and_empty_passthrough(self):
        run_main(attach, self.args())
        rows = read_jsonl(self.output)
        self.assertEqual(rows[1:3], self.rows[1:3])
        self.assertEqual(rows[0]["target_sentence"], " a ")
        self.assertEqual(rows[0]["value_items"][0]["proj_scores"], {"1": 0.8944, "2": 0.8944})
        self.assertEqual(rows[3]["value_items"][0]["proj_scores"], {"1": -0.4472, "2": -0.4472})
        self.assertEqual(Path(str(self.output) + ".errors.jsonl").read_text(), "")

    def test_same_idx_sentence_swap_rejected(self):
        write_jsonl(self.source, [self.rows[3], *self.rows[1:3], self.rows[0]])
        self.assert_rejected_before_write("sentence mismatch")

    def test_full_input_meta_npz_consumption_required(self):
        variants = [
            (self.rows[:1], self.meta_rows, "row-count mismatch"),
            (self.rows[:1], self.meta_rows[:1], "row-count mismatch"),
            (self.rows, self.meta_rows[:1], "row-count mismatch"),
            (self.rows, [*self.meta_rows, dict(self.meta_rows[1], row_id=2)], "row-count mismatch"),
            ([*self.rows, self.rows[3]], [*self.meta_rows, dict(self.meta_rows[1], row_id=2)], "more input/metadata"),
            (self.rows, [self.meta_rows[0], dict(self.meta_rows[1], row_id=0)], "each row_id"),
        ]
        for source, meta, message in variants:
            with self.subTest(message=message, source_count=len(source), meta_count=len(meta)):
                write_jsonl(self.source, source)
                write_jsonl(self.meta, meta)
                self.assert_rejected_before_write(message)

    def test_full_text_and_matching_idx_type_required(self):
        for key in ("target_sentence", "idx", "text_type"):
            with self.subTest(key=key):
                rows = [dict(row) for row in self.meta_rows]
                rows[0].pop(key)
                write_jsonl(self.meta, rows)
                self.assert_rejected_before_write("requires full target_sentence|sentence mismatch")
        args = self.args()
        index = args.index("--meta_jsonl")
        del args[index:index + 2]
        self.assert_rejected_before_write("requires --meta_jsonl", args)

    def test_all_layers_and_columns_checked_before_writing(self):
        for bad in (np.zeros((1, 1)), np.zeros((2, 2)), np.zeros((2,)), np.zeros((2, 1, 1))):
            with self.subTest(shape=bad.shape):
                np.savez_compressed(self.scores, layer_1=np.zeros((2, 1)), layer_2=bad)
                # Even a bad layer not requested for attachment must be detected.
                self.assert_rejected_before_write("shape", [*self.args(), "--layers", "1"])
        np.savez_compressed(self.scores, layer_1=np.zeros((2, 1)))
        self.assert_rejected_before_write("layer keys")
        self.columns.write_text(json.dumps(dict(self.column_data, selected_value_items=["care", "care"])))
        self.assert_rejected_before_write("duplicate columns")


class ToyTokenizer:
    padding_side = "right"

    def __call__(self, texts, **kwargs):
        rows = [[{"a": 1, "b": 2}[token] for token in text.split()] for text in texts]
        width = max(map(len, rows))
        return {
            "input_ids": torch.tensor([row + [0] * (width - len(row)) for row in rows]),
            "attention_mask": torch.tensor([[1] * len(row) + [0] * (width - len(row)) for row in rows]),
        }


class ToyModel:
    config = types.SimpleNamespace(num_hidden_layers=2, hidden_size=2)

    def __call__(self, input_ids, attention_mask, **kwargs):
        states = torch.tensor([[0.0, 0.0], [2.0, 0.0], [0.0, 1.0]])[input_ids]
        return types.SimpleNamespace(hidden_states=(states * 0, states, states * 2))


@unittest.skipUnless(torch is not None, "CPU torch is needed only for synthetic native activation execution")
class NativeActivationConnection(unittest.TestCase):
    def test_original_connection_and_opt_in_rejections(self):
        from activation_parser import activation_extractor as extractor
        bundle = extractor.ModelBundle(ToyModel(), ToyTokenizer(), torch.device("cpu"), 2)
        with tempfile.TemporaryDirectory(prefix="native_activation_test_") as temp:
            work = Path(temp)
            posneg = work / "posneg.jsonl"
            write_jsonl(posneg, [
                {"value_item": "care", "label": "pos", "sentence": "a"},
                {"value_item": "care", "label": "pos", "sentence": "a a"},
                {"value_item": "care", "label": "neg", "sentence": "b"},
                {"value_item": "care", "label": "neg", "sentence": "b b"},
            ])
            acts = work / "acts"
            activation_args = ["--input_jsonl", posneg, "--model_name", "toy/model", "--out_dir", acts,
                               "--layers", "1,2", "--pooling", "mean_last_k_tokens", "--last_k", "3",
                               "--batch_size", "2", "--shard_size", "2", "--require_fresh_output"]
            with patch.object(extractor, "load_model_and_tokenizer", return_value=bundle):
                with patch.object(bundle.tokenizer, "padding_side", "left"):
                    with self.assertRaisesRegex(ValueError, "requires right padding"):
                        run_main(activation, activation_args)
            self.assertFalse(acts.exists())
            with patch.object(extractor, "load_model_and_tokenizer", return_value=bundle):
                run_main(activation, activation_args)
            meta = acts / "toy__model_activations_meta.jsonl"
            original_meta = meta.read_bytes()
            with patch.object(extractor, "load_model_and_tokenizer", side_effect=AssertionError("model load forbidden")):
                with self.assertRaisesRegex(FileExistsError, "not resumable"):
                    run_main(activation, activation_args)
            self.assertEqual(meta.read_bytes(), original_meta)
            self.assertEqual(len(read_jsonl(meta)), 4)
            prefix = work / "toy"
            run_main(vectors, ["--meta_jsonl", meta, "--shard_glob", acts / "toy__model_activations_shard*.npz",
                               "--out_prefix", prefix])
            vector_path = Path(str(prefix) + ".value_vectors.npz")
            with np.load(vector_path) as z:
                np.testing.assert_array_equal(z["layer_1"], [[2, -1]])
                np.testing.assert_array_equal(z["layer_2"], [[4, -2]])
            target = work / "targets.jsonl"
            rows = [
                {"idx": 7, "text_type": "plain", "target_sentence": "a", "value_items": [{"value_item": "care"}]},
                {"idx": 7, "text_type": "plain", "target_sentence": "", "value_items": []},
                {"idx": 7, "text_type": "plain", "target_sentence": "b", "value_items": [{"value_item": "care"}]},
            ]
            write_jsonl(target, rows)
            defs = work / "defs.json"
            defs.write_text(json.dumps({"care": "Synthetic value."}), encoding="utf-8")
            scores = work / "scores"
            config = acts / "toy__model_activations_config.json"
            scoring_args = ["--input_jsonls", target, "--model_name", "toy/model", "--layers", "1,2",
                            "--pooling", "mean_last_k_tokens", "--last_k", "3", "--max_length", "256",
                            "--value_index_jsonl", str(prefix) + ".value_vectors_index.jsonl",
                            "--value_vectors_npz", vector_path, "--dedup_value_defs_json", defs,
                            "--out_dir", scores, "--quant", "int16", "--activation_config", config]
            native_config = config.read_text(encoding="utf-8")
            for field, bad in (("model_name", "other/model"), ("layers", [1]),
                               ("pooling", "last_token"), ("last_k", 2)):
                with self.subTest(config_mismatch=field):
                    modified = json.loads(native_config)
                    modified[field] = bad
                    config.write_text(json.dumps(modified), encoding="utf-8")
                    with patch.object(extractor, "load_model_and_tokenizer", side_effect=AssertionError("model load forbidden")):
                        with self.assertRaisesRegex(ValueError, "Activation config mismatch"):
                            run_main(scorer, scoring_args)
                    self.assertFalse(scores.exists())
            config.write_text(native_config, encoding="utf-8")
            original_vectors = vector_path.read_bytes()
            for arrays in (
                {"layer_1": np.zeros((1, 2)), "layer_2": np.zeros((1, 3))},
                {"layer_1": np.zeros((1, 2)), "layer_2": np.zeros((2, 2))},
                {"layer_1": np.zeros((1, 2)), "layer_2": np.zeros((1,))},
            ):
                with self.subTest(vector_shapes=[v.shape for v in arrays.values()]):
                    np.savez_compressed(vector_path, **arrays)
                    with patch.object(extractor, "load_model_and_tokenizer", side_effect=AssertionError("model load forbidden")):
                        with self.assertRaises(ValueError):
                            run_main(scorer, scoring_args)
                    self.assertFalse(scores.exists())
            vector_path.write_bytes(original_vectors)
            with patch.object(extractor, "load_model_and_tokenizer", return_value=bundle):
                with patch.object(bundle.model.config, "hidden_size", 3):
                    with self.assertRaisesRegex(ValueError, "hidden_size=3"):
                        run_main(scorer, scoring_args)
                self.assertFalse(scores.exists())
                with patch.object(bundle.tokenizer, "padding_side", "left"):
                    with self.assertRaisesRegex(ValueError, "requires right padding"):
                        run_main(scorer, scoring_args)
                self.assertFalse(scores.exists())
            with patch.object(extractor, "load_model_and_tokenizer", return_value=bundle):
                run_main(scorer, scoring_args)
            output = work / "attached.jsonl"
            run_main(attach, ["--input_jsonl", target, "--columns_json", scores / "targets..columns.json",
                              "--scores_npz", scores / "targets..scores.npz", "--meta_jsonl", scores / "targets..meta.jsonl",
                              "--out_jsonl", output, "--strict_meta_check", "--fail_fast"])
            attached = read_jsonl(output)
            self.assertEqual(attached[1], rows[1])
            self.assertEqual(attached[0]["value_items"][0]["proj_scores"], {"1": 0.8944, "2": 0.8944})
            self.assertEqual(attached[2]["value_items"][0]["proj_scores"], {"1": -0.4472, "2": -0.4472})


if __name__ == "__main__":
    unittest.main()
