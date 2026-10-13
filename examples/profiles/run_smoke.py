#!/usr/bin/env python3
"""Exercise the profiling CLIs with generated, entirely synthetic inputs.

Run from the repository root:
    python examples/profiles/run_smoke.py --out_dir outputs/profile_smoke

This checks native formats and execution, not annotation or scientific accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MODULES = [
    "profile_constructor/construct_distributional_surface_regimes.py",
    "profile_constructor/filter_surface_regimes.py",
    "profile_constructor/attach_valueitem_projection_scores.py",
    "profile_constructor/construct_distributional_raw_intensity.py",
    "context_parser/context_effect_ot.py",
    "context_parser/context_effect_ot_raw_score.py",
    "context_parser/context_effect_joint_ot_relations.py",
    "value_profiler/build_value_relation_distribution.py",
    "value_profiler/distributional_dataset.py",
]


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out_dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0")
    commands = []

    def run(module, *options, expected_failure=False):
        command = [sys.executable, "-B", str(ROOT / module), *map(str, options)]
        result = subprocess.run(command, cwd=out, env=env, text=True, capture_output=True)
        commands.append({"argv": command, "returncode": result.returncode})
        with (out / "run.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(command) + "\n" + result.stdout + result.stderr + "\n")
        if expected_failure:
            assert result.returncode != 0 and "nonempty and unique" in result.stderr, result.stderr
        elif result.returncode:
            raise RuntimeError(f"{module} failed:\n{result.stdout}\n{result.stderr}")

    for module in MODULES:
        run(module, "--help")

    values = ["care", "fairness"]
    categories = ["Opposed", "Weak", "Moderate", "Strong", "Not Invoked"]
    definitions = [{"value_item": value, "value_item_index": j,
                    "categories": [{"name": name, "definition": f"Synthetic {name} category", "signals": []}
                                   for name in categories]}
                   for j, value in enumerate(values)]
    factors = {"items": {value: {"factors": [
        {"factor": factor, "states": ["A", "B"], "definition": "Synthetic factor", "cues": []}
        for factor in ["situation", "audience"]]} for value in values}}
    (out / "factors.json").write_text(json.dumps(factors), encoding="utf-8")
    write_jsonl(out / "regimes.jsonl", definitions)
    surfaces, contexts, annotations, relations = [], [], [], []
    for i in range(24):
        key = {"idx": i, "target_sentence": f"Synthetic sentence {i}."}
        state = "A" if i < 12 else "B"
        items = [{"value_item": value, "status": "EXPRESSED"} for value in values]
        annotations.append({**key, "value_items": items})
        surfaces.append({**key, "surface_value_regimes": {
            value: {"category": categories[(i + j) % 4]} for j, value in enumerate(values)}})
        contexts.append({**key, "context_states": {value: {
            factor: {"selected_state": state} for factor in ["situation", "audience"]} for value in values}})
        relations.append({**key, "value_relation_graph": {"values": items, "edges": [
            {"source": "care", "target": "fairness", "relation": "SYNERGY" if state == "A" else "TRADEOFF"},
            {"source": "fairness", "target": "care", "relation": "SYNERGY" if state == "A" else "TRADEOFF"}]}})
    write_jsonl(out / "surface.jsonl", surfaces)
    write_jsonl(out / "context.jsonl", contexts)
    write_jsonl(out / "annotations.jsonl", annotations)
    write_jsonl(out / "relations.jsonl", relations)
    scores = np.array([[i / 12 - 1, 1 - i / 16] for i in range(24)], dtype=np.float32)
    np.savez_compressed(out / "scores.npz", layer_2=scores)
    (out / "columns.json").write_text(json.dumps({"selected_value_items": values, "layers": [2], "quant": "float32"}), encoding="utf-8")

    run(MODULES[0], "--labels_jsonl", "surface.jsonl", "--regimes_jsonl", "regimes.jsonl",
        "--out_jsonl", "counted/demo.surface_regimes.counted.jsonl")
    run(MODULES[1], "--in_dir", "counted", "--out_dir", "filtered", "--threshold", 0)
    run(MODULES[2], "--input_jsonl", "annotations.jsonl", "--scores_npz", "scores.npz",
        "--columns_json", "columns.json", "--out_jsonl", "activation.jsonl", "--fail_fast")

    continuous_args = ["--attribute_name", "demo", "--context_jsonl", "context.jsonl",
                       "--activation_jsonls", "activation.jsonl", "--model_names", "synthetic_model",
                       "--factors_json", "factors.json", "--min_value_item_count", 1]
    run(MODULES[3], *continuous_args, "--min_group_n", 1,
        "--out_summary_csv", "base_summary.csv", "--out_raw_npz", "base.npz",
        "--out_fit_jsonl", "continuous_base/demo/demo.fit.jsonl")
    # Factor/state groups are required by the continuous context --raw_npz path.
    run(MODULES[3], *continuous_args, "--group_mode", "attribute_value_factor_state", "--min_group_n", 1,
        "--out_summary_csv", "state_summary.csv", "--out_raw_npz", "states.npz",
        "--out_fit_jsonl", "states.fit.jsonl")
    run(MODULES[4], "--context_jsonl", "context.jsonl", "--surface_jsonl", "surface.jsonl",
        "--factors_json", "factors.json", "--surface_regime_defs_jsonl", "regimes.jsonl",
        "--min_value_item_count", 1, "--draws", 50, "--out_csv", "discrete_ot/demo.ot.csv",
        "--out_counts_csv", "discrete_counts.csv")
    run(MODULES[5], *continuous_args, "--draws", 50, "--out_csv", "continuous_ot/demo.ot.csv")
    run(MODULES[5], "--raw_npz", "states.npz", "--draws", 50, "--out_csv", "continuous_from_npz.csv")
    run(MODULES[6], "--context_jsonl", "context.jsonl", "--relation_jsonl", "relations.jsonl",
        "--factors_json", "factors.json", "--min_anchor_value_count", 1, "--min_partner_count", 1,
        "--draws", 50, "--out_csv", "joint.csv", "--out_counts_csv", "joint_counts.csv",
        "--out_delta_csv", "joint_deltas.csv")
    run(MODULES[7], "--input", "relations.jsonl", "--output", "relation_distribution.json", "--attribute", "demo")
    run(MODULES[8], "--attributes", "demo", "--distribution_dir", "filtered",
        "--distribution_suffix", ".surface_regimes.filtered_noNotInv_gt0.jsonl",
        "--ot_dir", "discrete_ot", "--ot_suffix", ".ot.csv",
        "--continuous_base_dir", "continuous_base", "--continuous_base_suffix", ".fit.jsonl",
        "--continuous_ot_dir", "continuous_ot", "--continuous_ot_suffix", ".ot.csv",
        "--factors_path", "factors.json", "--surface_regime_defs_path", "regimes.jsonl",
        "--output_dir", "assembled", "--quiet")

    base = read_jsonl(out / "assembled/distributional_profile_base_v2.jsonl")
    pairs = read_jsonl(out / "assembled/distributional_profile_pairwise_ot_v2.jsonl")
    assert len(base) == 2 and len(pairs) == 4
    assert all(row["total_count"] == 24 for row in base)
    # Two factor records per sentence give 48 samples / 24 unique sentences.
    assert all(row["continuous_distribution"]["n_samples"] == 48 for row in base)
    assert all(row["continuous_distribution"]["n_unique_sentences"] == 24 for row in base)
    assert all(row["discrete_ot"] and row["continuous_ot"] for row in pairs)
    with (out / "joint.csv").open(encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 4
    relation_profile = json.loads((out / "relation_distribution.json").read_text())
    assert relation_profile["value_relations"]["care"]["counts"]["cooccurrence"]["fairness"] == 24
    assert relation_profile["value_relations"]["care"]["distribution"]["relations"]["fairness"] == {"SYNERGY": 0.5, "TRADEOFF": 0.5}

    for module, output_options in [
        (MODULES[3], ["--out_summary_csv", "collision.csv", "--out_raw_npz", "collision.npz", "--out_fit_jsonl", "collision.jsonl"]),
        (MODULES[5], ["--out_csv", "collision_context.csv"]),
    ]:
        run(module, "--attribute_name", "demo", "--context_jsonl", "context.jsonl",
            "--activation_jsonls", "activation.jsonl,activation.jsonl", *output_options, expected_failure=True)

    result = {"status": "PASS", "help_pages": len(MODULES), "assembled_base_rows": len(base),
              "assembled_context_rows": len(pairs), "joint_relation_rows": 4,
              "duplicate_source_rejections": 2, "commands": commands}
    (out / "verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "commands"}, indent=2))


if __name__ == "__main__":
    main()
