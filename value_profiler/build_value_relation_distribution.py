import json
import argparse
from collections import defaultdict
from pathlib import Path

VALID_STATUSES = {"ACTIVATED", "EXPRESSED", "RELATED"}  # can modify

def nested_dict():
    return defaultdict(nested_dict)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--attribute", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_count", type=int, default=1)
    args = parser.parse_args()

    # -----------------------------
    # COUNT STORAGE
    # -----------------------------
    cooccur_count = defaultdict(lambda: defaultdict(int))
    relation_count = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # -----------------------------
    # READ JSONL
    # -----------------------------
    with open(args.input, "r") as f:
        for line in f:
            data = json.loads(line)

            graph = data.get("value_relation_graph", {})
            values = graph.get("values", [])
            edges = graph.get("edges", [])

            # filter valid values
            valid_values = [
                v["value_item"]
                for v in values
                if v.get("status") in VALID_STATUSES
            ]

            # co-occurrence (pairwise)
            for i in range(len(valid_values)):
                for j in range(len(valid_values)):
                    if i == j:
                        continue
                    v_i = valid_values[i]
                    v_j = valid_values[j]
                    cooccur_count[v_i][v_j] += 1

            # relations
            for e in edges:
                v_i = e["source"]
                v_j = e["target"]
                r = e["relation"]

                if v_i == v_j:
                    continue

                relation_count[v_i][v_j][r] += 1

    # -----------------------------
    # NORMALIZATION
    # -----------------------------
    result = {
        "attribute": args.attribute,
        "value_relations": {}
    }

    for v_i in cooccur_count:
        # filter by min_count
        filtered = {
            v_j: c for v_j, c in cooccur_count[v_i].items()
            if c >= args.min_count
        }

        # sort & top-k
        sorted_items = sorted(filtered.items(), key=lambda x: -x[1])[:args.top_k]

        total_co = sum(c for _, c in sorted_items)

        if total_co == 0:
            continue

        result["value_relations"][v_i] = {
            "counts": {
                "cooccurrence": dict(sorted_items),
                "relations": {}
            },
            "distribution": {
                "cooccurrence": {},
                "relations": {}
            }
        }

        # -----------------
        # Co-occurrence dist
        # -----------------
        for v_j, c in sorted_items:
            result["value_relations"][v_i]["distribution"]["cooccurrence"][v_j] = c / total_co

        # -----------------
        # Relation dist
        # -----------------
        for v_j, _ in sorted_items:
            rel_counts = relation_count[v_i][v_j]
            total_rel = sum(rel_counts.values())

            if total_rel == 0:
                continue

            result["value_relations"][v_i]["counts"]["relations"][v_j] = dict(rel_counts)
            result["value_relations"][v_i]["distribution"]["relations"][v_j] = {
                r: c / total_rel for r, c in rel_counts.items()
            }

    # -----------------------------
    # SAVE
    # -----------------------------
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"[Saved] {output_path}")


if __name__ == "__main__":
    main()