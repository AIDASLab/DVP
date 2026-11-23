"""Support-based selection of stored aggregate profiles.

The protocol selects whole profiles, states, or comparisons. It never changes
probabilities, removes individual relation cells, or recomputes statistics.
"""

from __future__ import annotations

import math


NON_INVOCATION_ALIASES = {("opposition to war", "Не упоминается антивоенность")}
PAIR_FAMILIES = {
    "context", "continuous_context", "context_relation_ot", "context_relation_deltas",
}


def _require(row: dict, *fields: str) -> None:
    if not isinstance(row, dict):
        raise ValueError("Expected a profile object")
    missing = set(fields) - row.keys()
    if missing:
        raise ValueError(f"Missing profile fields: {', '.join(sorted(missing))}")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _count(value: object) -> float | None:
    result = _number(value)
    return result if result is not None and result >= 0 and result.is_integer() else None


def _numbers(row: dict, fields: list[str]) -> dict[str, float] | None:
    _require(row, *fields)
    result = {field: _number(row[field]) for field in fields}
    return result if all(value is not None for value in result.values()) else None


def _distribution(values: list[object]) -> bool:
    numbers = [_number(value) for value in values]
    return (bool(numbers) and all(n is not None and 0 <= n <= 1 for n in numbers)
            and abs(sum(numbers) - 1) <= 0.001)


def _threshold(value: int) -> int:
    if type(value) is not int or value < 0:
        raise ValueError("Support thresholds must be nonnegative integers")
    return value


class SupportProtocol:
    """Apply configurable support and core-validity checks to native profiles."""

    def __init__(self, base_support: int, context_support: int,
                 allowed_states: dict | None = None):
        self.base_support = _threshold(base_support)
        self.context_support = _threshold(context_support)
        self.allowed_states = allowed_states

    def metadata(self) -> dict:
        return {
            "name": "support", "version": 1,
            "base_support": self.base_support,
            "context_support": self.context_support,
            "exclude_unassigned": True,
            "unassigned_state": "NONE",
            "require_catalog_states": True,
            "exclude_non_invocation_categories": True,
            "relation_anchor_gate": "eligible_categorical_base",
            "numeric_validity": "Finite core statistics, valid counts and probabilities, ordered intervals",
            "statistics": "Preserved; no renormalization or individual relation-cell pruning",
        }

    def eligible_base(self, row: dict, minimum: int | None = None) -> bool:
        """Check an invoked categorical profile without modifying its categories."""
        _require(row, "value_item", "categories", "total_count")
        threshold = self.base_support if minimum is None else _threshold(minimum)
        categories = row["categories"]
        total = _count(row["total_count"])
        if (not isinstance(row["value_item"], str) or not row["value_item"]
                or not isinstance(categories, list) or not categories
                or total is None or total <= 0):
            return False
        counts = []
        probabilities = []
        for category in categories:
            _require(category, "name", "count", "proportion")
            name = category["name"]
            if not isinstance(name, str):
                return False
            if ("not invoked" in name.casefold()
                    or (row["value_item"], name) in NON_INVOCATION_ALIASES):
                return False
            count = _count(category["count"])
            probability = _number(category["proportion"])
            if count is None or probability is None:
                return False
            counts.append(count)
            probabilities.append(probability)
        return (sum(counts) == total and total >= threshold
                and _distribution(probabilities)
                and all(abs(p - count / total) <= 0.001
                        for count, p in zip(counts, probabilities)))

    def _states_valid(self, family: str, row: dict) -> bool:
        if self.allowed_states is None:
            raise ValueError("The support protocol needs the context-factor catalog")
        value_field = "value_item" if family in {"context", "continuous_context"} else "anchor_value"
        state_fields = ["state"] if family == "context_relation_counts" else ["state_a", "state_b"]
        _require(row, value_field, "factor", *state_fields)
        if any(not isinstance(row[field], str) for field in [value_field, "factor", *state_fields]):
            return False
        states = [row[field] for field in state_fields]
        if "NONE" in states:
            return False
        allowed = self.allowed_states.get((row[value_field], row["factor"]), set())
        return all(state in allowed for state in states)

    def keep(self, family: str, row: dict, minimum: int | None = None) -> bool:
        """Return whether a record passes; missing required fields are errors."""
        if family == "base":
            return self.eligible_base(row, minimum)
        if family == "continuous_base":
            return self._continuous_base(row, minimum)
        if family not in PAIR_FAMILIES | {"context_relation_counts"}:
            raise ValueError(f"Unsupported profile family: {family}")
        threshold = self.context_support if minimum is None else _threshold(minimum)
        support_fields = ["support_N_state"] if family == "context_relation_counts" else ["N_a", "N_b"]
        _require(row, *support_fields)
        support = [_count(row[field]) for field in support_fields]
        if any(n is None or n < threshold for n in support):
            return False
        if not self._states_valid(family, row):
            return False
        if family in {"context", "continuous_context"}:
            return self._context(family, row)
        if family == "context_relation_ot":
            return self._relation_ot(row)
        if family == "context_relation_counts":
            _require(row, "count", "empirical_prob")
            count = _count(row["count"])
            p = _number(row["empirical_prob"])
            n = support[0]
            if not (count is not None and p is not None and 0 <= p <= 1
                    and n > 0 and abs(p - count / n) <= 1e-8):
                raise ValueError("Invalid relation-cell count or probability; refusing a partial state distribution")
            return True
        values = _numbers(row, ["pA_mean", "pB_mean", "delta_mean", "delta_q025", "delta_q975"])
        if not (values is not None
                and all(-1e-9 <= values[key] <= 1 + 1e-9 for key in ("pA_mean", "pB_mean"))
                and values["delta_q025"] <= values["delta_q975"]):
            raise ValueError("Invalid relation-cell probabilities or interval; refusing a partial comparison")
        return True

    def _continuous_base(self, row: dict, minimum: int | None) -> bool:
        _require(row, "value_item", "group_mode", "n_samples", "n_unique_sentences")
        threshold = self.base_support if minimum is None else _threshold(minimum)
        numbers = _numbers(row, ["mean", "std", "support_min", "support_max", "q05", "q50", "q95"])
        support = _count(row["n_unique_sentences"])
        return (isinstance(row["value_item"], str) and bool(row["value_item"])
                and row["group_mode"] == "attribute_value"
                and support is not None and support >= threshold
                and _count(row["n_samples"]) is not None and numbers is not None
                and numbers["std"] >= 0 and numbers["support_min"] <= numbers["support_max"]
                and numbers["q05"] <= numbers["q50"] <= numbers["q95"])

    def _context(self, family: str, row: dict) -> bool:
        fields = ["ot_mean", "ot_q025", "ot_q975", "ot_ci_width",
                  "mean_shift", "mean_shift_q025", "mean_shift_q975"]
        bins = None
        if family == "context":
            _require(row, "K_bins")
            bins = _count(row["K_bins"])
            if bins is None or bins < 1:
                return False
            bins = int(bins)
            fields += [f"{prefix}_bin{i}_{suffix}" for i in range(1, bins + 1)
                       for prefix, suffix in [("pA", "mean"), ("pB", "mean"),
                                              ("delta", "mean"), ("delta", "q025"),
                                              ("delta", "q975")]]
        else:
            fields += ["ot_empirical", "mean_shift_empirical", "mean_a", "mean_b", "std_a", "std_b",
                       "q05_a", "q50_a", "q95_a", "q05_b", "q50_b", "q95_b"]
        numbers = _numbers(row, fields)
        if numbers is None or any(n < 0 for key, n in numbers.items()
                                  if key.startswith("ot_") or key.startswith("std_")):
            return False
        if any(numbers[f"{prefix}_q025"] > numbers[f"{prefix}_q975"] for prefix in ("ot", "mean_shift")):
            return False
        if bins is not None:
            if any(not _distribution([numbers[f"p{side}_bin{i}_mean"] for i in range(1, bins + 1)])
                   for side in ("A", "B")):
                return False
            if any(numbers[f"delta_bin{i}_q025"] > numbers[f"delta_bin{i}_q975"]
                   for i in range(1, bins + 1)):
                return False
        return True

    @staticmethod
    def _relation_ot(row: dict) -> bool:
        fields = ["joint_ot_fast_mean", "joint_ot_fast_within_relation", "joint_ot_fast_across_partner",
                  "joint_jsd_mean", "joint_jsd_q025", "joint_jsd_q975", "value_jsd_mean", "relation_jsd_mean"]
        numbers = _numbers(row, fields)
        return (numbers is not None and all(n >= -1e-12 for n in numbers.values())
                and numbers["joint_jsd_q025"] <= numbers["joint_jsd_q975"])

    @staticmethod
    def keep_relation(profile: dict) -> bool:
        """Validate separate partner/conditional distributions, keeping all cells."""
        _require(profile, "counts", "distribution")
        counts, distributions = profile["counts"], profile["distribution"]
        _require(counts, "cooccurrence", "relations")
        _require(distributions, "cooccurrence", "relations")
        co_counts = counts["cooccurrence"]
        co_probs = distributions["cooccurrence"]
        relations = distributions["relations"]
        relation_counts = counts["relations"]
        if not all(isinstance(value, dict) for value in (co_counts, co_probs, relations, relation_counts)):
            return False
        if (not all(_count(n) is not None for n in co_counts.values())
                or not _distribution(list(co_probs.values()))):
            return False
        for partner, probs in relations.items():
            if not isinstance(probs, dict) or not _distribution(list(probs.values())):
                return False
            if partner not in relation_counts:
                raise ValueError(f"Missing relation counts for partner: {partner}")
            partner_counts = relation_counts[partner]
            if (not isinstance(partner_counts, dict)
                    or not all(_count(n) is not None for n in partner_counts.values())):
                return False
        return True
