#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Assemble base distributions and pairwise context effects into JSONL profiles.

The base file determines which values are included. Context rows are joined by
(value_item, factor, state_a, state_b) and filtered by minimum state support.
Continuous summaries are optional. Input directories, suffixes, catalogs and
minimum support are configurable through the constructor or command line.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ============================================================
# User-configurable section
# ============================================================

# Discrete base distributions
DISTRIBUTION_DIR = Path(
    "outputs/distributional_profiles/surface_regimes_filtered"
)
DISTRIBUTION_SUFFIX = ".surface_regimes.filtered_noNotInv_gt200.jsonl"

# Discrete pairwise OT
OT_DIR = Path(
    "outputs/context_effect_ot"
)
OT_SUFFIX = ".items_by_count200.ot_context_effects.csv"

# Continuous base Gaussian fit
CONTINUOUS_BASE_DIR = Path(
    "outputs/distributional_profiles/continuous_distribution"
)
CONTINUOUS_BASE_SUFFIX = ".items_by_count200.group_attribute_value.gaussian_fit.jsonl"

# Continuous pairwise OT
CONTINUOUS_OT_DIR = Path(
    "outputs/context_effect_ot_continuous"
)
CONTINUOUS_OT_SUFFIX = ".items_by_count200.norm_polarity_quantile.ot_context_effects.continuous.csv"

FACTORS_PATH = Path(
    "resources/original/context_factors_revised.json"
)
SURFACE_REGIME_DEFS_PATH = Path(
    "resources/original/surface_value_regimes_gpt.jsonl"
)

OUTPUT_DIR = Path(
    "outputs/distributional_profile_dataset"
)

SCHEMA_VERSION = "distributional_profile_v2"

# Reliability filters
MIN_BASE_TOTAL_COUNT = 1
MIN_OT_STATE_COUNT = 10


# ============================================================
# Data classes
# ============================================================

@dataclass
class CategoryEntry:
    bin_index: int
    name: str
    description: Optional[str]
    signals: List[str] = field(default_factory=list)
    count: Optional[int] = None
    proportion: Optional[float] = None


@dataclass
class FactorDefinitionEntry:
    group: Optional[str]
    factor: str
    definition: Optional[str]
    states: List[str] = field(default_factory=list)
    polarity: Optional[str] = None
    cues: List[str] = field(default_factory=list)


@dataclass
class BaseDistributionExample:
    schema_version: str
    example_type: str

    attribute_raw: str
    attribute_label: str

    value_item: str
    value_item_index: Optional[int]
    value_definition: Optional[str]

    categories: List[Dict[str, Any]]
    distribution: List[float]
    counts: List[int]
    total_count: int

    discrete_expected_score: Optional[float]

    continuous_distribution: Optional[Dict[str, Any]]

    factor_definitions: List[Dict[str, Any]]

    metadata: Dict[str, Any]


@dataclass
class PairwiseOTExample:
    schema_version: str
    example_type: str

    attribute_raw: str
    attribute_label: str

    value_item: str
    value_item_index: Optional[int]
    value_definition: Optional[str]

    factor: str
    factor_group: Optional[str]
    factor_definition: Optional[str]
    factor_states: List[str]
    factor_polarity: Optional[str]
    factor_cues: List[str]

    state_a: str
    state_b: str

    discrete_ot: Optional[Dict[str, Any]]
    continuous_ot: Optional[Dict[str, Any]]

    metadata: Dict[str, Any]


# ============================================================
# Helpers
# ============================================================

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse JSONL: {path} line {line_idx}") from e


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_csv_dicts(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _to_optional_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except ValueError:
        return None


def _to_int(x: Any, default: int = 0) -> int:
    out = _to_optional_int(x)
    return default if out is None else out


def _to_optional_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_float(x: Any, default: float = 0.0) -> float:
    out = _to_optional_float(x)
    return default if out is None else out


def _safe_get_attribute_label(attribute_raw: str) -> str:
    return attribute_raw


def _infer_value_definition_from_categories(categories: List[Dict[str, Any]]) -> Optional[str]:
    # Placeholder. You can plug in a real value-definition source later.
    return None


def _expected_score_from_distribution(distribution: List[float], one_indexed: bool = True) -> Optional[float]:
    if not distribution:
        return None
    offset = 1 if one_indexed else 0
    return sum((i + offset) * p for i, p in enumerate(distribution))


# ============================================================
# Definition loaders
# ============================================================

def load_factor_definitions(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Returns:
        factor_lookup[value_item][factor] = {
            "group": ...,
            "factor": ...,
            "definition": ...,
            "states": [...],
            "polarity": ...,
            "cues": [...],
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing factors file: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", {}) or {}

    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}

    for value_item, info in items.items():
        factors = info.get("factors", []) or []
        lookup[value_item] = {}

        for factor_entry in factors:
            factor_name = factor_entry.get("factor")
            if not factor_name:
                continue

            lookup[value_item][factor_name] = {
                "group": factor_entry.get("group"),
                "factor": factor_name,
                "definition": factor_entry.get("definition"),
                "states": list(factor_entry.get("states", []) or []),
                "polarity": factor_entry.get("polarity"),
                "cues": list(factor_entry.get("cues", []) or []),
            }

    return lookup


def load_surface_regime_definitions(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
        regime_lookup[value_item] = {
            "value_item_index": ...,
            "categories": [...]
        }
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing surface regime defs file: {path}")

    lookup: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        value_item = row.get("value_item")
        if value_item is None:
            continue
        lookup[value_item] = {
            "value_item_index": row.get("value_item_index"),
            "categories": list(row.get("categories", []) or []),
        }
    return lookup


# ============================================================
# Distribution helpers
# ============================================================

def _normalize_categories_with_defs(
    value_item: str,
    dist_row: Dict[str, Any],
    regime_lookup: Dict[str, Dict[str, Any]],
) -> Tuple[List[CategoryEntry], Optional[int]]:
    """
    Prefer descriptions/signals from regime defs; counts/proportions from dist row.
    Align by index.
    """
    dist_categories = list(dist_row.get("categories", []) or [])
    regime_info = regime_lookup.get(value_item, {})
    regime_categories = list(regime_info.get("categories", []) or [])
    value_item_index = regime_info.get("value_item_index")

    out: List[CategoryEntry] = []
    n = len(dist_categories)

    for idx in range(n):
        dist_cat = dist_categories[idx] if idx < len(dist_categories) else {}
        regime_cat = regime_categories[idx] if idx < len(regime_categories) else {}

        out.append(
            CategoryEntry(
                bin_index=idx,
                name=dist_cat.get("name") or regime_cat.get("name") or f"bin_{idx}",
                description=regime_cat.get("description") or dist_cat.get("description"),
                signals=list(regime_cat.get("signals", []) or dist_cat.get("signals", []) or []),
                count=_to_optional_int(dist_cat.get("count")),
                proportion=_to_optional_float(dist_cat.get("proportion")),
            )
        )

    return out, value_item_index


def _extract_distribution_and_counts(categories: List[CategoryEntry]) -> Tuple[List[float], List[int]]:
    distribution: List[float] = []
    counts: List[int] = []

    for cat in categories:
        distribution.append(float(cat.proportion) if cat.proportion is not None else 0.0)
        counts.append(int(cat.count) if cat.count is not None else 0)

    return distribution, counts


def _get_factor_definitions_for_value(
    value_item: str,
    factor_lookup: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[FactorDefinitionEntry]:
    factor_dict = factor_lookup.get(value_item, {})
    out: List[FactorDefinitionEntry] = []

    for factor_name in sorted(factor_dict.keys()):
        info = factor_dict[factor_name]
        out.append(
            FactorDefinitionEntry(
                group=info.get("group"),
                factor=factor_name,
                definition=info.get("definition"),
                states=list(info.get("states", []) or []),
                polarity=info.get("polarity"),
                cues=list(info.get("cues", []) or []),
            )
        )

    return out


# ============================================================
# Continuous lookup builders
# ============================================================

def _get_continuous_base_path(attribute: str) -> Path:
    return CONTINUOUS_BASE_DIR / attribute / f"{attribute}{CONTINUOUS_BASE_SUFFIX}"


def _get_continuous_ot_path(attribute: str) -> Path:
    return CONTINUOUS_OT_DIR / f"{attribute}{CONTINUOUS_OT_SUFFIX}"


def load_continuous_base_lookup(attribute: str, path: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    """
    Returns:
        lookup[value_item] = row
    Only group_mode == attribute_value is kept.
    """
    path = Path(path) if path is not None else _get_continuous_base_path(attribute)
    if not path.exists():
        return {}

    lookup: Dict[str, Dict[str, Any]] = {}
    for row in read_jsonl(path):
        if row.get("group_mode") != "attribute_value":
            continue
        value_item = row.get("value_item")
        if not value_item:
            continue
        lookup[value_item] = row
    return lookup


def _pair_key(value_item: str, factor: str, state_a: str, state_b: str) -> Tuple[str, str, str, str]:
    return (
        value_item.strip(),
        factor.strip(),
        state_a.strip(),
        state_b.strip(),
    )


def load_discrete_ot_lookup(attribute: str, path: Optional[Path] = None) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    path = Path(path) if path is not None else OT_DIR / f"{attribute}{OT_SUFFIX}"
    if not path.exists():
        return {}

    out: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in read_csv_dicts(path):
        key = _pair_key(
            row.get("value_item", ""),
            row.get("factor", ""),
            row.get("state_a", ""),
            row.get("state_b", ""),
        )
        out[key] = row
    return out


def load_continuous_ot_lookup(attribute: str, path: Optional[Path] = None) -> Dict[Tuple[str, str, str, str], Dict[str, str]]:
    path = Path(path) if path is not None else _get_continuous_ot_path(attribute)
    if not path.exists():
        return {}

    out: Dict[Tuple[str, str, str, str], Dict[str, str]] = {}
    for row in read_csv_dicts(path):
        key = _pair_key(
            row.get("value_item", ""),
            row.get("factor", ""),
            row.get("state_a", ""),
            row.get("state_b", ""),
        )
        out[key] = row
    return out


# ============================================================
# Normalization: base distribution
# ============================================================

def normalize_base_distribution_row(
    attribute_raw: str,
    row: Dict[str, Any],
    regime_lookup: Dict[str, Dict[str, Any]],
    factor_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    continuous_base_lookup: Dict[str, Dict[str, Any]],
    source_distribution_jsonl: Path,
) -> Optional[Dict[str, Any]]:
    value_item = row.get("value_item")
    if not value_item:
        return None

    total_count = _to_int(row.get("total_count"), default=0)
    if total_count < MIN_BASE_TOTAL_COUNT:
        return None

    categories, value_item_index = _normalize_categories_with_defs(
        value_item=value_item,
        dist_row=row,
        regime_lookup=regime_lookup,
    )
    distribution, counts = _extract_distribution_and_counts(categories)
    value_definition = _infer_value_definition_from_categories([asdict(c) for c in categories])

    factor_definitions = _get_factor_definitions_for_value(
        value_item=value_item,
        factor_lookup=factor_lookup,
    )

    continuous_row = continuous_base_lookup.get(value_item)
    continuous_distribution: Optional[Dict[str, Any]] = None
    if continuous_row is not None:
        continuous_distribution = {
            "group_id": continuous_row.get("group_id"),
            "group_mode": continuous_row.get("group_mode"),
            "normalization_mode": continuous_row.get("normalization_mode"),
            "clip_z": _to_optional_float(continuous_row.get("clip_z")),
            "tanh_temperature": _to_optional_float(continuous_row.get("tanh_temperature")),
            "outlier_quantile_low": _to_optional_float(continuous_row.get("outlier_quantile_low")),
            "outlier_quantile_high": _to_optional_float(continuous_row.get("outlier_quantile_high")),
            "outlier_action": continuous_row.get("outlier_action"),
            "distribution_type": continuous_row.get("distribution_type"),
            "mean": _to_optional_float(continuous_row.get("mean")),
            "std": _to_optional_float(continuous_row.get("std")),
            "support_min": _to_optional_float(continuous_row.get("support_min")),
            "support_max": _to_optional_float(continuous_row.get("support_max")),
            "n_samples": _to_optional_int(continuous_row.get("n_samples")),
            "n_unique_sentences": _to_optional_int(continuous_row.get("n_unique_sentences")),
            "gaussian_like_flag": _to_optional_int(continuous_row.get("gaussian_like_flag")),
            "ks_to_fitted_normal": _to_optional_float(continuous_row.get("ks_to_fitted_normal")),
            "skewness": _to_optional_float(continuous_row.get("skewness")),
            "excess_kurtosis": _to_optional_float(continuous_row.get("excess_kurtosis")),
            "jb_stat": _to_optional_float(continuous_row.get("jb_stat")),
            "q05": _to_optional_float(continuous_row.get("q05")),
            "q50": _to_optional_float(continuous_row.get("q50")),
            "q95": _to_optional_float(continuous_row.get("q95")),
        }

    example = BaseDistributionExample(
        schema_version=SCHEMA_VERSION,
        example_type="base_distribution",
        attribute_raw=attribute_raw,
        attribute_label=_safe_get_attribute_label(attribute_raw),
        value_item=value_item,
        value_item_index=value_item_index if value_item_index is not None else row.get("value_item_index"),
        value_definition=value_definition,
        categories=[asdict(c) for c in categories],
        distribution=distribution,
        counts=counts,
        total_count=total_count,
        discrete_expected_score=_expected_score_from_distribution(distribution, one_indexed=True),
        continuous_distribution=continuous_distribution,
        factor_definitions=[asdict(x) for x in factor_definitions],
        metadata={
            "source_distribution_jsonl": str(source_distribution_jsonl),
            "distribution_run_id": row.get("run_id"),
            "distribution_meta": row.get("meta", {}) or {},
            "has_continuous_distribution": continuous_distribution is not None,
        },
    )
    return asdict(example)


# ============================================================
# Normalization: OT blocks
# ============================================================

def _build_discrete_ot_block(row: Dict[str, str], min_state_count: int = MIN_OT_STATE_COUNT) -> Optional[Dict[str, Any]]:
    N_a = _to_int(row.get("N_a"), default=0)
    N_b = _to_int(row.get("N_b"), default=0)
    if N_a < min_state_count or N_b < min_state_count:
        return None

    K_bins = _to_int(row.get("K_bins"), default=0)
    if K_bins <= 0:
        return None

    state_a_distribution = [
        _to_float(row.get(f"pA_bin{i}_mean"), default=0.0)
        for i in range(1, K_bins + 1)
    ]
    state_b_distribution = [
        _to_float(row.get(f"pB_bin{i}_mean"), default=0.0)
        for i in range(1, K_bins + 1)
    ]
    delta_distribution = [
        _to_float(row.get(f"delta_bin{i}_mean"), default=0.0)
        for i in range(1, K_bins + 1)
    ]
    delta_q025 = [
        _to_float(row.get(f"delta_bin{i}_q025"), default=0.0)
        for i in range(1, K_bins + 1)
    ]
    delta_q975 = [
        _to_float(row.get(f"delta_bin{i}_q975"), default=0.0)
        for i in range(1, K_bins + 1)
    ]

    return {
        "state_a": row.get("state_a", "").strip(),
        "state_b": row.get("state_b", "").strip(),
        "N_a": N_a,
        "N_b": N_b,
        "low_n_flag": _to_int(row.get("low_n_flag"), default=0),
        "K_bins": K_bins,
        "lambda_prior": _to_float(row.get("lambda_prior"), default=0.0),
        "draws": _to_int(row.get("draws"), default=0),
        "seed": _to_int(row.get("seed"), default=0),
        "ot": {
            "ot_mean": _to_float(row.get("ot_mean"), default=0.0),
            "ot_q025": _to_float(row.get("ot_q025"), default=0.0),
            "ot_q975": _to_float(row.get("ot_q975"), default=0.0),
            "ot_ci_width": _to_float(row.get("ot_ci_width"), default=0.0),
        },
        "mean_shift": {
            "mean_shift": _to_float(row.get("mean_shift"), default=0.0),
            "q025": _to_float(row.get("mean_shift_q025"), default=0.0),
            "q975": _to_float(row.get("mean_shift_q975"), default=0.0),
        },
        "state_a_distribution": state_a_distribution,
        "state_b_distribution": state_b_distribution,
        "delta_distribution": delta_distribution,
        "delta_q025": delta_q025,
        "delta_q975": delta_q975,
    }


def _build_continuous_ot_block(row: Dict[str, str], min_state_count: int = MIN_OT_STATE_COUNT) -> Optional[Dict[str, Any]]:
    N_a = _to_int(row.get("N_a"), default=0)
    N_b = _to_int(row.get("N_b"), default=0)
    if N_a < min_state_count or N_b < min_state_count:
        return None

    return {
        "state_a": row.get("state_a", "").strip(),
        "state_b": row.get("state_b", "").strip(),
        "N_a": N_a,
        "N_b": N_b,
        "low_n_flag": _to_int(row.get("low_n_flag"), default=0),
        "draws": _to_int(row.get("draws"), default=0),
        "seed": _to_int(row.get("seed"), default=0),
        "normalization_mode": row.get("normalization_mode"),
        "outlier_quantile_low": _to_optional_float(row.get("outlier_quantile_low")),
        "outlier_quantile_high": _to_optional_float(row.get("outlier_quantile_high")),
        "outlier_action": row.get("outlier_action"),
        "ot_empirical": _to_optional_float(row.get("ot_empirical")),
        "mean_shift_empirical": _to_optional_float(row.get("mean_shift_empirical")),
        "ot": {
            "ot_mean": _to_optional_float(row.get("ot_mean")),
            "ot_q025": _to_optional_float(row.get("ot_q025")),
            "ot_q975": _to_optional_float(row.get("ot_q975")),
            "ot_ci_width": _to_optional_float(row.get("ot_ci_width")),
        },
        "mean_shift": {
            "mean_shift": _to_optional_float(row.get("mean_shift")),
            "q025": _to_optional_float(row.get("mean_shift_q025")),
            "q975": _to_optional_float(row.get("mean_shift_q975")),
        },
        "summary_a": {
            "mean": _to_optional_float(row.get("mean_a")),
            "std": _to_optional_float(row.get("std_a")),
            "q05": _to_optional_float(row.get("q05_a")),
            "q50": _to_optional_float(row.get("q50_a")),
            "q95": _to_optional_float(row.get("q95_a")),
        },
        "summary_b": {
            "mean": _to_optional_float(row.get("mean_b")),
            "std": _to_optional_float(row.get("std_b")),
            "q05": _to_optional_float(row.get("q05_b")),
            "q50": _to_optional_float(row.get("q50_b")),
            "q95": _to_optional_float(row.get("q95_b")),
        },
    }


# ============================================================
# Normalization: merged pairwise OT example
# ============================================================

def normalize_pairwise_ot_row(
    attribute_raw: str,
    value_item: str,
    factor: str,
    state_a: str,
    state_b: str,
    valid_value_items: Dict[str, Dict[str, Any]],
    factor_lookup: Dict[str, Dict[str, Dict[str, Any]]],
    discrete_ot_row: Optional[Dict[str, str]],
    continuous_ot_row: Optional[Dict[str, str]],
    source_discrete_ot_csv: Path,
    source_continuous_ot_csv: Path,
    min_ot_state_count: int = MIN_OT_STATE_COUNT,
) -> Optional[Dict[str, Any]]:
    if value_item not in valid_value_items:
        return None

    discrete_ot_block = _build_discrete_ot_block(discrete_ot_row, min_ot_state_count) if discrete_ot_row is not None else None
    continuous_ot_block = _build_continuous_ot_block(continuous_ot_row, min_ot_state_count) if continuous_ot_row is not None else None

    # keep only if at least one usable OT block remains
    if discrete_ot_block is None and continuous_ot_block is None:
        return None

    factor_info = (factor_lookup.get(value_item, {}) or {}).get(factor, {})

    base_info = valid_value_items[value_item]
    value_item_index = base_info.get("value_item_index")
    value_definition = base_info.get("value_definition")

    example = PairwiseOTExample(
        schema_version=SCHEMA_VERSION,
        example_type="pairwise_context_ot",
        attribute_raw=attribute_raw,
        attribute_label=_safe_get_attribute_label(attribute_raw),
        value_item=value_item,
        value_item_index=value_item_index,
        value_definition=value_definition,
        factor=factor,
        factor_group=factor_info.get("group"),
        factor_definition=factor_info.get("definition"),
        factor_states=list(factor_info.get("states", []) or []),
        factor_polarity=factor_info.get("polarity"),
        factor_cues=list(factor_info.get("cues", []) or []),
        state_a=state_a,
        state_b=state_b,
        discrete_ot=discrete_ot_block,
        continuous_ot=continuous_ot_block,
        metadata={
            "source_discrete_ot_csv": str(source_discrete_ot_csv),
            "source_continuous_ot_csv": str(source_continuous_ot_csv),
            "has_discrete_ot": discrete_ot_block is not None,
            "has_continuous_ot": continuous_ot_block is not None,
        },
    )
    return asdict(example)


# ============================================================
# Dataset builder
# ============================================================

class DistributionalProfileDataset:
    """
    Combine base and context-conditioned profile summaries.

    Produces:
      - base_examples
      - pairwise_ot_examples
      - all_examples

    Filtering policy:
      1) base distributions: keep the value items present in the filtered
         distributional profile JSONL
      2) discrete OT rows: keep only rows whose value_item is present in the base file
         and whose N_a and N_b meet min_ot_state_count
      3) continuous OT rows: same filter
      4) pairwise rows are merged by (value_item, factor, state_a, state_b)
    """

    def __init__(
        self,
        attributes: List[str],
        distribution_dir: Path = DISTRIBUTION_DIR,
        distribution_suffix: str = DISTRIBUTION_SUFFIX,
        ot_dir: Path = OT_DIR,
        ot_suffix: str = OT_SUFFIX,
        continuous_base_dir: Path = CONTINUOUS_BASE_DIR,
        continuous_base_suffix: str = CONTINUOUS_BASE_SUFFIX,
        continuous_ot_dir: Path = CONTINUOUS_OT_DIR,
        continuous_ot_suffix: str = CONTINUOUS_OT_SUFFIX,
        factors_path: Path = FACTORS_PATH,
        surface_regime_defs_path: Path = SURFACE_REGIME_DEFS_PATH,
        verbose: bool = True,
        min_ot_state_count: int = MIN_OT_STATE_COUNT,
    ) -> None:
        self.attributes = attributes
        self.distribution_dir = Path(distribution_dir)
        self.distribution_suffix = distribution_suffix
        self.ot_dir = Path(ot_dir)
        self.ot_suffix = ot_suffix
        self.continuous_base_dir = Path(continuous_base_dir)
        self.continuous_base_suffix = continuous_base_suffix
        self.continuous_ot_dir = Path(continuous_ot_dir)
        self.continuous_ot_suffix = continuous_ot_suffix
        self.factors_path = Path(factors_path)
        self.surface_regime_defs_path = Path(surface_regime_defs_path)
        self.verbose = verbose
        if min_ot_state_count < 0:
            raise ValueError("min_ot_state_count must be nonnegative")
        self.min_ot_state_count = min_ot_state_count

        self.factor_lookup = load_factor_definitions(self.factors_path)
        self.regime_lookup = load_surface_regime_definitions(self.surface_regime_defs_path)

        self.base_examples: List[Dict[str, Any]] = []
        self.pairwise_ot_examples: List[Dict[str, Any]] = []
        self.all_examples: List[Dict[str, Any]] = []

        self._build()

    def _get_distribution_path(self, attribute: str) -> Path:
        return self.distribution_dir / f"{attribute}{self.distribution_suffix}"

    def _get_discrete_ot_path(self, attribute: str) -> Path:
        return self.ot_dir / f"{attribute}{self.ot_suffix}"

    def _get_continuous_base_path(self, attribute: str) -> Path:
        return self.continuous_base_dir / attribute / f"{attribute}{self.continuous_base_suffix}"

    def _get_continuous_ot_path(self, attribute: str) -> Path:
        return self.continuous_ot_dir / f"{attribute}{self.continuous_ot_suffix}"

    def _build(self) -> None:
        total_base = 0
        total_ot = 0

        for attribute in self.attributes:
            distribution_path = self._get_distribution_path(attribute)
            discrete_ot_path = self._get_discrete_ot_path(attribute)
            continuous_base_path = self._get_continuous_base_path(attribute)
            continuous_ot_path = self._get_continuous_ot_path(attribute)

            if not distribution_path.exists():
                raise FileNotFoundError(f"Missing distribution file: {distribution_path}")
            if not discrete_ot_path.exists():
                raise FileNotFoundError(f"Missing discrete OT file: {discrete_ot_path}")

            if self.verbose:
                print(f"[INFO] Processing attribute: {attribute}")
                print(f"[INFO]   distribution    : {distribution_path}")
                print(f"[INFO]   discrete OT    : {discrete_ot_path}")
                print(f"[INFO]   continuous base: {continuous_base_path} {'(missing)' if not continuous_base_path.exists() else ''}")
                print(f"[INFO]   continuous OT  : {continuous_ot_path} {'(missing)' if not continuous_ot_path.exists() else ''}")

            # ----------------------------------------
            # Step 1. Continuous base lookup
            # ----------------------------------------
            continuous_base_lookup = load_continuous_base_lookup(attribute, continuous_base_path)

            # ----------------------------------------
            # Step 2. Base distributions define valid set
            # ----------------------------------------
            attribute_base_examples: List[Dict[str, Any]] = []
            valid_value_items: Dict[str, Dict[str, Any]] = {}

            for row in read_jsonl(distribution_path):
                ex = normalize_base_distribution_row(
                    attribute_raw=attribute,
                    row=row,
                    regime_lookup=self.regime_lookup,
                    factor_lookup=self.factor_lookup,
                    continuous_base_lookup=continuous_base_lookup,
                    source_distribution_jsonl=distribution_path,
                )
                if ex is None:
                    continue

                attribute_base_examples.append(ex)
                valid_value_items[ex["value_item"]] = {
                    "value_item_index": ex.get("value_item_index"),
                    "value_definition": ex.get("value_definition"),
                }

            # ----------------------------------------
            # Step 3. Load OT lookups
            # ----------------------------------------
            discrete_ot_lookup = load_discrete_ot_lookup(attribute, discrete_ot_path)
            continuous_ot_lookup = load_continuous_ot_lookup(attribute, continuous_ot_path)

            # union of keys across discrete/continuous OT
            all_pair_keys = set(discrete_ot_lookup.keys()) | set(continuous_ot_lookup.keys())

            # ----------------------------------------
            # Step 4. Merge pairwise OT rows
            # ----------------------------------------
            attribute_pairwise_ot_examples: List[Dict[str, Any]] = []

            for key in sorted(all_pair_keys):
                value_item, factor, state_a, state_b = key

                ex = normalize_pairwise_ot_row(
                    attribute_raw=attribute,
                    value_item=value_item,
                    factor=factor,
                    state_a=state_a,
                    state_b=state_b,
                    valid_value_items=valid_value_items,
                    factor_lookup=self.factor_lookup,
                    discrete_ot_row=discrete_ot_lookup.get(key),
                    continuous_ot_row=continuous_ot_lookup.get(key),
                    source_discrete_ot_csv=discrete_ot_path,
                    source_continuous_ot_csv=continuous_ot_path,
                    min_ot_state_count=self.min_ot_state_count,
                )
                if ex is None:
                    continue

                attribute_pairwise_ot_examples.append(ex)

            self.base_examples.extend(attribute_base_examples)
            self.pairwise_ot_examples.extend(attribute_pairwise_ot_examples)

            total_base += len(attribute_base_examples)
            total_ot += len(attribute_pairwise_ot_examples)

            if self.verbose:
                n_base_with_cont = sum(
                    1 for x in attribute_base_examples
                    if x.get("continuous_distribution") is not None
                )
                n_ot_with_disc = sum(
                    1 for x in attribute_pairwise_ot_examples
                    if x.get("discrete_ot") is not None
                )
                n_ot_with_cont = sum(
                    1 for x in attribute_pairwise_ot_examples
                    if x.get("continuous_ot") is not None
                )

                print(f"[INFO]   kept base examples      : {len(attribute_base_examples)}")
                print(f"[INFO]   base w/ continuous     : {n_base_with_cont}")
                print(f"[INFO]   kept pairwise OT rows   : {len(attribute_pairwise_ot_examples)}")
                print(f"[INFO]   OT w/ discrete block    : {n_ot_with_disc}")
                print(f"[INFO]   OT w/ continuous block  : {n_ot_with_cont}")

        self.all_examples = list(self.base_examples) + list(self.pairwise_ot_examples)

        if self.verbose:
            print(f"[INFO] Total base examples    : {total_base}")
            print(f"[INFO] Total pairwise OT rows : {total_ot}")
            print(f"[INFO] Total all examples     : {len(self.all_examples)}")

    def __len__(self) -> int:
        return len(self.all_examples)

    def save_base_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        write_jsonl(path, self.base_examples)
        if self.verbose:
            print(f"[INFO] Saved base examples to: {path}")

    def save_pairwise_ot_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        write_jsonl(path, self.pairwise_ot_examples)
        if self.verbose:
            print(f"[INFO] Saved pairwise OT examples to: {path}")

    def save_all_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        write_jsonl(path, self.all_examples)
        if self.verbose:
            print(f"[INFO] Saved all examples to: {path}")


# ============================================================
# Command line
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble base and pairwise context profiles.")
    parser.add_argument("--attributes", nargs="+", required=True)
    parser.add_argument("--distribution_dir", type=Path, default=DISTRIBUTION_DIR)
    parser.add_argument("--distribution_suffix", default=DISTRIBUTION_SUFFIX)
    parser.add_argument("--ot_dir", type=Path, default=OT_DIR)
    parser.add_argument("--ot_suffix", default=OT_SUFFIX)
    parser.add_argument("--continuous_base_dir", type=Path, default=CONTINUOUS_BASE_DIR)
    parser.add_argument("--continuous_base_suffix", default=CONTINUOUS_BASE_SUFFIX)
    parser.add_argument("--continuous_ot_dir", type=Path, default=CONTINUOUS_OT_DIR)
    parser.add_argument("--continuous_ot_suffix", default=CONTINUOUS_OT_SUFFIX)
    parser.add_argument("--factors_path", "--factors_json", type=Path, default=FACTORS_PATH)
    parser.add_argument("--surface_regime_defs_path", "--surface_regime_defs_jsonl", type=Path, default=SURFACE_REGIME_DEFS_PATH)
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--min_ot_state_count", type=int, default=MIN_OT_STATE_COUNT)
    args = parser.parse_args()

    dataset = DistributionalProfileDataset(
        attributes=args.attributes,
        distribution_dir=args.distribution_dir,
        distribution_suffix=args.distribution_suffix,
        ot_dir=args.ot_dir,
        ot_suffix=args.ot_suffix,
        continuous_base_dir=args.continuous_base_dir,
        continuous_base_suffix=args.continuous_base_suffix,
        continuous_ot_dir=args.continuous_ot_dir,
        continuous_ot_suffix=args.continuous_ot_suffix,
        factors_path=args.factors_path,
        surface_regime_defs_path=args.surface_regime_defs_path,
        verbose=not args.quiet,
        min_ot_state_count=args.min_ot_state_count,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_out = args.output_dir / "distributional_profile_base_v2.jsonl"
    ot_out = args.output_dir / "distributional_profile_pairwise_ot_v2.jsonl"
    all_out = args.output_dir / "distributional_profile_all_v2.jsonl"

    dataset.save_base_jsonl(base_out)
    dataset.save_pairwise_ot_jsonl(ot_out)
    dataset.save_all_jsonl(all_out)

    if not args.quiet and len(dataset.base_examples) > 0:
        print("\n[INFO] Example base row:\n")
        print(json.dumps(dataset.base_examples[0], ensure_ascii=False, indent=2))

    if not args.quiet and len(dataset.pairwise_ot_examples) > 0:
        print("\n[INFO] Example pairwise OT row:\n")
        print(json.dumps(dataset.pairwise_ot_examples[0], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
