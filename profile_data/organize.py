"""Select native aggregate profiles from a local or Hugging Face dataset bundle.

Only remote access needs the optional ``huggingface_hub`` package. Statistics,
partner relations, and support denominators are never recomputed or normalized.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sys
import tempfile
from decimal import Decimal, InvalidOperation


SCHEMA_VERSION = "profile-bundle-v1"
# format, value field, and the support fields that must ALL meet a threshold.
FAMILIES = {
    "base": ("jsonl", "value_item", ("total_count",)),
    "continuous_base": ("jsonl", "value_item", ("n_unique_sentences",)),
    "context": ("csv", "value_item", ("N_a", "N_b")),
    "continuous_context": ("csv", "value_item", ("N_a", "N_b")),
    "relations": ("json", "value_relations", ()),
    "context_relation_ot": ("csv", "anchor_value", ("N_a", "N_b")),
    "context_relation_counts": ("csv", "anchor_value", ("support_N_state",)),
    "context_relation_deltas": ("csv", "anchor_value", ("N_a", "N_b")),
}
ENTRY_FIELDS = {
    "path", "family", "group", "format", "value_field", "support_fields",
    "rows", "bytes", "sha256",
}
SHA256 = re.compile(r"^[0-9a-f]{64}$")
GROUP = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class BundleError(ValueError):
    """The bundle or requested selection cannot be safely processed."""


def digest(path: Path) -> tuple[int, str]:
    checksum = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
            size += len(block)
    return size, checksum.hexdigest()


def relative_path(value: object) -> str:
    if not isinstance(value, str) or value in {"", "."} or "\\" in value or "\x00" in value:
        raise BundleError(f"Invalid relative bundle path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or path.as_posix() != value or any(
        part in {".", ".."} or ":" in part for part in path.parts
    ):
        raise BundleError(f"Unsafe relative bundle path: {value!r}")
    return value


def within_root(root: Path, relative: str) -> Path:
    path = (root / relative_path(relative)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise BundleError(f"Bundle path escapes its root: {relative}") from exc
    if not path.is_file():
        raise BundleError(f"Missing regular bundle file: {relative}")
    return path


def reject_constant(value: str) -> None:
    raise BundleError(f"Non-finite JSON constant: {value}")


def read_manifest(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    manifest_hash = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw, parse_constant=reject_constant)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != SCHEMA_VERSION:
        raise BundleError(f"Expected manifest schema_version={SCHEMA_VERSION!r}")
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise BundleError("Manifest files must be a nonempty list")
    for field in ("groups", "families"):
        if field in manifest and not isinstance(manifest[field], dict):
            raise BundleError(f"Optional manifest {field} metadata must be an object")
    paths = set()
    outputs = set()
    identities = set()
    for entry in entries:
        if not isinstance(entry, dict) or not ENTRY_FIELDS <= entry.keys():
            raise BundleError("Manifest file entries must contain all required fields")
        if entry.keys() - ENTRY_FIELDS - {"source_sha256"}:
            raise BundleError(f"Unexpected file-entry fields: {sorted(entry.keys() - ENTRY_FIELDS - {'source_sha256'})}")
        path = relative_path(entry["path"])
        output = path.removesuffix(".gz")
        if path in paths or output in outputs:
            raise BundleError(f"Duplicate bundle path or output path: {path}")
        paths.add(path)
        outputs.add(output)
        family = entry["family"]
        group = entry["group"]
        if not isinstance(family, str) or not isinstance(entry["format"], str):
            raise BundleError(f"Family and format must be strings: {path}")
        if family == "catalog":
            if group is not None or entry["format"] not in {"json", "jsonl"}:
                raise BundleError(f"Invalid catalog contract: {path}")
            if entry["value_field"] is not None or entry["support_fields"] != []:
                raise BundleError(f"Catalogs cannot define value/support fields: {path}")
            if not path.startswith("resources/") or not output.endswith("." + entry["format"]):
                raise BundleError(f"Catalog path must be resources/... with its declared format: {path}")
        else:
            if family not in FAMILIES or not isinstance(group, str) or not GROUP.fullmatch(group):
                raise BundleError(f"Unknown family or invalid group: {family!r}/{group!r}")
            fmt, value_field, support_fields = FAMILIES[family]
            if (entry["format"], entry["value_field"], entry["support_fields"]) != (
                fmt, value_field, list(support_fields)
            ):
                raise BundleError(f"File does not match the registered {family} contract: {path}")
            expected = f"data/{family}/{group}.{fmt}"
            if output != expected:
                raise BundleError(f"Expected path {expected}[.gz], got {path}")
            identity = (family, group)
            if identity in identities:
                raise BundleError(f"Duplicate family/group entry: {family}/{group}")
            identities.add(identity)
        for name in ("rows", "bytes"):
            if type(entry[name]) is not int or entry[name] < 0:
                raise BundleError(f"Invalid {name} in {path}")
        for name in ("sha256", "source_sha256"):
            if name in entry and (not isinstance(entry[name], str) or not SHA256.fullmatch(entry[name])):
                raise BundleError(f"Invalid {name} in {path}")
    # Prevent a file from being used as another entry's parent directory.
    for path in outputs:
        if any(parent.as_posix() in outputs for parent in PurePosixPath(path).parents):
            raise BundleError(f"Conflicting parent/file paths: {path}")
    return manifest, manifest_hash


class Source:
    def __init__(self, args: argparse.Namespace):
        self.repo_id = args.repo_id
        self.revision = None
        self.download = None
        if args.source_dir is not None:
            if args.revision is not None:
                raise BundleError("--revision is only valid with --repo-id")
            self.root = Path(args.source_dir).resolve()
            if not self.root.is_dir():
                raise BundleError("--source-dir must be an existing bundle directory")
        else:
            try:
                from huggingface_hub import HfApi, hf_hub_download
            except ImportError as exc:
                raise BundleError("Remote bundles require: pip install huggingface_hub") from exc
            self.root = Path(args.cache_dir).resolve()
            info = HfApi().dataset_info(repo_id=args.repo_id, revision=args.revision)
            self.revision = info.sha
            if not isinstance(self.revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40,64}", self.revision):
                raise BundleError("The dataset API did not return a resolved commit SHA")
            self.download = hf_hub_download

    def get(self, relative: str) -> Path:
        relative_path(relative)
        if self.download is None:
            return within_root(self.root, relative)
        result = Path(self.download(
            repo_id=self.repo_id, filename=relative, repo_type="dataset",
            revision=self.revision, cache_dir=str(self.root),
        )).resolve()
        try:
            result.relative_to(self.root)
        except ValueError as exc:
            raise BundleError("Downloaded file resolves outside the requested cache directory") from exc
        if not result.is_file():
            raise BundleError(f"Downloaded bundle file is not a regular file: {relative}")
        return result


def thresholds(args: argparse.Namespace, selected_families: set[str]) -> dict[str, int]:
    result = {}
    for item in args.min_support:
        family, sep, raw = item.partition("=")
        if not sep or family not in FAMILIES or not FAMILIES[family][2]:
            raise BundleError("--min-support requires FAMILY=N for a family with defined support counts")
        if family not in selected_families:
            raise BundleError(f"Support filter refers to an unselected family: {family}")
        if family in result or not re.fullmatch(r"[0-9]+", raw):
            raise BundleError(f"Duplicate or invalid support filter: {item}")
        result[family] = int(raw)
    return result


def select_files(manifest: dict, args: argparse.Namespace) -> tuple[list[dict], dict[str, int]]:
    available = [e for e in manifest["files"] if e["family"] != "catalog"]
    available_families = {e["family"] for e in available}
    families = set(args.family or available_families)
    unknown = families - available_families
    if unknown:
        raise BundleError(f"Requested families are absent from the bundle: {sorted(unknown)}")
    groups = set(args.group or [])
    candidates = [e for e in available if e["family"] in families]
    unknown = groups - {e["group"] for e in candidates}
    if unknown:
        raise BundleError(f"Requested groups are absent from the selected families: {sorted(unknown)}")
    selected = [e for e in candidates if not groups or e["group"] in groups]
    if not selected:
        raise BundleError("No profile files match the selection")
    return selected, thresholds(args, {e["family"] for e in selected})


def open_text(path: Path):
    # Cache blob filenames do not retain the original suffix: detect gzip magic.
    with path.open("rb") as stream:
        compressed = stream.read(2) == b"\x1f\x8b"
    opener = gzip.open if compressed else open
    return opener(path, "rt", encoding="utf-8", newline="")


def check_row(row: object, entry: dict, values: set[str], minimum: int | None) -> tuple[str, bool]:
    if not isinstance(row, dict):
        raise BundleError(f"Expected object row in {entry['path']}")
    name = row.get(entry["value_field"])
    if not isinstance(name, str) or not name:
        raise BundleError(f"Missing string value field in {entry['path']}")
    enough = True
    for field in entry["support_fields"]:
        value = row.get(field)
        if isinstance(value, bool) or value is None:
            raise BundleError(f"Invalid {field} support count in {entry['path']}")
        try:
            count = Decimal(str(value))
        except InvalidOperation as exc:
            raise BundleError(f"Invalid {field} support count in {entry['path']}") from exc
        if not count.is_finite() or count < 0 or count != count.to_integral_value():
            raise BundleError(f"Invalid {field} support count in {entry['path']}")
        if minimum is not None and count < minimum:
            enough = False
    return name, enough and (not values or name in values)


def filter_file(source: Path, target: Path, entry: dict, values: set[str], minimum: int | None) -> tuple[int, set[str]]:
    seen = kept = 0
    matched = set()
    with open_text(source) as stream, target.open("w", encoding="utf-8", newline="") as out:
        if entry["format"] == "jsonl":
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line, parse_constant=reject_constant)
                seen += 1
                name, keep = check_row(row, entry, values, minimum)
                if keep:
                    # Keep the original serialized values as well as the statistics.
                    out.write(line.rstrip("\r\n") + "\n")
                    kept += 1
                    matched.add(name)
        elif entry["format"] == "csv":
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            if not fields or len(fields) != len(set(fields)) or not set(
                [entry["value_field"], *entry["support_fields"]]
            ) <= set(fields):
                raise BundleError(f"Invalid CSV header in {entry['path']}")
            writer = csv.DictWriter(out, fieldnames=fields)
            writer.writeheader()
            for row in reader:
                if None in row or any(v is None for v in row.values()):
                    raise BundleError(f"Malformed CSV row in {entry['path']}")
                seen += 1
                name, keep = check_row(row, entry, values, minimum)
                if keep:
                    writer.writerow(row)
                    kept += 1
                    matched.add(name)
        else:
            document = json.load(stream, parse_constant=reject_constant)
            if not isinstance(document, dict) or not isinstance(document.get("value_relations"), dict):
                raise BundleError(f"Missing value_relations object in {entry['path']}")
            anchors = document["value_relations"]
            if any(not isinstance(name, str) or not name for name in anchors):
                raise BundleError(f"Invalid relation anchor in {entry['path']}")
            seen = len(anchors)
            document["value_relations"] = {
                name: partners for name, partners in anchors.items() if not values or name in values
            }
            matched.update(document["value_relations"])
            kept = len(matched)
            json.dump(document, out, ensure_ascii=False, indent=2, allow_nan=False)
            out.write("\n")
    if seen != entry["rows"]:
        raise BundleError(f"Manifest row count mismatch for {entry['path']}: {entry['rows']} != {seen}")
    return kept, matched


def copy_catalog(source: Path, target: Path, entry: dict) -> None:
    with open_text(source) as stream:
        if entry["format"] == "json":
            json.load(stream, parse_constant=reject_constant)
        else:
            rows = 0
            for line in stream:
                if line.strip():
                    json.loads(line, parse_constant=reject_constant)
                    rows += 1
            if rows != entry["rows"]:
                raise BundleError(f"Catalog row count mismatch for {entry['path']}")
    with open_text(source) as stream, target.open("w", encoding="utf-8", newline="") as out:
        shutil.copyfileobj(stream, out)


def update_coverage(result: dict, original: dict, files: list[dict]) -> None:
    """Keep provenance annotations while replacing derived source-wide totals."""
    if "groups" in original:
        result["groups"] = {
            group: original["groups"].get(group, {})
            for group in sorted({entry["group"] for entry in files if entry["group"] is not None})
        }
    if "families" in original:
        result["families"] = {}
        for family in sorted({entry["family"] for entry in files if entry["family"] != "catalog"}):
            previous = original["families"].get(family, {})
            metadata = dict(previous) if isinstance(previous, dict) else {}
            members = [entry for entry in files if entry["family"] == family]
            metadata.update(
                files=len(members), rows=sum(entry["rows"] for entry in members),
                groups=sorted({entry["group"] for entry in members}),
                support_fields=list(FAMILIES[family][2]),
            )
            result["families"][family] = metadata


def organize(source: Source, manifest: dict, manifest_hash: str, selected: list[dict], minimums: dict[str, int], args: argparse.Namespace) -> None:
    output = Path(args.output_dir).absolute()
    if os.path.lexists(output):
        raise BundleError("--output-dir must not already exist; existing outputs are never overwritten")
    output.parent.mkdir(parents=True, exist_ok=True)
    output = output.parent.resolve() / output.name
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent))
    claimed_output = False
    try:
        values = set(args.value or [])
        matched_values = set()
        matched_groups = set()
        files = []
        catalogs = [e for e in manifest["files"] if e["family"] == "catalog"]
        for entry in selected + catalogs:
            path = source.get(entry["path"])
            size, checksum = digest(path)
            if size != entry["bytes"] or checksum != entry["sha256"]:
                raise BundleError(f"Size or SHA-256 mismatch for {entry['path']}")
            relative = entry["path"].removesuffix(".gz")
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if entry["family"] == "catalog":
                # Preserve catalog content; JSON catalogs have no universal row semantics.
                copy_catalog(path, target, entry)
                rows = entry["rows"]
            else:
                rows, names = filter_file(path, target, entry, values, minimums.get(entry["family"]))
                matched_values.update(names)
                if not rows:
                    target.unlink()
                    continue
                matched_groups.add(entry["group"])
            size, checksum = digest(target)
            result = dict(entry)
            result.update(path=relative, rows=rows, bytes=size, sha256=checksum)
            files.append(result)
        if not matched_values:
            raise BundleError("No profile records remain after value/support filtering")
        if values - matched_values:
            raise BundleError(f"Requested values are absent or have no records meeting support: {sorted(values - matched_values)}")
        if set(args.group or []) - matched_groups:
            raise BundleError("A requested group has no records after value/support filtering")
        result = dict(manifest)
        result["files"] = files
        update_coverage(result, manifest, files)
        result["selection"] = {
            "families": sorted({e["family"] for e in files if e["family"] != "catalog"}),
            "requested_families": sorted(set(args.family or [])),
            "groups": sorted(set(args.group or [])),
            "values": sorted(values), "min_support": minimums,
        }
        result["source_bundle"] = {"manifest_sha256": manifest_hash}
        if "selection" in manifest:
            result["source_bundle"]["selection"] = manifest["selection"]
        if "source_bundle" in manifest:
            result["source_bundle"]["upstream"] = manifest["source_bundle"]
        if source.repo_id is not None:
            result["source_bundle"].update(repo_id=source.repo_id, resolved_commit=source.revision)
        with (stage / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
        # Claim a NEW destination atomically, then replace only our empty directory.
        # This also rejects a concurrently created destination, including symlinks.
        output.mkdir()
        claimed_output = True
        os.replace(stage, output)
        claimed_output = False
        print(f"Organized {sum(e['family'] != 'catalog' for e in files)} profile files into {output}")
    finally:
        if stage.exists():
            shutil.rmtree(stage)
        if claimed_output:
            # Never remove content another process may have placed in the directory.
            try:
                output.rmdir()
            except OSError:
                pass


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    source = ap.add_mutually_exclusive_group(required=True)
    source.add_argument("--source-dir", help="Local directory containing manifest.json")
    source.add_argument("--repo-id", help="Hugging Face dataset repository OWNER/NAME")
    ap.add_argument("--revision", help="Remote revision; resolved once to a commit SHA")
    ap.add_argument("--cache-dir", default="outputs/profile_cache", help="Remote cache (default: outputs/profile_cache)")
    action = ap.add_mutually_exclusive_group(required=True)
    action.add_argument("--list", action="store_true", help="List manifest family/group counts without fetching data files")
    action.add_argument("--output-dir", help="New output directory; must not exist")
    ap.add_argument("--family", action="append", choices=sorted(FAMILIES), help="Repeat to select families; default all available")
    ap.add_argument("--group", action="append", help="Repeat to select exact group IDs")
    ap.add_argument("--value", action="append", help="Repeat to select exact values (relation anchor keys only)")
    ap.add_argument("--min-support", action="append", default=[], metavar="FAMILY=N", help="Repeat per family; all registered support fields must be >=N; unavailable for relations/catalogs")
    return ap


def main(argv: list[str] | None = None) -> int:
    ap = parser()
    args = ap.parse_args(argv)
    try:
        if args.list and (args.value or args.min_support):
            raise BundleError("--list is manifest-only; value/support filters require --output-dir")
        if args.value and any(not value for value in args.value):
            raise BundleError("--value cannot be empty")
        source = Source(args)
        manifest, manifest_hash = read_manifest(source.get("manifest.json"))
        selected, minimums = select_files(manifest, args)
        if args.list:
            print("family\tgroup\trows\tbytes")
            for entry in selected:
                print(f"{entry['family']}\t{entry['group']}\t{entry['rows']}\t{entry['bytes']}")
        else:
            organize(source, manifest, manifest_hash, selected, minimums, args)
        return 0
    except (BundleError, OSError, ValueError, csv.Error, EOFError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
