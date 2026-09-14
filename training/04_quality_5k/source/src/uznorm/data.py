"""Frozen JSONL release validation; training text is never cleaned or truncated."""

from collections import Counter
import hashlib
import logging
from pathlib import Path
import re
import unicodedata

from .io import contained, read_json, read_jsonl, sha256

LOG = logging.getLogger(__name__)
SPLITS = ("train", "validation", "test")
CATEGORIES = ("lexical", "legacy_format", "punctuation", "casing", "og_apostrophe", "tutuq", "mixed", "identity")
APOSTROPHES = "'‘’ʻʼ`´ʹ"
SPECIAL = re.compile(r"<(?:extra_id_\d+|pad|/s|unk)>")
MONITOR = "evaluation/validation-monitor.jsonl"


def release_manifest(config, all_files=False):
    root = Path(config.data.root)
    if sha256(root / "manifest.json") != config.data.manifest_sha256:
        raise ValueError("Dataset manifest changed; do not resume with different data")
    manifest = read_json(root / "manifest.json")
    expected_main = [f"data/{split}.jsonl" for split in SPLITS]
    if set(manifest.get("main_data_files", [])) != set(expected_main):
        raise ValueError("Dataset must declare exactly train, validation and test files")
    if not manifest.get("all_targets_gold", False) and not config.data.allow_unverified_targets:
        raise ValueError("Targets are not Gold; explicitly acknowledge allow_unverified_targets")
    files = manifest["files"]
    for name in files:
        contained(root, name)
    for relative in files if all_files else [*expected_main, MONITOR]:
        path = contained(root, relative)
        if not path.is_file() or sha256(path) != files[relative]:
            raise ValueError(f"Dataset checksum mismatch: {relative}")
    return manifest


def validate_row(row, split, model):
    if not isinstance(row, dict):
        raise ValueError("A dataset record must be an object")
    for field in ("id", "input", "target", "group_id", "document_id", "source_id", "source_kind"):
        if not isinstance(row.get(field), str) or not row[field]:
            raise ValueError(f"Record requires nonempty {field}")
    if row.get("split") != split or row.get("category") not in CATEGORIES:
        raise ValueError(f"Invalid split/category for record {row['id']}")
    identity = row["input"] == row["target"]
    if type(row.get("is_identity")) is not bool or row["is_identity"] != identity:
        raise ValueError(f"Incorrect identity flag: {row['id']}")
    if identity != (row["category"] == "identity"):
        raise ValueError(f"Identity/category disagreement: {row['id']}")
    for field, limit in (("input", model.max_source_tokens), ("target", model.max_target_tokens)):
        # ByT5: each UTF-8 byte is a token, plus EOS. Special-token text is excluded.
        text = row[field]
        if not text.strip() or SPECIAL.search(text) or "\x00" in text:
            raise ValueError(f"Unsupported text in {row['id']}:{field}")
        if len(text.encode("utf-8")) + 1 > limit:
            raise ValueError(f"Token limit exceeded: {row['id']}:{field}; truncation is forbidden")


def load_split(config, split):
    if split not in SPLITS:
        raise ValueError("Unknown split")
    rows = list(read_jsonl(Path(config.data.root) / "data" / f"{split}.jsonl"))
    for row in rows:
        validate_row(row, split, config.model)
    return rows


def _canonical(text):
    return "".join(c for c in unicodedata.normalize("NFC", text).casefold()
                   if c.isalnum() and c not in APOSTROPHES)


def _grams(text):
    text = unicodedata.normalize("NFC", text).casefold().translate(str.maketrans("", "", APOSTROPHES))
    words = re.findall(r"[^\W_]+", text)
    return {hashlib.sha256(" ".join(words[i:i + 8]).encode()).digest() for i in range(len(words) - 7)}


def validate_data(config, *, full=True):
    """Read-only audit. No new split assignment, filtering, or target rewriting."""
    manifest = release_manifest(config, all_files=full)
    indexes = {f: {} for f in ("group_id", "source_id", "document_id", "document_url_key", "text", "8gram")}
    seen_ids, pairs, labels = set(), set(), {}
    counts, by_category, lengths = {}, {}, {"input": 0, "target": 0}
    validation_rows = {}
    human_verified = 0
    for split in SPLITS:
        categories = Counter()
        for row in read_jsonl(Path(config.data.root) / "data" / f"{split}.jsonl"):
            validate_row(row, split, config.model)
            if row["id"] in seen_ids:
                raise ValueError("Duplicate record id")
            seen_ids.add(row["id"])
            categories[row["category"]] += 1
            human_verified += int(row.get("human_verified") is True)
            if split == "validation":
                validation_rows[row["id"]] = row
            for field in ("group_id", "source_id", "document_id", "document_url_key"):
                key = row.get(field)
                if key:
                    if indexes[field].setdefault(key, split) != split:
                        raise ValueError(f"Cross-split {field} leakage")
            for field in ("input", "target"):
                text = row[field]
                lengths[field] = max(lengths[field], len(text.encode()) + 1)
                key = hashlib.sha256(_canonical(text).encode()).digest()
                if indexes["text"].setdefault(key, split) != split:
                    raise ValueError("Cross-split normalized text leakage")
                if full:
                    for gram in _grams(text):
                        if indexes["8gram"].setdefault(gram, split) != split:
                            raise ValueError("Cross-split shared 8-word sequence")
            inp = hashlib.sha256(row["input"].encode()).digest()
            target = hashlib.sha256(row["target"].encode()).digest()
            if labels.setdefault(inp, target) != target:
                raise ValueError("Exact input has conflicting targets")
            if (inp, target) in pairs:
                raise ValueError("Duplicate input/target pair")
            pairs.add((inp, target))
            if len(seen_ids) % 25000 == 0:
                LOG.info("Validated %d records", len(seen_ids))
        counts[split] = sum(categories.values())
        by_category[split] = dict(categories)
        if set(categories) != set(CATEGORIES):
            raise ValueError(f"Missing categories in {split}")
    if counts != manifest["split_counts"] or sum(counts.values()) != manifest["dataset_rows"]:
        raise ValueError("Split counts disagree with the pinned manifest")
    aggregate = Counter()
    for categories in by_category.values():
        aggregate.update(categories)
    if dict(aggregate) != manifest["category_counts"]:
        raise ValueError("Category counts disagree with the pinned manifest")
    monitor = list(read_jsonl(Path(config.data.root) / MONITOR))
    if not monitor or len({r["id"] for r in monitor}) != len(monitor):
        raise ValueError("Empty/duplicate monitoring set")
    if any(validation_rows.get(r["id"]) != r for r in monitor):
        raise ValueError("Monitor must contain exact existing validation records only")
    monitor_counts = Counter(r["category"] for r in monitor)
    if set(monitor_counts) != set(CATEGORIES) or len(set(monitor_counts.values())) != 1:
        raise ValueError("Monitor must be balanced across all eight categories")
    return {"status": "passed", "dataset_sha256": config.data.manifest_sha256,
            "split_counts": counts, "category_counts": by_category, "monitor_counts": dict(monitor_counts),
            "max_byt5_tokens_including_eos": lengths, "shared_8word_check": full,
            "all_release_files_hashed": full, "human_verified_rows": human_verified,
            "linguistic_gold_certified": False, "semantic_dedup_exhaustive": False}


class TextDataset:
    """Lazy tokenization bounds memory; keep all original spaces and UTF-8 bytes."""

    def __init__(self, rows, tokenizer, model_config):
        self.rows, self.tokenizer, self.config = rows, tokenizer, model_config

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        item = self.tokenizer(row["input"], truncation=False)
        labels = self.tokenizer(text_target=row["target"], truncation=False)["input_ids"]
        if len(item["input_ids"]) > self.config.max_source_tokens or len(labels) > self.config.max_target_tokens:
            raise ValueError(f"Token limit exceeded: {row['id']}; never silently truncate")
        item["labels"] = labels
        return item
