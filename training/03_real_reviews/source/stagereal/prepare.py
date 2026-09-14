"""Frozen, auditable real-review split. No target rewriting and no API calls."""
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
import argparse
import hashlib
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from uznorm.data import _canonical, _grams, SPECIAL, validate_row
from uznorm.io import read_json, read_jsonl, sha256, write_json, write_jsonl

SOURCE = ROOT / 'outputs/uznorm-real-three-sources-2026-09-08/revisions/007-user-feedback-final-2026-09-09'
OLD = ROOT / 'outputs/uznorm-merged-v2-2026-09-08'
SOURCE_SHA = '617d80f667beeeecd31ccd3c10d9e6a80904cfea7206d2a1a931cc32eff03e92'
OLD_MANIFEST_SHA = 'af37b4590e094e83233d4ecd4364796a7b9b2062a7e4cc2d28f4ccaa968b7dcf'
DOMAINS = ('uzum', 'commeta', 'google_play')


def keys(row):
    """Union input AND target keys, including cross-direction overlap."""
    result = set()
    for field in ('input', 'target'):
        text = row[field]
        canonical = _canonical(text)
        if canonical:
            result.add(('text', canonical))
        result.update(('8gram', g.hex()) for g in _grams(text))
    return result


def components(rows):
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    owners = {}
    for i, row in enumerate(rows):
        features = keys(row) | {('declared', row['near_duplicate_group'])}
        for key in features:
            if key in owners:
                a, b = find(i), find(owners[key])
                parent[max(a, b)] = min(a, b)
            else:
                owners[key] = i
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[find(i)].append(row)
    return list(groups.values())


def split_rows(rows):
    result = {name: [] for name in ('train', 'validation', 'test')}
    groups = components(rows)
    for group in groups:
        ids = sorted(r['id'] for r in group)
        fingerprint = hashlib.sha256(('real-v1:' + '\n'.join(ids)).encode()).hexdigest()
        # Stable 90/5/5 group assignment. Very large template families stay in train.
        bucket = int(fingerprint[:8], 16) % 10000
        split = 'train' if len(group) > 100 or bucket < 9000 else ('validation' if bucket < 9500 else 'test')
        for row in group:
            result[split].append({**row, 'group_id': 'real-' + fingerprint[:24], 'split': split})
    for rows_in_split in result.values():
        rows_in_split.sort(key=lambda r: r['id'])
    return result, sorted((len(g) for g in groups), reverse=True)


def check_splits(splits):
    seen, group_split, feature_split = set(), {}, {}
    for split, rows in splits.items():
        for row in rows:
            validate_row(row, split, SimpleNamespace(max_source_tokens=512, max_target_tokens=512))
            if row['id'] in seen:
                raise ValueError('Duplicate ID')
            seen.add(row['id'])
            if group_split.setdefault(row['group_id'], split) != split:
                raise ValueError('Cross-split group')
            for key in keys(row):
                if feature_split.setdefault(key, split) != split:
                    raise ValueError('Cross-split text / shared 8-word sequence')


def prepare(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    source_path = SOURCE / 'data/input-output.jsonl'
    if sha256(source_path) != SOURCE_SHA or sha256(OLD / 'manifest.json') != OLD_MANIFEST_SHA:
        raise ValueError('Original frozen source changed')
    source = list(read_jsonl(source_path))
    if len(source) != 13955:
        raise ValueError('Unexpected source row count')
    manifest = read_json(SOURCE / 'manifest.json')
    full_path = SOURCE / 'data/surface-pairs.jsonl'
    if sha256(full_path) != manifest['files']['data/surface-pairs.jsonl']:
        raise ValueError('Metadata source changed')
    full = {r['id']: r for r in read_jsonl(full_path)}
    reasons = defaultdict(set)
    targets_by_input = defaultdict(set)
    rows = []
    for original in source:
        row = {**original, 'target': original['output'], 'source_kind': 'real_review_silver',
               'source_id': original['id'], 'document_id': original['id'],
               'is_identity': original['input'] == original['output'], 'evaluation_domain': 'real'}
        # Not a hand-verified error taxonomy: "mixed" simply denotes general real correction.
        row['category'] = 'identity' if row['is_identity'] else 'mixed'
        row['category_basis'] = 'coarse_identity_or_general_correction_not_expert_error_label'
        targets_by_input[row['input']].add(row['target'])
        if row['source_name'] not in DOMAINS or row['split'] != 'unassigned':
            raise ValueError('Unexpected source or previous split')
        for field in ('input', 'target'):
            value = row[field]
            if len(value.encode()) + 1 > 512:
                reasons[row['id']].add('over_512_byt5_tokens_no_truncation')
            if not value.strip() or '\x00' in value or SPECIAL.search(value) or not _canonical(value):
                reasons[row['id']].add('unsupported_text')
        if full[row['id']].get('privacy_flags'):
            reasons[row['id']].add('unresolved_privacy_flag')
        rows.append(row)
    for row in rows:
        if len(targets_by_input[row['input']]) != 1:
            reasons[row['id']].add('same_exact_input_conflicting_targets')
    # Conservative: screen against ALL original train/validation/test rows, even though
    # the parent did not finish its old training. No old data or split is changed.
    key_owners = defaultdict(set)
    for row in rows:
        for key in keys(row):
            key_owners[key].add(row['id'])
    old_manifest = read_json(OLD / 'manifest.json')
    old_checked = {}
    for split in ('train', 'validation', 'test'):
        name = f'data/{split}.jsonl'
        path = OLD / name
        if sha256(path) != old_manifest['files'][name]:
            raise ValueError('Old split changed: ' + name)
        n = 0
        for row in read_jsonl(path):
            n += 1
            for key in keys(row):
                for identifier in key_owners.get(key, ()):
                    reasons[identifier].add('old_' + split + '_' + key[0] + '_overlap')
        old_checked[split] = n
        print('OLD_OVERLAP_SCAN', split, n, flush=True)
    eligible = [r for r in rows if not reasons[r['id']]]
    # Even an unusable row can bridge duplicates; quarantine its connected family too.
    for group in components(rows):
        if any(reasons[r['id']] for r in group):
            for row in group:
                if not reasons[row['id']]:
                    reasons[row['id']].add('connected_to_quarantined_family')
    eligible = [r for r in rows if not reasons[r['id']]]
    splits, sizes = split_rows(eligible)
    check_splits(splits)
    if any(len(splits[s]) < 200 for s in ('validation', 'test')) or len(splits['train']) < 10000:
        raise ValueError('Unexpectedly small split; inspect before training')
    for split, split_data in splits.items():
        write_jsonl(output / f'data/{split}.jsonl', split_data)
    held = [{**r, 'exclusion_reasons': sorted(reasons[r['id']])} for r in rows if reasons[r['id']]]
    write_jsonl(output / 'data/preflight-quarantine.jsonl', held)
    # Fixed source-balanced DEVELOPMENT sample, not a Gold test set.
    monitor = []
    for domain in DOMAINS:
        pool = [r for r in splits['validation'] if r['source_name'] == domain]
        pool.sort(key=lambda r: hashlib.sha256(('real-monitor-v1:' + r['id']).encode()).hexdigest())
        if len(pool) < 48:
            raise ValueError('Not enough development examples for ' + domain)
        monitor.extend(pool[:48])
    write_jsonl(output / 'data/real-monitor-144.jsonl', monitor)
    report = {
        'source_rows': len(rows), 'source_sha256': SOURCE_SHA,
        'counts': {s: len(v) for s, v in splits.items()}, 'preflight_quarantine': len(held),
        'counts_by_source': {s: dict(Counter(r['source_name'] for r in v)) for s, v in splits.items()},
        'identity_by_split': {s: sum(r['is_identity'] for r in v) for s, v in splits.items()},
        'quarantine_reasons': dict(Counter(reason for r in held for reason in r['exclusion_reasons'])),
        'group_count': len(sizes), 'largest_groups': sizes[:10],
        'group_policy': '90/5/5 stable hash; families >100 rows in train; transitive canonical/8gram grouping',
        'old_rows_overlap_checked': old_checked, 'old_manifest_sha256': OLD_MANIFEST_SHA,
        'source_text_and_targets_unchanged': True, 'monitor_n': len(monitor),
        'all_targets_gold': False, 'human_semantic_accuracy_measured': False,
        'privacy_check': 'existing regex flags only; not complete PII clearance',
        'test_used_for_training_or_model_selection': False,
        'old_clean_50k_changed': False,
        'source_caveat': 'Uzum publisher normalized text, not guaranteed untouched raw comments',
    }
    write_json(output / 'reports/split.json', report)
    write_json(output / 'manifest.json', {'schema': 1, 'stage': 'real-reviews-v1',
        'files': {p.relative_to(output).as_posix(): sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}})
    assert sha256(source_path) == SOURCE_SHA
    print(report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    prepare(parser.parse_args().output)
