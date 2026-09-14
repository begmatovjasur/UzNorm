"""Screen a pinned external source BEFORE inspecting any model predictions."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import sys
import unicodedata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))
from uznorm.data import _canonical, _grams, APOSTROPHES
from uznorm.io import read_jsonl, read_json, sha256

OLD = ROOT / 'outputs/uznorm-merged-v2-2026-09-08'
REAL = ROOT / 'outputs/uznorm-real-three-sources-2026-09-08/revisions/007-user-feedback-final-2026-09-09/data/input-output.jsonl'
DATA = HERE / 'data'

def keys(text):
    return {('text', _canonical(text)), *(('8gram', x.hex()) for x in _grams(text))}

def apost(text):
    return unicodedata.normalize('NFC', text).translate(str.maketrans({c: "'" for c in APOSTROPHES}))

def category(source, target):
    a, b = apost(source), apost(target)
    if a == b:
        return 'glyph_only'
    if a.casefold() == b.casefold():
        return 'casing'
    if a.replace("'", '') == b.replace("'", ''):
        return 'apostrophe'
    if _canonical(a) == _canonical(b):
        return 'punctuation_spacing_mixed'
    return 'spelling'

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')

def main():
    source = list(read_jsonl(DATA / 'source/correct_word.jsonl'))
    provenance = read_json(DATA / 'source/PROVENANCE.json')
    if sha256(DATA / 'source/correct_word.jsonl') != provenance['jsonl_sha256']:
        raise RuntimeError('Downloaded source changed')
    owners = defaultdict(set)
    for row in source:
        for option in 'abcd':
            for key in keys(row['option_' + option]):
                owners[key].add(row['id'])
    hits = defaultdict(list)
    old_manifest = read_json(OLD / 'manifest.json')
    if sha256(OLD / 'manifest.json') != 'af37b4590e094e83233d4ecd4364796a7b9b2062a7e4cc2d28f4ccaa968b7dcf':
        raise RuntimeError('Old manifest changed')
    paths = [(OLD / f'data/{s}.jsonl', old_manifest['files'][f'data/{s}.jsonl'])
             for s in ('train', 'validation', 'test')]
    paths.append((REAL, '617d80f667beeeecd31ccd3c10d9e6a80904cfea7206d2a1a931cc32eff03e92'))
    scanned = []
    for path, expected in paths:
        if sha256(path) != expected:
            raise RuntimeError('Known training source changed: ' + str(path))
        n = 0
        for old in read_jsonl(path):
            n += 1
            for field in ('input', 'target' if 'target' in old else 'output'):
                for key in keys(old[field]):
                    for source_id in owners.get(key, ()):
                        if len(hits[source_id]) < 3:
                            hits[source_id].append({'old_file': str(path.relative_to(ROOT)),
                                                    'old_id': old['id'], 'match': key[0]})
        scanned.append({'path': str(path.relative_to(ROOT)), 'sha256': expected, 'rows': n})
        print('OVERLAP_SCANNED', path.name, n, flush=True)
    candidates, rejected, seen = [], [], set()
    for row in source:
        sid = row['id']
        target = row['option_' + row['answer'].lower()]
        reason = ('known_corpus_overlap' if sid in hits else
                  'source_target_family_duplicate' if _canonical(target) in seen else
                  'long_target' if len(target.encode()) > 180 else None)
        if reason:
            rejected.append({'id': sid, 'reason': reason, 'evidence': hits.get(sid, [])})
            continue
        seen.add(_canonical(target))
        options = []
        for opt in 'ABCD':
            if opt == row['answer']:
                continue
            text = row['option_' + opt.lower()]
            cat = category(text, target)
            if cat != 'glyph_only' and len(text.encode()) <= 180:
                options.append({'option': opt, 'input': text, 'category': cat})
        if options:
            candidates.append({'source_id': sid, 'target': target, 'answer': row['answer'], 'options': options})
    write(DATA / 'overlap-report.json', {'source': provenance, 'scanned': scanned,
          'known_rows_checked': sum(x['rows'] for x in scanned), 'source_rows': len(source),
          'method': 'all four source options versus both sides of every known pair; case/punctuation/apostrophe-insensitive full text and shared 8-word sequences',
          'limitations': ['Not a claim of unseen words or absence from ByT5 pretraining.',
                         'Short substrings inside longer training sentences and semantic paraphrases are not excluded.'],
          'rejected': rejected, 'candidate_questions': len(candidates)})
    write(DATA / 'candidates.json', candidates)
    print('CANDIDATES', len(candidates), 'REJECTIONS', Counter(x['reason'] for x in rejected))
    # A readable pool only, not the frozen evaluation set. Model predictions are not used.
    lines = []
    for row in candidates:
        options = ' | '.join(f"{o['option']}:{o['input']} ({o['category']})" for o in row['options'])
        lines.append(f"{row['source_id']} TARGET={row['target']} | {options}")
    (DATA / 'candidates.txt').write_text('\n'.join(lines), encoding='utf-8')

if __name__ == '__main__':
    main()
