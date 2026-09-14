"""Paired input→reference vs model→reference evaluation. Never trains or edits targets."""
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import time
import unicodedata
import uuid

from rapidfuzz.distance import Levenshtein
from .artifacts import ModelError, read_json, require, sha, write_new
from .metrics import APOSTROPHES, PUNCT_LABELS, score

METRICS = [
    ('error_text_pct', 'Xatoli matnlar', 'lower'),
    ('raw_cer_pct', 'CER · barcha belgilar', 'lower'),
    ('content_wer_pct', 'WER · mazmuniy so‘zlar', 'lower'),
    ('word_wer_pct', 'WER · bo‘shliq bo‘yicha so‘zlar', 'lower'),
    ('spelling_cer_pct', 'CER · imlo', 'lower'),
    ('exact_match_pct', 'To‘liq mos javob', 'higher'),
    ('punctuation_macro_f1_pct', 'Tinish belgilari · macro F1', 'higher'),
    ('apostrophe_f1_pct', 'Apostrof va tutuq · F1', 'higher'),
    ('casing_end_to_end_accuracy_pct', 'Registr · end-to-end', 'higher'),
    ('identity_change_rate_pct', 'Toza matnni buzish', 'lower'),
    ('entity_regex_retention_pct', 'Raqam / manzilni saqlash', 'higher')]


def apost(text):
    return unicodedata.normalize('NFC', text).translate(str.maketrans({c: "'" for c in APOSTROPHES}))


def measure(rows, predictions):
    result = score(rows, predictions, breakdown=False)
    edits = sum(Levenshtein.distance(r['target'].split(), p.split()) for r, p in zip(rows, predictions))
    denominator = sum(len(r['target'].split()) for r in rows)
    result['word_wer_pct'] = 100 * edits / denominator if denominator else None
    result['word_errors'] = edits
    result['reference_words'] = denominator
    result['character_errors'] = sum(Levenshtein.distance(r['target'], p) for r, p in zip(rows, predictions))
    result['reference_characters'] = sum(len(r['target']) for r in rows)
    result['error_texts'] = sum(r['target'] != p for r, p in zip(rows, predictions))
    result['error_text_pct'] = 100 * result['error_texts'] / len(rows)
    return result


def comparison(rows, predictions, normalize=False):
    if normalize:
        rows = [{**r, 'input': apost(r['input']), 'target': apost(r['target'])} for r in rows]
        predictions = [apost(p) for p in predictions]
    before, after = measure(rows, [r['input'] for r in rows]), measure(rows, predictions)
    # Use the SAME active punctuation classes in both macro averages.
    labels = [label for label in PUNCT_LABELS if any(m[label + '_reference_support'] + m[label + '_prediction_support'] for m in (before, after))]
    for m in (before, after):
        m['punctuation_macro_f1_pct'] = sum(m[label + '_f1_pct'] or 0 for label in labels) / len(labels) if labels else None
        m['punctuation_active_classes'] = len(labels)
    counters = Counter()
    for row, prediction in zip(rows, predictions):
        old = Levenshtein.distance(row['target'], row['input'])
        new = Levenshtein.distance(row['target'], prediction)
        counters['fully_fixed'] += int(old > 0 and new == 0)
        counters['partly_improved'] += int(0 < new < old)
        counters['worsened'] += int(new > old)
        counters['unchanged_error_distance'] += int(new == old)
        counters['clean_preserved'] += int(old == new == 0)
    table = []
    for key, label, direction in METRICS:
        a, b = before[key], after[key]
        table.append(dict(key=key, label=label, direction=direction, before=a, after=b,
            delta_pp=None if a is None or b is None else b-a,
            error_reduction_pct=100*(a-b)/a if direction == 'lower' and a is not None and a > 0 and b is not None else None))
    return dict(before=before, after=after, table=table, transitions=dict(counters), punctuation_classes=labels)


def load_suites(home):
    folder = Path(home) / 'evaluation-data/v1'
    manifest = read_json(folder / 'MANIFEST.json')
    require(manifest.get('schema') == 1 and manifest['model_run_id'] == 'uq5k-c22f2b158d306bfc'
            and manifest['model_step'] == 157, 'Baholash manifesti mos emas.')
    require(sha(folder / 'AUDIT.json') == manifest['audit_sha256'], 'Baholash auditi o‘zgargan.')
    suites = []
    for item in manifest['suites']:
        require(item['file'] in ('heldout.jsonl', 'uzlib.jsonl'), 'Noma’lum baholash fayli.')
        path = folder / item['file']
        require(sha(path) == item['sha256'], 'Etalon fayli freeze’dan keyin o‘zgargan.')
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
        require(len(rows) == item['n'] and len({r['id'] for r in rows}) == len(rows), 'Baholash qatorlari mos emas.')
        suites.append((item, rows))
    return manifest, suites


def evaluate(service, home, progress=lambda _: None):
    manifest, suites = load_suites(home)
    if not service.ready:
        service.load(progress)
    require(service.metadata['run_id'] == manifest['model_run_id'] and service.metadata['step'] == 157, 'Baholash modeli boshqa.')
    parent = Path(home) / 'evaluations'
    parent.mkdir(parents=True, exist_ok=True)
    folder = parent / ('eval-' + datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    folder.mkdir()
    write_new(folder / 'STARTED.json', dict(dataset_manifest_sha256=sha(Path(home) / 'evaluation-data/v1/MANIFEST.json'),
        model_weights_sha256=service.metadata['files']['model.safetensors'], step=157))
    all_results, suite_reports = [], []
    started = time.perf_counter()
    total, done = sum(len(rows) for _, rows in suites), 0
    with (folder / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for meta, rows in suites:
            results = []
            for row in rows:
                result = service.correct(row['input'])
                entry = dict(row, **{'output': result.output, 'seconds': result.seconds,
                    'ended_with_eos': result.ended_with_eos, 'suite': meta['id']})
                old, new = Levenshtein.distance(apost(row['target']), apost(row['input'])), Levenshtein.distance(apost(row['target']), apost(result.output))
                entry.update(input_character_errors=old, output_character_errors=new,
                    outcome='fixed' if old > 0 and new == 0 else 'improved' if new < old else 'worsened' if new > old else 'preserved' if new == 0 else 'unchanged')
                results.append(entry)
                all_results.append(entry)
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
                done += 1
                elapsed = time.perf_counter()-started
                progress(f'Baholash: {done}/{total} · {elapsed:.0f} s · taxminan {(total-done)*elapsed/done:.0f} s qoldi')
            predictions = [r['output'] for r in results]
            groups = []
            for key, title in [('category', 'Kategoriya'), ('source_name', 'Manba'), ('source_kind', 'Matn turi')]:
                for label in sorted({r.get(key, 'uzlib' if key == 'source_name' else 'publisher_mcq_distractor_adaptation') for r in rows}):
                    indices = [i for i, r in enumerate(rows) if r.get(key, 'uzlib' if key == 'source_name' else 'publisher_mcq_distractor_adaptation') == label]
                    subset, outputs = [rows[i] for i in indices], [predictions[i] for i in indices]
                    groups.append(dict(group=title, label=label, n=len(subset), comparison=comparison(subset, outputs, True)))
            suite_reports.append(dict(meta, strict=comparison(rows, predictions), equivalent=comparison(rows, predictions, True), groups=groups, examples=results))
    report = dict(schema=1, created_utc=datetime.now(timezone.utc).isoformat(), run_id=service.metadata['run_id'], step=157,
        model_weights_sha256=service.metadata['files']['model.safetensors'], dataset_manifest_sha256=sha(Path(home) / 'evaluation-data/v1/MANIFEST.json'),
        backend='CPU FP32 / greedy / no postprocessing', n=total, suites=suite_reports, limitations=manifest['limitations'],
        timing=dict(total_seconds=time.perf_counter()-started, median_seconds=statistics.median(r['seconds'] for r in all_results)),
        missing_eos_n=sum(not r['ended_with_eos'] for r in all_results), model_unchanged=True,
        metric_note='Apostrof-teng ko‘rinish faqat NFC va apostrof glifini tenglashtiradi; harf, so‘z, registr va punktuatsiya o‘zgarmaydi. CER/WER korpus bo‘yicha yig‘ilgan edit-distance; F1 0–100, accuracy emas.')
    write_new(folder / 'REPORT.json', report)
    lines = ['# Quality 5k / 157 — xato kirish → model javobi', '', 'Taqqoslash: INPUT → bir xil TARGET; MODEL OUTPUT → o‘sha TARGET. Trening bajarilmadi.', '', report['metric_note'], '']
    for suite in suite_reports:
        lines += ['## ' + suite['title'], '', f"N={suite['n']} · {suite['label_quality']}", '', '| Ko‘rsatkich | Kirish | Model | Farq (foiz punkt) |', '|---|---:|---:|---:|']
        for metric in suite['equivalent']['table']:
            fmt = lambda x: '—' if x is None else f'{x:.3f}'
            lines.append(f"| {metric['label']} | {fmt(metric['before'])} | {fmt(metric['after'])} | {fmt(metric['delta_pp'])} |")
        lines += ['', 'Apostrof gliflari tenglashtirilgan. Strict natijalar REPORT.json va lokal sayt ichida alohida.', '', str(suite['equivalent']['transitions']), '']
    lines += ['## Chegaralar', '', *('- ' + x for x in manifest['limitations']), '', 'Barcha xom javoblar: predictions.jsonl. Etalonlar va model fayllari o‘zgartirilmadi.']
    with (folder / 'REPORT.md').open('x', encoding='utf-8') as out:
        out.write('\n'.join(lines) + '\n')
    write_new(folder / 'COMPLETE.json', dict(schema=1, n=total, files={name: sha(folder / name) for name in ('REPORT.json', 'REPORT.md', 'predictions.jsonl', 'STARTED.json')}))
    progress('Baholash tugadi. Natijalar saqlandi.')
    return report


def latest_report(home):
    parent = Path(home) / 'evaluations'
    for folder in sorted(parent.glob('eval-*'), reverse=True):
        if (folder / 'COMPLETE.json').is_file():
            seal = read_json(folder / 'COMPLETE.json')
            require(sha(folder / 'REPORT.json') == seal['files']['REPORT.json'], 'Hisobot checksum mos emas.')
            with (folder / 'REPORT.json').open(encoding='utf-8') as stream:
                return json.load(stream)
    return None
