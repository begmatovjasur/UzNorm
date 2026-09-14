"""Offline evaluation on the sealed, independently sourced correction challenge."""
from collections import Counter
import argparse
import datetime
import html
import json
import os
from pathlib import Path
import statistics
import sys
import time

from engine import Corrector, file_sha, MODEL
from prepare_benchmark import apost

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'src'))
from uznorm.metrics import score, content_tokens

def measured(rows, predictions):
    raw = score(rows, predictions, breakdown=False)
    normal_rows = [{**r, 'input': apost(r['input']), 'target': apost(r['target'])} for r in rows]
    normalized = score(normal_rows, [apost(p) for p in predictions], breakdown=False)
    fields = ('n', 'exact_match_pct', 'raw_cer_pct', 'content_wer_pct', 'spelling_cer_pct',
              'identity_change_rate_pct', 'identity_support', 'punctuation_macro_f1_pct',
              'apostrophe_f1_pct', 'casing_end_to_end_accuracy_pct')
    return {'strict': {k: raw[k] for k in fields},
            'apostrophe_equivalent': {k: normalized[k] for k in fields},
            'content_exact_pct': 100 * sum(content_tokens(r['target']) == content_tokens(p)
                                           for r, p in zip(rows, predictions)) / len(rows)}

def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=HERE / 'results' / 'uzlib-v1-cpu')
    parser.add_argument('--threads', type=int, default=4, choices=range(1, 9))
    args = parser.parse_args()
    data_path = HERE / 'data/uzlib-correction-v1.jsonl'
    frozen = json.loads((HERE / 'data/BENCHMARK.json').read_text(encoding='utf-8'))
    if file_sha(data_path) != frozen['benchmark_sha256']:
        raise RuntimeError('Benchmark changed after freeze')
    rows = [json.loads(line) for line in data_path.read_text(encoding='utf-8').splitlines()]
    if len(rows) != frozen['n'] or len({r['id'] for r in rows}) != len(rows):
        raise RuntimeError('Benchmark inventory mismatch')
    args.output.mkdir(parents=True, exist_ok=False)
    corrector = Corrector(threads=args.threads)
    meta = {'started_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'benchmark_sha256': frozen['benchmark_sha256'],
            'model_manifest_sha256': file_sha(MODEL / 'LOCAL_MODEL.json'),
            'model_weights_sha256': corrector.manifest['files']['model.safetensors'],
            'backend': corrector.backend, 'load_seconds': corrector.load_seconds}
    write_json(args.output / 'EVALUATION.json', meta)
    results = []
    started = time.perf_counter()
    with (args.output / 'predictions.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
        for i, row in enumerate(rows, 1):
            prediction = corrector.correct(row['input'])
            result = {**row, **prediction,
                      'strict_match': prediction['output'] == row['target'],
                      'apostrophe_equivalent_match': apost(prediction['output']) == apost(row['target'])}
            results.append(result)
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())
            elapsed = time.perf_counter() - started
            if i == 1 or i % 8 == 0 or i == len(rows):
                print(f'EVAL {i}/{len(rows)} elapsed={elapsed:.1f}s ETA={elapsed / i * (len(rows)-i):.1f}s', flush=True)
    predictions = [r['output'] for r in results]
    report = {'metadata': meta, 'n': len(rows), 'model': measured(rows, predictions),
              'unchanged_input_baseline': measured(rows, [r['input'] for r in rows]),
              'categories': {}, 'ended_with_eos_n': sum(r['ended_with_eos'] for r in results),
              'timing': {'total_inference_seconds': time.perf_counter() - started,
                         'median_seconds': statistics.median(r['seconds'] for r in results),
                         'max_seconds': max(r['seconds'] for r in results)},
              'limitations': frozen['limitations']}
    for category in sorted({r['category'] for r in rows}):
        selected = [i for i, r in enumerate(rows) if r['category'] == category]
        report['categories'][category] = measured([rows[i] for i in selected], [predictions[i] for i in selected])
    write_json(args.output / 'metrics.json', report)
    raw, norm = report['model']['strict'], report['model']['apostrophe_equivalent']
    lines = ['# Lokal model — tashqi UzLiB sinovi', '',
             f"Model: real-review 392-qadam. {len(rows)} misol, CPU FP32. Trening bajarilmadi.", '',
             '| Ko‘rsatkich | Qiymat |', '|---|---:|',
             f"| Xom javob to‘liq mosligi | {raw['exact_match_pct']:.2f}% |",
             f"| Apostrof shakli tenglashtirilgan to‘liq moslik | {norm['exact_match_pct']:.2f}% |",
             f"| Xom CER (kamroq yaxshi) | {raw['raw_cer_pct']:.2f}% |",
             f"| Mazmuniy token WER (kamroq yaxshi) | {raw['content_wer_pct']:.2f}% |",
             f"| Toza matnni o‘zgartirish (apostrof shakli hisobga olinmaydi) | {norm['identity_change_rate_pct']:.2f}% |",
             f"| Median javob vaqti | {report['timing']['median_seconds']:.2f} soniya |", '',
             '## Kategoriyalar', '', '| Kategoriya | N | To‘liq moslik, apostrof shakli tenglashtirilgan |', '|---|---:|---:|']
    for cat, metrics in report['categories'].items():
        m = metrics['apostrophe_equivalent']
        lines.append(f"| {cat} | {m['n']} | {m['exact_match_pct']:.2f}% |")
    lines += ['', '## Chegaralar', '', *('- ' + x for x in frozen['limitations']), '',
              '## Barcha xom javoblar', '', '| ID | Input | Nashriyot targeti | Model javobi | Mos |', '|---|---|---|---|---|']
    def cell(s):
        return s.replace('|', '\\|').replace('\n', '<br>')
    for r in results:
        lines.append('| ' + ' | '.join([r['id'], cell(r['input']), cell(r['target']), cell(r['output']),
                                       'ha' if r['apostrophe_equivalent_match'] else 'yo‘q']) + ' |')
    (args.output / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    write_json(args.output / 'COMPLETE.json', {'n': len(rows), 'benchmark_sha256': frozen['benchmark_sha256'],
               'files': {n: file_sha(args.output / n) for n in ('EVALUATION.json', 'predictions.jsonl', 'metrics.json', 'REPORT.md')}})
    print('EVALUATION_COMPLETE', args.output, flush=True)
    print(json.dumps({'strict': raw, 'normalized': norm, 'timing': report['timing']}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
