"""Reference-based reconstruction metrics, percentages; never a semantic accuracy claim."""

from collections import Counter
import re
import unicodedata

from rapidfuzz.distance import Levenshtein

from .data import APOSTROPHES, CATEGORIES

PUNCT = {".": "period", ",": "comma", "?": "question", "!": "exclamation", ":": "colon",
         ";": "semicolon", "-": "dash", "—": "dash", "–": "dash", "(": "open_paren",
         ")": "close_paren", "/": "slash", '"': "quote", "«": "quote", "»": "quote",
         "“": "quote", "”": "quote", "…": "ellipsis"}
PUNCT_LABELS = tuple(sorted(set(PUNCT.values())))
AP_LABELS = ("apostrophe_og", "apostrophe_tutuq")
ENTITY = re.compile(r"https?://[^\s]+|www\.[^\s]+|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|(?<!\w)\d+(?:[.,:/-]\d+)*(?!\w)")
NAME = re.compile(r"\b[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ]+(?:['‘’ʻʼ][A-Za-z]+)*\b")


def percent(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else None


def content_tokens(text):
    canonical = unicodedata.normalize("NFC", text).casefold().translate(
        str.maketrans({c: "'" for c in APOSTROPHES}))
    return re.findall(r"[^\W_]+(?:'[^\W_]+)*(?:(?<=[og])')?", canonical)


def scaffold(text):
    text = unicodedata.normalize("NFC", text)
    base, original, events = [], [], Counter()
    for i, char in enumerate(text):
        # Modifier apostrophes are Unicode letters! Check them before isalnum().
        if char in APOSTROPHES:
            left = text[i - 1] if i else ""
            right = text[i + 1] if i + 1 < len(text) else ""
            word_ap = left.isalpha() and (right.isalpha() or left.casefold() in ("o", "g"))
            label = ("apostrophe_og" if left.casefold() in ("o", "g") else "apostrophe_tutuq") if word_ap else "quote"
            events[(len(base), label)] += 1
        elif char.isalnum():
            base.extend(char.casefold())
            original.extend([char] * len(char.casefold()))
        elif char in PUNCT:
            events[(len(base), PUNCT[char])] += 1
    return "".join(base), original, events


def aligned_counts(reference, prediction):
    rb, ro, revents = scaffold(reference)
    pb, po, pevents = scaffold(prediction)
    matches = {}
    for op in Levenshtein.opcodes(rb, pb):
        if op.tag == "equal":
            matches.update((op.src_start + i, op.dest_start + i) for i in range(op.src_end - op.src_start))
    gaps = {}
    if not rb and not pb or matches.get(0) == 0:
        gaps[0] = 0
    if rb and matches.get(len(rb) - 1) == len(pb) - 1:
        gaps[len(rb)] = len(pb)
    for i, j in matches.items():
        if matches.get(i + 1) == j + 1:
            gaps[i + 1] = j + 1
    mapped = Counter()
    for (position, label), count in revents.items():
        if position in gaps:
            mapped[(gaps[position], label)] += count
    hits = mapped & pevents
    counts = {label: [sum(n for (_, label2), n in source.items() if label2 == label)
                      for source in (hits, pevents, revents)] for label in (*PUNCT_LABELS, *AP_LABELS)}
    case = (sum(ro[i].isalpha() and ro[i] == po[j] for i, j in matches.items()),
            sum(ro[i].isalpha() for i in matches), sum(c.isalpha() for c in ro))
    return counts, case


def score(rows, predictions, *, breakdown=True):
    if not rows or len(rows) != len(predictions) or any(not isinstance(p, str) for p in predictions):
        raise ValueError("Metrics require equally sized nonempty row/prediction lists")
    total = Counter()
    events = {label: [0, 0, 0] for label in (*PUNCT_LABELS, *AP_LABELS)}
    for row, prediction in zip(rows, predictions):
        target, source = row["target"], row["input"]
        rt, pt = content_tokens(target), content_tokens(prediction)
        rc, pc = "".join(rt), "".join(pt)
        total.update({"exact": int(target == prediction), "empty": int(not prediction),
                      "raw_err": Levenshtein.distance(target, prediction), "raw_den": len(target),
                      "word_err": Levenshtein.distance(rt, pt), "word_den": len(rt),
                      "spell_err": Levenshtein.distance(rc, pc), "spell_den": len(rc)})
        counts, case = aligned_counts(target, prediction)
        for label, values in counts.items():
            events[label] = [a + b for a, b in zip(events[label], values)]
        total.update(dict(zip(("case_ok", "case_aligned", "case_den"), case)))
        if source == target:
            total["identity_support"] += 1
            total["identity_changed"] += int(prediction != target)
        for name, pattern in (("entity_regex", ENTITY), ("capitalized_name_proxy", NAME)):
            before, after = Counter(pattern.findall(source)), Counter(pattern.findall(prediction))
            total[name + "_support"] += sum(before.values())
            total[name + "_kept"] += sum((before & after).values())
    result = {"n": len(rows), "exact_match_pct": percent(total["exact"], len(rows)),
              "empty_prediction_pct": percent(total["empty"], len(rows)),
              "raw_cer_pct": percent(total["raw_err"], total["raw_den"]),
              "content_wer_pct": percent(total["word_err"], total["word_den"]),
              "spelling_cer_pct": percent(total["spell_err"], total["spell_den"]),
              "casing_end_to_end_accuracy_pct": percent(total["case_ok"], total["case_den"]),
              "casing_alignment_coverage_pct": percent(total["case_aligned"], total["case_den"]),
              "identity_change_rate_pct": percent(total["identity_changed"], total["identity_support"]),
              "identity_support": total["identity_support"]}
    active = []
    for label, (tp, pred, ref) in events.items():
        result[f"{label}_f1_pct"] = percent(2 * tp, pred + ref)
        result[f"{label}_reference_support"] = ref
        result[f"{label}_prediction_support"] = pred
        if label in PUNCT_LABELS and pred + ref:
            active.append(percent(2 * tp, pred + ref))
    result["punctuation_macro_f1_pct"] = sum(active) / len(active) if active else None
    result["punctuation_active_classes"] = len(active)
    ap = [sum(events[label][i] for label in AP_LABELS) for i in range(3)]
    result["apostrophe_f1_pct"] = percent(2 * ap[0], ap[1] + ap[2])
    for name in ("entity_regex", "capitalized_name_proxy"):
        result[name + "_retention_pct"] = percent(total[name + "_kept"], total[name + "_support"])
        result[name + "_support"] = total[name + "_support"]
    if breakdown:
        category_cers = []
        for category in CATEGORIES:
            indices = [i for i, row in enumerate(rows) if row["category"] == category]
            if not indices:
                continue
            measured = score([rows[i] for i in indices], [predictions[i] for i in indices], breakdown=False)
            for key in ("n", "raw_cer_pct", "content_wer_pct", "spelling_cer_pct", "exact_match_pct"):
                result[f"category_{category}_{key}"] = measured[key]
            category_cers.append(measured["raw_cer_pct"])
        result["category_macro_cer_pct"] = sum(category_cers) / len(category_cers) if category_cers else None
        result["category_support"] = len(category_cers)
    return result


def numeric_metrics(metrics):
    return {key: value for key, value in metrics.items() if type(value) in (int, float)}


def make_compute_metrics(tokenizer, rows):
    import numpy as np

    def compute(output):
        ids = output.predictions[0] if isinstance(output.predictions, tuple) else output.predictions
        ids = np.where(ids < 0, tokenizer.pad_token_id, ids)
        predictions = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        result = numeric_metrics(score(rows, predictions))
        result["generation_missing_eos_pct"] = percent(
            sum(tokenizer.eos_token_id not in sequence for sequence in ids), len(ids))
        return result
    return compute
