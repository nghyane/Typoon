"""Aggregate per-chapter audit summaries into one rollup.

Reads every `debug-runs/audit/<chapter_id>/issues.jsonl` and produces
totals + per-language breakdowns + the worst offending bubbles.

Usage:
    python scripts/audit_summary.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path


def main() -> None:
    root = Path("debug-runs") / "audit"
    if not root.exists():
        print("no audit directory — run audit_all.py first")
        return

    all_records: list[dict] = []
    per_chapter: dict[int, dict] = {}

    for chapter_dir in sorted(root.iterdir()):
        if not chapter_dir.is_dir():
            continue
        issues_file = chapter_dir / "issues.jsonl"
        summary_file = chapter_dir / "summary.json"
        if not issues_file.exists():
            continue
        records = [json.loads(l) for l in issues_file.read_text("utf-8").splitlines() if l.strip()]
        all_records.extend([{**r, "chapter_id": int(chapter_dir.name)} for r in records])
        if summary_file.exists():
            per_chapter[int(chapter_dir.name)] = json.loads(summary_file.read_text("utf-8"))

    if not all_records:
        print("no records found")
        return

    # Get source_lang from per_chapter summary (fall back to "unknown")
    lang_by_chapter = {cid: s.get("source_lang", "?") for cid, s in per_chapter.items()}
    label_by_chapter = {cid: s.get("label", "?") for cid, s in per_chapter.items()}

    verdicts = Counter(r["verdict"] for r in all_records)
    issue_types: Counter = Counter()
    severity_by_type: dict[str, list[int]] = defaultdict(list)
    by_lang_type: dict[str, Counter] = defaultdict(Counter)
    by_lang_bubbles: dict[str, int] = defaultdict(int)

    for r in all_records:
        lang = lang_by_chapter.get(r["chapter_id"], "?")
        by_lang_bubbles[lang] += 1
        for it in r.get("issues", []):
            t = it.get("type", "other")
            sev = int(it.get("severity", 1))
            issue_types[t] += 1
            severity_by_type[t].append(sev)
            by_lang_type[lang][t] += 1

    n = len(all_records)
    print("=" * 72)
    print(f"AUDIT ROLLUP — {n} bubbles across {len(per_chapter)} chapters")
    print("=" * 72)

    print("\nVerdicts:")
    for v, count in verdicts.most_common():
        print(f"  {v:8s} {count:5d}  ({count / n * 100:5.1f}%)")

    print("\nIssue types (frequency):")
    for t, count in issue_types.most_common():
        sev = severity_by_type[t]
        avg = sum(sev) / len(sev)
        rate = count / n * 100
        print(f"  {t:30s} {count:5d}  ({rate:5.1f}% of bubbles)  avg_sev={avg:.2f}")

    print("\nBy source language:")
    for lang in sorted(by_lang_type):
        nb = by_lang_bubbles[lang]
        total_issues = sum(by_lang_type[lang].values())
        rate = total_issues / nb * 100
        print(f"\n  [{lang}]  {nb} bubbles, {total_issues} issues ({rate:.1f}% rate)")
        for t, c in by_lang_type[lang].most_common():
            print(f"      {t:28s} {c:4d}  ({c / nb * 100:5.1f}%)")

    print("\nPer-chapter rollup:")
    for cid in sorted(per_chapter):
        s = per_chapter[cid]
        verdicts_str = " ".join(f"{k}={v}" for k, v in s["verdicts"].items())
        print(
            f"  c{cid:>4} [{s.get('source_lang','?'):2s}] "
            f"n={s['n_judged']:>3d}  {verdicts_str}  "
            f"| {s.get('label','?')[:50]}",
        )


if __name__ == "__main__":
    main()
