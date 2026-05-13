"""Probe idea 9 — OCR text clustering for speaker grouping.

Hypothesis: each character has a distinct "voice" (vocabulary, length,
register) that an embedding model can capture. If we embed all bubble OCR
text and cluster them, clusters may align with speakers.

If this works, material-level scaling becomes O(clusters) anchor calls
instead of O(bubbles) per chapter:
  - Embed all bubbles in N chapters → ~K clusters total.
  - Anchor 1 bubble/cluster with a vision call (or use combined probe).
  - All other bubbles in that cluster inherit the speaker → free.

Uses Chainsaw Man ch.1 page 5-13 (same 9 pages, 37 bubbles, combined-probe
ground truth available).

Output: debug-runs/storyboard_proto/probe_clustering.md
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from speaker_probe_3x3 import WebpPreparedReader, OUT

from typoon.adapters.vision_runtime import VisionRuntime
from typoon.config import load_config
from typoon.stages.keys import assign_keys
from typoon.stages.scan import scan_chapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[1]
CHAP = ROOT / "cache" / "probe_chapter"
SLICE = list(sorted(CHAP.glob("*.png")))[5:14]


def main() -> None:
    config, paths = load_config()
    runtime = VisionRuntime.from_config(config, paths, source_lang="en")[0]
    reader = WebpPreparedReader(SLICE)
    prepared = reader.chapter("chainsaw")

    log.info("scanning…")
    t0 = time.monotonic()
    out = scan_chapter(prepared, reader, runtime, source_lang="en")
    scan_t = time.monotonic() - t0
    log.info("scan: %.1fs, %d bubbles", scan_t, len(out.chapter.all_bubbles))

    keyed = assign_keys(out.chapter.all_bubbles, chapter_id=1)
    flat = []
    for bk in keyed:
        b = bk.bubble
        flat.append({
            "key": bk.key, "page": b.page_index, "idx": b.idx,
            "text": b.source_text, "shape_kind": b.shape_kind,
        })

    from probe_deterministic import COMBINED_GT
    # Use combined-probe results as labels for ARI/NMI.
    labels_true: list[int] = []
    label_map: dict[str, int] = {}
    for b in flat:
        sp = COMBINED_GT.get(b["key"], "unknown")
        if sp not in label_map:
            label_map[sp] = len(label_map)
        labels_true.append(label_map[sp])

    log.info("loading embedding model (all-MiniLM-L6-v2, ~22MB)…")
    t0 = time.monotonic()
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    log.info("model loaded: %.1fs", time.monotonic() - t0)

    texts = [b["text"] for b in flat]
    t0 = time.monotonic()
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embed_t = time.monotonic() - t0
    log.info("embed %d texts: %.2fs (%.1f ms/text), shape=%s",
             len(texts), embed_t, embed_t * 1000 / len(texts), emb.shape)

    # GT has 4 distinct labels: Denji, Pochita, unknown, sfx. Try k=2..6.
    print("\n--- clustering quality vs combined-probe labels ---")
    print(f"{'algo':<25s} {'k':>4s} {'ARI':>8s} {'NMI':>8s} {'note'}")
    results = []
    for k in (2, 3, 4, 5, 6):
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(emb)
        ari = adjusted_rand_score(labels_true, km.labels_)
        nmi = normalized_mutual_info_score(labels_true, km.labels_)
        results.append(("kmeans", k, ari, nmi, km.labels_))
        print(f"{'KMeans':<25s} {k:>4d} {ari:>8.3f} {nmi:>8.3f}")

    for k in (2, 3, 4, 5, 6):
        ac = AgglomerativeClustering(n_clusters=k, linkage="average", metric="cosine").fit(emb)
        ari = adjusted_rand_score(labels_true, ac.labels_)
        nmi = normalized_mutual_info_score(labels_true, ac.labels_)
        results.append(("agg-avg-cos", k, ari, nmi, ac.labels_))
        print(f"{'Agglomerative(cos,avg)':<25s} {k:>4d} {ari:>8.3f} {nmi:>8.3f}")

    # Pick best result, show clusters with text.
    best = max(results, key=lambda r: r[2])  # by ARI
    algo, k, ari, nmi, labels = best
    print(f"\n=== Best: {algo} k={k} (ARI={ari:.3f}, NMI={nmi:.3f}) ===")
    cluster_texts: dict[int, list[tuple[str, str, str]]] = {}
    for b, lab in zip(flat, labels):
        cluster_texts.setdefault(int(lab), []).append(
            (b["key"], COMBINED_GT.get(b["key"], "—"), b["text"])
        )

    report_lines = [
        "# Clustering probe (idea 9)",
        "",
        f"- bubbles: {len(flat)}",
        f"- embedding: all-MiniLM-L6-v2 (~22MB)",
        f"- embed time: {embed_t*1000/len(texts):.1f} ms/text",
        f"- scan: {scan_t:.1f}s",
        f"- best result: {algo} k={k} → ARI={ari:.3f}, NMI={nmi:.3f}",
        "",
        "## Score table",
        "",
        f"| algo | k | ARI | NMI |",
        f"|---|---|---|---|",
    ]
    for r in results:
        report_lines.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} |")
    report_lines.append("")
    report_lines.append("ARI=1.0 means perfect agreement; ARI=0 means random.")
    report_lines.append("NMI≥0.6 typically considered useful for downstream tasks.")
    report_lines.append("")
    report_lines.append(f"## Best clustering ({algo} k={k})")
    report_lines.append("")
    for cid in sorted(cluster_texts.keys()):
        gt_counts: dict[str, int] = {}
        for _, gt, _ in cluster_texts[cid]:
            gt_counts[gt] = gt_counts.get(gt, 0) + 1
        purity = max(gt_counts.values()) / sum(gt_counts.values())
        dominant = max(gt_counts.items(), key=lambda kv: kv[1])[0]
        report_lines.append(
            f"### Cluster {cid} — {len(cluster_texts[cid])} bubbles "
            f"(purity {purity*100:.0f}%, dominant={dominant})"
        )
        report_lines.append("")
        for key, gt, txt in cluster_texts[cid]:
            t = txt.replace("\n", " ")[:60]
            report_lines.append(f"- `{key}` (gt={gt}) {t!r}")
        report_lines.append("")

    out_path = OUT / "probe_clustering.md"
    out_path.write_text("\n".join(report_lines), "utf-8")
    log.info("wrote %s", out_path)

    print("\n=== SUMMARY ===")
    print(f"best ARI {ari:.3f}, NMI {nmi:.3f} ({algo} k={k})")
    print(f"embedding cost: {embed_t*1000/len(texts):.1f} ms/bubble (CPU)")


if __name__ == "__main__":
    main()
