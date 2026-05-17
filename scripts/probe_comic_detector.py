"""Probe — comic-text-and-bubble-detector (ogkalu/...).

RT-DETRv2-small int8 ONNX, 10.6MB, fine-tuned on 11k manga / webtoon /
comic images. Three classes: bubble (0), text_bubble (1), text_free (2).

This probe just exercises the model on the 3 probe fixtures and:
  1. Confirms it loads and runs (CoreML on macOS).
  2. Times one inference per page.
  3. Counts detections per class.
  4. Checks whether the dropped `他们俩` region on probe3 is covered
     by any `text_bubble` / `text_free` detection (the recovery
     hypothesis).
  5. Visualises bboxes overlaid on each page.

Output: debug-runs/lens_bubble_probe{,2,3}/comicdetr.png + summary print.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REPO_ID = "ogkalu/comic-text-and-bubble-detector"
MODEL_FILE = "detector-v4-s_int8.onnx"
INPUT_SIZE = 640
CLASS_NAMES = {0: "bubble", 1: "text_bubble", 2: "text_free"}
CLASS_COLORS = {
    0: (255, 200, 0),    # bubble — yellow
    1: (0, 255, 0),      # text_bubble — green
    2: (0, 180, 255),    # text_free — cyan
}


def _ensure_model() -> Path:
    from huggingface_hub import hf_hub_download
    path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILE)
    return Path(path)


def _build_session(model_path: Path):
    import onnxruntime as ort
    providers = []
    available = ort.get_available_providers()
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")
    providers.append("CPUExecutionProvider")
    sess = ort.InferenceSession(str(model_path), providers=providers)
    return sess


def _preprocess(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize to (INPUT_SIZE, INPUT_SIZE), convert to float32 normalised CHW."""
    h, w = img.shape[:2]
    resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR)
    arr = resized.astype(np.float32) / 255.0
    arr = arr.transpose(2, 0, 1)[None]   # NCHW
    return arr, (w, h)


def _run(sess, img: np.ndarray) -> list[dict]:
    """Returns list of detections: {class, conf, bbox_xyxy_page}."""
    arr, (orig_w, orig_h) = _preprocess(img)
    # RT-DETR inputs: pixel_values + (sometimes) orig_target_sizes
    input_names = [i.name for i in sess.get_inputs()]
    print(f"  inputs: {input_names}")
    print(f"  outputs: {[o.name for o in sess.get_outputs()]}")
    feeds = {input_names[0]: arr}
    if len(input_names) > 1 and "size" in input_names[1].lower():
        feeds[input_names[1]] = np.array([[orig_h, orig_w]], dtype=np.int64)
    elif len(input_names) > 1:
        feeds[input_names[1]] = np.array([[orig_h, orig_w]], dtype=np.int64)
    outputs = sess.run(None, feeds)
    return outputs, (orig_w, orig_h)


def _decode_rtdetr(outputs: list[np.ndarray], page_size: tuple[int, int],
                   conf_threshold: float = 0.3) -> list[dict]:
    """Parse RT-DETRv2 outputs. Output layout depends on export.

    Common formats:
      - 'labels' [N], 'boxes' [N, 4], 'scores' [N]
      - or single [N, 6] = [x1, y1, x2, y2, score, class]
    """
    print(f"  output shapes: {[o.shape for o in outputs]}")
    print(f"  output dtypes: {[o.dtype for o in outputs]}")
    w, h = page_size
    dets: list[dict] = []
    # Try common formats. RT-DETRv2 HF export typically returns
    # [batch, num_queries, 4] boxes (cxcywh normalized) + [batch, num_queries, num_classes] logits.
    if len(outputs) == 2:
        boxes, logits = outputs
        boxes = boxes[0]    # [N, 4]
        logits = logits[0]  # [N, C]
        # logits → sigmoid scores
        scores = 1.0 / (1.0 + np.exp(-logits))
        classes = scores.argmax(axis=1)
        confs = scores.max(axis=1)
        for i in range(len(boxes)):
            if confs[i] < conf_threshold:
                continue
            cx, cy, bw, bh = boxes[i]
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            dets.append({
                "class": int(classes[i]),
                "conf": float(confs[i]),
                "bbox": (max(0, x1), max(0, y1), min(w, x2), min(h, y2)),
            })
    elif len(outputs) == 3:
        # labels, boxes, scores
        labels, boxes, scores = outputs
        for i in range(len(labels[0]) if labels.ndim > 1 else len(labels)):
            sc = float(scores[0][i] if scores.ndim > 1 else scores[i])
            if sc < conf_threshold:
                continue
            b = boxes[0][i] if boxes.ndim > 2 else boxes[i]
            x1, y1, x2, y2 = b
            # Determine if normalized
            if x2 <= 1.0:
                x1 *= w; x2 *= w; y1 *= h; y2 *= h
            dets.append({
                "class": int(labels[0][i] if labels.ndim > 1 else labels[i]),
                "conf": sc,
                "bbox": (max(0, int(x1)), max(0, int(y1)),
                         min(w, int(x2)), min(h, int(y2))),
            })
    return dets


def _draw(img: np.ndarray, dets: list[dict]) -> np.ndarray:
    out = img.copy()
    for d in dets:
        x1, y1, x2, y2 = d["bbox"]
        color = CLASS_COLORS[d["class"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{CLASS_NAMES[d['class']]} {d['conf']:.2f}"
        cv2.putText(
            out, label, (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA,
        )
    return out


def main() -> None:
    print(f"Loading {REPO_ID} / {MODEL_FILE} ...")
    model_path = _ensure_model()
    print(f"  cached at: {model_path}  size={model_path.stat().st_size/1024/1024:.2f} MB")

    sess = _build_session(model_path)

    for name in ["lens_bubble_probe", "lens_bubble_probe2", "lens_bubble_probe3"]:
        src = ROOT / "debug-runs" / name / "source.png"
        if not src.exists():
            print(f"  skip: {src} not found")
            continue
        print(f"\n=== {name} ===")
        img = np.asarray(Image.open(src).convert("RGB"))
        print(f"  page: {img.shape}")
        t0 = time.perf_counter()
        outputs, page_size = _run(sess, img)
        latency = (time.perf_counter() - t0) * 1000
        print(f"  inference: {latency:.1f}ms")
        dets = _decode_rtdetr(outputs, page_size)
        print(f"  detections (conf >= 0.3): {len(dets)}")
        counts = {0: 0, 1: 0, 2: 0}
        for d in dets:
            counts[d["class"]] += 1
        for cls_id, n in counts.items():
            print(f"    {CLASS_NAMES[cls_id]:12s}: {n}")

        # Probe3 specific: check if any text_bubble covers `他们俩` zone
        if "probe3" in name:
            # Target zone (eyeballed from user screenshot): around x ~300-400, y ~150-550
            covered = [
                d for d in dets
                if d["class"] in (1, 2)
                and not (d["bbox"][2] < 300 or d["bbox"][0] > 450
                         or d["bbox"][3] < 150 or d["bbox"][1] > 550)
            ]
            print(f"  zone-overlap candidates for 他们俩 region: {len(covered)}")
            for c in covered:
                print(f"    {CLASS_NAMES[c['class']]} conf={c['conf']:.2f} bbox={c['bbox']}")

        # Save overlay
        overlay = _draw(img, dets)
        out_path = src.parent / "comicdetr.png"
        Image.fromarray(overlay).save(out_path)
        print(f"  overlay → {out_path}")


if __name__ == "__main__":
    main()
