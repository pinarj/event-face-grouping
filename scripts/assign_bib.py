# -*- coding: utf-8 -*-
"""
assign_bib.py — Assign bib numbers to face groups using QR/DataMatrix results.

Logging: stderr only
Output:  stdout — single JSON line on success

Usage:
    python assign_bib.py --groups refined_groups.json --qr qr.json --output output/
"""

import os
import sys
import json
import logging
import argparse
import time
from collections import defaultdict

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assign_bib")


def _emit_json_and_exit(data, code=0):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.write("done\n")
    sys.stdout.flush()
    sys.exit(code)


def load_groups(groups_path):
    with open(groups_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"], data.get("noise", [])


def load_qr(qr_path):
    with open(qr_path, encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "images" in data:
        items = data["images"]
    elif isinstance(data, dict) and "orig_img" in data:
        items = [data]
    else:
        items = data

    bib_lookup = defaultdict(list)
    for item in items:
        orig_img = item.get("orig_img", "")
        filename = os.path.basename(orig_img)
        for box in item.get("boxes", []):
            if box.get("discrimination_result", "Visible") != "Visible":
                continue
            bib = (box.get("datamatrix_result1") or box.get("ocr_result") or "").strip()
            if not bib:
                continue
            confidence = float(box.get("box_confidences", box.get("ocr_confidence", 0)))
            bib_lookup[filename].append((bib, confidence))

    return bib_lookup


def face_id_to_photo_name(face_id):
    base = "_".join(face_id.rsplit("_", 1)[:-1])
    return base + ".jpg"


def assign_bib_to_group(photos, bib_lookup, min_confidence=0.5):
    bib_scores = defaultdict(float)
    bib_best_conf = defaultdict(float)

    for photo in photos:
        lookup_key = photo if photo in bib_lookup else face_id_to_photo_name(photo)
        for bib, conf in bib_lookup.get(lookup_key, []):
            if conf < min_confidence:
                continue
            bib_scores[bib] += conf
            if conf > bib_best_conf[bib]:
                bib_best_conf[bib] = conf

    if not bib_scores:
        return None, 0.0

    best_bib = max(bib_scores, key=lambda b: bib_scores[b])
    return best_bib, round(bib_best_conf[best_bib], 4)


def photo_to_base(photo):
    name = face_id_to_photo_name(photo) if not photo.endswith(".jpg") else photo
    return os.path.basename(name)


def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Assign bib numbers to face groups")
    parser.add_argument("--groups", required=True, help="Path to refined_groups.json")
    parser.add_argument("--qr", required=True, help="Path to qr.json")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Min box_confidence to use a bib detection (default: 0.5)")
    args = parser.parse_args()

    json_dir = os.path.join(args.output, "json")
    os.makedirs(json_dir, exist_ok=True)

    try:
        logger.info("Loading groups...")
        groups, noise = load_groups(args.groups)
        logger.info(f"Groups: {len(groups)}")

        logger.info("Loading QR results...")
        bib_lookup = load_qr(args.qr)
        logger.info(f"Photos with bib detections: {len(bib_lookup)}")

        bib_matches = defaultdict(list)
        assigned = 0
        unassigned = 0

        for group_id, photos in groups.items():
            bib, confidence = assign_bib_to_group(photos, bib_lookup, args.min_confidence)
            if bib:
                assigned += 1
                for photo in photos:
                    base = photo_to_base(photo)
                    bib_matches[bib].append({
                        "base": base,
                        "sim": confidence,
                        "pass_threshold": True
                    })
            else:
                unassigned += 1

        processed_ok = 0
        processed_error = 0
        for bib, matches in bib_matches.items():
            out_path = os.path.join(json_dir, f"{bib}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"bib": bib, "matches": matches}, f, indent=2, ensure_ascii=False)
                processed_ok += 1
            except Exception as e:
                logger.error(f"Failed to write {out_path}: {e}")
                processed_error += 1

        duration = round(time.time() - t0, 3)
        logger.info(f"Done. Bibs: {assigned} | Unassigned: {unassigned} | Duration: {duration}s")

        _emit_json_and_exit({
            "status": "success",
            "bib_count": len(bib_matches),
            "processed_ok": processed_ok,
            "processed_error": processed_error,
            "symlinks_created_total": 0,
            "duration_seconds": duration
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
