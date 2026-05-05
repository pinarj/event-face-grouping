# -*- coding: utf-8 -*-
"""
assign_bib.py — Assign bib numbers to face groups using QR or Brian's OCR output.
Optionally runs FAISS search to find additional candidate photos.

Logging : stderr
Output  : stdout — single JSON line

Usage (QR, no FAISS):
    python assign_bib.py \
        --groups refined_groups.json \
        --qr qr.json \
        --output output/

Usage (QR + FAISS — includes low-similarity candidates with pass_threshold:false):
    python assign_bib.py \
        --groups refined_groups.json \
        --qr qr.json \
        --output output/ \
        --faiss-dir faiss/ \
        --embeddings embeddings/

Usage (Brian OCR, no FAISS):
    python assign_bib.py \
        --groups refined_groups.json \
        --ocr-json brian_output.json \
        --output output/

Usage (Brian OCR + FAISS):
    python assign_bib.py \
        --groups refined_groups.json \
        --ocr-json brian_output.json \
        --output output/ \
        --faiss-dir faiss/ \
        --embeddings embeddings/
"""

import os
import sys
import json
import pickle
import logging
import argparse
import time
import numpy as np
from collections import defaultdict

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("assign_bib")


def _emit_json_and_exit(data, code=0):
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.write("done\n")
    sys.stdout.flush()
    sys.exit(code)


# ── groups ────────────────────────────────────────────────────────────────────

def load_groups(groups_path):
    with open(groups_path, encoding="utf-8") as f:
        data = json.load(f)
    return data["groups"], data.get("noise", [])


# ── QR ────────────────────────────────────────────────────────────────────────

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
            xyxy = box.get("xyxy")
            bib_lookup[filename].append((bib, confidence, xyxy))

    return bib_lookup


def face_id_to_photo_name(face_id):
    base = "_".join(face_id.rsplit("_", 1)[:-1])
    return base + ".jpg"


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _bbox_dist(c1, c2):
    return ((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2) ** 0.5


def _load_face_bbox(face_id, embeddings_dir):
    if not embeddings_dir:
        return None
    meta_path = os.path.join(embeddings_dir, f"{face_id}_meta.json")
    if not os.path.isfile(meta_path):
        return None
    try:
        with open(meta_path, encoding="utf-8") as f:
            return json.load(f).get("bbox")
    except Exception:
        return None


def _select_bib_for_face(face_bbox, bibs_in_photo, min_confidence):
    """
    Pick the closest bib to a face within 3x face diagonal.
    Falls back to highest-confidence bib if face bbox is unknown.
    Prevents neighbouring runners from inheriting each other's bib.
    """
    valid = [(bib, conf, xyxy) for bib, conf, xyxy in bibs_in_photo if conf >= min_confidence]
    if not valid:
        return None, 0.0

    if face_bbox is not None:
        face_cx, face_cy = _bbox_center(face_bbox)
        face_w = face_bbox[2] - face_bbox[0]
        face_h = face_bbox[3] - face_bbox[1]
        face_diag = (face_w ** 2 + face_h ** 2) ** 0.5
        max_dist = face_diag * 3.0

        best_bib, best_conf, best_dist = None, 0.0, float("inf")
        for bib, conf, xyxy in valid:
            if xyxy is None:
                continue
            dist = _bbox_dist((face_cx, face_cy), _bbox_center(xyxy))
            if dist < best_dist:
                best_dist, best_bib, best_conf = dist, bib, conf

        if best_bib is not None:
            if best_dist <= max_dist:
                return best_bib, best_conf
            else:
                return None, 0.0

    best_bib, best_conf, _ = max(valid, key=lambda x: x[1])
    return best_bib, best_conf


def assign_bib_to_group(photos, bib_lookup, min_confidence=0.5, embeddings_dir=None):
    """Vote for the most common bib across all photos in the group."""
    bib_scores = defaultdict(float)
    bib_best_conf = defaultdict(float)

    for face_id in photos:
        lookup_key = face_id if face_id in bib_lookup else face_id_to_photo_name(face_id)
        bibs_in_photo = bib_lookup.get(lookup_key, [])
        if not bibs_in_photo:
            continue

        face_bbox = _load_face_bbox(face_id, embeddings_dir)
        bib, conf = _select_bib_for_face(face_bbox, bibs_in_photo, min_confidence)
        if bib is None:
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


# ── FAISS ─────────────────────────────────────────────────────────────────────

def load_faiss_index(faiss_dir):
    try:
        import faiss as _faiss
    except ImportError:
        raise RuntimeError("faiss not installed — cannot run FAISS search")

    index_path = os.path.join(faiss_dir, "index.faiss")
    meta_path  = os.path.join(faiss_dir, "index.pkl")
    if not (os.path.isfile(index_path) and os.path.isfile(meta_path)):
        raise FileNotFoundError(f"FAISS index not found in {faiss_dir}")

    index = _faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    filenames = meta.get("filenames") if isinstance(meta, dict) else meta
    metric    = meta.get("metric", "L2") if isinstance(meta, dict) else "L2"
    return index, filenames, str(metric).upper()


def score_to_similarity(metric, raw):
    if metric in ("IP", "COSINE"):
        return (1.0 + float(raw)) / 2.0
    return max(0.0, 1.0 - float(raw) / 4.0)


def load_group_embedding(face_ids, embeddings_dir):
    embs = []
    for fid in face_ids:
        base = fid if fid.endswith(".npy") else fid + ".npy"
        path = os.path.join(embeddings_dir, base)
        if not os.path.isfile(path):
            alt = os.path.join(embeddings_dir, os.path.splitext(fid)[0] + ".npy")
            if os.path.isfile(alt):
                path = alt
            else:
                continue
        try:
            emb = np.load(path).astype("float32")
            norm = np.linalg.norm(emb)
            if norm > 0:
                embs.append(emb / norm)
        except Exception:
            continue

    if not embs:
        return None

    avg = np.mean(embs, axis=0).astype("float32")
    best_idx = int(np.argmax([np.dot(e, avg) for e in embs]))
    best = embs[best_idx]
    n = np.linalg.norm(best)
    return best / n if n > 0 else None


def faiss_search_for_group(face_ids, embeddings_dir, index, filenames, metric,
                            threshold, candidate_threshold, top_k):
    emb = load_group_embedding(face_ids, embeddings_dir)
    if emb is None:
        return []

    search_k = min(top_k * 3, len(filenames))
    q = emb.reshape(1, -1)
    D, I = index.search(q, search_k)

    best_sim = {}
    best_raw = {}
    best_idx = {}

    for raw_score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(filenames):
            continue
        base = os.path.basename(filenames[idx])
        sim  = score_to_similarity(metric, float(raw_score))

        if sim < candidate_threshold:
            continue

        if base not in best_sim or sim > best_sim[base]:
            best_sim[base] = sim
            best_raw[base] = float(raw_score)
            best_idx[base] = int(idx)

    matches = []
    for base, sim in best_sim.items():
        matches.append({
            "base":           base,
            "sim":            round(float(sim), 6),
            "pass_threshold": bool(sim >= threshold),
            "raw":            float(best_raw[base]),
            "idx":            int(best_idx[base]),
        })

    matches.sort(key=lambda x: x["sim"], reverse=True)
    return matches[:top_k]


# ── Brian OCR ─────────────────────────────────────────────────────────────────

def load_brian_ocr(ocr_path, status_filter=None):
    with open(ocr_path, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "groups" not in data:
        raise ValueError(
            "Not a Brian OCR output — expected top-level 'groups' key. "
            "Use --qr for QR/DataMatrix JSON."
        )

    allowed = set(status_filter) if status_filter else {"confident"}
    group_bib_map = {}

    for group_id, gdata in data["groups"].items():
        if not isinstance(gdata, dict):
            continue
        if gdata.get("status", "") not in allowed:
            continue
        bib = str(gdata.get("best_guess") or "").strip()
        conf = float(gdata.get("confidence", 0.0))
        if bib:
            group_bib_map[group_id] = (bib, conf)

    return group_bib_map


def _is_brian_format(path):
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return False
        groups = data.get("groups")
        if not isinstance(groups, dict):
            return False
        sample = next(iter(groups.values()), None)
        return isinstance(sample, dict) and "best_guess" in sample
    except Exception:
        return False


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Assign bib numbers to face groups")
    parser.add_argument("--groups",    required=True,  help="Path to refined_groups.json")
    parser.add_argument("--qr",        default=None,   help="Path to qr.json (QR/DataMatrix format)")
    parser.add_argument("--ocr-json",  default=None,   help="Path to Brian's raceocr output.json")
    parser.add_argument("--output",    required=True,  help="Output folder")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Min box_confidence to use a bib detection (default: 0.5)")
    parser.add_argument("--embeddings",  default=None,
                        help="Embeddings folder (required with --faiss-dir)")
    parser.add_argument("--faiss-dir",   default=None,
                        help="FAISS index folder (enables candidate photo search)")
    parser.add_argument("--threshold",   type=float, default=0.75,
                        help="Similarity threshold for pass_threshold=true (default: 0.75)")
    parser.add_argument("--candidate-threshold", type=float, default=0.65,
                        help="Lower threshold for candidate photos (default: 0.65)")
    parser.add_argument("--top-k",  type=int, default=200,
                        help="Max FAISS results per bib (default: 200)")
    parser.add_argument("--ocr-status", nargs="+", default=["confident"],
                        help="Brian OCR statuses to accept (default: confident)")
    args = parser.parse_args()

    if not args.qr and not args.ocr_json:
        parser.error("one of --qr or --ocr-json is required")
    if args.qr and args.ocr_json:
        parser.error("--qr and --ocr-json are mutually exclusive")

    if args.qr and _is_brian_format(args.qr):
        logger.warning("Detected Brian's raceocr format in --qr; switching to --ocr-json mode.")
        args.ocr_json = args.qr
        args.qr = None

    json_dir = os.path.join(args.output, "json")
    os.makedirs(json_dir, exist_ok=True)

    faiss_index = faiss_filenames = faiss_metric = None
    use_faiss = bool(args.faiss_dir)
    if use_faiss:
        if not args.embeddings:
            logger.error("--embeddings is required when --faiss-dir is set")
            _emit_json_and_exit({"status": "error", "error": "--embeddings required"}, 1)
        try:
            faiss_index, faiss_filenames, faiss_metric = load_faiss_index(args.faiss_dir)
            logger.info(f"FAISS index loaded: {len(faiss_filenames)} vectors, metric={faiss_metric}")
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            _emit_json_and_exit({"status": "error", "error": str(e)}, 1)

    try:
        logger.info("Loading groups...")
        groups, noise = load_groups(args.groups)
        logger.info(f"Groups: {len(groups)}")

        bib_matches  = defaultdict(list)
        bib_face_ids = defaultdict(list)
        assigned   = 0
        unassigned = 0

        if args.ocr_json:
            logger.info("Loading Brian OCR results...")
            group_bib_map = load_brian_ocr(args.ocr_json, status_filter=args.ocr_status)
            logger.info(f"Groups with confident bib: {len(group_bib_map)}")

            for group_id, photos in groups.items():
                entry = group_bib_map.get(group_id)
                if entry:
                    bib, confidence = entry
                    assigned += 1
                    bib_face_ids[bib].extend(photos)
                    if not use_faiss:
                        seen = {m["base"] for m in bib_matches[bib]}
                        for photo in photos:
                            base = photo_to_base(photo)
                            if base not in seen:
                                bib_matches[bib].append({
                                    "base":           base,
                                    "sim":            confidence,
                                    "pass_threshold": True,
                                })
                                seen.add(base)
                else:
                    unassigned += 1

        else:
            logger.info("Loading QR results...")
            bib_lookup = load_qr(args.qr)
            logger.info(f"Photos with bib detections: {len(bib_lookup)}")

            for group_id, photos in groups.items():
                bib, confidence = assign_bib_to_group(photos, bib_lookup, args.min_confidence,
                                                       embeddings_dir=args.embeddings)
                if bib:
                    assigned += 1
                    bib_face_ids[bib].extend(photos)
                    if not use_faiss:
                        seen = {m["base"] for m in bib_matches[bib]}
                        for photo in photos:
                            base = photo_to_base(photo)
                            if base not in seen:
                                bib_matches[bib].append({
                                    "base":           base,
                                    "sim":            confidence,
                                    "pass_threshold": True,
                                })
                                seen.add(base)
                else:
                    unassigned += 1

        if use_faiss:
            for bib, face_ids in bib_face_ids.items():
                matches = faiss_search_for_group(
                    face_ids,
                    args.embeddings,
                    faiss_index,
                    faiss_filenames,
                    faiss_metric,
                    threshold=args.threshold,
                    candidate_threshold=args.candidate_threshold,
                    top_k=args.top_k,
                )
                bib_matches[bib] = matches

        processed_ok    = 0
        processed_error = 0
        for bib, matches in bib_matches.items():
            out_path = os.path.join(json_dir, f"{bib}.json")
            try:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump({"bib": bib, "matches": matches}, f,
                              indent=2, ensure_ascii=False)
                processed_ok += 1
            except Exception as e:
                logger.error(f"Failed to write {out_path}: {e}")
                processed_error += 1

        duration = round(time.time() - t0, 3)
        logger.info(f"Done. Bibs: {assigned} | Unassigned: {unassigned} | Duration: {duration}s")

        _emit_json_and_exit({
            "status":                "success",
            "bib_count":             len(bib_matches),
            "processed_ok":          processed_ok,
            "processed_error":       processed_error,
            "symlinks_created_total": 0,
            "duration_seconds":      duration,
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
