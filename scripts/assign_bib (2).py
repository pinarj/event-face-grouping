# -*- coding: utf-8 -*-
"""
assign_bib_production.py — Assign bib numbers to face groups using QR/DataMatrix results.
Optionally runs FAISS search to include candidate photos (pass_threshold: false).

Logging: stderr only
Output:  stdout — single JSON line on success

Usage (QR + spatial check — recommended):
    python assign_bib_production.py --groups refined_groups.json --qr qr.json \
        --output output/ --embeddings embeddings/

    --embeddings is required for spatial face↔bib matching.
    Without it, all faces in a multi-face photo receive the same bib vote
    regardless of position, which can cause wrong bib assignments.

Usage (QR + spatial check + FAISS candidates):
    python assign_bib_production.py --groups refined_groups.json --qr qr.json \
        --output output/ --embeddings embeddings/ --faiss-dir faiss/
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
            xyxy = box.get("xyxy")  # bib bbox — used for spatial face↔bib matching
            bib_lookup[filename].append((bib, confidence, xyxy))

    return bib_lookup


def face_id_to_photo_name(face_id):
    base = "_".join(face_id.rsplit("_", 1)[:-1])
    return base + ".jpg"


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0)


def _load_face_bbox(face_id, embeddings_dir):
    """Return face [x1,y1,x2,y2] from _meta.json, or None."""
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


def _select_bib_for_face(face_bbox, bibs, min_confidence):
    """
    Pick the best bib for a single face using spatial proximity.

    For every photo (single or multiple bibs):
    - If face bbox is known: find the closest bib whose center is within
      3× the face diagonal. If closest bib is still too far, return None
      (this face is not the owner of any detected bib in this photo).
    - If face bbox is unknown: fall back to highest-confidence bib.

    This prevents runners standing next to each other from inheriting
    a neighbour's bib number even when only one bib is visible.
    """
    valid = [(bib, conf, xyxy) for bib, conf, xyxy in bibs if conf >= min_confidence]
    if not valid:
        return None, 0.0

    if face_bbox is not None:
        fx, fy = _bbox_center(face_bbox)
        face_w = face_bbox[2] - face_bbox[0]
        face_h = face_bbox[3] - face_bbox[1]
        face_diag = (face_w ** 2 + face_h ** 2) ** 0.5
        max_dist = face_diag * 3.0          # bib must be within 3× face diagonal

        best_bib, best_conf, best_dist = None, 0.0, float("inf")
        for bib, conf, xyxy in valid:
            if xyxy is None:
                continue
            bx, by = _bbox_center(xyxy)
            dist = ((fx - bx) ** 2 + (fy - by) ** 2) ** 0.5
            if dist < best_dist:
                best_dist, best_bib, best_conf = dist, bib, conf

        if best_bib is not None:
            # spatial data available — accept only if close enough
            if best_dist <= max_dist:
                return best_bib, best_conf
            else:
                return None, 0.0    # all bibs too far → not this face's bib

    # no face bbox (or all bibs lack xyxy) → fallback: highest confidence
    b, c, _ = max(valid, key=lambda x: x[1])
    return b, c


def assign_bib_to_group(photos, bib_lookup, min_confidence=0.5, embeddings_dir=None):
    """
    Vote for the most common bib across all photos in the group.

    Uses spatial proximity (face bbox vs bib bbox) for every photo,
    regardless of how many bibs are detected in that photo. This ensures
    that a runner standing next to the bib owner does not receive the
    wrong bib number even when only one bib is visible in the photo.
    """
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
    # L2 squared distance
    return max(0.0, 1.0 - float(raw) / 4.0)


def load_group_embedding(face_ids, embeddings_dir):
    """Average embedding of all faces in the group."""
    embs = []
    for fid in face_ids:
        # face_id may already include extension or not
        base = fid if fid.endswith(".npy") else fid + ".npy"
        path = os.path.join(embeddings_dir, base)
        if not os.path.isfile(path):
            # try without extension suffix
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
    n = np.linalg.norm(avg)
    return avg / n if n > 0 else None


def faiss_search_for_group(face_ids, embeddings_dir, index, filenames, metric,
                            threshold, candidate_threshold, top_k):
    """
    Search FAISS for a group.
    Returns list of match dicts with pass_threshold True/False.
    """
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


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    parser = argparse.ArgumentParser(description="Assign bib numbers to face groups")
    parser.add_argument("--groups",    required=True,  help="Path to refined_groups.json")
    parser.add_argument("--qr",        required=True,  help="Path to qr.json")
    parser.add_argument("--output",    required=True,  help="Output folder")
    parser.add_argument("--min-confidence", type=float, default=0.5,
                        help="Min box_confidence to use a bib detection (default: 0.5)")
    # embeddings folder — used for spatial bib matching AND FAISS search
    parser.add_argument("--embeddings",  default=None,
                        help="Embeddings folder with .npy/_meta.json files "
                             "(enables spatial face-bib matching; required with --faiss-dir)")
    # FAISS (optional)
    parser.add_argument("--faiss-dir",   default=None,
                        help="FAISS index folder (enables candidate photo search)")
    parser.add_argument("--threshold",   type=float, default=0.75,
                        help="Similarity threshold for pass_threshold=true (default: 0.75)")
    parser.add_argument("--candidate-threshold", type=float, default=0.65,
                        help="Lower threshold for candidate photos (default: 0.65)")
    parser.add_argument("--top-k",  type=int, default=200,
                        help="Max FAISS results per bib (default: 200)")
    args = parser.parse_args()

    json_dir = os.path.join(args.output, "json")
    os.makedirs(json_dir, exist_ok=True)

    # load FAISS if requested
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

        logger.info("Loading QR results...")
        bib_lookup = load_qr(args.qr)
        logger.info(f"Photos with bib detections: {len(bib_lookup)}")

        bib_matches  = defaultdict(list)
        bib_face_ids = defaultdict(list)
        bib_seen     = defaultdict(set)   # dedup per bib across all groups
        assigned   = 0
        unassigned = 0

        for group_id, photos in groups.items():
            bib, confidence = assign_bib_to_group(photos, bib_lookup, args.min_confidence,
                                                   embeddings_dir=args.embeddings)
            if bib:
                assigned += 1
                bib_face_ids[bib].extend(photos)
                if not use_faiss:
                    for photo in photos:
                        base = photo_to_base(photo)
                        if base in bib_seen[bib]:
                            continue
                        bib_seen[bib].add(base)
                        bib_matches[bib].append({
                            "base":           base,
                            "sim":            confidence,
                            "pass_threshold": True,
                        })
            else:
                unassigned += 1

        # FAISS search — build matches with pass_threshold True/False
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
