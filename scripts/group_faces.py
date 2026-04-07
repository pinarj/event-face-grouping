# -*- coding: utf-8 -*-
#
# Face clustering — DBSCAN on ArcFace embeddings
# stdout: final JSON only
# stderr: progress and error logs

import os
import sys
import json
import glob
import logging
import argparse
import numpy as np
from datetime import datetime
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import normalize

logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("group_faces")


def _emit_json_and_exit(data, code=0):
    # write JSON result to stdout so the caller (PHP) can parse it cleanly
    sys.stdout.write(json.dumps(data, ensure_ascii=False) + "\n")
    sys.stdout.flush()
    sys.exit(code)


def load_embeddings(embeddings_dir):
    embeddings = []
    face_keys = []
    photo_names = set()

    # each face is stored as a .npy file + a _meta.json with the original filename
    npy_files = sorted(glob.glob(os.path.join(embeddings_dir, "*.npy")))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {embeddings_dir}")

    for npy_path in npy_files:
        meta_path = npy_path.replace(".npy", "_meta.json")
        # skip embeddings that have no metadata (incomplete pipeline run)
        if not os.path.exists(meta_path):
            continue
        with open(meta_path, "r") as f:
            meta = json.load(f)
        embedding = np.load(npy_path).astype(np.float32)
        embeddings.append(embedding)
        # face_key identifies the face: e.g. "DSC001_0" = first face from DSC001.jpg
        face_key = os.path.basename(npy_path).replace(".npy", "")
        face_keys.append(face_key)
        photo_names.add(meta["original_filename"])

    logger.info(f"Loaded {len(embeddings)} embeddings from {len(photo_names)} unique photos.")
    return np.array(embeddings), face_keys


def cluster_embeddings(embeddings, eps=0.7, min_samples=2):
    # normalize to unit vectors before clustering — ArcFace embeddings are cosine-space,
    # but DBSCAN uses euclidean distance; L2-normalizing first makes euclidean ≈ cosine
    normed = normalize(embeddings, norm="l2")

    # DBSCAN is used instead of K-means because:
    # - the number of people is unknown in advance
    # - it can mark ambiguous/outlier faces as noise (label=-1) instead of forcing a match
    # - eps controls how similar two embeddings must be to belong to the same cluster
    db = DBSCAN(eps=eps, min_samples=min_samples, metric="euclidean", n_jobs=-1)
    labels = db.fit_predict(normed)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    logger.info(f"Clusters: {n_clusters} | Noise: {n_noise}")
    return labels


def build_groups(labels, face_keys):
    groups = {}
    noise = []

    for label, face_key in zip(labels, face_keys):
        if label == -1:
            # noise faces: detected but not similar enough to form a group
            noise.append(face_key)
        else:
            key = f"group_{label + 1}"
            if key not in groups:
                groups[key] = []
            groups[key].append(face_key)

    # sort groups by size descending — largest groups (most-photographed people) come first
    groups = dict(sorted(groups.items(), key=lambda x: len(x[1]), reverse=True))
    return groups, noise


def main():
    t0 = datetime.now()
    parser = argparse.ArgumentParser(description="Group faces by identity using DBSCAN")
    parser.add_argument("--embeddings", required=True, help="Path to embeddings folder")
    parser.add_argument("--output", required=True, help="Path to output folder")
    # eps: distance threshold — lower = stricter grouping, higher = more merging
    # 0.7 works well for ArcFace buffalo_l embeddings in practice
    parser.add_argument("--eps", type=float, default=0.7, help="DBSCAN eps (default: 0.7)")
    # min_samples=2: a person must appear in at least 2 photos to form a group
    parser.add_argument("--min-samples", type=int, default=2, help="DBSCAN min_samples (default: 2)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    try:
        embeddings, face_keys = load_embeddings(args.embeddings)
        labels = cluster_embeddings(embeddings, eps=args.eps, min_samples=args.min_samples)
        groups, noise = build_groups(labels, face_keys)

        # write groups.json — maps each group label to its list of face_keys
        output = {
            "groups": groups,
            "noise": sorted(list(set(noise)))
        }
        out_path = os.path.join(args.output, "groups.json")
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        duration = round((datetime.now() - t0).total_seconds(), 3)
        logger.info(f"Done. Groups: {len(groups)} | Noise: {len(set(noise))} | Duration: {duration}s")

        _emit_json_and_exit({
            "status": "success",
            "groups": len(groups),
            "noise": len(set(noise)),
            "largest_group": len(list(groups.values())[0]) if groups else 0,
            "output": out_path,
            "duration_seconds": duration
        })

    except Exception as e:
        logger.error(str(e))
        _emit_json_and_exit({"status": "error", "error": str(e)}, 1)


if __name__ == "__main__":
    main()
