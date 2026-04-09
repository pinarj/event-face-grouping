import os
import sys
import json
import pickle
import argparse
import logging
import time
from datetime import datetime

import numpy as np
import faiss
from tqdm import tqdm

# ==========================================================
# Logging (stderr for logs, stdout for JSON/done)
# ==========================================================
logger = logging.getLogger("faiss_index_builder")
logger.setLevel(logging.INFO)
logger.handlers.clear()
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter("%(asctime)s [INFO] %(message)s"))
logger.addHandler(_handler)
logger.propagate = False


def _exit_json(payload: dict, code: int = 0) -> int:
    """Emit exactly one JSON to stdout, then 'done' and return exit code."""
    print(json.dumps(payload, ensure_ascii=False), flush=True)
    print("done", flush=True)
    return code


def load_embeddings(embedding_dir: str) -> tuple:
    """Load and normalize embeddings from .npy files in the specified directory"""
    embeddings = []
    filenames = []

    if not os.path.exists(embedding_dir):
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")

    npy_files = [f for f in os.listdir(embedding_dir) if f.endswith(".npy")]
    if not npy_files:
        return None, []

    # tqdm disabled — keeps stdout clean for the PHP pipeline
    total_files = len(npy_files)
    for i, filename in enumerate(tqdm(npy_files, desc="Loading embeddings", disable=True)):
        try:
            emb = np.load(os.path.join(embedding_dir, filename))

            # skip silently — invalid embeddings are not worth surfacing as warnings
            if not isinstance(emb, np.ndarray):
                continue

            base = os.path.splitext(filename)[0]
            meta_path = os.path.join(embedding_dir, f"{base}_meta.json")

            original_name = base
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        original_name = meta.get("original_filename", base)
                except Exception:
                    original_name = base

            # Normalize embeddings
            if emb.ndim == 2:
                norms = np.linalg.norm(emb, axis=1, keepdims=True)
                emb = emb / (norms + 1e-10)
                embeddings.append(emb)
                filenames.extend([original_name] * emb.shape[0])
            else:
                norm = np.linalg.norm(emb)
                if norm != 0:
                    emb = emb / norm
                embeddings.append(emb.reshape(1, -1))
                filenames.append(original_name)

            if (i + 1) % 500 == 0 or (i + 1) == total_files:
                logger.info(f"Progress: {i + 1}/{total_files} embeddings loaded")

        except Exception:
            # skip silently on any load error
            continue

    if not embeddings:
        return None, []

    all_embeddings = np.vstack(embeddings)
    return all_embeddings, filenames


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build FAISS IP index (cosine similarity)"""
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype("float32"))
    return index


def save_index(index: faiss.IndexFlatIP, filenames: list, faiss_dir: str) -> None:
    os.makedirs(faiss_dir, exist_ok=True)

    index_path = os.path.join(faiss_dir, "index.faiss")
    faiss.write_index(index, index_path)

    meta = {
        "dimension": index.d,
        "metric": "IP",
        "total_vectors": index.ntotal,
        "filenames": filenames,
        "index_file": "index.faiss",
    }

    with open(os.path.join(faiss_dir, "index.pkl"), "wb") as f:
        pickle.dump(meta, f)


def main():
    start_ts = time.time()
    started_at = datetime.now()
    logger.info(f"START faiss_index_builder started_at={started_at.isoformat(timespec='seconds')}")

    parser = argparse.ArgumentParser(description="Build FAISS index for event")
    parser.add_argument("--event", type=str, required=True, help="Event ID")
    parser.add_argument("--embedding-dir", type=str, help="Optional custom path to embeddings directory")
    parser.add_argument("--output-dir", type=str, help="Optional output base dir (default: data/events/<event>)")
    args = parser.parse_args()

    # Default directories
    base_dir = args.output_dir or os.path.join("data", "events", args.event)
    embedding_dir = args.embedding_dir or os.path.join(base_dir, "embeddings")

    try:
        embeddings, filenames = load_embeddings(embedding_dir)

        if embeddings is None or len(embeddings) == 0:
            # no embeddings found — write empty metadata so downstream scripts fail gracefully
            os.makedirs(base_dir, exist_ok=True)
            meta = {
                "dimension": 0,
                "metric": "IP",
                "total_vectors": 0,
                "filenames": [],
                "index_file": "",
            }
            with open(os.path.join(base_dir, "index.pkl"), "wb") as f:
                pickle.dump(meta, f)

            duration_s = round(time.time() - start_ts, 3)
            finished_at = datetime.now()
            logger.info(f"END faiss_index_builder finished_at={finished_at.isoformat(timespec='seconds')} duration_s={duration_s} status=ok message='No embeddings'")
            return _exit_json(
                {
                    "status": "ok",
                    "duration_seconds": duration_s,
                    "message": "No embeddings to index",
                    "event": args.event,
                    "output_dir": base_dir,
                },
                0,
            )

        index = build_faiss_index(embeddings)
        save_index(index, filenames, base_dir)

        duration_s = round(time.time() - start_ts, 3)
        finished_at = datetime.now()
        logger.info(f"END faiss_index_builder finished_at={finished_at.isoformat(timespec='seconds')} duration_s={duration_s} status=ok total_vectors={len(filenames)}")
        return _exit_json(
            {
                "status": "ok",
                "duration_seconds": duration_s,
                "event": args.event,
                "output_dir": base_dir,
            },
            0,
        )

    except Exception as e:
        logger.error(str(e))
        duration_s = round(time.time() - start_ts, 3)
        return _exit_json(
            {
                "status": "error",
                "duration_seconds": duration_s,
                "message": str(e),
            },
            1,
        )


if __name__ == "__main__":
    raise SystemExit(main())
