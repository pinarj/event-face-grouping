# Face Grouping Pipeline — Production Scripts

Clusters all faces from a race event into person-based groups, then merges those groups with bib number detections to produce a final `bib_groups.json` per person.

---

## Pipeline Overview

```
photos/
  └─ embeddings.py      → embeddings/ (.npy + _meta.json)
       └─ build_index.py → faiss_index/ (selfie search — unchanged)
       └─ group_faces.py    → groups/groups.json
       └─ clothing_extractor.py → clothing/ (color + texture features)
            └─ refine_groups.py → groups_refined/refined_groups.json  ← final output
```

Run steps in order for each event.

---

## Scripts

### 1. `embeddings.py` — Face Embedding Generation

Detects faces in all event photos and generates a 512-dim ArcFace embedding per face.

```bash
python embeddings.py \
    --input /events/event_id/import \
    --output /events/event_id/embeddings
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input` | `dataset` | Folder containing original photos |
| `--output` | `embeddings` | Output folder |
| `--force` | off | Reprocess photos that already have embeddings |
| `--max-workers` | auto | Parallel processes (auto = CPU count, capped by RAM) |

**Output per face:**
- `PHOTONAME_N.npy` — 512-dim float32 embedding
- `PHOTONAME_N_meta.json` — `{"original_filename": "...", "bbox": [x1, y1, x2, y2], ...}`

---

### 2. `build_index.py` — FAISS Index (Selfie Search)

Builds the FAISS index used by selfie search. Unchanged from previous milestone.

```bash
python build_index.py \
    --event event_id \
    --embedding-dir /events/event_id/embeddings \
    --output-dir /events/event_id/faiss_index
```

---

### 3. `group_faces.py` — Face Clustering

Groups all face embeddings by person identity using FAISS nearest-neighbour search + Union-Find. No prior knowledge of the number of people is required.

```bash
python group_faces.py \
    --embeddings /events/event_id/embeddings \
    --output /events/event_id/groups \
    --eps 0.7
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--embeddings` | required | Embeddings folder |
| `--output` | required | Output folder |
| `--eps` | `0.7` | Distance threshold. Lower = stricter (fewer false merges). Higher = looser (fewer splits). |
| `--min-samples` | `2` | Minimum faces per group. Set to `1` to include single-appearance faces. |
| `--workers` | all cores | FAISS thread count. Set to available core count on AWS. |

**Output:** `groups/groups.json`

**Scalability:** Uses FAISS IndexFlatL2 (< 10k faces) or IndexIVFFlat (≥ 10k faces). Tested up to 250k images on 4 AWS instances.

---

### 4. `clothing_extractor.py` — Clothing Feature Extraction

Extracts upper and lower body clothing features (HSV colour histogram + LBP texture) for each face. Used by `refine_groups.py` as an additional signal when merging borderline groups.

```bash
python clothing_extractor.py \
    --photos /events/event_id/import \
    --embeddings /events/event_id/embeddings \
    --output /events/event_id/clothing \
    --workers 8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--photos` | required | Original photos folder |
| `--embeddings` | required | Embeddings folder |
| `--output` | required | Output folder |
| `--workers` | `4` | Parallel workers. Set to available CPU core count. |
| `--save-crops` | off | Save upper/lower crop images for visual inspection |

**Output:**
- `clothing_features.npy` — upper body feature matrix
- `clothing_features_lower.npy` — lower body feature matrix
- `clothing_index.json` — face ID → row mapping

---

### 5. `refine_groups.py` — Group Refinement

Merges small groups that likely represent the same person, using face similarity + clothing similarity as evidence. Large groups are never merged — they are already reliable.

```bash
python refine_groups.py \
    --groups /events/event_id/groups/groups.json \
    --embeddings /events/event_id/embeddings \
    --clothing /events/event_id/clothing \
    --output /events/event_id/groups_refined
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--groups` | required | Path to `groups.json` |
| `--embeddings` | required | Face embeddings folder |
| `--clothing` | required | Clothing features folder |
| `--output` | required | Output folder |
| `--clothing-threshold` | `0.6` | Minimum score to merge. `face_sim × 0.7 + clothing_sim × 0.3 ≥ threshold` |
| `--lower-veto-threshold` | `0.5` | If lower body similarity is below this, block merge regardless of other scores |
| `--max-group-size` | `5` | Only attempt to merge groups smaller than this |

**Merge conditions (all three must pass):**
1. Both groups are small (≤ `max-group-size`) AND have consecutive photo numbers in the same camera series
2. Face centroid cosine similarity ≥ 0.25 (hard veto below this)
3. Combined score = `face_sim × 0.7 + clothing_sim × 0.3` ≥ `clothing-threshold`

**Output:** `groups_refined/refined_groups.json` ← **use this for bib assignment**

---

## Output Format

### `refined_groups.json`

```json
{
  "groups": {
    "group_1": ["FAJ_2847_0", "FAJ_2848_0", "LIL_0042_1"],
    "group_2": ["FAL_0112_0", "WAR_3310_0"]
  },
  "noise": ["FAJ_2101_0", "MAR_3059_2"]
}
```


### Understanding face IDs

Each entry in `groups` is a **face ID**, format: `PHOTONAME_FACEINDEX`

- `FAJ_2847_0` → face #0 in photo `FAJ_2847.jpg`
- `FAJ_2847_1` → face #1 in the same photo

To get the photo filename and bounding box, read the `_meta.json` file:
```
embeddings/FAJ_2847_0_meta.json → {"original_filename": "FAJ_2847.jpg", "bbox": [x1, y1, x2, y2]}
```

The `bbox` tells Brian's OCR exactly where on the photo the face is, so it can focus on the nearest bib rather than reading all visible bibs.

---

## Folder Structure

```
/events/event_id/
    import/               ← original photos
    embeddings/           ← .npy + _meta.json (one pair per detected face)
    faiss_index/          ← index.faiss + index.pkl (selfie search)
    groups/               ← groups.json (raw clustering output)
    clothing/             ← clothing_features.npy + clothing_index.json
    groups_refined/       ← refined_groups.json
    bib_groups.json       ← final output
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `No .npy files found` | Wrong path or Step 1 not run | Check `--embeddings` path |
| `WARNING: N embeddings have no bbox` | Old embeddings without bbox | Re-run `embeddings.py --force` |
| `Clothing features not found` | Step 4 not run | Run `clothing_extractor.py` first |
| Too many faces in noise | `eps` too strict | Try `--eps 0.8` |
| Wrong persons merged | `eps` too loose | Try `--eps 0.6` or `--clothing-threshold 0.8` |

---

## stdout / stderr Convention

All scripts follow the same convention (required by the PHP pipeline):

- `stdout` — JSON line, then `done` on the next line
- `stderr` — all progress logs and warnings

```bash
# PHP-safe usage:
python group_faces.py ... 2>>/var/log/pipeline.log
```
