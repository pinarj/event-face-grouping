# Face Grouping & Clothing Analysis Module

This repository contains the production scripts for automatically clustering athlete faces across race event photos, using facial identity (embeddings) and clothing color/texture for high-accuracy grouping.

## Repository Structure

- `scripts/`: Implementation scripts (Python).
- `results/largevent92/`: Sample results and group montages from a test on 1,398 photos.
- `requirements.txt`: Python package dependencies.

---

## Installation

Install dependencies using `pip`:

```bash
pip install -r requirements.txt
```

---

## Full Pipeline (Per Event)

Run these steps in order for each event folder.

### Step 1 — Generate Embeddings
```bash
python scripts/embeddings.py --input /path/to/photos --output /path/to/embeddings --force
```

### Step 2 — Group Faces (Initial DBSCAN)
```bash
python scripts/group_faces.py --embeddings /path/to/embeddings --output /path/to/groups --eps 0.7
```

### Step 3 — Extract Clothing Features
```bash
python scripts/clothing_extractor.py --photos /path/to/photos --embeddings /path/to/embeddings --output /path/to/clothing
```

### Step 4 — Refine and Merge Groups
```bash
python scripts/refine_groups.py \
    --groups /path/to/groups/groups.json \
    --embeddings /path/to/embeddings \
    --clothing /path/to/clothing \
    --output /path/to/groups_refined
```

**Final Output:** `groups_refined/refined_groups.json` (maps face IDs to person-based albums).

---

## Sample Test Results

Successfully tested on a real-world dataset of **1,398 photos** across 5 cameras:
- **Embeddings extracted:** 2,614
- **Final refined groups:** 340 (representing 340 unique athletes)
- **Clothing Analysis:** Successfully used dual-layer (upper/lower body) color & texture analysis to avoid false merges in crowded scenes.
- **Accuracy:** Zero false merges found in randomly verified samples.
- **Cross-camera grouping:** Verified across multiple camera sources.

Visual confirmation of the groups can be found in the results directory.

---

## OCR Integration Guide

The system provides face location data (**bbox**) saved in the `_meta.json` files within the embeddings folder. This is designed to help OCR engines focus on the correct bib number.

1.  Each `refined_groups.json` entry contains a list of Face IDs.
2.  Each Face ID maps to a `{face_id}_meta.json` metadata file.
3.  The metadata contains the `bbox` `[x1, y1, x2, y2]` to pinpoint the face location.

Using these together enables high-accuracy multi-shot OCR and majority voting for each group cluster.
