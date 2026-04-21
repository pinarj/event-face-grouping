#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# Face embedding generator — CPU, fork multiprocessing
# stdout: final JSON only
# stderr: progress and error logs

# must be set before importing anything that uses multiprocessing
import multiprocessing
import platform
_start_method = "fork" if platform.system() != "Windows" else "spawn"
multiprocessing.set_start_method(_start_method, force=True)

# limit internal threadpools before importing numpy/onnx/cv2
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
os.environ.setdefault("KMP_AFFINITY", "granularity=fine,compact,1,0")

# suppress irrelevant warnings
import warnings
warnings.filterwarnings(
    "ignore",
    message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
    category=UserWarning,
    module="onnxruntime"
)
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import json
import time
import glob
import psutil
import logging
import argparse
import numpy as np
from datetime import datetime
from PIL import Image, ExifTags, ImageEnhance
from concurrent.futures import ProcessPoolExecutor, as_completed

import cv2
cv2.setNumThreads(1)

# logging — all logs go to stderr, stdout is reserved for the final JSON
logger = logging.getLogger("embeddings")
logger.setLevel(logging.INFO)
logger.handlers.clear()
logger.propagate = False

_console = logging.StreamHandler(sys.stderr)
_console.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
logger.addHandler(_console)


def _emit_json_and_exit(summary: dict, exit_code: int) -> None:
    # write JSON result to stdout so the caller (PHP) can parse it cleanly
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    print("done", flush=True)
    sys.exit(exit_code)


# CLI
def parse_arguments():
    parser = argparse.ArgumentParser(description="Process images and generate face embeddings (CPU, fork, shared model).")
    parser.add_argument("--input", "-i", type=str, default="dataset",
                        help="Input folder containing images (default: dataset)")
    parser.add_argument("--output", "-o", type=str, default="embeddings",
                        help="Output folder for embeddings (default: embeddings)")
    parser.add_argument("--force", "-f", action="store_true",
                        help="Force reprocessing even if outputs exist")
    parser.add_argument("--max-workers", "-w", type=int, default=None,
                        help="Number of parallel processes (default: auto)")
    parser.add_argument("--batch-size", "-b", type=int, default=None,
                        help="Submit window size (default: 2 x workers)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (CUDA) for inference — requires onnxruntime-gpu")
    return parser.parse_args()




# image helpers

def adjust_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        shadow = brightness if brightness > 0 else 0
        highlight = 255 if brightness > 0 else 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.convertScaleAbs(image, alpha=alpha_b, beta=gamma_b)

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        image = cv2.convertScaleAbs(image, alpha=alpha_c, beta=gamma_c)

    return image


def enhance_image(img_bgr):
    # for dark images: boost brightness + use detail enhancement
    # for normal images: mild contrast + sharpness boost
    try:
        if len(img_bgr.shape) == 3 and img_bgr.shape[2] == 3:
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = img_bgr

        avg_brightness = float(np.mean(gray))

        if avg_brightness < 60:
            enhanced = adjust_brightness_contrast(
                img_bgr,
                brightness=min(100, int(100 - avg_brightness)),
                contrast=30
            )
            enhanced = cv2.detailEnhance(enhanced, sigma_s=10, sigma_r=0.15)
            return enhanced

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_img = ImageEnhance.Contrast(pil_img).enhance(1.2)
        pil_img = ImageEnhance.Sharpness(pil_img).enhance(1.5)
        enhanced = np.array(pil_img)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.2, beta=20)
        return cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR)
    except Exception:
        return img_bgr


def apply_exif_orientation(image_path):
    # rotate image according to EXIF orientation tag, then enhance
    try:
        image = Image.open(image_path)
        if hasattr(image, "_getexif") and image._getexif() is not None:
            try:
                exif_dict = image._getexif()
                if exif_dict:
                    exif = {ExifTags.TAGS.get(k, k): v for k, v in exif_dict.items() if k in ExifTags.TAGS}
                    orientation = exif.get("Orientation", 1)
                    if orientation == 2:
                        image = image.transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 3:
                        image = image.rotate(180, expand=True)
                    elif orientation == 4:
                        image = image.rotate(180, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 5:
                        image = image.rotate(-90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 6:
                        image = image.rotate(-90, expand=True)
                    elif orientation == 7:
                        image = image.rotate(90, expand=True).transpose(Image.FLIP_LEFT_RIGHT)
                    elif orientation == 8:
                        image = image.rotate(90, expand=True)
            except Exception:
                pass

        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        return enhance_image(img_cv)
    except Exception:
        try:
            return cv2.imread(image_path)
        except Exception:
            return None


from insightface.app import FaceAnalysis

# fork: model loaded once in parent before workers spawn, inherited via copy-on-write
# spawn (Windows): model loaded per worker in _init_worker
_IS_FORK = (platform.system() != "Windows")
APP_SHARED = None


def _init_worker(ctx_id=-1):
    global APP_SHARED
    import warnings as _w
    _w.filterwarnings("ignore", category=FutureWarning)
    _w.filterwarnings(
        "ignore",
        message="Specified provider 'CUDAExecutionProvider' is not in available provider names",
        category=UserWarning,
        module="onnxruntime"
    )
    if APP_SHARED is None:
        APP_SHARED = FaceAnalysis(name="buffalo_l")
        APP_SHARED.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.7)


def _process_image_worker(input_folder, filename):
    try:
        img_path = os.path.join(input_folder, filename)
        img = apply_exif_orientation(img_path)
        if img is None:
            return filename, None, f"Error: {filename} could not be loaded"

        # skip very dark images (e.g. flash-off indoor shots)
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray_full.mean() < 30:
            return filename, [], None

        # first pass: detect on enhanced image
        # second pass: re-enhance if no faces found on first try
        faces = APP_SHARED.get(img)
        if not faces:
            enhanced_img = enhance_image(img)
            faces = APP_SHARED.get(enhanced_img)

        if not faces:
            return filename, [], None

        out = []
        for i, face in enumerate(faces):
            emb = getattr(face, "embedding", None)
            if emb is None:
                continue

            bbox = face.bbox.astype(int)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]

            # skip small faces (< 40px) — too low resolution for reliable matching
            if w < 40 or h < 40:
                continue

            # skip low-confidence detections
            score = float(getattr(face, "det_score", 1.0))
            if score < 0.5:
                continue

            # skip faces touching image borders (likely cut off)
            img_h, img_w = img.shape[:2]
            x1r, y1r, x2r, y2r = bbox[0], bbox[1], bbox[2], bbox[3]
            if x1r < 10 or y1r < 10 or x2r > img_w - 10 or y2r > img_h - 10:
                continue

            try:
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), bbox[2], bbox[3]
                crop = img[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                # skip blurry faces — Laplacian variance below 100 indicates motion blur or out of focus
                if cv2.Laplacian(gray, cv2.CV_64F).var() < 100:
                    continue
                # skip dark face crops (underexposed, shadow, etc.)
                if gray.mean() < 40:
                    continue
            except Exception:
                continue

            nrm = float(np.linalg.norm(emb))
            if nrm == 0.0:
                continue

            # normalize embedding to unit vector before saving
            out.append((i, (emb / nrm).astype(np.float32), bbox.tolist()))

        return filename, out, None
    except Exception as e:
        return filename, None, f"Error: {str(e)}"


def already_processed(output_folder, base_filename):
    pattern = os.path.join(output_folder, f"{base_filename}_*.npy")
    return any(glob.iglob(pattern))


def chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def main():
    args = parse_arguments()
    INPUT_FOLDER = args.input
    OUTPUT_FOLDER = args.output

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # worker count — capped by both CPU count and available RAM (~2 GB per worker)
    cpu_count = os.cpu_count() or 1
    total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    max_by_ram = max(1, int(total_ram_gb // 2))
    auto_workers = min(cpu_count, max_by_ram)

    MAX_WORKERS = max(1, args.max_workers if args.max_workers else auto_workers)
    BATCH_SIZE = max(1, args.batch_size if args.batch_size else MAX_WORKERS * 2)

    import onnxruntime as ort
    gpu_available = "CUDAExecutionProvider" in ort.get_available_providers()
    use_gpu = args.gpu or gpu_available
    ctx_id = 0 if use_gpu else -1
    device_label = f"GPU (ctx_id={ctx_id})" if use_gpu else "CPU"

    if _IS_FORK:
        global APP_SHARED
        APP_SHARED = FaceAnalysis(name="buffalo_l")
        APP_SHARED.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.7)

    started_at = datetime.now()
    t0 = time.time()
    logger.info(f"START embeddings started_at={started_at.isoformat(timespec='seconds')} device={device_label}")

    failed_files = []
    processed = 0
    skipped = 0

    # collect image list
    try:
        image_files = [
            f for f in os.listdir(INPUT_FOLDER)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".gif"))
        ]
    except FileNotFoundError:
        logger.error(f"Input folder not found: {INPUT_FOLDER}")
        finished_at = datetime.now()
        duration_s = round(time.time() - t0, 3)
        logger.info(
            "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
            finished_at.isoformat(timespec="seconds"),
            duration_s,
            "error",
            0,
            0
        )
        _emit_json_and_exit({
            "status": "error",
            "total_images": 0,
            "submitted": 0,
            "skipped_existing": 0,
            "processed": 0,
            "failed": 0,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_s,
            "error_message": f"Input folder not found: {INPUT_FOLDER}",
        }, 1)

    if not image_files:
        logger.error(f"No image files found in '{INPUT_FOLDER}'")
        finished_at = datetime.now()
        duration_s = round(time.time() - t0, 3)
        logger.info(
            "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
            finished_at.isoformat(timespec="seconds"),
            duration_s,
            "error",
            0,
            0
        )
        _emit_json_and_exit({
            "status": "error",
            "total_images": 0,
            "submitted": 0,
            "skipped_existing": 0,
            "processed": 0,
            "failed": 0,
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": duration_s,
            "error_message": f"No image files found in '{INPUT_FOLDER}'",
        }, 1)

    # skip already processed files unless --force
    submit_list = []
    if not args.force:
        for fname in image_files:
            base = os.path.splitext(fname)[0]
            if already_processed(OUTPUT_FOLDER, base):
                skipped += 1
                continue
            submit_list.append(fname)
    else:
        submit_list = image_files

    # run
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, initializer=_init_worker, initargs=(ctx_id,)) as executor:
        total_to_process = len(submit_list)
        processed_count = 0
        for chunk in chunks(submit_list, BATCH_SIZE):
            futures = {executor.submit(_process_image_worker, INPUT_FOLDER, fname): fname for fname in chunk}
            for future in as_completed(futures):
                filename, embeddings, error = future.result()
                base_filename = os.path.splitext(filename)[0]

                if error:
                    logger.error(error)
                    failed_files.append((filename, error))
                else:
                    if embeddings:
                        for face_idx, embedding, bbox in embeddings:
                            # each face gets its own .npy file and a meta.json with original filename + bbox
                            unique_face_id = f"{base_filename}_{face_idx}"
                            np.save(os.path.join(OUTPUT_FOLDER, f"{unique_face_id}.npy"), embedding)
                            with open(os.path.join(OUTPUT_FOLDER, f"{unique_face_id}_meta.json"), "w", encoding="utf-8") as meta_file:
                                json.dump({"original_filename": filename, "bbox": bbox}, meta_file, ensure_ascii=False)

                processed += 1
                processed_count += 1
                if processed_count % 500 == 0 or processed_count == total_to_process:
                    logger.info(f"Progress: {processed_count}/{total_to_process} images processed")

    # summary
    finished_at = datetime.now()
    duration_s = round(time.time() - t0, 3)

    status = "error" if processed == 0 else ("partial" if len(failed_files) > 0 else "success")

    summary = {
        "status": status,
        "total_images": len(image_files),
        "submitted": len(submit_list),
        "skipped_existing": skipped,
        "processed": processed,
        "failed": len(failed_files),
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds"),
        "duration_seconds": duration_s,
    }

    logger.info(
        "END embeddings finished_at=%s duration_s=%.3f status=%s processed=%d failed=%d",
        finished_at.isoformat(timespec="seconds"),
        duration_s,
        summary["status"],
        summary["processed"],
        summary["failed"]
    )

    _emit_json_and_exit(summary, 0 if status != "error" else 1)


if __name__ == "__main__":
    main()
