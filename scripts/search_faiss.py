import os
import contextlib
import numpy as np
import json
import cv2
import pickle
from PIL import Image
from typing import List, Tuple, Optional, Dict, Any
import argparse
import logging
import sys

# Configure logging to stderr
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger(__name__)

# Redirect stdout to stderr temporarily (to avoid polluting stdout with logs)
@contextlib.contextmanager
def redirect_stdout_to_stderr():
    old_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        yield
    finally:
        sys.stdout = old_stdout


# Configuration
SIMILARITY_THRESHOLD = 0.75
HIGH_CONFIDENCE_THRESHOLD = 0.80
MEDIUM_CONFIDENCE_THRESHOLD = 0.70
MIN_FACE_CONFIDENCE = 0.1
MIN_FACE_SIZE = 3
MAX_FACES_TO_PROCESS = 200
DETECTION_SCALES = [0.6, 1.0, 1.4]
FACE_ALIGNMENT_ITERATIONS = 6

# Image enhancement settings
SHARPEN_FACTOR = 1.2
CONTRAST_FACTOR = 1.4
BRIGHTNESS_FACTOR = 1.2

# Model configuration
MODEL_CONFIG = {
    'name': 'buffalo_l',
    'root': '~/.insightface/models',
    'allowed_modules': ['detection', 'recognition'],
    'providers': ['CPUExecutionProvider'],
    'det_size': (640, 640),
    'det_thresh': MIN_FACE_CONFIDENCE
}

# ANSI color codes for colored output
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def load_models():
    """Loads face recognition models"""
    logger.info(f"{Colors.HEADER}🚀 Loading face recognition models...{Colors.ENDC}")

    try:
        logger.info(f"Model configuration: {MODEL_CONFIG}")

        logger.info("Model loading...")
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name=MODEL_CONFIG['name'],
                         root=MODEL_CONFIG['root'],
                         allowed_modules=MODEL_CONFIG['allowed_modules'],
                         providers=MODEL_CONFIG['providers'])

        # Prepare the model
        logger.info("Preparing model...")
        app.prepare(ctx_id=0,
                   det_size=MODEL_CONFIG.get('det_size', (640, 640)),
                   det_thresh=MODEL_CONFIG.get('det_thresh', 0.2))

        logger.info(f"{Colors.OKGREEN}✅ Models loaded successfully{Colors.ENDC}")
        return app
    except Exception as e:
        logger.info(f"{Colors.FAIL}❌ Error loading model: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None

def load_embeddings():
    """Loads all embeddings using associated meta.json files"""
    logger.info(f"{Colors.OKCYAN}🔍 Loading embeddings with original filenames...{Colors.ENDC}")
    embeddings = {}

    for filename in os.listdir(EMBEDDINGS_DIR):
        if filename.endswith(".npy"):
            try:
                base_name = os.path.splitext(filename)[0]  # GLM01812_0
                emb_path = os.path.join(EMBEDDINGS_DIR, filename)
                meta_path = os.path.join(EMBEDDINGS_DIR, f"{base_name}_meta.json")

                if not os.path.exists(meta_path):
                    logger.info(f"{Colors.WARNING}⚠️  Missing meta file for {filename}, skipping.{Colors.ENDC}")
                    continue

                # Load meta
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    original_file = meta.get("original_filename", None)

                if not original_file:
                    logger.info(f"{Colors.WARNING}⚠️  No original filename in {meta_path}, skipping.{Colors.ENDC}")
                    continue

                emb = np.load(emb_path)

                if emb.shape[0] != 512:
                    continue
                if np.isclose(np.linalg.norm(emb), 0.0):
                    continue

                emb = emb / np.linalg.norm(emb)

                if original_file in embeddings:
                    embeddings[original_file]['embeddings'].append(emb)
                else:
                    embeddings[original_file] = {
                        'embeddings': [emb],
                        'original_file': original_file
                    }

            except Exception as e:
                logger.info(f"{Colors.WARNING}⚠️  Error loading {filename}: {str(e)}{Colors.ENDC}")

    # Final processing
    final_embeddings = {}
    for original_file, data in embeddings.items():
        avg_emb = np.mean(data['embeddings'], axis=0)
        avg_emb = avg_emb / np.linalg.norm(avg_emb)

        for i, emb in enumerate(data['embeddings']):
            final_embeddings[f"{original_file}_ver{i}"] = {
                'embedding': emb,
                'original_file': original_file,
                'version': i,
                'avg_embedding': avg_emb
            }

    logger.info(f"{Colors.OKGREEN}✅ {len(final_embeddings)} embeddings loaded with filenames.{Colors.ENDC}")
    return final_embeddings

# EXIF correction
def apply_exif_orientation(image_path):
    try:
        img_pil = Image.open(image_path)
        exif = img_pil._getexif()
        ORIENTATION_TAG = 274
        if exif is not None:
            for tag, value in exif.items():
                if tag == ORIENTATION_TAG:
                    if value == 2:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
                    elif value == 3:
                        img_pil = img_pil.transpose(Image.ROTATE_180)
                    elif value == 4:
                        img_pil = img_pil.transpose(Image.FLIP_TOP_BOTTOM)
                    elif value == 5:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_270)
                    elif value == 6:
                        img_pil = img_pil.transpose(Image.ROTATE_270)
                    elif value == 7:
                        img_pil = img_pil.transpose(Image.FLIP_LEFT_RIGHT).transpose(Image.ROTATE_90)
                    elif value == 8:
                        img_pil = img_pil.transpose(Image.ROTATE_90)
                    break
        img_cv2 = np.array(img_pil)
        if len(img_cv2.shape) == 3 and img_cv2.shape[2] == 3:
            img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_RGB2BGR)
        return img_cv2
    except:
        return cv2.imread(image_path)

def enhance_image_quality(image: np.ndarray) -> np.ndarray:
    """Enhances image quality"""
    try:
        # Color correction
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

        # Sharpening
        kernel = np.array([[-1, -1, -1],
                          [-1, 9, -1],
                          [-1, -1, -1]])
        enhanced = cv2.filter2D(enhanced, -1, kernel)

        return enhanced
    except Exception as e:
        logger.info(f"Image enhancement error: {str(e)}")
        return image

def align_face(img: np.ndarray, face, iteration: int = 0) -> np.ndarray:
    """Advanced face alignment and normalization"""
    try:
        if img is None or not hasattr(face, 'landmark_2d_106'):
            logger.info("⚠️ Landmarks not found for alignment, using original image")
            return img  # Return the original image

        # Get the landmarks
        lm = face.landmark_2d_106
        if lm is None or len(lm) < 5:
            logger.info("⚠️ Insufficient landmarks found, using original image")
            return img  # Return the original image

        # Critical landmarks
        left_eye = np.mean(lm[33:42], axis=0)  # Left eye
        right_eye = np.mean(lm[87:96], axis=0)  # Right eye
        nose = lm[97]  # Nose tip
        mouth_left = lm[52]  # Left mouth corner
        mouth_right = lm[61]  # Right mouth corner

        # Face center
        face_center = np.mean([left_eye, right_eye, nose, mouth_left, mouth_right], axis=0)

        # Face width and height
        face_width = np.linalg.norm(left_eye - right_eye) * 3.0
        face_height = face_width * 1.3  # Standard face ratio

        # Calculate the face angle (more accurate)
        dY = right_eye[1] - left_eye[1]
        dX = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))

        # Scale factor (adjust based on face size)
        desired_width = 256
        scale = desired_width / face_width

        # Rotation matrix
        M = cv2.getRotationMatrix2D(tuple(face_center), angle, scale)

        # Move the face to the center
        tX = img.shape[1] / 2 - face_center[0] * scale
        tY = img.shape[0] / 2 - face_center[1] * scale
        M[0, 2] += tX
        M[1, 2] += tY

        # Transform the image
        aligned = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)

        # Crop the face region
        x = int(face_center[0] * scale + tX - desired_width/2)
        y = int(face_center[1] * scale + tY - desired_width/2)

        # Boundary checks
        x = max(0, x)
        y = max(0, y)
        w = min(desired_width, img.shape[1] - x)
        h = min(desired_width, img.shape[0] - y)

        if w <= 0 or h <= 0:
            logger.info("⚠️ Invalid aligned region, using original image")
            return img  # Return the original image

        cropped = aligned[y:y+h, x:x+w]

        # Final checks
        if cropped is None or cropped.size == 0:
            logger.info("⚠️ Invalid aligned region, using original image")
            return img  # Return the original image

        # Minimum size check
        if cropped.shape[0] < 20 or cropped.shape[1] < 20:
            logger.info("⚠️ Aligned region too small, using original image")
            return img  # Return the original image

        return cropped

    except Exception as e:
        logger.info(f"⚠️ Alignment error: {str(e)}")
        logger.info("⚠️ Using original image in case of error")
        return img  # Return the original image in case of error

def detect_faces_multi_scale(img: np.ndarray, app, scales: list = None) -> list:
    """Performs face detection at different scales.
    Tries scale=1.0 first (single inference). Falls back to full multi-scale only if no face found.
    """
    h, w = img.shape[:2]

    # Fast path: try original scale first (covers the vast majority of selfies)
    try:
        fast_faces = app.get(img)
        if fast_faces:
            valid = []
            for face in fast_faces:
                face_w = face.bbox[2] - face.bbox[0]
                face_h = face.bbox[3] - face.bbox[1]
                if face_w * face_h >= (MIN_FACE_SIZE * MIN_FACE_SIZE):
                    valid.append(face)
            if valid:
                logger.info("[SPEED] Face found at scale=1.0, skipping multi-scale")
                return valid
    except Exception as e:
        logger.info(f"⚠️ Fast-path detection error: {str(e)}")

    # Fallback: multi-scale (used only when face not found at scale=1.0)
    logger.info("[FALLBACK] No face at scale=1.0, trying multi-scale...")
    if scales is None:
        scales = DETECTION_SCALES

    all_faces = []

    for scale in scales:
        # Resize the image
        new_w, new_h = int(w * scale), int(h * scale)
        if new_w < MIN_FACE_SIZE or new_h < MIN_FACE_SIZE:
            continue

        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Perform face detection
        try:
            faces = app.get(resized)

            # Adjust the bounding boxes to the original size
            for face in faces:
                # Scale the bounding boxes
                scale_factor = 1.0 / scale
                face.bbox = (face.bbox * scale_factor).astype('int32')

                if hasattr(face, 'kps') and face.kps is not None:
                    face.kps = (face.kps * scale_factor).astype('int32')

                # Check the face size
                face_w = face.bbox[2] - face.bbox[0]
                face_h = face.bbox[3] - face.bbox[1]
                face_size = face_w * face_h

                if face_size >= (MIN_FACE_SIZE * MIN_FACE_SIZE):
                    all_faces.append(face)

        except Exception as e:
            logger.info(f"{Colors.WARNING}⚠️  Face detection error (scale: {scale}): {str(e)}{Colors.ENDC}")

    return all_faces

def select_best_face(faces: list) -> tuple:
    """Selects the best face from detected faces"""
    if not faces:
        return None, -1

    best_face = None
    best_score = -1

    for face in faces:
        # Score the face based on size and confidence
        face_w = face.bbox[2] - face.bbox[0]
        face_h = face.bbox[3] - face.bbox[1]
        face_size = face_w * face_h

        # Face aspect ratio (closer to 1 is better)
        aspect_ratio = min(face_w, face_h) / max(face_w, face_h) if max(face_w, face_h) > 0 else 0

        # Face angle (0-30 degrees is best)
        angle_score = 1.0
        if hasattr(face, 'kps') and face.kps is not None and len(face.kps) >= 2:
            left_eye = face.kps[0]
            right_eye = face.kps[1]
            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = abs(np.degrees(np.arctan2(dY, dX)))
            # 0-30 degrees is best (frontal)
            angle_score = max(0, 1 - (min(angle, 180 - angle) / 30))

        # Final score (size, confidence, aspect ratio, and angle)
        score = (face_size * face.det_score * aspect_ratio * angle_score) ** 0.25

        if score > best_score:
            best_score = score
            best_face = face

    return best_face, best_score

def process_selfie(image_path: str, app) -> tuple:
    """Face detection and embedding extraction from selfie"""
    logger.info(f"\n{Colors.HEADER}📸 Processing selfie: {os.path.basename(image_path)}{Colors.ENDC}")

    # EXIF correction
    img = apply_exif_orientation(image_path)
    if img is None:
        logger.info(f"{Colors.FAIL}❌ Error loading image!{Colors.ENDC}")
        return None, None

    # Store the original image
    original_img = img.copy()

    # SPEED FIX: Downscale selfie images to 1024px 
    # 1024px is a safe middle ground: reduces latency but keeps enough detail to avoid false positives
    _max_dim = 1024.0
    _h, _w = img.shape[:2]
    if max(_h, _w) > _max_dim:
        _scale = _max_dim / float(max(_h, _w))
        img = cv2.resize(img, (int(_w * _scale), int(_h * _scale)), interpolation=cv2.INTER_AREA)
        logger.info(f"[SPEED FIX] Image resized from {_w}x{_h} to {int(_w*_scale)}x{int(_h*_scale)}")

    # Stage 1: Face detection at different scales
    all_faces = detect_faces_multi_scale(img, app)

    if not all_faces:
        logger.info(f"{Colors.FAIL}❌ Error: No faces detected!{Colors.ENDC}")
        return None, None

    # Stage 2: Select the best face
    best_face, best_score = select_best_face(all_faces)

    if best_face is None:
        logger.info(f"{Colors.FAIL}❌ Error: No suitable face found!{Colors.ENDC}")
        return None, None

    # Stage 3: Align the face
    aligned_face = align_face(original_img, best_face)
    if aligned_face is None:
        logger.info("⚠️ Aligned face not found, using original image")
        aligned_face = img

    # Stage 4: Extract embedding from the aligned face (single inference)
    try:
        embedding = augment_and_average_embedding(aligned_face, app, fallback_embedding=best_face.embedding if hasattr(best_face, 'embedding') and best_face.embedding is not None else None)
        return embedding, original_img

    except Exception as e:
        logger.info(f"{Colors.FAIL}❌ Error processing face: {str(e)}{Colors.ENDC}")
        import traceback
        traceback.print_exc()
        return None, None

def augment_and_average_embedding(img, app, fallback_embedding=None):
    """Extracts embedding from selfie (Single-pass for maximum speed)"""
    try:
        faces = app.get(img)
        if not faces:
            # Use fallback embedding from multi-scale detection if available
            if fallback_embedding is not None:
                logger.info("⚠️ No face on aligned image, using best_face embedding as fallback")
                norm = np.linalg.norm(fallback_embedding)
                return fallback_embedding / norm if norm > 0 else None
            return None

        # Get the highest confidence face
        main_face = max(faces, key=lambda x: x.det_score)
        emb = main_face.embedding

        # Normalize
        final_emb = emb / np.linalg.norm(emb)
        return final_emb
    except Exception as e:
        logger.info(f"⚠️ Embedding extraction error: {str(e)}")
        return None

def build_faiss_index(embeddings):
    """Creates FAISS index from embeddings"""
    if not embeddings:
        return None, []

    # Get the embedding dimension (from the first valid embedding)
    emb_dim = None
    for emb_data in embeddings.values():
        if 'embedding' in emb_data and emb_data['embedding'] is not None:
            emb_dim = emb_data['embedding'].shape[0]
            break

    if emb_dim is None:
        return None, []

    # Create the FAISS index (for Dot Product)
    # We use Inner Product since the embeddings are already normalized
    import faiss
    index = faiss.IndexFlatIP(emb_dim)

    # Collect all embeddings
    all_embeddings = []
    valid_keys = []

    for filename, data in embeddings.items():
        if 'embedding' in data and data['embedding'] is not None:
            emb = data['embedding'].astype('float32')
            all_embeddings.append(emb)
            valid_keys.append(filename)

    if not all_embeddings:
        return None, []

    # Stack the embeddings into a matrix
    embeddings_matrix = np.vstack(all_embeddings)

    # Add the embeddings to the FAISS index
    index.add(embeddings_matrix)

    return index, valid_keys

def search_similar_faces(query_embedding, embeddings, top_k=200):  # Return more results
    """Face search function using FAISS"""
    if query_embedding is None or not embeddings:
        return []

    # Create the FAISS index
    index, valid_keys = build_faiss_index(embeddings)
    if index is None or not valid_keys:
        return []

    # Prepare the query vector
    query_vector = query_embedding.astype('float32').reshape(1, -1)

    # Search using FAISS (Dot Product)
    # Search for more results to get the top k
    search_k = min(top_k * 3, len(valid_keys))
    similarities, indices = index.search(query_vector, search_k)

    # Process the results
    results = []
    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
        if idx < 0 or idx >= len(valid_keys):  # Invalid index check
            continue

        filename = valid_keys[idx]
        data = embeddings[filename]

        # Normalize the Dot Product result to 0-1 range
        # Dot Product is in the range -1 to 1, so we convert it to 0-1 range
        similarity = float((1.0 + sim) / 2.0)

        # Filter results below the threshold
        if similarity < SIMILARITY_THRESHOLD:
            continue

        # Embeddings themselves
        query_emb = query_embedding
        target_emb = data['embedding']
        avg_emb = data.get('avg_embedding', target_emb)

        # Calculate cosine similarity
        cos_sim = np.dot(query_emb, target_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(target_emb))
        avg_cos_sim = np.dot(query_emb, avg_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(avg_emb))

        # Calculate Euclidean distance
        euclidean_dist = np.linalg.norm(query_emb - target_emb)
        avg_euclidean_dist = np.linalg.norm(query_emb - avg_emb)

        results.append({
            'filename': filename,
            'original_file': data['original_file'],
            'version': data.get('version', 0),
            'similarity': similarity,
            'distance': float(1.0 - similarity),
            'cos_sim': float(cos_sim),
            'avg_cos_sim': float(avg_cos_sim),
            'euclidean_score': float(euclidean_dist),
            'avg_euclidean_score': float(avg_euclidean_dist)
        })

    # Sort by similarity (high to low)
    results.sort(key=lambda x: x['similarity'], reverse=True)

    # Filter results below the threshold
    filtered_results = [r for r in results if r['similarity'] >= SIMILARITY_THRESHOLD]

    # Return the top k results
    return filtered_results[:top_k]

def print_result(result):
    """Prints search results in color"""
    if not isinstance(result, dict):
        return

    similarity = result.get('similarity', 0)
    color = get_similarity_color(similarity)

    # Get the file name
    original_file = result.get('original_file', 'None')
    version = result.get('version', 0)

    # Similarity comment
    logger.info(f"   {get_similarity_comment(similarity)}")
    logger.info("-" * 70)

    rank = result.get('rank', 0)
    filename = result.get('filename', 'None')

    # Version information
    if version > 0:
        logger.info(f"   {Colors.OKCYAN}-Version:{Colors.ENDC} {version}")

    # Cluster information
    cluster_info = result.get('cluster_info', {})

    if cluster_info:
        size = cluster_info.get('size', 0)
        samples = cluster_info.get('sample_images', [])

        logger.info(f"   {Colors.OKCYAN}📊 Cluster Size:{Colors.ENDC} {size} faces")
        if samples:
            logger.info(f"   {Colors.OKCYAN}🔍 Sample Images:{Colors.ENDC}")
            for img in samples:
                logger.info(f"      • {img}")
    logger.info("-" * 70)

def load_prebuilt_faiss(faiss_dir):
    if not faiss_dir:
        return None, None

    index_path = os.path.join(faiss_dir, "index.faiss")
    meta_path = os.path.join(faiss_dir, "index.pkl")

    if not os.path.exists(index_path) or not os.path.exists(meta_path):
        logger.info("⚠️ FAISS index not found")
        return None, None

    try:
        import faiss
        index = faiss.read_index(index_path)

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        filenames = meta.get("filenames", [])

        logger.info(f"✅ Loaded FAISS index ({index.ntotal} vectors)")
        return index, filenames

    except Exception as e:
        logger.info(f"❌ Error loading FAISS index: {e}")
        return None, None

def get_similarity_color(similarity):
    """Returns color based on similarity score"""
    if similarity >= 0.7:
        return Colors.OKGREEN
    elif similarity >= 0.6:
        return Colors.OKCYAN
    elif similarity >= 0.5:
        return Colors.WARNING
    else:
        return Colors.FAIL

def get_similarity_comment(similarity):
    """Returns comment based on similarity score"""
    if similarity >= HIGH_CONFIDENCE_THRESHOLD:
        return f"   {Colors.BOLD}✅ High Confidence (Same person - 95%+){Colors.ENDC}"
    elif similarity >= MEDIUM_CONFIDENCE_THRESHOLD:
        return f"   {Colors.OKGREEN}✅ Medium Confidence (Likely same person - 80%+){Colors.ENDC}"
    elif similarity >= SIMILARITY_THRESHOLD:
        return f"   {Colors.WARNING}⚠️  Low Confidence (May be same person - 70%+){Colors.ENDC}"
    else:
        return f"   {Colors.FAIL}❌ Low Similarity (Different person){Colors.ENDC}"

def main():
    with redirect_stdout_to_stderr():
        # Parse command line arguments
        parser = argparse.ArgumentParser(description='Face similarity search using FAISS')

        parser.add_argument('embeddings_dir', type=str, help='Path to the embeddings directory')
        parser.add_argument('selfie_path', type=str, help='Path to the selfie image to search')

        parser.add_argument('--faiss-dir', type=str, help='Path to prebuilt FAISS index directory')
        parser.add_argument('--json', action='store_true', default=True, help='Output results in JSON format (default)')

        args = parser.parse_args()

        if args.json:
            Colors.HEADER = Colors.OKBLUE = Colors.OKCYAN = Colors.OKGREEN = Colors.WARNING = Colors.FAIL = Colors.ENDC = Colors.BOLD = ''

        # Validate paths
        if not os.path.exists(args.embeddings_dir):
            logger.info(f"{Colors.FAIL}❌ Error: Embeddings directory '{args.embeddings_dir}' does not exist{Colors.ENDC}")
            return

        if not os.path.isfile(args.selfie_path):
            logger.info(f"{Colors.FAIL}❌ Error: Selfie image '{args.selfie_path}' does not exist{Colors.ENDC}")
            return

        # Update global variables
        global EMBEDDINGS_DIR
        EMBEDDINGS_DIR = args.embeddings_dir

        try:
            # Load models
            app = load_models()
            if not app:
                logger.info(f"{Colors.FAIL}❌ Failed to load model!{Colors.ENDC}")
                return

            logger.info(f"\n{Colors.HEADER}📸 Processing selfie: {os.path.basename(args.selfie_path)}{Colors.ENDC}")

            # Process selfie
            query_embedding, original_img = process_selfie(args.selfie_path, app)
            if query_embedding is None:
                logger.info(f"{Colors.FAIL}❌ Error processing selfie!{Colors.ENDC}")
                return

            # Build or load FAISS index, then search
            if args.faiss_dir:
                # Use prebuilt index — much faster than rebuilding from embeddings
                index, valid_keys = load_prebuilt_faiss(args.faiss_dir)
                if index is not None and valid_keys:
                    query_vector = query_embedding.astype('float32').reshape(1, -1)
                    search_k = min(200 * 3, len(valid_keys))
                    similarities, indices = index.search(query_vector, search_k)
                    results = []
                    for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                        if idx < 0 or idx >= len(valid_keys):
                            continue
                        similarity = float((1.0 + sim) / 2.0)
                        if similarity < SIMILARITY_THRESHOLD:
                            continue
                        results.append({
                            'filename': valid_keys[idx],
                            'original_file': valid_keys[idx],
                            'similarity': similarity,
                            'cos_sim': float(sim),
                            'euclidean_score': float(1.0 - similarity),
                        })
                    results.sort(key=lambda x: x['similarity'], reverse=True)
                    results = results[:200]
                else:
                    logger.info("⚠️ Falling back to manual embedding load...")
                    embeddings = load_embeddings()
                    if not embeddings:
                        logger.info(f"{Colors.FAIL}❌ No embeddings found!{Colors.ENDC}")
                        return
                    results = search_similar_faces(query_embedding, embeddings, top_k=200)
            else:
                embeddings = load_embeddings()
                if not embeddings:
                    logger.info(f"{Colors.FAIL}❌ No embeddings found!{Colors.ENDC}")
                    return
                results = search_similar_faces(query_embedding, embeddings, top_k=200)

            if not results:
                logger.info(f"{Colors.WARNING}⚠️  No similar faces found!{Colors.ENDC}")
                return

            # Prepare JSON output
            output = {
                "status": "success",
                "matches": [],
                "summary": {
                    "total_faces": len(results),
                    "unique_files": len(set(r['original_file'] for r in results if 'original_file' in r))
                }
            }

            # Process results for JSON
            for result in results:
                if not isinstance(result, dict) or 'similarity' not in result:
                    continue

                if result['similarity'] < SIMILARITY_THRESHOLD:
                    continue

                match = {
                    "filename": os.path.basename(result['original_file']),
                    "similarity": float(result['similarity']),
                    "cosine_similarity": float(result.get('cos_sim', 0)),
                    "euclidean_distance": float(result.get('euclidean_score', 0))
                }
                output["matches"].append(match)

            # Sort matches by similarity
            output["matches"].sort(key=lambda x: x["similarity"], reverse=True)

            # Output results
            if args.json:
                return output
            else:
                # Display results
                logger.info(f"\n{Colors.HEADER}🔍 Similar Faces Found (Total {len(output['matches'])} results):{Colors.ENDC}")
                logger.info("=" * 60)

                # Print summary
                logger.info(f"\n{Colors.OKCYAN}📊 Detected {output['summary']['total_faces']} faces from {output['summary']['unique_files']} different files.{Colors.ENDC}")
                logger.info("-" * 60)

                # Print detailed results
                for i, match in enumerate(output['matches'], 1):
                    color = get_similarity_color(match['similarity'])

                    logger.info(f"\n{i}. {color}✅ {match['filename']}{Colors.ENDC} → Similarity: {match['similarity']:.4f}")
                    if 'full_path' in match:
                        logger.info(f"    Full Path: {match['full_path']}")
                    if 'confidence' in match:
                        logger.info(f"    Confidence: {match['confidence'].title()} Confidence")
                    logger.info(f"    Cosine Similarity: {match.get('cosine_similarity', 'N/A'):.4f} | Euclidean Distance: {match.get('euclidean_distance', 'N/A'):.4f}")
                    logger.info()

                return output

        except Exception as e:
            logger.info(f"{Colors.FAIL}❌ Unexpected error occurred: {str(e)}{Colors.ENDC}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    result = main()
    if not isinstance(result, dict):
        result = {
            "status": "error",
            "message": "Invalid or empty result from script"
        }
    print(json.dumps(result, ensure_ascii=False))
