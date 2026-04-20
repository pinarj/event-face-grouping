import os
import sys
import time
import logging
import traceback
from flask import Flask, request, jsonify

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_faiss_production import (
    load_models,
    process_selfie,
    load_prebuilt_faiss,
    SIMILARITY_THRESHOLD,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("search_server")
logging.getLogger("werkzeug").setLevel(logging.ERROR)

app = Flask(__name__)

# Global instances — loaded once at startup
faiss_index_cache = {}   # faiss_dir → (index, valid_keys)

logger.info("Initializing face recognition models...")
face_app = load_models()
if not face_app:
    logger.error("Failed to load models!")
    sys.exit(1)
logger.info("Models loaded and server ready.")


def get_faiss_index(faiss_dir):
    """Load FAISS index once and keep in memory for all subsequent requests."""
    if faiss_dir not in faiss_index_cache:
        logger.info(f"Loading FAISS index from {faiss_dir} into memory...")
        index, valid_keys = load_prebuilt_faiss(faiss_dir)
        if index is None:
            return None, None
        faiss_index_cache[faiss_dir] = (index, valid_keys)
        logger.info(f"FAISS index cached: {index.ntotal} vectors")
    return faiss_index_cache[faiss_dir]

@app.route('/search', methods=['POST'])
def search_endpoint():
    try:
        data = request.json
        if not data:
            return jsonify({"status": "error", "message": "No JSON data provided"}), 400
        
        selfie_path = data.get('selfie_path')
        faiss_dir = data.get('faiss_dir')
        top_k = data.get('top_k', 200)
        threshold = data.get('threshold', SIMILARITY_THRESHOLD)

        if not selfie_path or not os.path.exists(selfie_path):
            return jsonify({"status": "error", "message": f"Selfie path invalid: {selfie_path}"}), 400

        start_time = time.time()

        # 1. Process Selfie (Extract Embedding)
        # Note: We use the globally loaded face_app
        query_embedding, _ = process_selfie(selfie_path, face_app)
        
        if query_embedding is None:
            return jsonify({"status": "error", "message": "Failed to extract embedding from selfie"}), 400

        # 2. Search using FAISS (index stays in memory between requests)
        results = []
        if faiss_dir:
            index, valid_keys = get_faiss_index(faiss_dir)
            if index is not None and valid_keys:
                query_vector = query_embedding.astype('float32').reshape(1, -1)
                search_k = min(top_k * 3, len(valid_keys))
                similarities, indices = index.search(query_vector, search_k)
                
                for i, (sim, idx) in enumerate(zip(similarities[0], indices[0])):
                    if idx < 0 or idx >= len(valid_keys):
                        continue
                    
                    similarity = float((1.0 + sim) / 2.0)
                    if similarity < threshold:
                        continue

                    results.append({
                        'filename': os.path.basename(valid_keys[idx]),
                        'similarity': similarity,
                        'cosine_similarity': float(sim),
                        'euclidean_distance': float(1.0 - similarity),
                    })
                results.sort(key=lambda x: x['similarity'], reverse=True)
                results = results[:top_k]
            else:
                # Fallback to manual load if index not found
                # Note: This is slow, but keeps the server functional
                logger.warning("FAISS index not found, falling back to manual load (SLOW)")
                # This would require setting global EMBEDDINGS_DIR if we use search_similar_faces directly
                # However, for the server, we want to stay fast.
                return jsonify({"status": "error", "message": "FAISS index not found and manual fallback not implemented for server performance safety"}), 404
        else:
            return jsonify({"status": "error", "message": "faiss_dir is required for fast search"}), 400

        total_time = time.time() - start_time
        
        # Prepare response
        output = {
            "status": "success",
            "matches": results,
            "summary": {
                "total_faces": len(results),
                "unique_files": len(set(r['filename'] for r in results))
            },
            "processing_time": f"{total_time:.2f}s"
        }
        
        logger.info(f"Search completed in {total_time:.2f}s. Found {len(results)} matches.")
        return jsonify(output)

    except Exception as e:
        logger.error(f"Error during search: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
