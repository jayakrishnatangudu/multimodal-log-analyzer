"""
anomaly_detector.py
-------------------
Detects anomalies by computing the cosine similarity between a CLIP image
embedding and a CLIP text embedding.

Rule: if cosine_similarity(image_emb, text_emb) < THRESHOLD → anomaly.
"""

import numpy as np


# Similarity below this value is flagged as an anomaly
ANOMALY_THRESHOLD = 0.25


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two 1-D vectors.

    Both vectors are expected to be L2-normalised (as returned by
    clip_encoder.encode_image / clip_encoder.encode_text), in which case
    cosine similarity is simply their dot product.

    Args:
        vec_a: 1-D numpy float32 array (e.g. image embedding, shape (512,)).
        vec_b: 1-D numpy float32 array (e.g. text  embedding, shape (512,)).

    Returns:
        Scalar float in [-1, 1].
    """
    vec_a = vec_a / (np.linalg.norm(vec_a) + 1e-8)
    vec_b = vec_b / (np.linalg.norm(vec_b) + 1e-8)
    return float(np.dot(vec_a, vec_b))


def detect_anomaly(
    image_embedding: np.ndarray,
    text_embedding: np.ndarray,
    threshold: float = ANOMALY_THRESHOLD,
) -> dict:
    """
    Compare an image embedding against a text embedding and decide whether
    the pair constitutes an anomaly.

    Args:
        image_embedding: L2-normalised CLIP image embedding, shape (512,).
        text_embedding:  L2-normalised CLIP text  embedding, shape (512,).
        threshold:       Minimum cosine similarity to be considered normal.
                         Pairs below this value are flagged as anomalies.

    Returns:
        dict with keys:
            similarity  (float)  – cosine similarity score
            is_anomaly  (bool)   – True if similarity < threshold
            threshold   (float)  – the threshold used
    """
    similarity = cosine_similarity(image_embedding, text_embedding)
    return {
        "similarity": round(similarity, 6),
        "is_anomaly": similarity < threshold,
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Simulate two nearly identical embeddings → should be normal
    base = rng.standard_normal(512).astype(np.float32)
    base /= np.linalg.norm(base)
    noise = base + rng.standard_normal(512).astype(np.float32) * 0.05
    noise /= np.linalg.norm(noise)

    result_normal = detect_anomaly(base, noise)
    print(f"Normal pair  -> {result_normal}")

    # Simulate two random orthogonal embeddings → should be anomaly
    rand_a = rng.standard_normal(512).astype(np.float32)
    rand_b = rng.standard_normal(512).astype(np.float32)
    rand_a /= np.linalg.norm(rand_a)
    rand_b /= np.linalg.norm(rand_b)

    result_anomaly = detect_anomaly(rand_a, rand_b)
    print(f"Random pair  -> {result_anomaly}")
