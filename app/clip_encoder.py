"""
clip_encoder.py
---------------
Provides two functions for encoding images and text into normalised
CLIP embeddings using openai/clip-vit-base-patch32 (CPU only).
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

# ---------------------------------------------------------------------------
# Model loading — loaded once at import time, pinned to CPU
# ---------------------------------------------------------------------------

_MODEL_NAME = "openai/clip-vit-base-patch32"

print(f"[clip_encoder] Loading {_MODEL_NAME} on CPU …")
_processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
_model = CLIPModel.from_pretrained(_MODEL_NAME).to("cpu")
_model.eval()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def encode_image(image_path: str) -> np.ndarray:
    """
    Load an image from *image_path* and return its normalised CLIP embedding.

    Args:
        image_path: Absolute or relative path to a JPEG/PNG (or any PIL-supported) image.

    Returns:
        1-D numpy float32 array of shape (512,), L2-normalised.
    """
    image = Image.open(image_path).convert("RGB")
    inputs = _processor(images=image, return_tensors="pt")
    # Move to CPU explicitly (defensive — model is already on CPU)
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        # Use the vision model directly; pooler_output is the [CLS] embedding
        vision_out = _model.vision_model(**inputs)
        features = _model.visual_projection(vision_out.pooler_output)  # (1, 512)

    features = F.normalize(features, p=2, dim=-1)  # L2 normalise
    return features.squeeze(0).cpu().numpy().astype(np.float32)


def encode_text(text: str) -> np.ndarray:
    """
    Encode a text string and return its normalised CLIP embedding.

    Args:
        text: Any string (long strings are truncated to 77 tokens by the tokeniser).

    Returns:
        1-D numpy float32 array of shape (512,), L2-normalised.
    """
    inputs = _processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    inputs = {k: v.to("cpu") for k, v in inputs.items()}

    with torch.no_grad():
        # Use the text model directly; pooler_output is the [EOS] embedding
        text_out = _model.text_model(**inputs)
        features = _model.text_projection(text_out.pooler_output)  # (1, 512)

    features = F.normalize(features, p=2, dim=-1)  # L2 normalise
    return features.squeeze(0).cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Text encoding
    emb = encode_text("ERROR: disk I/O failure on /dev/sda1")
    print(f"Text embedding shape : {emb.shape}")   # (512,)
    print(f"L2 norm (should be 1): {np.linalg.norm(emb):.6f}")

    # Image encoding — create a tiny dummy image so the test runs standalone
    import tempfile, os
    dummy = Image.new("RGB", (224, 224), color=(128, 64, 32))
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        dummy.save(f.name)
        tmp_path = f.name

    img_emb = encode_image(tmp_path)
    print(f"Image embedding shape : {img_emb.shape}")  # (512,)
    print(f"L2 norm (should be 1) : {np.linalg.norm(img_emb):.6f}")
    os.unlink(tmp_path)
