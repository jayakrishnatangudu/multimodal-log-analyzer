"""
finetune_clip_lora.py
---------------------
Fine-tunes the openai/clip-vit-base-patch32 model on custom IT log data
using Low-Rank Adaptation (LoRA) adapters.

This script is designed to run in a constrained environment (e.g., Google Colab T4 GPU).
It injects LoRA into the attention matrices (q_proj, v_proj), reducing the trainable
parameter count drastically while maintaining model performance.

Usage:
    # Requires: pip install transformers torch Pillow scikit-learn peft
    python scripts/finetune_clip_lora.py
"""

import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig
from PIL import Image

def main():
    print("Loading base CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    # ── 1. Inject LoRA ────────────────────────────────────────────────────────
    print("Injecting LoRA adapters...")
    lora_config = LoraConfig(
        r=4,                                  # low rank: fast training
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # attention layers
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # Expect: ~0.5% of params trainable

    # Use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Moving model to {device}...")
    model = model.to(device)

    # ── 2. Dataset ────────────────────────────────────────────────────────────
    # Synthetic dataset (replace with real HDFS logs for actual training)
    print("Preparing dataset...")
    normal_logs = [
        "INFO: Connection established to node 192.168.1.1",
        "INFO: Heartbeat received from datanode",
        "INFO: Block replication complete",
        "INFO: Checkpoint saved successfully",
        "INFO: Memory usage at 45%, within normal range",
    ]
    anomaly_logs = [
        "ERROR: Connection timeout on node 192.168.1.5",
        "CRITICAL: Datanode failed to respond after 3 retries",
        "ERROR: Block corruption detected on disk",
        "CRITICAL: Memory usage at 98%, system overloaded",
        "ERROR: Network packet loss exceeding 30% threshold",
    ]
    
    # Labels: 1 = normal, 0 = anomaly
    texts = normal_logs + anomaly_logs
    labels = [1] * len(normal_logs) + [0] * len(anomaly_logs)

    # ── 3. Training Loop ──────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CosineEmbeddingLoss()

    print("Beginning training loop (10 epochs)...")
    model.train()
    
    for epoch in range(10):
        total_loss = 0
        # Compare adjacent pairs in the list
        for i in range(0, len(texts)-1, 2):
            t1 = processor(text=texts[i], return_tensors="pt", padding=True, truncation=True).to(device)
            t2 = processor(text=texts[i+1], return_tensors="pt", padding=True, truncation=True).to(device)

            e1 = model.get_text_features(**t1)
            e2 = model.get_text_features(**t2)

            # Same label = pull together (+1), different = push apart (-1)
            target = torch.tensor([1.0 if labels[i] == labels[i+1] else -1.0]).to(device)
            loss = loss_fn(e1, e2, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:2d}/10 — Loss: {total_loss:.4f}")

    # ── 4. Save ───────────────────────────────────────────────────────────────
    save_dir = "./clip-lora-finetuned"
    print(f"\nTraining complete. Saving LoRA adapters to {save_dir}...")
    model.save_pretrained(save_dir)
    print("Model saved! You can now load these adapters back into the CLIP pipeline.")


if __name__ == "__main__":
    main()
