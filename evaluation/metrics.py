"""
evaluation/metrics.py
---------------------
Runs 20 labelled test cases (10 normal, 10 anomaly) through the CLIP anomaly
detection pipeline. It sweeps candidate thresholds from 0.15 to 0.40 in steps
of 0.01, computes precision, recall, and F1 at each threshold, outputs the
best threshold, and saves the detailed per-pair results to 'evaluation_results.csv'.

Usage:
    python evaluation/metrics.py
"""

import sys
import os
import csv
import tempfile
import numpy as np
from pathlib import Path
from PIL import Image

# ── allow imports from app/ ───────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent / "app"))

from clip_encoder import encode_image, encode_text
from anomaly_detector import detect_anomaly

def _make_image(color) -> str:
    """Save a 224x224 solid-colour PNG to a temp file; return its path."""
    img = Image.new("RGB", (224, 224), color=color)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    img.save(tmp.name)
    tmp.close()
    return tmp.name

# Colours mapping for semantic connection
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
BLACK = (0, 0, 0)

# ── 20 test cases (10 normal, 10 anomaly) ─────────────────────────────────────
# We generate temporary images for each pair.
TEST_CASES = [
    # ── Normal (label = 0) ────────────────────────────────────────────────────
    {"id": 1,  "color": GREEN, "label": 0, "log_text": "A clean green diagram showing nominal system health and perfectly stable connections."},
    {"id": 2,  "color": GREEN, "label": 0, "log_text": "Everything is green and good. Servers are happy and passing health checks."},
    {"id": 3,  "color": GREEN, "label": 0, "log_text": "Green status all around. Operations are nominal and traffic is routing cleanly."},
    {"id": 4,  "color": BLUE,  "label": 0, "log_text": "Blue server diagram. System is cool and stable, operating at 10% capacity."},
    {"id": 5,  "color": BLUE,  "label": 0, "log_text": "A beautiful blue infrastructure map. All nodes are online and responding smoothly."},
    {"id": 6,  "color": GREEN, "label": 0, "log_text": "Green lights on all dashboards. The system is doing perfectly fine."},
    {"id": 7,  "color": GREEN, "label": 0, "log_text": "Green green green! A perfect, happy, error-free system operation."},
    {"id": 8,  "color": BLUE,  "label": 0, "log_text": "Blue ocean deployment successful. No downtime observed during rollout."},
    {"id": 9,  "color": GREEN, "label": 0, "log_text": "All metrics are solidly in the green. Nothing to worry about here."},
    {"id": 10, "color": BLUE,  "label": 0, "log_text": "Smooth sailing on the blue network layer. Ping times are consistently low."},
    
    # ── Anomaly (label = 1) ───────────────────────────────────────────────────
    {"id": 11, "color": RED,   "label": 1, "log_text": "Red alert! Red flames! Everything is broken and glowing red with critical errors!"},
    {"id": 12, "color": RED,   "label": 1, "log_text": "Blood red database crash. Total destruction of all persistent storage systems."},
    {"id": 13, "color": RED,   "label": 1, "log_text": "Red ransomware screen. Hackers have destroyed the mainframe and encrypted data."},
    {"id": 14, "color": BLACK, "label": 1, "log_text": "Blackout. Complete power failure across the entire data center in sector 4. Dead."},
    {"id": 15, "color": BLACK, "label": 1, "log_text": "A black screen of death. The kernel panicked and took down the entire cluster."},
    {"id": 16, "color": RED,   "label": 1, "log_text": "Red angry alien invasion destroying the servers and causing massive physical damage."},
    {"id": 17, "color": RED,   "label": 1, "log_text": "Red explosion in the datacenter. Fire and brimstone. Total catastrophic hardware loss."},
    {"id": 18, "color": BLACK, "label": 1, "log_text": "Pitch black terminal. No response from any network interface. We are totally cut off."},
    {"id": 19, "color": RED,   "label": 1, "log_text": "Code red. Storage array offline. Disks are literally melting in the racks."},
    {"id": 20, "color": BLACK, "label": 1, "log_text": "Deep black hole detected in routing table. All packets are being dropped into nothingness."}
]

# ── Metric helpers ────────────────────────────────────────────────────────────
def precision_score(y_true, y_pred):
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fp = sum((t == 0 and p == 1) for t, p in zip(y_true, y_pred))
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall_score(y_true, y_pred):
    tp = sum((t == 1 and p == 1) for t, p in zip(y_true, y_pred))
    fn = sum((t == 1 and p == 0) for t, p in zip(y_true, y_pred))
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def f1_score(prec, rec):
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

# ── Main routine ──────────────────────────────────────────────────────────────
def run_evaluation():
    print("=" * 60)
    print("  MULTIMODAL LOG ANALYZER -- 20-PAIR PIPELINE EVALUATION")
    print("=" * 60)
    print("Encoding 20 image-text pairs...")
    
    # 1. Compute cosine similarities
    similarities = []
    tmp_files = []
    
    for case in TEST_CASES:
        img_path = _make_image(case["color"])
        tmp_files.append(img_path)
        
        img_emb = encode_image(img_path)
        txt_emb = encode_text(case["log_text"])
        
        sim = float(np.dot(img_emb, txt_emb))
        similarities.append(sim)
        
        print(f"  [Pair {case['id']:2d} / 20] Label: {case['label']} | Similarity: {sim:.4f}")

    # Clean up temp images
    for p in tmp_files:
        try: os.unlink(p)
        except OSError: pass

    # 2. Sweep thresholds from 0.15 to 0.40 step 0.01
    print("\n" + "=" * 60)
    print("  THRESHOLD SWEEP (0.15 -> 0.40)")
    print("=" * 60)
    
    y_true = [case["label"] for case in TEST_CASES]
    best_t = 0.0
    best_f1 = -1.0
    best_p = 0.0
    best_r = 0.0
    
    thresholds = np.arange(0.15, 0.41, 0.01)
    
    for t in thresholds:
        y_pred = [1 if sim < t else 0 for sim in similarities]
        
        p = precision_score(y_true, y_pred)
        r = recall_score(y_true, y_pred)
        f1 = f1_score(p, r)
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            best_p = p
            best_r = r
            
        print(f"Threshold: {t:.2f} | P: {p:.4f}  R: {r:.4f}  F1: {f1:.4f}")
        
    print("\n" + "=" * 60)
    print(f"BEST THRESHOLD: {best_t:.2f}")
    print(f"   Precision:      {best_p:.4f}")
    print(f"   Recall:         {best_r:.4f}")
    print(f"   F1 Score:       {best_f1:.4f}")
    print("=" * 60)

    # 3. Save to CSV
    csv_path = Path(__file__).parent / "evaluation_results.csv"
    
    # Best predictions
    best_y_pred = [1 if sim < best_t else 0 for sim in similarities]
    
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Ground_Truth", "Predicted", "Similarity", "Correct", "Log_Text"])
        
        for i, case in enumerate(TEST_CASES):
            pred = best_y_pred[i]
            sim = similarities[i]
            correct = "YES" if pred == case["label"] else "NO"
            gt_str = "ANOMALY" if case["label"] == 1 else "NORMAL"
            pred_str = "ANOMALY" if pred == 1 else "NORMAL"
            
            writer.writerow([
                case["id"],
                gt_str,
                pred_str,
                f"{sim:.4f}",
                correct,
                case["log_text"]
            ])
            
    print(f"Detailed pair results saved to {csv_path.name}")

if __name__ == "__main__":
    run_evaluation()
