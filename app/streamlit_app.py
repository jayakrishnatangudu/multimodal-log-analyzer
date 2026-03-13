"""
streamlit_app.py
----------------
Multimodal Log Analyzer — full Streamlit UI.

Connects:
  • clip_encoder.encode_image()   — CLIP embedding from uploaded network diagram
  • clip_encoder.encode_text()    — CLIP embedding from pasted log text
  • anomaly_detector.detect_anomaly() — cosine similarity + threshold flag
  • llm_interface.explain_anomaly()   — Groq llama3-8b-8192 explanation

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import tempfile
import os
from pathlib import Path

import streamlit as st

# ── make app/ importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from clip_encoder import encode_image, encode_text
from anomaly_detector import detect_anomaly
from llm_interface import explain_anomaly

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Multimodal Log Analyzer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #4F8BF9 0%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
}
.hero-sub {
    color: #94a3b8;
    font-size: 1rem;
    margin-top: 0.2rem;
    margin-bottom: 1.8rem;
}
.section-label {
    font-size: 0.75rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}

/* ── result cards ── */
.result-card {
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
}
.card-normal {
    background: linear-gradient(135deg, #052e16 0%, #14532d 100%);
    border: 1px solid #16a34a;
}
.card-anomaly {
    background: linear-gradient(135deg, #2d0a0a 0%, #450a0a 100%);
    border: 1px solid #dc2626;
}
.badge {
    display: inline-block;
    padding: 0.25rem 0.75rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}
.badge-normal  { background: #16a34a22; color: #4ade80; border: 1px solid #16a34a; }
.badge-anomaly { background: #dc262622; color: #f87171; border: 1px solid #dc2626; }

.score-big {
    font-size: 2.8rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1;
}
.score-normal  { color: #4ade80; }
.score-anomaly { color: #f87171; }

.log-box {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #94a3b8;
    word-break: break-all;
    margin-top: 0.6rem;
}

/* ── sidebar ── */
section[data-testid="stSidebar"] { background: #0f172a; }
</style>
""", unsafe_allow_html=True)

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    threshold = st.slider(
        "Anomaly threshold",
        min_value=0.05, max_value=0.60, value=0.25, step=0.05,
        help="Cosine similarity below this value → anomaly. Default: 0.25",
    )
    use_llm = st.toggle("LLM explanation", value=True,
                        help="Calls Groq llama3-8b-8192. Requires GROQ_API_KEY in .env")
    st.markdown("---")
    st.caption("CLIP · cosine similarity · Groq LLaMA 3")
    st.caption(f"Threshold: **{threshold}**")

# ── hero ──────────────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 Multimodal Log Analyzer</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Upload a network diagram image and paste a log entry — '
    "the AI compares them and flags anomalies using CLIP + Groq.</p>",
    unsafe_allow_html=True,
)

st.divider()

# ── two-column input layout ───────────────────────────────────────────────────
col_img, col_log = st.columns([1, 1], gap="large")

with col_img:
    st.markdown('<p class="section-label">📷 Network Diagram Image</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Upload PNG or JPG",
        type=["png", "jpg", "jpeg"],
        label_visibility="collapsed",
    )
    if uploaded_file:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

with col_log:
    st.markdown('<p class="section-label">📋 Log Text</p>', unsafe_allow_html=True)
    log_text = st.text_area(
        label="Paste log entry",
        height=260,
        placeholder="Paste a single log line or a short block of log text here…\n\nExample:\nCRITICAL: Disk I/O error on /dev/sda1 – bad sector at 0x1A3F",
        label_visibility="collapsed",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── analyse button ────────────────────────────────────────────────────────────
btn_col, _ = st.columns([1, 3])
with btn_col:
    analyse = st.button("🚀 Analyze", type="primary", use_container_width=True)

# ── analysis pipeline ─────────────────────────────────────────────────────────
if analyse:
    # ── validation ────────────────────────────────────────────────────────────
    errors = []
    if not uploaded_file:
        errors.append("Please upload a network diagram image (PNG/JPG).")
    if not log_text.strip():
        errors.append("Please paste some log text.")
    if errors:
        for e in errors:
            st.error(e)
        st.stop()

    st.divider()
    st.markdown("## 📊 Results")

    # ── Step 1: encode image ──────────────────────────────────────────────────
    with st.spinner("Encoding image with CLIP…"):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
        try:
            img_embedding = encode_image(tmp_path)
        finally:
            os.unlink(tmp_path)

    # ── Step 2: encode log text ───────────────────────────────────────────────
    with st.spinner("Encoding log text with CLIP…"):
        txt_embedding = encode_text(log_text.strip())

    # ── Step 3: detect anomaly ────────────────────────────────────────────────
    detection = detect_anomaly(img_embedding, txt_embedding, threshold=threshold)
    similarity  = detection["similarity"]
    is_anomaly  = detection["is_anomaly"]

    card_cls  = "card-anomaly" if is_anomaly else "card-normal"
    badge_cls = "badge-anomaly" if is_anomaly else "badge-normal"
    score_cls = "score-anomaly" if is_anomaly else "score-normal"
    badge_lbl = "⚠️ ANOMALY" if is_anomaly else "✅ NORMAL"
    status_txt = (
        "The image and log text are **semantically mismatched** — this combination is flagged as anomalous."
        if is_anomaly else
        "The image and log text are **semantically consistent** — no anomaly detected."
    )

    # ── Results card ──────────────────────────────────────────────────────────
    res_left, res_right = st.columns([1, 2], gap="large")

    with res_left:
        st.markdown(f"""
        <div class="result-card {card_cls}">
            <div style="margin-bottom:0.6rem">
                <span class="badge {badge_cls}">{badge_lbl}</span>
            </div>
            <div class="score-big {score_cls}">{similarity:.4f}</div>
            <div style="color:#94a3b8;font-size:0.78rem;margin-top:0.4rem">
                cosine similarity
                &nbsp;|&nbsp; threshold {threshold}
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.caption(status_txt)

    with res_right:
        st.markdown('<p class="section-label">Log Entry Analysed</p>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="log-box">{log_text.strip()}</div>',
            unsafe_allow_html=True,
        )

        # ── similarity bar ────────────────────────────────────────────────────
        st.markdown('<p class="section-label" style="margin-top:1rem">Similarity Score</p>',
                    unsafe_allow_html=True)
        bar_val = max(0.0, min(1.0, (similarity + 1) / 2))   # map [-1,1] → [0,1]
        st.progress(bar_val)
        st.caption(
            f"Raw cosine similarity: **{similarity:.4f}**  ·  "
            f"{'Below' if is_anomaly else 'At or above'} anomaly threshold of {threshold}"
        )

    # ── Step 4: LLM explanation ───────────────────────────────────────────────
    if use_llm:
        st.markdown("---")
        st.markdown("### 🤖 LLM Explanation  *(Groq · llama3-8b-8192)*")
        try:
            with st.spinner("Querying Groq llama3-8b-8192…"):
                explanation = explain_anomaly(log_text.strip(), similarity)
            st.markdown(explanation)
        except ValueError as e:
            st.warning(f"LLM unavailable: {e}")
        except Exception as e:
            st.error(f"LLM error: {e}")
    else:
        st.info("LLM explanation disabled. Toggle it on in the sidebar to get AI-powered analysis.")



st.sidebar.title("Quick Test Examples")

examples = {
    "Normal System": {
        "log": "INFO 2024-01-15 08:23:11 NameNode: Block replicated to 3 nodes successfully\nINFO 2024-01-15 08:24:02 DataNode: Heartbeat received from node 192.168.1.10\nINFO 2024-01-15 08:25:30 HDFS: Checkpoint completed, memory usage at 42%",
        "label": "Expected: Normal"
    },
    "Memory Anomaly": {
        "log": "CRITICAL 2024-01-15 09:11:22 System: Memory usage at 97%\nERROR 2024-01-15 09:12:45 NameNode: OutOfMemoryError, heap dump triggered\nCRITICAL 2024-01-15 09:13:10 DataNode: Node 192.168.1.5 unresponsive",
        "label": "Expected: Anomaly"
    },
    "Network Failure": {
        "log": "ERROR 2024-01-15 09:10:05 Network: Packet loss 45%, 3 nodes unreachable\nCRITICAL 2024-01-15 09:11:30 Switch: Port 4 failure detected\nERROR 2024-01-15 09:12:00 DataNode: Failed to connect after 5 retries",
        "label": "Expected: Anomaly"
    },
    "Disk Failure": {
        "log": "ERROR 2024-01-15 10:05:11 DataNode: Disk I/O timeout on node 192.168.1.20\nCRITICAL 2024-01-15 10:06:22 HDFS: Block blk_9999 lost, no replicas found\nERROR 2024-01-15 10:07:45 System: Disk failure, data recovery initiated",
        "label": "Expected: Anomaly"
    }
}

for name, data in examples.items():
    if st.sidebar.button(f"Load: {name}"):
        st.session_state["log_input"] = data["log"]
        st.sidebar.success(data["label"])
