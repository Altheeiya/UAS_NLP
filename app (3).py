import streamlit as st
import json
import joblib
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

# ─────────────────────────────────────────────
# Konfigurasi halaman
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Cognitive Distortion Detector",
    page_icon="brain",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-family: 'DM Serif Display', serif;
        font-size: 2.2rem;
        color: #1a1a2e;
        line-height: 1.2;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        font-size: 0.95rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }

    .result-card {
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1.2rem;
        border-left: 5px solid;
    }

    .result-distortion {
        background-color: #fff5f5;
        border-color: #e53e3e;
    }

    .result-clean {
        background-color: #f0fff4;
        border-color: #38a169;
    }

    .result-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #9ca3af;
        margin-bottom: 0.3rem;
    }

    .result-value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.1rem;
    }

    .result-confidence {
        font-size: 0.85rem;
        color: #6b7280;
    }

    .distortion-badge {
        display: inline-block;
        background-color: #e53e3e;
        color: white;
        font-size: 0.8rem;
        font-weight: 600;
        padding: 0.3rem 0.8rem;
        border-radius: 99px;
        margin-top: 0.8rem;
    }

    .metric-row {
        display: flex;
        gap: 1rem;
        margin-top: 0.8rem;
    }

    .metric-box {
        flex: 1;
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        text-align: center;
    }

    .metric-box .metric-val {
        font-size: 1.4rem;
        font-weight: 700;
        color: #1a1a2e;
    }

    .metric-box .metric-lbl {
        font-size: 0.72rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 0.1rem;
    }

    .info-box {
        background: #f8faff;
        border: 1px solid #dbeafe;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
        font-size: 0.88rem;
        color: #374151;
        line-height: 1.6;
    }

    .section-divider {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }

    textarea {
        font-family: 'Inter', sans-serif !important;
        font-size: 0.95rem !important;
    }

    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        font-weight: 600;
        font-size: 0.9rem;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.8rem;
        width: 100%;
        transition: background-color 0.2s;
    }

    .stButton > button:hover {
        background-color: #2d2d4e;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Load model (cache agar tidak reload tiap interaksi)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Memuat model, harap tunggu...")
def load_all():
    with open("streamlit_artifacts/metadata.json", encoding="utf-8") as f:
        meta = json.load(f)

    model_bin    = joblib.load("streamlit_artifacts/model_binary.pkl")
    model_multi  = joblib.load("streamlit_artifacts/model_multi.pkl")
    scaler_bin   = joblib.load("streamlit_artifacts/scaler_binary.pkl")
    scaler_multi = joblib.load("streamlit_artifacts/scaler_multi.pkl")
    le           = joblib.load("streamlit_artifacts/label_encoder.pkl")

    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model_name"])
    indobert  = AutoModel.from_pretrained(meta["model_name"]).to(device).eval()

    return meta, model_bin, model_multi, scaler_bin, scaler_multi, le, tokenizer, indobert, device


meta, model_bin, model_multi, scaler_bin, scaler_multi, le, tokenizer, indobert, device = load_all()


# ─────────────────────────────────────────────
# Fungsi prediksi
# ─────────────────────────────────────────────
def predict(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=meta["max_length"],
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = indobert(**inputs)

    cls_emb        = outputs.last_hidden_state[:, 0, :]
    attention_mask = inputs["attention_mask"].unsqueeze(-1).float()
    mean_emb       = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
    emb            = torch.cat([cls_emb, mean_emb], dim=1).cpu().numpy()

    emb_bin_sc = scaler_bin.transform(emb)
    prob_bin   = model_bin.predict_proba(emb_bin_sc)[0]
    pred_bin   = 1 if prob_bin[1] >= meta["binary_threshold"] else 0

    result = {
        "is_distortion"       : bool(pred_bin),
        "binary_confidence"   : float(prob_bin[pred_bin]),
        "prob_distortion"     : float(prob_bin[1]),
        "distortion_type"     : None,
        "type_confidence"     : None,
        "all_probs"           : None,
    }

    if pred_bin == 1:
        emb_multi_sc      = scaler_multi.transform(emb)
        pred_multi        = model_multi.predict(emb_multi_sc)[0]
        prob_multi        = model_multi.predict_proba(emb_multi_sc)[0]
        result["distortion_type"] = le.classes_[pred_multi]
        result["type_confidence"] = float(prob_multi[pred_multi])
        result["all_probs"]       = {
            cls: float(p) for cls, p in zip(le.classes_, prob_multi)
        }

    return result


# ─────────────────────────────────────────────
# Sidebar: Info model
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Informasi Model")
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    perf = meta["performance"]

    st.markdown("**Binary Classifier**")
    c1, c2 = st.columns(2)
    c1.metric("Test Accuracy", f"{perf['binary']['test_accuracy_opt_threshold']*100:.1f}%")
    c2.metric("AUC-ROC", f"{perf['binary']['auc_roc']:.3f}")

    st.markdown("**Multi-Class Classifier**")
    c3, c4 = st.columns(2)
    c3.metric("Test Accuracy", f"{perf['multi_class']['test_accuracy']*100:.1f}%")
    c4.metric("Kelas", str(len(meta["multi_classes"])))

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("**Arsitektur**")
    st.markdown(f"""
- Embedding: IndoBERT (CLS + Mean Pooling)
- Classifier: XGBoost
- Dimensi: {meta['embedding_dim']} fitur
- Threshold: {meta['binary_threshold']:.3f}
    """)

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    st.markdown("**Kelas Distorsi**")
    for i, cls in enumerate(sorted(meta["multi_classes"]), 1):
        st.markdown(f"{i}. {cls}")


# ─────────────────────────────────────────────
# Halaman utama
# ─────────────────────────────────────────────
st.markdown('<p class="main-title">Cognitive Distortion Detector</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Deteksi pola pikir tidak sehat dalam teks Bahasa Indonesia '
    'menggunakan IndoBERT dan XGBoost.</p>',
    unsafe_allow_html=True,
)

st.markdown("""
<div class="info-box">
Distorsi kognitif adalah pola pikir otomatis yang cenderung tidak akurat dan negatif.
Masukkan teks yang ingin dianalisis — sistem akan mendeteksi apakah terdapat distorsi
kognitif dan mengidentifikasi jenisnya.
</div>
""", unsafe_allow_html=True)

# Input teks
input_text = st.text_area(
    label="Teks untuk dianalisis",
    placeholder="Contoh: Saya merasa tidak ada yang peduli dengan saya dan saya tidak berguna...",
    height=140,
    label_visibility="collapsed",
)

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    run = st.button("Analisis Teks", use_container_width=True)
with col_clear:
    clear = st.button("Hapus", use_container_width=True)

if clear:
    st.rerun()

# ─────────────────────────────────────────────
# Hasil prediksi
# ─────────────────────────────────────────────
if run:
    text = input_text.strip()

    if not text:
        st.warning("Masukkan teks terlebih dahulu.")
    elif len(text.split()) < 3:
        st.warning("Teks terlalu pendek. Masukkan minimal 3 kata.")
    else:
        with st.spinner("Menganalisis..."):
            res = predict(text)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("#### Hasil Analisis")

        if res["is_distortion"]:
            st.markdown(f"""
<div class="result-card result-distortion">
    <div class="result-label">Status</div>
    <div class="result-value">Distorsi Kognitif Terdeteksi</div>
    <div class="result-confidence">Keyakinan model: {res['binary_confidence']*100:.1f}%</div>
    <span class="distortion-badge">{res['distortion_type']}</span>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-val">{res['binary_confidence']*100:.1f}%</div>
        <div class="metric-lbl">Prob. Distorsi</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">{res['type_confidence']*100:.1f}%</div>
        <div class="metric-lbl">Keyakinan Jenis</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">{res['distortion_type']}</div>
        <div class="metric-lbl">Jenis Distorsi</div>
    </div>
</div>
""", unsafe_allow_html=True)

            # Top 3 probabilitas jenis distorsi
            if res["all_probs"]:
                st.markdown("<br>**Probabilitas per kelas (Top 5)**", unsafe_allow_html=True)
                sorted_probs = sorted(res["all_probs"].items(), key=lambda x: x[1], reverse=True)[:5]
                for cls, prob in sorted_probs:
                    is_top = cls == res["distortion_type"]
                    bar_color = "#e53e3e" if is_top else "#cbd5e0"
                    st.markdown(f"""
<div style="margin-bottom:0.5rem;">
    <div style="display:flex; justify-content:space-between; font-size:0.83rem;
                font-weight:{'600' if is_top else '400'}; color:#374151; margin-bottom:3px;">
        <span>{cls}</span>
        <span>{prob*100:.1f}%</span>
    </div>
    <div style="background:#e5e7eb; border-radius:99px; height:6px;">
        <div style="background:{bar_color}; width:{prob*100:.1f}%; height:6px; border-radius:99px;"></div>
    </div>
</div>
""", unsafe_allow_html=True)

        else:
            st.markdown(f"""
<div class="result-card result-clean">
    <div class="result-label">Status</div>
    <div class="result-value">Tidak Terdeteksi Distorsi</div>
    <div class="result-confidence">Keyakinan model: {res['binary_confidence']*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

            st.markdown(f"""
<div class="metric-row">
    <div class="metric-box">
        <div class="metric-val">{(1 - res['prob_distortion'])*100:.1f}%</div>
        <div class="metric-lbl">Prob. Normal</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">{res['prob_distortion']*100:.1f}%</div>
        <div class="metric-lbl">Prob. Distorsi</div>
    </div>
</div>
""", unsafe_allow_html=True)

        # Tampilkan teks yang dianalisis
        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        st.markdown("**Teks yang dianalisis:**")
        st.markdown(
            f'<div style="background:#f9fafb; border:1px solid #e5e7eb; border-radius:8px; '
            f'padding:0.9rem 1.1rem; font-size:0.9rem; color:#374151; line-height:1.7;">'
            f'{text}</div>',
            unsafe_allow_html=True,
        )

# ─────────────────────────────────────────────
# Contoh teks
# ─────────────────────────────────────────────
st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
st.markdown("**Contoh teks untuk dicoba:**")

examples = [
    "Semua orang pasti membenci saya, tidak ada yang bisa dipercaya.",
    "Hari ini saya makan siang bersama teman-teman dan sangat menyenangkan.",
    "Kalau saya gagal sekali, berarti saya memang orang yang tidak kompeten.",
    "Saya tahu dia pasti berpikir buruk tentang saya meskipun dia tidak mengatakannya.",
]

for ex in examples:
    if st.button(ex, key=ex):
        st.session_state["_example"] = ex
        st.rerun()
