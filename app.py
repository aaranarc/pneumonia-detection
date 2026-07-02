"""
app.py
------
Streamlit demo for the Pneumonia Detection model.

How it works:
    1. User uploads a chest X-ray image.
    2. App preprocesses it to match the training-time pipeline.
    3. EfficientNetB0 (fine-tuned) predicts pneumonia probability.
    4. App shows the raw probability and a verdict at the chosen threshold.
    5. User can slide the decision threshold to explore the security-vs-usability
       (in medical terms: sensitivity-vs-specificity) trade-off.

Educational notes shown alongside the prediction:
    - Class imbalance in training data (2.88:1 Pneumonia:Normal)
    - Youden's J threshold selection used during evaluation
    - Model limitations (train/test distribution shift, not for clinical use)
"""

import io
import math

import numpy as np
import streamlit as st
from PIL import Image

from model import (
    CLASS_NAMES,
    classify,
    load_pneumonia_model,
    predict,
    preprocess_image,
)

# -------------------------------------------------------------------
# Page configuration  (MUST be the first Streamlit command)
# -------------------------------------------------------------------
st.set_page_config(
    page_title="PneumoScan",
    page_icon="🫁",
    layout="wide",
)


# -------------------------------------------------------------------
# Cached model loader
# -------------------------------------------------------------------
@st.cache_resource
def get_model():
    return load_pneumonia_model()


# -------------------------------------------------------------------
# Sidebar
# -------------------------------------------------------------------
st.sidebar.title("PneumoScan")
st.sidebar.divider()

threshold = st.sidebar.slider(
    "Decision threshold",
    min_value=0.05,
    max_value=0.95,
    value=0.50,
    step=0.05,
    help=(
        "The probability cutoff above which we flag the X-ray as pneumonia. "
        "Higher threshold = fewer false positives but more missed cases. "
        "Lower threshold = catches more pneumonia but more false alarms. "
        "The notebook used Youden's J statistic to pick an optimal threshold."
    ),
)

st.sidebar.divider()

with st.sidebar.expander("About"):
    st.markdown(
        "**Architecture:** EfficientNetB0 (transfer learning)  \n"
        "**Training:** Two-phase — feature extraction, then fine-tuning "
        "top 30 layers  \n"
        "**Dataset:** Kaggle Chest X-Ray Images — 5,232 train images  \n"
        "**Class balance:** 2.88 : 1 Pneumonia to Normal (class weights)  \n"
        "**Threshold:** Youden's J statistic"
    )
    st.markdown("---")
    st.markdown("**Model Performance**")
    st.markdown(
        "| Metric | Value |\n"
        "|---|---|\n"
        "| Accuracy | 89% |\n"
        "| AUC | 0.99 |\n"
        "| Sensitivity | 88% |\n"
        "| Specificity | 79% |"
    )

with st.sidebar.expander("Disclaimer"):
    st.markdown(
        "**Not for clinical use.** This is an educational demo showing an "
        "applied deep-learning workflow. Real medical diagnosis requires "
        "specialist review of the full clinical picture."
    )

# -------------------------------------------------------------------
# Main area — header with tagline
# -------------------------------------------------------------------
st.title("Pneumonia Detection")
st.markdown(
    "<p style='font-size:13px; color:#8b95a5; margin-top:-10px; "
    "margin-bottom:4px;'>"
    "EfficientNetB0 · Trained on 5,216 chest X-rays · 92% validation accuracy"
    "</p>",
    unsafe_allow_html=True,
)
st.caption(
    "Upload a chest X-ray to classify it as Normal or Pneumonia using a "
    "fine-tuned EfficientNetB0."
)

# -------------------------------------------------------------------
# Styled upload zone
# -------------------------------------------------------------------
st.markdown(
    """
    <div style="
        border: 2px dashed #3a3f4b;
        border-radius: 12px;
        padding: 28px 16px;
        text-align: center;
        margin-bottom: 12px;
        background: #12151c;
        height: 140px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    ">
        <span style="font-size: 40px;">🫁</span>
        <p style="color: #8b95a5; font-size: 14px; margin: 8px 0 0 0;">
            Drop chest X-ray here
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload a chest X-ray (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    help="The model expects a frontal chest X-ray. Other image types will "
         "still produce a prediction but it will not be meaningful.",
)

if uploaded_file is None:
    st.info(
        "Upload a chest X-ray image to get started. Sample images are "
        "available at "
        "[kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia]"
        "(https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)."
    )
    st.stop()

# -------------------------------------------------------------------
# Divider between upload and preview/prediction
# -------------------------------------------------------------------
st.divider()

# -------------------------------------------------------------------
# Image + inference
# -------------------------------------------------------------------
image_bytes = uploaded_file.read()

col_img, col_pred = st.columns(2, gap="large")

with col_img:
    st.subheader("X-ray preview")
    display_image = Image.open(io.BytesIO(image_bytes))
    st.image(display_image, use_container_width=True)

    # Image info pill
    img_info_text = (
        f"{display_image.size[0]} × {display_image.size[1]} px  ·  "
        f"{display_image.mode}  ·  resized to 224 × 224 for inference"
    )
    st.markdown(
        f"<span style='"
        f"display:inline-block; background:#1e222a; color:#8b95a5; "
        f"font-size:11px; padding:4px 10px; border-radius:20px; "
        f"margin-top:4px;"
        f"'>{img_info_text}</span>",
        unsafe_allow_html=True,
    )

with col_pred:
    st.subheader("Prediction")

    with st.spinner("Running model..."):
        model = get_model()
        image_tensor = preprocess_image(io.BytesIO(image_bytes))
        probability = predict(model, image_tensor)
        result = classify(probability, threshold)

    # --- Determine accent color ---
    is_pneumonia = result["verdict"] == "Pneumonia"
    accent_color = "#e74c3c" if is_pneumonia else "#2ecc71"
    verdict = result["verdict"].upper()

    # --- Prediction result card ---
    st.markdown(
        f"""
        <div style="
            background: #161a22;
            border-radius: 12px;
            padding: 24px;
            border-left: 5px solid {accent_color};
            margin-bottom: 16px;
        ">
            <div style="
                font-size: 2.4rem;
                font-weight: 800;
                color: {accent_color};
                letter-spacing: 2px;
                margin-bottom: 6px;
            ">{verdict}</div>
            <div style="
                font-size: 13px;
                color: #8b95a5;
            ">Threshold: {threshold:.2f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Raw probability ---
    st.markdown(
        f"<span style='font-size:2rem; font-weight:700; color:#e4e6eb;'>"
        f"{probability:.3f}</span>",
        unsafe_allow_html=True,
    )
    st.caption("pneumonia probability")

    # --- Circular confidence gauge (SVG) ---
    conf = result["confidence"]
    conf_pct = conf * 100
    radius = 62
    stroke_width = 15
    center = radius + stroke_width
    svg_size = 2 * center
    circumference = 2 * math.pi * radius
    dash_offset = circumference * (1 - conf)
    gauge_color = "#00d4ff"

    st.markdown(
        f"""
        <div style="display:flex; flex-direction:column; align-items:center;
                    margin: 12px 0 8px 0;">
            <svg width="{svg_size}" height="{svg_size}"
                 viewBox="0 0 {svg_size} {svg_size}">
                <!-- Background ring -->
                <circle cx="{center}" cy="{center}" r="{radius}"
                        fill="none" stroke="#2a2e36"
                        stroke-width="{stroke_width}"
                        stroke-linecap="round"/>
                <!-- Foreground arc -->
                <circle cx="{center}" cy="{center}" r="{radius}"
                        fill="none" stroke="{gauge_color}"
                        stroke-width="{stroke_width}"
                        stroke-linecap="round"
                        stroke-dasharray="{circumference}"
                        stroke-dashoffset="{dash_offset:.2f}"
                        transform="rotate(-90 {center} {center})"
                        style="transition: stroke-dashoffset 0.6s ease;"/>
                <!-- Percentage text -->
                <text x="{center}" y="{center - 4}"
                      text-anchor="middle" dominant-baseline="central"
                      fill="#e4e6eb" font-size="28" font-weight="700">
                    {conf_pct:.1f}%
                </text>
                <!-- Label -->
                <text x="{center}" y="{center + 22}"
                      text-anchor="middle" dominant-baseline="central"
                      fill="#8b95a5" font-size="12">
                    confidence
                </text>
            </svg>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --- Metric chips ---
    chip_style = (
        "display:inline-block; background:#1e222a; color:#c0c6d0; "
        "font-size:12px; padding:5px 12px; border-radius:20px; "
        "margin-right:8px; margin-top:8px; letter-spacing:0.3px;"
    )
    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:12px;">
            <span style="{chip_style}">Sensitivity: 88%</span>
            <span style="{chip_style}">Specificity: 79%</span>
            <span style="{chip_style}">AUC: 0.99</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -------------------------------------------------------------------
# Probability scale bar — single small HTML block
# -------------------------------------------------------------------
st.divider()
st.subheader("Probability scale")

prob_pct = probability * 100
thresh_pct = threshold * 100

st.markdown(
    f"""<div style="position:relative; width:100%; height:28px;
        background:linear-gradient(90deg, #00d4ff 0%, #1a1d24 50%, #ffb84d 100%);
        border-radius:6px; margin:0.5rem 0;">
      <div style="position:absolute; top:50%; left:{prob_pct:.1f}%;
          transform:translate(-50%,-50%); width:14px; height:14px;
          background:#e4e6eb; border-radius:50%;
          border:2px solid #0e1117; z-index:2;"></div>
      <div style="position:absolute; top:-2px; left:{thresh_pct:.1f}%;
          transform:translateX(-50%); width:2px; height:32px;
          background:#e4e6eb; opacity:0.6; z-index:1;"></div>
    </div>""",
    unsafe_allow_html=True,
)

lcol, mcol, rcol = st.columns(3)
lcol.caption("0.0 — Normal")
mcol.markdown(
    f"<p style='text-align:center; font-size:0.8rem; color:#888;'>"
    f"threshold = {threshold:.2f}</p>",
    unsafe_allow_html=True,
)
rcol.markdown(
    "<p style='text-align:right; font-size:0.8rem; color:#888;'>"
    "1.0 — Pneumonia</p>",
    unsafe_allow_html=True,
)

# -------------------------------------------------------------------
# How this decision was made
# -------------------------------------------------------------------
st.divider()
st.subheader("How this decision was made")

if probability > threshold:
    st.markdown(
        f"The model's pneumonia probability is **{probability:.3f}**, which is "
        f"**above** the threshold of **{threshold:.2f}** — flagged as **Pneumonia**."
    )
else:
    st.markdown(
        f"The model's pneumonia probability is **{probability:.3f}**, which is "
        f"**at or below** the threshold of **{threshold:.2f}** — classified as **Normal**."
    )

st.markdown(
    "Try moving the threshold slider in the sidebar to see how the decision "
    "boundary changes. In medical screening this trade-off is the difference "
    "between missing pneumonia cases (patient goes untreated) and false "
    "alarms (unnecessary follow-up tests). Clinical systems usually "
    "prioritise sensitivity over specificity because the cost of a missed "
    "diagnosis far exceeds a false alarm."
)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.divider()
st.caption(
    "EfficientNetB0 · 5,232 training images · Youden's J threshold · "
    "2.88 : 1 class weighting · Educational demo — not for clinical use."
)
