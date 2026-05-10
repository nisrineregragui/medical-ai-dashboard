import streamlit as st
import plotly.graph_objects as go
import numpy as np
from PIL import Image
import io

from utils.mlp_model import predict_breast_cancer, load_mlp_model, get_scaler
from utils.cnn_model import predict_pneumonia, load_cnn_model

#feature definitions
BREAST_FEATURES = [
    ("radius_mean",             "Radius Mean",             14.13),
    ("texture_mean",            "Texture Mean",            19.29),
    ("perimeter_mean",          "Perimeter Mean",          91.97),
    ("area_mean",               "Area Mean",               654.89),
    ("smoothness_mean",         "Smoothness Mean",         0.0964),
    ("compactness_mean",        "Compactness Mean",        0.1043),
    ("concavity_mean",          "Concavity Mean",          0.0888),
    ("concave_points_mean",     "Concave Points Mean",     0.0489),
    ("symmetry_mean",           "Symmetry Mean",           0.1812),
    ("fractal_dim_mean",        "Fractal Dimension Mean",  0.0628),
    ("radius_se",               "Radius SE",               0.4051),
    ("texture_se",              "Texture SE",              1.2169),
    ("perimeter_se",            "Perimeter SE",            2.8661),
    ("area_se",                 "Area SE",                 40.34),
    ("smoothness_se",           "Smoothness SE",           0.0070),
    ("compactness_se",          "Compactness SE",          0.0255),
    ("concavity_se",            "Concavity SE",            0.0319),
    ("concave_points_se",       "Concave Points SE",       0.0118),
    ("symmetry_se",             "Symmetry SE",             0.0200),
    ("fractal_dim_se",          "Fractal Dimension SE",    0.0037),
    ("radius_worst",            "Radius Worst",            16.27),
    ("texture_worst",           "Texture Worst",           25.68),
    ("perimeter_worst",         "Perimeter Worst",         107.26),
    ("area_worst",              "Area Worst",              880.58),
    ("smoothness_worst",        "Smoothness Worst",        0.1323),
    ("compactness_worst",       "Compactness Worst",       0.2543),
    ("concavity_worst",         "Concavity Worst",         0.2722),
    ("concave_points_worst",    "Concave Points Worst",    0.1146),
    ("symmetry_worst",          "Symmetry Worst",          0.2900),
    ("fractal_dim_worst",       "Fractal Dimension Worst", 0.0839),
]


#Helpers
def donut_chart(values, labels, colors, title=""):
    fig = go.Figure(go.Pie(
        values=values,
        labels=labels,
        hole=0.72,
        marker=dict(colors=colors, line=dict(width=0)),
        textinfo="none",
        hovertemplate="%{label}: %{percent:.1%}<extra></extra>",
    ))
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.05, y=0.5,
            font=dict(size=11, color="#4a5568"),
        ),
        margin=dict(l=0, r=90, t=10, b=10),
        height=130,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def mlp_arch_html():
    nodes = [
        ("30",  "Input"),
        ("64",  "ReLU + Dropout\n(0.3)"),
        ("32",  "ReLU + Dropout\n(0.3)"),
        ("2",   "Output\n(Softmax)"),
    ]
    parts = []
    for i, (val, lbl) in enumerate(nodes):
        lbl_html = lbl.replace("\n", "<br>")
        parts.append(f"""
            <div class="arch-node">
                <div class="arch-node-val">{val}</div>
                <div class="arch-node-lbl">{lbl_html}</div>
            </div>""")
        if i < len(nodes) - 1:
            parts.append('<div class="arch-arrow">→</div>')
    return f"""
    <div class="arch-section">
        <div class="arch-title">Model Architecture (MLP)</div>
        <div class="arch-pipeline">{''.join(parts)}</div>
    </div>"""


def cnn_arch_html():
    nodes = [
        ("Conv2D", "64 filters\n5×5"),
        ("BatchNorm", "ReLU"),
        ("Conv2D", "64 filters\n5×5"),
        ("MaxPool", "2×2"),
        ("Conv2D", "64 filters\n1×1"),
        ("FC", "2\n(Output)"),
    ]
    parts = []
    for i, (val, lbl) in enumerate(nodes):
        lbl_html = lbl.replace("\n", "<br>")
        parts.append(f"""
            <div class="arch-node">
                <div class="arch-node-val" style="font-size:12px;">{val}</div>
                <div class="arch-node-lbl">{lbl_html}</div>
            </div>""")
        if i < len(nodes) - 1:
            parts.append('<div class="arch-arrow">→</div>')
    return f"""
    <div class="arch-section">
        <div class="arch-title blue">Model Architecture (FlexCNN)</div>
        <div class="arch-pipeline">{''.join(parts)}</div>
    </div>"""


# Main show()
def show():
    # Pre-load models silently
    load_mlp_model()
    get_scaler()
    load_cnn_model()

    # Page header
    col_title, col_btn = st.columns([5, 1])
    with col_title:
        st.markdown("""
        <div class="page-title">Welcome to Medical AI Diagnosis Dashboard</div>
        <div class="page-subtitle">AI models for early disease detection and classification</div>
        """, unsafe_allow_html=True)

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)

    # Metric row
    m1, m2, m3, m4 = st.columns(4)
    metrics = [
        ("📊", "purple", "2",     "Models\nDeployed"),
        ("🎯", "green",  "98.8%", "Breast Cancer\nValidation Accuracy"),
        ("🫁", "blue",   "88.5%", "Pneumonia\nTest Accuracy"),
        ("📈", "red",    "0.91",  "Pneumonia\nF1-Score"),
    ]
    for col, (icon, color, val, label) in zip([m1, m2, m3, m4], metrics):
        with col:
            label_html = label.replace("\n", "<br>")
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-icon {color}">{icon}</div>
                <div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{label_html}</div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<div style="height:20px"></div>', unsafe_allow_html=True)

    # Two-column panels
    col_left, col_right = st.columns(2, gap="medium")

    # LEFT — Breast Cancer Classification
    with col_left:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-icon purple">🎗️</div>
                <div>
                    <div class="panel-title">Breast Cancer Classification</div>
                    <div class="panel-subtitle">MLP Model for Wisconsin Breast Cancer Dataset</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        # Tabs
        tab_manual, tab_csv = st.tabs(["📋  Manual Input", "📁  CSV Upload"])

        # ── Manual Input tab ──────────────────────────
        with tab_manual:
            st.markdown("""
            <div class="input-section-label">Clinical Metrics Input (30 features) ℹ️</div>
            """, unsafe_allow_html=True)

            # First 6 features
            feat_vals = {}
            cols = st.columns(3)
            for i, (key, label, default) in enumerate(BREAST_FEATURES[:6]):
                with cols[i % 3]:
                    feat_vals[key] = st.number_input(
                        label, value=float(default), format="%.4f",
                        key=f"bc_{key}", label_visibility="visible"
                    )

            # Expander for remaining 24
            with st.expander("➕ Show More Features (24 remaining)"):
                exp_cols = st.columns(3)
                for i, (key, label, default) in enumerate(BREAST_FEATURES[6:]):
                    with exp_cols[i % 3]:
                        feat_vals[key] = st.number_input(
                            label, value=float(default), format="%.4f",
                            key=f"bc_{key}"
                        )

            # Fill any missing (shouldn't happen)
            for key, _, default in BREAST_FEATURES:
                if key not in feat_vals:
                    feat_vals[key] = float(default)

            # Predict button
            st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
            bc_predict = st.button("🔍  Predict", key="bc_predict_manual", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            if bc_predict or st.session_state.get("bc_result_manual"):
                if bc_predict:
                    features = [feat_vals[k] for k, _, _ in BREAST_FEATURES]
                    label, conf, probs = predict_breast_cancer(features)
                    st.session_state["bc_result_manual"] = (label, conf, probs)

                label, conf, probs = st.session_state["bc_result_manual"]
                css_class = "benign" if label == "Benign" else "malignant"
                check = "✅" if label == "Benign" else "❌"
                benign_pct  = probs[1] * 100
                mal_pct     = probs[0] * 100

                res_col, chart_col = st.columns([1, 1])
                with res_col:
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label-title">Prediction Result</div>
                        <div style="font-size:26px;margin:4px 0 2px">{check}</div>
                        <div class="result-label-value {css_class}">{label}</div>
                        <div class="result-confidence">Confidence: {conf*100:.2f}%</div>
                    </div>""", unsafe_allow_html=True)
                with chart_col:
                    fig = donut_chart(
                        [benign_pct, mal_pct],
                        [f"Benign: {benign_pct:.2f}%", f"Malignant: {mal_pct:.2f}%"],
                        ["#22c55e", "#ef4444"],
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # ── CSV Upload tab ────────────────────────────
        with tab_csv:
            st.markdown("""
            <div class="input-hint">Upload a CSV file with 30 feature columns (one row per patient).</div>
            """, unsafe_allow_html=True)
            csv_file = st.file_uploader(
                "Upload CSV", type=["csv"], key="bc_csv",
                label_visibility="collapsed"
            )
            if csv_file:
                import pandas as pd
                try:
                    df = pd.read_csv(csv_file)
                    # Try to grab first 30 numeric columns
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if len(numeric_cols) < 30:
                        st.error(f"Need at least 30 numeric columns, found {len(numeric_cols)}.")
                    else:
                        features = df[numeric_cols[:30]].iloc[0].tolist()
                        st.markdown('<div class="predict-btn">', unsafe_allow_html=True)
                        if st.button("🔍  Predict from CSV", key="bc_predict_csv", use_container_width=True):
                            label, conf, probs = predict_breast_cancer(features)
                            css_class = "benign" if label == "Benign" else "malignant"
                            check = "✅" if label == "Benign" else "❌"
                            benign_pct = probs[1] * 100
                            mal_pct    = probs[0] * 100
                            res_col, chart_col = st.columns([1, 1])
                            with res_col:
                                st.markdown(f"""
                                <div class="result-box">
                                    <div class="result-label-title">Prediction Result</div>
                                    <div style="font-size:26px;margin:4px 0 2px">{check}</div>
                                    <div class="result-label-value {css_class}">{label}</div>
                                    <div class="result-confidence">Confidence: {conf*100:.2f}%</div>
                                </div>""", unsafe_allow_html=True)
                            with chart_col:
                                fig = donut_chart(
                                    [benign_pct, mal_pct],
                                    [f"Benign: {benign_pct:.2f}%", f"Malignant: {mal_pct:.2f}%"],
                                    ["#22c55e", "#ef4444"],
                                )
                                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
                        st.markdown('</div>', unsafe_allow_html=True)
                        st.dataframe(df.head(3), use_container_width=True)
                except Exception as e:
                    st.error(f"Error reading CSV: {e}")

        # Architecture
        st.markdown(mlp_arch_html(), unsafe_allow_html=True)

    # ════════════════════════════════════════════════════
    # RIGHT — Pneumonia Detection
    # ════════════════════════════════════════════════════
    with col_right:
        st.markdown("""
        <div class="panel-card">
            <div class="panel-header">
                <div class="panel-icon blue">🫁</div>
                <div>
                    <div class="panel-title blue">Pneumonia Detection</div>
                    <div class="panel-subtitle">FlexCNN Model for PneumoniaMNIST Dataset</div>
                </div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("**Upload Chest X-ray Image**")

        up_col, prev_col = st.columns([1.4, 1])
        with up_col:
            xray_file = st.file_uploader(
                "Drag and drop an image here or click to browse",
                type=["jpg", "jpeg", "png"],
                key="xray_upload",
                label_visibility="visible"
            )
            st.markdown(
                '<div class="upload-hint">Supports: JPG, PNG, JPEG (Max. 5MB)</div>',
                unsafe_allow_html=True
            )

        with prev_col:
            st.markdown("**Preview**")
            if xray_file:
                img = Image.open(xray_file)
                st.image(img, use_container_width=True)
                # Show processed size
                gray28 = img.convert("L").resize((28, 28))
                st.markdown(
                    '<div class="upload-hint">Processed: 28 × 28 Grayscale</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown("""
                <div style="background:#f8fafc;border-radius:10px;height:140px;
                     display:flex;align-items:center;justify-content:center;
                     color:#c0c9d8;font-size:28px;border:1px solid #e2e8f0;">
                    🫁
                </div>""", unsafe_allow_html=True)

        # Predict button
        st.markdown('<div class="predict-btn-blue">', unsafe_allow_html=True)
        pn_predict = st.button("🔍  Predict", key="pn_predict", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if pn_predict:
            if xray_file is None:
                st.warning("Please upload a chest X-ray image first.")
            else:
                img = Image.open(xray_file)
                label, pneu_prob, probs = predict_pneumonia(img)
                st.session_state["pn_result"] = (label, pneu_prob, probs)

        if st.session_state.get("pn_result"):
            label, pneu_prob, probs = st.session_state["pn_result"]
            css_class = "pneumonia" if label == "Pneumonia" else "normal"
            pneu_pct   = probs[1] * 100
            normal_pct = probs[0] * 100

            res_col2, chart_col2 = st.columns([1, 1])
            with res_col2:
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label-title">Prediction Result</div>
                    <div class="result-label-value {css_class}" style="margin-top:6px;">{label}</div>
                    <div class="result-confidence">Confidence: {pneu_prob*100:.2f}%</div>
                </div>""", unsafe_allow_html=True)
            with chart_col2:
                fig2 = donut_chart(
                    [pneu_pct, normal_pct],
                    [f"Pneumonia: {pneu_pct:.2f}%", f"Normal: {normal_pct:.2f}%"],
                    ["#ef4444", "#22c55e"],
                )
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

        # Architecture
        st.markdown(cnn_arch_html(), unsafe_allow_html=True)

    # Disclaimer
    st.markdown("""
    <div class="disclaimer-box">
        <div class="disclaimer-icon">⚠️</div>
        <div>
            <div class="disclaimer-title">Disclaimer</div>
            <div class="disclaimer-text">
                This application was developed as a university project for educational and research purposes only.
                The AI predictions are experimental and are not intended for real medical diagnosis or clinical use.
            </div>
        </div>
    </div>""", unsafe_allow_html=True)
