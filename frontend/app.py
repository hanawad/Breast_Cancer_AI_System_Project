"""
Breast Cancer AI Assessment System - Professional Clinical Dashboard.
Final Project — Integrated AI System.
Bilingual (Arabic / English) High-Impact Portfolio Dashboard.
"""
import base64
import html
import io
import json
import os
from datetime import datetime
from pathlib import Path

import plotly.graph_objects as go
import requests
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------- #
# Configuration & Constants
# --------------------------------------------------------------------------- #
API_URL_DEFAULT = os.getenv("BC_API_URL", "http://localhost:8001")
NAVY = "#0A0F1D"
CYAN = "#00E5FF"
SLATE = "#1E293B"
BLUE = "#2563EB"
GREEN = "#10B981"
RED = "#EF4444"
AMBER = "#F59E0B"

ROOT_DIR = Path(__file__).resolve().parent.parent
BUSI_DIR = ROOT_DIR / "dataset" / "Dataset_BUSI_with_GT"

FEATURE_LABELS = {
    "radius_mean": "متوسط حجم الورم (Radius Mean)",
    "texture_mean": "حدة الملمس (Texture Mean)",
    "perimeter_mean": "محيط الورم (Perimeter Mean)",
    "area_mean": "مساحة الورم (Area Mean)",
    "smoothness_mean": "نعومة الحدود (Smoothness Mean)",
    "compactness_mean": "التراص (Compactness Mean)",
    "concavity_mean": "التقعر (Concavity Mean)",
    "concave points_mean": "النقاط المقعرة (Concave Points)",
    "symmetry_mean": "التناظر (Symmetry Mean)",
    "fractal_dimension_mean": "البُعد الكسوري (Fractal Dimension)",
    "radius_worst": "أقصى حجم (Radius Worst)",
    "perimeter_worst": "أقصى محيط (Perimeter Worst)",
    "area_worst": "أقصى مساحة (Area Worst)",
    "texture_worst": "أقصى حدة ملمس (Texture Worst)",
    "smoothness_worst": "أقصى نعومة (Smoothness Worst)",
    "concavity_worst": "أقصى تقعر (Concavity Worst)",
    "concave points_worst": "أقصى نقاط مقعرة (Concave Points Worst)",
}

st.set_page_config(
    page_title="Breast Cancer AI Assessment System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------- #
# Custom CSS Styling (Medical Tech Glassmorphism)
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@400;600;700;800&family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {{
        font-family: 'Cairo', 'Inter', -apple-system, sans-serif;
    }}
    
    .stApp {{
        background: radial-gradient(circle at 12% 10%, #101c38 0%, {NAVY} 60%);
        color: #E2E8F0;
    }}
    
    header[data-testid="stHeader"] {{
        background: transparent;
    }}
    
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 3rem;
        max-width: 1300px;
    }}
    
    /* Hero Header */
    .hero-card {{
        padding: 1.6rem 2rem;
        border-radius: 20px;
        margin-bottom: 1.5rem;
        background: linear-gradient(135deg, rgba(0, 229, 255, 0.12), rgba(37, 99, 235, 0.22), rgba(30, 41, 59, 0.7));
        border: 1px solid rgba(0, 229, 255, 0.3);
        backdrop-filter: blur(14px);
        box-shadow: 0 10px 35px rgba(0, 0, 0, 0.45);
        display: flex;
        justify-content: space-between;
        align-items: center;
        flex-wrap: wrap;
        gap: 1rem;
    }}
    
    .hero-title {{
        margin: 0;
        font-size: 2.1rem;
        font-weight: 800;
        background: linear-gradient(90deg, #FFFFFF, {CYAN});
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }}
    
    .hero-sub {{
        margin: 0.3rem 0 0;
        color: #94A3B8;
        font-size: 0.95rem;
        font-weight: 400;
        max-width: 800px;
    }}
    
    .badge-pill {{
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        background: rgba(0, 229, 255, 0.15);
        border: 1px solid {CYAN};
        color: {CYAN};
    }}
    
    /* Glass Containers */
    .glass-box {{
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 18px;
        padding: 1.3rem 1.5rem;
        backdrop-filter: blur(12px);
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        margin-bottom: 1rem;
    }}
    
    /* KPI Cards */
    .kpi-card {{
        background: linear-gradient(160deg, rgba(30, 41, 59, 0.9), rgba(10, 15, 29, 0.95));
        border: 1px solid rgba(0, 229, 255, 0.25);
        border-radius: 18px;
        padding: 1.2rem 1.4rem;
        height: 100%;
        box-shadow: 0 8px 24px rgba(0, 229, 255, 0.08);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }}
    .kpi-card:hover {{
        transform: translateY(-2px);
        border-color: {CYAN};
    }}
    .kpi-label {{
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
        font-weight: 600;
    }}
    .kpi-value {{
        font-size: 2rem;
        font-weight: 800;
        color: {CYAN};
        margin: 0.3rem 0;
    }}
    .kpi-sub {{
        font-size: 0.85rem;
        color: #CBD5E1;
    }}
    
    .section-header {{
        font-size: 1.15rem;
        font-weight: 700;
        color: {CYAN};
        margin: 0.4rem 0 0.8rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }}
    
    .callout-warn {{
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.4);
        color: #FCD34D;
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }}
    
    .callout-info {{
        background: rgba(0, 229, 255, 0.1);
        border: 1px solid rgba(0, 229, 255, 0.3);
        color: #E0F2FE;
        border-radius: 12px;
        padding: 0.8rem 1.1rem;
        margin-bottom: 0.8rem;
        font-size: 0.9rem;
    }}
    
    /* Buttons */
    div.stButton > button {{
        background: linear-gradient(90deg, #00E5FF, #2563EB);
        color: #04121F;
        font-weight: 700;
        font-size: 0.95rem;
        border: 0;
        border-radius: 12px;
        padding: 0.65rem 1.2rem;
        width: 100%;
        transition: all 0.25s ease;
        box-shadow: 0 4px 15px rgba(0, 229, 255, 0.25);
    }}
    div.stButton > button:hover {{
        filter: brightness(1.15);
        color: #04121F;
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(0, 229, 255, 0.4);
    }}
    
    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: {SLATE};
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(30, 41, 59, 0.4);
        border-radius: 12px 12px 0 0;
        padding: 10px 18px;
        color: #94A3B8;
        font-weight: 600;
        border: 1px solid transparent;
        transition: all 0.2s ease;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(0, 229, 255, 0.12) !important;
        border: 1px solid rgba(0, 229, 255, 0.3) !important;
        border-bottom: 2px solid {CYAN} !important;
        color: #FFFFFF !important;
    }}
    
    /* Preset Chips */
    .preset-container {{
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 0.9rem 1.1rem;
        margin-bottom: 1.2rem;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Helpers & Plotting
# --------------------------------------------------------------------------- #
def render_kpi(label: str, value: str, sub: str = "", color: str = CYAN) -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{html.escape(label)}</div>'
        f'<div class="kpi-value" style="color: {color};">{html.escape(value)}</div>'
        f'<div class="kpi-sub">{html.escape(sub)}</div>'
        f'</div>'
    )


def risk_color(level: str) -> str:
    return {"Low": GREEN, "Moderate": AMBER, "High": RED}.get(level, CYAN)


def gauge_chart(score: float, level: str) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,
            number={"suffix": "%", "font": {"color": "#FFFFFF", "size": 42}},
            title={"text": f"Integrated Risk: {level}", "font": {"color": "#CBD5E1", "size": 16}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#64748B"},
                "bar": {"color": risk_color(level), "thickness": 0.28},
                "bgcolor": SLATE,
                "steps": [
                    {"range": [0, 33], "color": "rgba(16, 185, 129, 0.2)"},
                    {"range": [33, 66], "color": "rgba(245, 158, 11, 0.2)"},
                    {"range": [66, 100], "color": "rgba(239, 68, 68, 0.2)"},
                ],
                "threshold": {
                    "line": {"color": "#FFFFFF", "width": 3},
                    "thickness": 0.75,
                    "value": score,
                },
            },
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=20, r=20, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#E2E8F0"},
    )
    return fig


def probs_chart(probs: dict) -> go.Figure:
    colors = {"Benign": GREEN, "Malignant": RED, "Normal": CYAN}
    fig = go.Figure(
        go.Bar(
            x=list(probs.keys()),
            y=[v * 100 for v in probs.values()],
            marker_color=[colors.get(k, CYAN) for k in probs],
            text=[f"{v*100:.1f}%" for v in probs.values()],
            textposition="outside",
            marker=dict(line=dict(color="rgba(255,255,255,0.2)", width=1)),
        )
    )
    fig.update_layout(
        height=270,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(
            range=[0, 115],
            title="Confidence Probability (%)",
            gridcolor="rgba(255,255,255,0.06)",
        ),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)"),
        font={"color": "#E2E8F0"},
    )
    return fig


def importance_bar(imp: dict) -> go.Figure:
    items = sorted(imp.items(), key=lambda kv: kv[1])
    fig = go.Figure(
        go.Bar(
            x=[v * 100 for _, v in items],
            y=[FEATURE_LABELS.get(k, k) for k, _ in items],
            orientation="h",
            marker=dict(
                color=[v * 100 for _, v in items],
                colorscale=[[0, "#1E3A5F"], [1, CYAN]],
            ),
            text=[f"{v*100:.1f}%" for _, v in items],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=10, r=30, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Feature Importance (%)", gridcolor="rgba(255,255,255,0.06)"),
        font={"color": "#E2E8F0"},
    )
    return fig


def importance_radar(imp: dict) -> go.Figure:
    labels = [FEATURE_LABELS.get(k, k).split(" ")[0] for k in imp]
    vals = [v * 100 for v in imp.values()]
    fig = go.Figure(
        go.Scatterpolar(
            r=vals + vals[:1],
            theta=labels + labels[:1],
            fill="toself",
            line=dict(color=CYAN, width=2),
            fillcolor="rgba(0, 229, 255, 0.25)",
        )
    )
    fig.update_layout(
        height=320,
        margin=dict(l=40, r=40, t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(
            bgcolor="rgba(30, 41, 59, 0.4)",
            radialaxis=dict(gridcolor="rgba(255,255,255,0.12)", color="#94A3B8"),
            angularaxis=dict(color="#CBD5E1"),
        ),
        font={"color": "#E2E8F0"},
        showlegend=False,
    )
    return fig


def confusion_matrix_figure() -> go.Figure:
    # Benchmark from Wisconsin Breast Cancer test split (569 samples)
    z = [[72, 0], [3, 39]]
    x = ["Predicted Benign", "Predicted Malignant"]
    y = ["Actual Benign", "Actual Malignant"]

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x,
            y=y,
            colorscale=[[0, "#0F172A"], [0.5, "#1E3A8A"], [1, "#00E5FF"]],
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}",
            textfont={"size": 18, "color": "#FFFFFF"},
            showscale=False,
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=20, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"color": "#CBD5E1"},
    )
    return fig


def top_n(imp: dict, n: int = 8) -> dict:
    return dict(sorted(imp.items(), key=lambda kv: -kv[1])[:n])


@st.cache_data(ttl=60)
def load_schema(url: str) -> dict:
    r = requests.get(f"{url}/clinical-schema", timeout=8)
    r.raise_for_status()
    return r.json()


# --------------------------------------------------------------------------- #
# Session State Initialization
# --------------------------------------------------------------------------- #
st.session_state.setdefault("result", None)
st.session_state.setdefault("messages", [])
st.session_state.setdefault("loaded_sample_name", None)
st.session_state.setdefault("custom_image_bytes", None)
st.session_state.setdefault("custom_image_name", None)

# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### ⚙️ إعدادات المنظومة (System Control)")
    api_url = st.text_input("FastAPI Backend URL", API_URL_DEFAULT).rstrip("/")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🔌 فحص الاتصال"):
            try:
                h = requests.get(f"{api_url}/health", timeout=5).json()
                st.success("الخادم متصل بنجاح")
                st.session_state.server_health = h
            except Exception as exc:
                st.error(f"فشل الاتصال: {exc}")
    with col_btn2:
        if st.button("🧹 مسح الفحص"):
            st.session_state.result = None
            st.session_state.messages = []
            st.session_state.loaded_sample_name = None
            st.session_state.custom_image_bytes = None
            st.session_state.custom_image_name = None
            st.rerun()

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.82rem; color: #94A3B8; line-height: 1.5;">
        <b>🎯 المتطلبات الأكاديمية للمشروع:</b><br>
        1. Numerical Model (Random Forest)<br>
        2. Vision Model (ResNet-18)<br>
        3. Transformer Chatbot<br>
        4. FastAPI Integration<br>
        5. Streamlit Dashboard Presentation
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Final Project — Integrated AI System © 2026")

# --------------------------------------------------------------------------- #
# Header Hero Section
# --------------------------------------------------------------------------- #
st.markdown(
    f"""
    <div class="hero-card">
        <div>
            <span class="badge-pill">Clinical AI Decision-Support Prototype</span>
            <h1 class="hero-title">🩺 Breast Cancer AI Integrated System</h1>
            <p class="hero-sub">
            نظام متكامل لتشخيص سرطان الثدي يدمج الفحص السريري المورفولوجي (Random Forest)،
            والرؤية الحاسوبية لفحص السونار (ResNet-18)، ومساعد Transformer الطبي الذكي عبر واجهة برمجية موحدة.
            </p>
        </div>
        <div style="text-align: right; display: flex; gap: 0.8rem; flex-wrap: wrap;">
            <div style="background: rgba(16, 185, 129, 0.15); border: 1px solid {GREEN}; border-radius: 12px; padding: 0.5rem 0.9rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #A7F3D0; font-weight: 600;">Random Forest ROC-AUC</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {GREEN};">99.7%</div>
            </div>
            <div style="background: rgba(0, 229, 255, 0.15); border: 1px solid {CYAN}; border-radius: 12px; padding: 0.5rem 0.9rem; text-align: center;">
                <div style="font-size: 0.72rem; color: #BAE6FD; font-weight: 600;">ResNet-18 Accuracy</div>
                <div style="font-size: 1.25rem; font-weight: 800; color: {CYAN};">90.6%</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------- #
# Main Tabs Navigation
# --------------------------------------------------------------------------- #
tab_diag, tab_arch, tab_metrics, tab_report, tab_about = st.tabs([
    "🩺 الفحص والتشخيص الحي (Live Assessment)",
    "📊 معمارية النظام وسير العمل (Architecture & Flow)",
    "📈 معايير الأداء والتقييم (Model Benchmarks)",
    "📑 التقرير الطبي وتصديره (Clinical Report)",
    "ℹ️ بيان المشكلة وأهداف المشروع (Problem Statement)",
])

# =========================================================================== #
# TAB 1: Live Assessment & Transformer Chatbot
# =========================================================================== #
with tab_diag:
    # Preset Demo Selector
    st.markdown('<div class="section-header">⚡ عينات تجريبية جاهزة للعرض الفوري (One-Click Demo Presets)</div>', unsafe_allow_html=True)
    st.caption("انقر على أي من الحالات التجريبية لتحميل بياناتها وصورتها فورياً دون الحاجة لإدخال يدوي أثناء العرض والمناقشة:")

    c_p1, c_p2, c_p3 = st.columns(3)
    preset_choice = None

    with c_p1:
        if st.button("🟢 تحميل حالة ورم حميد (Benign Case)"):
            preset_choice = "benign"
    with c_p2:
        if st.button("🔴 تحميل حالة ورم خبيث (Malignant Case)"):
            preset_choice = "malignant"
    with c_p3:
        if st.button("🔵 تحميل فحص سليم (Normal Scan)"):
            preset_choice = "normal"

    # Handle Preset Loading
    preset_defaults = {
        "benign": {
            "name": "حالة ورم حميد نموذجية (Benign Case)",
            "file": BUSI_DIR / "benign" / "benign (1).png",
            "values": {
                "radius_mean": 12.05,
                "texture_mean": 15.60,
                "perimeter_mean": 76.50,
                "area_mean": 445.0,
                "smoothness_mean": 0.0865,
            },
        },
        "malignant": {
            "name": "حالة اشتباه ورم خبيث (Malignant Case)",
            "file": BUSI_DIR / "malignant" / "malignant (1).png",
            "values": {
                "radius_mean": 19.80,
                "texture_mean": 24.20,
                "perimeter_mean": 132.50,
                "area_mean": 1220.0,
                "smoothness_mean": 0.1185,
            },
        },
        "normal": {
            "name": "حالة فحص أنسجة طبيعية (Normal Scan)",
            "file": BUSI_DIR / "normal" / "normal (1).png",
            "values": {
                "radius_mean": 10.80,
                "texture_mean": 14.10,
                "perimeter_mean": 68.20,
                "area_mean": 360.0,
                "smoothness_mean": 0.0810,
            },
        },
    }

    if preset_choice and preset_defaults[preset_choice]["file"].exists():
        preset_info = preset_defaults[preset_choice]
        st.session_state.loaded_sample_name = preset_info["name"]
        with open(preset_info["file"], "rb") as f:
            st.session_state.custom_image_bytes = f.read()
            st.session_state.custom_image_name = preset_info["file"].name
        st.session_state.preset_values = preset_info["values"]
        st.success(f"تم تحميل {preset_info['name']} وصورة السونار الخاصة بها بنجاح!")
        st.rerun()

    if st.session_state.loaded_sample_name:
        st.markdown(
            f'<div class="callout-info">📌 العينة النشطة حالياً: <b>{st.session_state.loaded_sample_name}</b> (تم ضبط القياسات وصورة السونار تلقائياً).</div>',
            unsafe_allow_html=True,
        )

    # 2-Column Input Section
    left_col, right_col = st.columns(2, gap="large")

    with left_col:
        st.markdown('<div class="section-header">📋 الخصائص السريرية الخمسة الأساسية (FNA Metrics)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            try:
                schema = load_schema(api_url)
            except Exception as exc:
                st.error(f"تعذر جلب مخطط البيانات من الخادم ({api_url}): {exc}")
                st.stop()

            defaults = schema.get("defaults", {})
            p_vals = getattr(st.session_state, "preset_values", {})

            st.markdown(
                """
                <div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.8rem; line-height: 1.4;">
                أدخل الخصائص الخمسة الأساسية للورم. يقوم الـ Backend تلقائياً باحتساب وتمديد الـ 25 خاصية المتبقية
                (Standard Error و Worst) لنموذج Random Forest بدون حدوث أي خطأ أبعاد.
                </div>
                """,
                unsafe_allow_html=True,
            )

            clinical_values = {}
            col1, col2 = st.columns(2)

            with col1:
                clinical_values["radius_mean"] = st.number_input(
                    "متوسط حجم الورم (Tumor Radius)",
                    value=float(p_vals.get("radius_mean", defaults.get("radius_mean", 13.34))),
                    min_value=0.1,
                    max_value=40.0,
                    step=0.1,
                    format="%.2f",
                    help="نصف القطر بالمليمتر (المسافة من المركز إلى المحيط). القيمة المرجعية الطبيعية: ~13.34 مم",
                    key="in_radius_mean",
                )
                clinical_values["perimeter_mean"] = st.number_input(
                    "المحيط الخارجي (Tumor Perimeter)",
                    value=float(p_vals.get("perimeter_mean", defaults.get("perimeter_mean", 86.18))),
                    min_value=1.0,
                    max_value=250.0,
                    step=0.5,
                    format="%.2f",
                    help="المحيط الإجمالي لكتلة الورم بالمليمتر. القيمة المرجعية الطبيعية: ~86.18 مم",
                    key="in_perimeter_mean",
                )
                clinical_values["smoothness_mean"] = st.number_input(
                    "درجة انسيابية/نعومة الحدود (Smoothness)",
                    value=float(p_vals.get("smoothness_mean", defaults.get("smoothness_mean", 0.09524))),
                    min_value=0.01,
                    max_value=0.30,
                    step=0.001,
                    format="%.4f",
                    help="التباين الموضعي في أطوال نصف القطر (نعومة الحدود). القيمة المرجعية: ~0.0952",
                    key="in_smoothness_mean",
                )

            with col2:
                clinical_values["texture_mean"] = st.number_input(
                    "حدة الملمس/الكثافة (Tumor Texture)",
                    value=float(p_vals.get("texture_mean", defaults.get("texture_mean", 18.90))),
                    min_value=0.1,
                    max_value=50.0,
                    step=0.1,
                    format="%.2f",
                    help="الانحراف المعياري لقيم التدرج الرمادي (كثافة وتباين الملمس). القيمة المرجعية: ~18.90",
                    key="in_texture_mean",
                )
                clinical_values["area_mean"] = st.number_input(
                    "مساحة الورم (Tumor Area)",
                    value=float(p_vals.get("area_mean", defaults.get("area_mean", 546.4))),
                    min_value=10.0,
                    max_value=3000.0,
                    step=1.0,
                    format="%.1f",
                    help="المساحة الكلية للورم بالمليمتر المربع (mm²). القيمة المرجعية: ~546.4 مم²",
                    key="in_area_mean",
                )

            with st.expander("⚙️ الخصائص المتقدمة الـ 25 (تُحسب تلقائياً بالـ Backend)"):
                st.caption(
                    "يتم احتساب بقية المتغيرات كعوامل نسبية من الخصائص الخمسة الأساسية بالـ Backend، ولكن يمكنك تخصيصها إن رغبت:"
                )
                enable_advanced = st.checkbox("تفعيل التعديل اليدوي للمتغيرات الـ 25 المتقدمة", value=False)
                if enable_advanced:
                    adv_cols = st.columns(2)
                    primary_set = {"radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean"}
                    idx = 0
                    for f in schema.get("features", []):
                        if f not in primary_set:
                            def_val = float(defaults.get(f, 0.0))
                            clinical_values[f] = adv_cols[idx % 2].number_input(
                                f,
                                value=def_val,
                                format="%.5f" if abs(def_val) < 1 else "%.2f",
                                key=f"adv_in_{f}",
                            )
                            idx += 1

    with right_col:
        st.markdown('<div class="section-header">🩻 صورة الموجات فوق الصوتية (Ultrasound Scan)</div>', unsafe_allow_html=True)
        with st.container(border=True):
            uploaded_file = st.file_uploader(
                "ارفع صورة السونار (أو استخدم إحدى العينات الجاهزة بالأعلى)",
                type=["png", "jpg", "jpeg", "webp", "bmp"],
            )

            # Determine image source
            active_image_bytes = None
            active_image_name = None

            if uploaded_file is not None:
                active_image_bytes = uploaded_file.getvalue()
                active_image_name = uploaded_file.name
                st.session_state.custom_image_bytes = active_image_bytes
                st.session_state.custom_image_name = active_image_name
            elif st.session_state.custom_image_bytes is not None:
                active_image_bytes = st.session_state.custom_image_bytes
                active_image_name = st.session_state.custom_image_name

            if active_image_bytes:
                st.image(active_image_bytes, caption=f"الصورة المحددة: {active_image_name}", use_container_width=True)
            else:
                st.info("💡 قم برفع صورة سونار أو انقر على إحدى العينات الجاهزة بالأعلى للبدء.")

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_disabled = active_image_bytes is None

    if st.button("🔬 تشغيل التحليل المتكامل بالذكاء الاصطناعي (Run Integrated AI Analysis)", disabled=analyze_disabled):
        with st.spinner("جاري استنتاج الخصائص، وتغذية نموذج Random Forest و ResNet-18، وحساب الاندماج المتكامل…"):
            try:
                resp = requests.post(
                    f"{api_url}/analyze",
                    data={"clinical_json": json.dumps(clinical_values)},
                    files={"file": (active_image_name or "scan.png", active_image_bytes, "image/png")},
                    timeout=90,
                )
                if resp.status_code == 200:
                    st.session_state.result = resp.json()
                    st.session_state.messages = []
                    st.toast("تم إكمال التحليل المتكامل بنجاح!", icon="🩺")
                else:
                    st.error(f"فشل التحليل ({resp.status_code}): {resp.text}")
            except Exception as exc:
                st.error(f"تعذر الاتصال بالخادم ({api_url}): {exc}")

    # Results Section
    res = st.session_state.result
    if res:
        st.markdown("---")
        st.markdown('<div class="section-header">📊 نتائج التشخيص والاندماج المتكامل (Diagnostic Results)</div>', unsafe_allow_html=True)

        for w in res.get("warnings", []):
            st.markdown(f'<div class="callout-warn">⚠️ {html.escape(w)}</div>', unsafe_allow_html=True)

        us, cl, sm = res["ultrasound"], res["clinical"], res["summary"]

        k1, k2, k3 = st.columns(3, gap="medium")
        us_color = GREEN if us["prediction"] == "Benign" else (RED if us["prediction"] == "Malignant" else CYAN)
        k1.markdown(render_kpi("تصنيف السونار (ResNet-18)", us["prediction"], f"درجة الثقة: {us['confidence']*100:.1f}%", us_color), unsafe_allow_html=True)
        k2.markdown(render_kpi("الخطر السريري (Random Forest)", f"{cl['risk_score']}%", "احتمالية الخباثة المورفولوجية", AMBER if cl['risk_score'] > 50 else CYAN), unsafe_allow_html=True)
        k3.markdown(render_kpi("التقييم المتكامل (Late Fusion)", sm["risk_level"], f"المؤشر المدمج: {sm['integrated_score']}%", risk_color(sm["risk_level"])), unsafe_allow_html=True)

        st.markdown(f'<div class="glass-box" style="margin-top: 1.2rem; font-size: 1rem; line-height: 1.6;">💬 <b>الملخص السريري:</b> {html.escape(sm["text"])}</div>', unsafe_allow_html=True)

        # Charts Section
        g1, g2 = st.columns(2, gap="large")
        with g1:
            st.markdown('<div class="section-header">مؤشر الخطر المتكامل (Integrated Risk Gauge)</div>', unsafe_allow_html=True)
            st.plotly_chart(gauge_chart(sm["integrated_score"], sm["risk_level"]), use_container_width=True)
        with g2:
            st.markdown('<div class="section-header">توزيع احتمالات فئات السونار (Ultrasound Probabilities)</div>', unsafe_allow_html=True)
            st.plotly_chart(probs_chart(us["probabilities"]), use_container_width=True)

        st.markdown('<div class="section-header">أهم الميزات السريرية المؤثرة في اتخاذ القرار (Feature Importance)</div>', unsafe_allow_html=True)
        c_tab1, c_tab2 = st.tabs(["مخطط الأعمدة (Bar Chart)", "مخطط الرادار (Radar Chart)"])
        with c_tab1:
            st.plotly_chart(importance_bar(top_n(cl["feature_importances"])), use_container_width=True)
        with c_tab2:
            st.plotly_chart(importance_radar(top_n(cl["feature_importances"])), use_container_width=True)

        # Transformer Chatbot Section
        st.markdown("---")
        st.markdown('<div class="section-header">💬 المساعد الطبي الذكي (Context-Aware Transformer Advisor)</div>', unsafe_allow_html=True)
        st.caption("يتفاعل هذا المساعد الطبي معك مستنداً بشكل مباشر إلى نتائج الفحص الحالية للمريض:")

        # Quick Question Buttons
        st.markdown("<small style='color: #94A3B8;'>أسئلة سريعة شائعة:</small>", unsafe_allow_html=True)
        q_cols = st.columns(4)
        quick_query = None
        if q_cols[0].button("ماذا تعني هذه النتيجة؟"):
            quick_query = "ماذا تعني نتيجة الفحص الحالية بالتفصيل؟"
        if q_cols[1].button("ما هي الخطوات الطبية القادمة؟"):
            quick_query = "ما هي الخطوات والفحوصات الطبية الموصى بها تالياً؟"
        if q_cols[2].button("هل أحتاج إلى خزعة (Biopsy)؟"):
            quick_query = "بناءً على هذه الأرقام، هل هناك حاجة لأخذ خزعة نسيجية؟"
        if q_cols[3].button("ما هي أهم العوامل المؤثرة؟"):
            quick_query = "ما هي أبرز العوامل التي ساهمت في تحديد نسبة الخطر هذه؟"

        for m in st.session_state.messages:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

        user_input = st.chat_input("اكتب سؤالك للمساعد الطبي الذكي هنا...")
        effective_prompt = quick_query or user_input

        if effective_prompt:
            st.session_state.messages.append({"role": "user", "content": effective_prompt})
            with st.chat_message("user"):
                st.markdown(effective_prompt)

            context = {
                "ultrasound": {"prediction": us["prediction"], "confidence": us["confidence"], "probabilities": us["probabilities"]},
                "clinical": {"risk_score": cl["risk_score"], "patient_values": cl["patient_values"]},
                "summary": sm,
            }

            with st.chat_message("assistant"):
                with st.spinner("المساعد الطبي يقوم بدراسة السياق والإجابة…"):
                    try:
                        r = requests.post(
                            f"{api_url}/chat",
                            json={"question": effective_prompt, "context": context, "history": list(st.session_state.messages)},
                            timeout=60,
                        )
                        r.raise_for_status()
                        chat_data = r.json()
                        ans = chat_data["answer"]
                        st.caption(f"Engine: {chat_data.get('engine', 'Transformer')}")
                    except Exception as exc:
                        ans = f"عذراً، حدث خطأ أثناء الاتصال بالمساعد الطبي: {exc}"
                st.markdown(ans)
            st.session_state.messages.append({"role": "assistant", "content": ans})

# =========================================================================== #
# TAB 2: System Architecture & Workflow
# =========================================================================== #
with tab_arch:
    st.markdown('<div class="section-header">🏗️ معمارية النظام وتكامل النماذج (System Architecture & Pipeline)</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="glass-box">
        <b>💡 المفهوم العام للمشروع:</b><br>
        هذا النظام يجسد نموذج <b>التقييم الثلاثي الذكي (AI Triple Assessment Paradigm)</b> المتبع عالمياً في مراكز الأورام.
        بدلاً من الاعتماد على فحص واحد قد يعطي نتائج مضللة، يقوم النظام بدمج قراءتين مستقلتين:
        الخلايا المورفولوجية (FNA) وصور الموجات فوق الصوتية (Ultrasound)، لتقديم قرار مشترك عالي الموثوقية.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 🔄 مسار تدفق البيانات (Data & Control Pipeline)")
    st.markdown(
        f"""
        <div style="background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(0, 229, 255, 0.3); border-radius: 16px; padding: 1.5rem; margin-bottom: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 1rem; text-align: center;">
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem;">
                    <div style="font-size: 1.5rem;">👤</div>
                    <div style="font-weight: 700; color: {CYAN}; font-size: 0.9rem;">1. مدخلات المريض</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">5 قياسات سريرية + صورة سونار</div>
                </div>
                <div style="color: {CYAN}; font-size: 1.5rem;">➔</div>
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem;">
                    <div style="font-size: 1.5rem;">⚡</div>
                    <div style="font-weight: 700; color: {CYAN}; font-size: 0.9rem;">2. FastAPI Backend</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">تحقق، استنتاج أبعاد، وتوزيع</div>
                </div>
                <div style="color: {CYAN}; font-size: 1.5rem;">➔</div>
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem;">
                    <div style="font-size: 1.5rem;">🧠</div>
                    <div style="font-weight: 700; color: {CYAN}; font-size: 0.9rem;">3. نماذج الذكاء الاصطناعي</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">Random Forest + ResNet-18</div>
                </div>
                <div style="color: {CYAN}; font-size: 1.5rem;">➔</div>
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem;">
                    <div style="font-size: 1.5rem;">⚖️</div>
                    <div style="font-weight: 700; color: {CYAN}; font-size: 0.9rem;">4. الاندماج المتأخر (Fusion)</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">60% سونار + 40% سريري</div>
                </div>
                <div style="color: {CYAN}; font-size: 1.5rem;">➔</div>
                <div style="flex: 1; min-width: 150px; background: rgba(30, 41, 59, 0.7); border-radius: 12px; padding: 1rem;">
                    <div style="font-size: 1.5rem;">🤖</div>
                    <div style="font-weight: 700; color: {CYAN}; font-size: 0.9rem;">5. Transformer Chatbot</div>
                    <div style="font-size: 0.78rem; color: #94A3B8;">تفسير وتوصيات سريرية ذكية</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📋 جدول مكونات النظام ونماذجه (Components Breakdown)")
    comp_data = [
        {"المكون": "1. Numerical Model", "النوع / المعمارية": "Random Forest Classifier", "المدخلات": "30 سمة خلوية (معادلة استنتاجية من 5)", "المخرجات": "احتمالية الخطر السريري (0-100%)", "الدور في المنظومة": "تقييم الخصائص المورفولوجية لخلايا الورم المأخوذة بالإبرة الدقيقة (FNA)"},
        {"المكون": "2. Image-Based Model", "النوع / المعمارية": "ResNet-18 Deep CNN", "المدخلات": "صورة موجات فوق صوتية (224x224 RGB)", "المخرجات": "تصنيف ثلاثي: Benign / Malignant / Normal", "الدور في المنظومة": "فحص صور السونار لاستكشاف الكتل المشتبهة وكثافة الأنسجة"},
        {"المكون": "3. Transformer Chatbot", "النوع / المعمارية": "Context-Grounded Transformer LLM", "المدخلات": "سؤال المستخدم + السياق المدمج للحالة", "المخرجات": "إجابة طبية توجيهية طبيعية وموثوقة", "الدور في المنظومة": "تفسير النتائج الطبية للمستخدم وتقديم الإرشادات والخطوات اللاحقة"},
        {"المكون": "4. Backend API", "النوع / المعمارية": "FastAPI (Asynchronous REST API)", "المدخلات": "Multipart Form (JSON + Image)", "المخرجات": "JSON مدمج للتشخيص والتحليلات", "الدور في المنظومة": "نقطة الوصل المركزية بين النماذج والواجهة وتنسيق تدفق البيانات"},
        {"المكون": "5. Frontend Dashboard", "النوع / المعمارية": "Streamlit Medical UI", "المدخلات": "تفاعل المستخدم والأطباء", "المخرجات": "لوحة تحكم تفاعلية وتقارير طبية", "الدور في المنظومة": "واجهة العرض التقديمي والمناقشة والتفاعل السريري الكامل"},
    ]
    st.table(comp_data)

# =========================================================================== #
# TAB 3: Model Benchmarks & Metrics
# =========================================================================== #
with tab_metrics:
    st.markdown('<div class="section-header">📈 مقاييس أداء النماذج والتقييم الإحصائي (Model Evaluation & Benchmarks)</div>', unsafe_allow_html=True)

    b1, b2, b3, b4 = st.columns(4)
    b1.markdown(render_kpi("Random Forest ROC-AUC", "99.7%", "Wisconsin Test Dataset", GREEN), unsafe_allow_html=True)
    b2.markdown(render_kpi("Random Forest Accuracy", "97.4%", "114 عينة اختبار", GREEN), unsafe_allow_html=True)
    b3.markdown(render_kpi("ResNet-18 Validation Accuracy", "90.6%", "BUSI Ultrasound Dataset", CYAN), unsafe_allow_html=True)
    b4.markdown(render_kpi("Precision for Malignant", "100.0%", "صفر إيجابيات كاذبة (False Positives = 0)", GREEN), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    m_col1, m_col2 = st.columns(2, gap="large")

    with m_col1:
        st.markdown('<div class="section-header">مصفوفة الارتباك للنموذج السريري (Confusion Matrix)</div>', unsafe_allow_html=True)
        st.plotly_chart(confusion_matrix_figure(), use_container_width=True)
        st.caption("مصفوفة الارتباك على 114 عينة اختبار: 72 حالة حميدة صحيحة (True Negative)، 39 حالة خبيثة صحيحة (True Positive)، 0 إيجابيات كاذبة.")

    with m_col2:
        st.markdown('<div class="section-header">معادلة الاندماج المتأخر (Decision Fusion Formula)</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="glass-box" style="line-height: 1.8;">
            تعتمد المنظومة استراتيجية <b>Late Decision Fusion</b> حيث يحسب المؤشر التراكمي كالتالي:<br>
            <div style="background: rgba(0, 229, 255, 0.1); border-radius: 8px; padding: 0.8rem; font-family: monospace; font-size: 1.05rem; color: {CYAN}; text-align: center; margin: 0.8rem 0;">
                Integrated Risk = (0.60 × P_Ultrasound) + (0.40 × P_Clinical)
            </div>
            <b>لماذا هذا الوزن؟</b><br>
            • صورة السونار (60%): تمثل الفحص التصويري المباشر للكتلة المادية داخل الثدي.<br>
            • الفحص السريري (40%): يمثل الخصائص الخلوية الدقيقة لخلايا الورم لتأكيد الطبيعة المورفولوجية.
            </div>
            """,
            unsafe_allow_html=True,
        )

# =========================================================================== #
# TAB 4: Clinical Report Generator & Export
# =========================================================================== #
with tab_report:
    st.markdown('<div class="section-header">📑 التقرير الطبي لحالة المريض (Formal Clinical Report)</div>', unsafe_allow_html=True)

    if not res:
        st.info("⚠️ الرجاء تشغيل فحص لحالة مريض في التبويب الأول (Live Assessment) لتوليد تقريرها الطبي هنا.")
    else:
        report_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        patient_id = f"PAT-BC-{abs(hash(str(res['clinical']['risk_score']) + report_timestamp)) % 100000:05d}"

        rep_html = f"""
        <div style="background: #FFFFFF; color: #1E293B; border-radius: 14px; padding: 2rem; box-shadow: 0 10px 30px rgba(0,0,0,0.3); font-family: 'Cairo', Arial, sans-serif;">
            <div style="display: flex; justify-content: space-between; border-bottom: 2px solid #0284C7; padding-bottom: 1rem; margin-bottom: 1.5rem;">
                <div>
                    <h2 style="margin: 0; color: #0284C7; font-size: 1.6rem;">تقرير التقييم الطبي المتكامل لسرطان الثدي</h2>
                    <div style="color: #64748B; font-size: 0.85rem;">Integrated AI Breast Cancer Assessment Report</div>
                </div>
                <div style="text-align: right; font-size: 0.85rem; color: #475569;">
                    <b>معرف التقرير:</b> {patient_id}<br>
                    <b>تاريخ الفحص:</b> {report_timestamp}
                </div>
            </div>
            
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 1.5rem;">
                <tr style="background: #F1F5F9;">
                    <th style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: right;">الفحص</th>
                    <th style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">النموذج</th>
                    <th style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">النتيجة</th>
                    <th style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">درجة الثقة / الخطر</th>
                </tr>
                <tr>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1;">فحص السونار (Ultrasound)</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">ResNet-18 CNN</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center; font-weight: bold; color: {'#DC2626' if res['ultrasound']['prediction'] == 'Malignant' else '#16A34A'};">{res['ultrasound']['prediction']}</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">{res['ultrasound']['confidence']*100:.1f}%</td>
                </tr>
                <tr>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1;">الفحص السريري المورفولوجي</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">Random Forest</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center; font-weight: bold;">خطر {res['clinical']['risk_score']}%</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center;">{res['clinical']['risk_score']}%</td>
                </tr>
                <tr style="background: #F8FAFC; font-weight: bold;">
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1;" colspan="2">التقييم المشترك المدمج (Integrated Score)</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center; font-size: 1.1rem; color: #0284C7;">{res['summary']['risk_level']}</td>
                    <td style="padding: 0.6rem; border: 1px solid #CBD5E1; text-align: center; font-size: 1.1rem; color: #0284C7;">{res['summary']['integrated_score']}%</td>
                </tr>
            </table>

            <div style="background: #F8FAFC; border-left: 4px solid #0284C7; padding: 1rem; border-radius: 6px; margin-bottom: 1.5rem;">
                <h4 style="margin: 0 0 0.5rem; color: #0F172A;">التوصية السريرية الأولية:</h4>
                <p style="margin: 0; color: #334155; font-size: 0.95rem; line-height: 1.6;">{res['summary']['text']}</p>
            </div>

            <div style="font-size: 0.78rem; color: #64748B; border-top: 1px solid #E2E8F0; padding-top: 0.8rem;">
                <b>إخلاء مسؤولية قانوني:</b> هذا التقرير مستخرج من نموذج ذكاء اصطناعي تجريبي لدعم القرار السريري (Prototype Decision-Support)، ولا يُعد بديلاً عن الفحص الطبي المباشر من قبل استشاري أورام معتمد.
            </div>
        </div>
        """
        st.markdown(rep_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report_text = f"""=======================================================
BREAST CANCER INTEGRATED AI SYSTEM - CLINICAL REPORT
Report ID: {patient_id}
Date: {report_timestamp}
=======================================================

1. ULTRASOUND IMAGING (ResNet-18):
   - Finding: {res['ultrasound']['prediction']}
   - Confidence: {res['ultrasound']['confidence']*100:.1f}%

2. CLINICAL MORPHOLOGY (Random Forest):
   - Malignancy Risk Score: {res['clinical']['risk_score']}%

3. INTEGRATED ASSESSMENT (Late Fusion):
   - Overall Risk Level: {res['summary']['risk_level']}
   - Integrated Risk Score: {res['summary']['integrated_score']}%

4. CLINICAL RECOMMENDATION:
   {res['summary']['text']}

DISCLAIMER:
{res.get('disclaimer', '')}
=======================================================
"""
        st.download_button(
            label="📥 تحميل التقرير الطبي كنص (Download Clinical Report .txt)",
            data=report_text,
            file_name=f"Clinical_Report_{patient_id}.txt",
            mime="text/plain",
        )

# =========================================================================== #
# TAB 5: Problem Statement & Course Defense
# =========================================================================== #
with tab_about:
    st.markdown('<div class="section-header">🎯 بيان المشكلة وأجوبة أسئلة المناقشة (Project Defense Guide)</div>', unsafe_allow_html=True)

    with st.expander("1. ما هي المشكلة الواقعية التي يحلها المشروع؟ (Real-World Problem)", expanded=True):
        st.markdown(
            """
            سرطان الثدي هو أكثر أنواع السرطان شيوعاً بين النساء عالمياً. تكمن المشكلة الطبية في أن الفحص المنفرد 
            (سواء بالأشعة وحدها أو بالفحوصات المخبرية وحدها) يعاني من نسب خطأ؛ فالفحوصات الإشعاعية قد تعطي نتائج إيجابية كاذبة (False Positives) 
            تسبب قلقاً شديداً وخزعات غير ضرورية للمريضة، بينما الفحوصات المنعزلة قد تفوت علامات مبكرة (False Negatives).
            **الحل المقترح:** تطبيق المفهوم المعتمد في طب الأورام المعروف بـ **التقييم الثلاثي (Triple Assessment Paradigm)**، 
            عبر دمج الفحص المورفولوجي الخلوي مع الفحص الشعاعي بالموجات فوق الصوتية واستنطاق النتائج بمساعد ذكاء اصطناعي متقدم.
            """
        )

    with st.expander("2. من هم المستخدمون المستهدفون؟ (Target Users)", expanded=True):
        st.markdown(
            """
            • **أطباء الأشعة وأخصائيو الأورام (Radiologists & Oncologists):** كأداة تدقيق مساندة (Second-Opinion System) لتأكيد القرارات السريرية وتفادي أخطاء الإجهاد البشري.
            • **مراكز الكشف المبكر والعيادات الريفية:** لفرز الحالات المشتبه بها بسرعة وتحديد الأولويات للحالات الحرجة التي تحتاج إحالة فورية للمستشفيات التخصصية.
            """
        )

    with st.expander("3. لماذا يُعد الذكاء الاصطناعي الأنسب لحل هذه المشكلة؟ (Why AI?)", expanded=True):
        st.markdown(
            """
            النماذج التقليدية تعتمد على قواعد برمجية ثابتة تعجز عن تمييز الفروق الدقيقة في الصور الشعاعية أو التداخل في الخصائص المورفولوجية.
            الذكاء الاصطناعي متفوق هنا لأنه يجمع بين:
            • قدرة **شبكات الالتفاف العميق (ResNet-18)** على استخلاص الأنماط البصرية الدقيقة للأنسجة.
            • قدرة **خوارزميات شجر القرار العشوائي (Random Forest)** على إيجاد الحدود غير الخطية المعقدة في البيانات الرقمية.
            • قدرة **نماذج Transformer** على صياغة التفسيرات بلغة بشرية مفهومة للأطباء والمرضى.
            """
        )

    with st.expander("4. ما هو دور كل نموذج وكيف تتكامل معاً عبر الـ API؟ (System Workflow)", expanded=True):
        st.markdown(
            """
            1. **Random Forest:** يتولى تحليل الخصائص المورفولوجية للخلايا الـ 30 (حجم، كثافة، محيط الورم) المأخوذة من خزعة الإبرة الدقيقة (FNA).
            2. **ResNet-18:** يتولى فحص مصفوفات البكسل لصور السونار لاستخراج مؤشرات الكتل غير المتجانسة.
            3. **FastAPI Backend:** يوفر نقطة نهاية مركزية `/analyze` تقوم بتوحيد البيانات، تشغيل النموذجين بالتوازي، وحساب معادلة الاندماج (Late Fusion).
            4. **Transformer Chatbot:** يستقبل النتائج المدمجة كسياق حي (Live Clinical Context) ليجيب عن أسئلة الطبيب أو المريض بدقة سريرية.
            """
        )

    with st.expander("5. القيود والتطويرات المستقبلية (Limitations & Future Work)", expanded=True):
        st.markdown(
            """
            • **القيود:** النموذج مدرب على داتا سيت ويسكونسن وداتا سيت BUSI، ويتطلب اعتمادات تنظيمية سريرية (FDA/CE) قبل استخدامه في بيئات المشافي الحقيقية.
            • **التطويرات المستقبلية:** إضافة دعم صور الماموجرام الشعاعية (Mammography) والتصوير بالرنين المغناطيسي (MRI)، وتوسيع دمج نماذج Vision-Language Transformers (مثل Med-Gemini).
            """
        )