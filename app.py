"""
BuirenggoBoxFit
Sistema Klasifikasaun Prontidaun Fiziku Atleta Boxing
Uza Algoritmu Random Forest
Dili Institute of Technology - Machine Learning 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from generate_dataset import generate_boxing_dataset
from model import (
    FEATURE_COLS, FEATURE_LABELS, CLASS_ORDER, CLASS_COLORS, CLASS_DESC,
    load_data, train_model, predict
)
from database import init_db, save_result, get_history, get_stats, MYSQL_SCHEMA
from manual_calc import (
    gini_impurity, gini_weighted, information_gain,
    step1_root_gini, step2_split_analysis, SAMPLE_DATA
)
import os, warnings
warnings.filterwarnings('ignore')
init_db()

st.set_page_config(
    page_title="BuirenggoBoxFit | Klasifikasaun Atleta Boxing",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# CUSTOM CSS — PROFESSIONAL DARK NAVY + GOLD THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Poppins:wght@600;700;800;900&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Background */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf1 100%);
    }

    /* HERO BANNER on home page */
    .hero-banner {
        background: linear-gradient(135deg, #0f1e3d 0%, #1a3060 50%, #243b6e 100%);
        border-radius: 24px;
        padding: 3rem 2rem;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 20px 60px rgba(15, 30, 61, 0.35);
        position: relative;
        overflow: hidden;
    }
    .hero-banner::before {
        content: '';
        position: absolute;
        top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(218,165,32,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-emoji {
        font-size: 5rem;
        display: block;
        margin-bottom: 0.5rem;
        filter: drop-shadow(0 8px 20px rgba(0,0,0,0.4));
    }
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.5rem; font-weight: 900;
        background: linear-gradient(135deg, #ffd700 0%, #ffa500 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0; letter-spacing: -1px;
        position: relative; z-index: 1;
    }
    .hero-subtitle {
        color: #cbd5e1; font-size: 1.15rem; font-weight: 500;
        margin-top: 0.5rem; position: relative; z-index: 1;
    }
    .hero-badge {
        display: inline-block;
        background: rgba(218,165,32,0.2);
        color: #ffd700;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 1rem;
        border: 1px solid rgba(218,165,32,0.4);
    }

    /* Standard page title (no emoji) */
    .page-title {
        font-family: 'Poppins', sans-serif;
        font-size: 2.4rem; font-weight: 800; color: #0f1e3d;
        margin-bottom: 0.3rem; letter-spacing: -0.5px;
    }
    .page-subtitle {
        color: #64748b; font-size: 1.05rem; font-weight: 500;
        margin-bottom: 2rem;
    }

    /* Section header */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.4rem; font-weight: 700; color: #0f1e3d;
        padding: 0.6rem 0;
        border-bottom: 3px solid #daa520;
        margin: 1.5rem 0 1rem 0;
        display: inline-block;
    }

    /* METRIC CARDS */
    div[data-testid="stMetric"] {
        background: white;
        padding: 1.2rem 1.4rem;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(15, 30, 61, 0.08);
        border-left: 5px solid #daa520;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(15, 30, 61, 0.12);
    }
    div[data-testid="stMetric"] label {
        color: #64748b !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #0f1e3d !important;
        font-weight: 800 !important;
        font-family: 'Poppins', sans-serif !important;
    }

    /* CARDS */
    .info-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem;
        box-shadow: 0 4px 20px rgba(15, 30, 61, 0.07);
        margin-bottom: 1rem;
        border-top: 4px solid #0f1e3d;
    }
    .info-card h3 {
        color: #0f1e3d; font-family: 'Poppins', sans-serif;
        font-weight: 700; margin-top: 0;
    }

    .feature-item {
        background: white;
        padding: 0.8rem 1rem;
        border-radius: 10px;
        margin-bottom: 0.5rem;
        border-left: 3px solid #daa520;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }
    .feature-item strong { color: #0f1e3d; }
    .feature-item span { color: #64748b; font-size: 0.9rem; }

    /* RESULT CARDS */
    .result-card {
        border-radius: 20px;
        padding: 2rem 1.5rem;
        text-align: center;
        box-shadow: 0 12px 40px rgba(0,0,0,0.12);
        margin: 1.5rem 0;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .result-card::before {
        content: '';
        position: absolute;
        top: -100px; right: -100px;
        width: 250px; height: 250px;
        border-radius: 50%;
        background: rgba(255,255,255,0.1);
    }
    .result-excellent { background: linear-gradient(135deg, #16a34a 0%, #15803d 100%); }
    .result-good      { background: linear-gradient(135deg, #2563eb 0%, #1e40af 100%); }
    .result-moderate  { background: linear-gradient(135deg, #ea580c 0%, #c2410c 100%); }
    .result-poor      { background: linear-gradient(135deg, #dc2626 0%, #991b1b 100%); }
    .result-card h1 {
        font-family: 'Poppins', sans-serif;
        font-size: 3rem; font-weight: 900;
        margin: 0; letter-spacing: 2px;
        position: relative; z-index: 1;
    }
    .result-card .result-conf {
        font-size: 1.5rem; font-weight: 700;
        margin: 0.5rem 0; opacity: 0.95;
        position: relative; z-index: 1;
    }
    .result-card .result-name {
        font-size: 1.1rem; opacity: 0.85;
        position: relative; z-index: 1;
    }
    .result-card .result-desc {
        font-size: 0.95rem; margin-top: 1rem;
        background: rgba(0,0,0,0.15);
        padding: 0.8rem;
        border-radius: 10px;
        position: relative; z-index: 1;
    }

    /* RECOMMENDATION CARD */
    .rec-card {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.05);
        border-left: 4px solid #0f1e3d;
        color: #1e293b;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* BUTTONS */
    .stButton>button {
        background: linear-gradient(135deg, #0f1e3d 0%, #243b6e 100%);
        color: white; font-weight: 700;
        border: none;
        border-radius: 12px;
        padding: 0.8rem 2rem;
        font-size: 1.05rem;
        font-family: 'Inter', sans-serif;
        box-shadow: 0 4px 20px rgba(15, 30, 61, 0.25);
        transition: all 0.2s ease;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #daa520 0%, #b8860b 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(218, 165, 32, 0.35);
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1e3d 0%, #1a3060 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        color: #ffd700 !important;
        font-family: 'Poppins', sans-serif !important;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label {
        background: rgba(255,255,255,0.05);
        padding: 0.6rem 1rem !important;
        border-radius: 10px;
        margin-bottom: 0.4rem;
        transition: all 0.2s ease;
    }
    [data-testid="stSidebar"] [data-baseweb="radio"] label:hover {
        background: rgba(218, 165, 32, 0.15);
    }

    .sidebar-stat {
        background: rgba(218,165,32,0.1);
        border-radius: 10px;
        padding: 0.8rem;
        margin: 0.4rem 0;
        border-left: 3px solid #daa520;
    }
    .sidebar-stat-label {
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #cbd5e1 !important;
        opacity: 0.8;
    }
    .sidebar-stat-value {
        font-size: 1.3rem;
        font-weight: 800;
        color: #ffd700 !important;
        font-family: 'Poppins', sans-serif;
    }

    /* INPUT FIELDS */
    .stNumberInput input, .stTextInput input {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 0.6rem 0.8rem !important;
        font-weight: 500 !important;
        background: white !important;
        color: #0f1e3d !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: #daa520 !important;
        box-shadow: 0 0 0 3px rgba(218,165,32,0.15) !important;
    }

    /* WIDGET LABELS — force dark color even in dark mode */
    .main [data-testid="stWidgetLabel"],
    .main [data-testid="stWidgetLabel"] *,
    .main label,
    .main label p,
    .main .stNumberInput label,
    .main .stTextInput label,
    .main .stSelectbox label,
    .main .stMultiSelect label,
    .main .stRadio label,
    .main .stCheckbox label {
        color: #0f1e3d !important;
        font-weight: 600 !important;
        opacity: 1 !important;
    }

    /* Main area markdown text — keep dark in both modes */
    .main .stMarkdown,
    .main .stMarkdown p,
    .main .stMarkdown li,
    .main .stMarkdown strong,
    .main .stMarkdown td,
    .main .stMarkdown th {
        color: #1e293b !important;
    }
    .main .stMarkdown h1, .main .stMarkdown h2,
    .main .stMarkdown h3, .main .stMarkdown h4 {
        color: #0f1e3d !important;
    }

    /* Number input step buttons */
    .stNumberInput button {
        background: white !important;
        color: #0f1e3d !important;
    }

    /* Multiselect & dataframe text */
    .main [data-baseweb="select"] *,
    .main [data-baseweb="tag"] * {
        color: #0f1e3d !important;
    }

    /* Force light background on main content area regardless of theme */
    .main, [data-testid="stAppViewContainer"] > .main {
        background: transparent !important;
    }

    /* CLASS LEGEND */
    .class-legend {
        background: white;
        border-radius: 14px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06);
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    .class-dot {
        width: 18px; height: 18px;
        border-radius: 50%;
        flex-shrink: 0;
    }
    .class-legend-title {
        font-weight: 700; color: #0f1e3d;
        font-size: 1.05rem; margin: 0;
    }
    .class-legend-desc {
        color: #64748b; font-size: 0.85rem; margin: 0;
    }

    /* HIDE STREAMLIT BRANDING */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Treina modelu Random Forest...")
def get_model():
    data_path = 'data/boxing_dataset.csv'
    if not os.path.exists(data_path):
        df = generate_boxing_dataset()
        os.makedirs('data', exist_ok=True)
        df.to_csv(data_path, index=False)
    X, y, df = load_data(data_path)
    rf, le, metrics, X_train, X_test, y_train, y_test = train_model(X, y)
    return rf, le, metrics, df

rf_model, le, metrics, df_full = get_model()

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<h2 style='text-align:center; margin-bottom:0;'>BuirenggoBoxFit</h2>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center; color:#cbd5e1; font-size:0.85rem; margin-top:0.2rem;'>Klasifikasaun Atleta Boxing</p>", unsafe_allow_html=True)
    st.markdown("---")

    page = st.radio(
        "Navigasaun",
        ["Beranda", "Klasifikasaun", "EDA", "Kontagen Manual", "Dataset", "Modelu", "Database", "Kona-ba"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.markdown(f"""
        <div class='sidebar-stat'>
            <div class='sidebar-stat-label'>Dataset</div>
            <div class='sidebar-stat-value'>{len(df_full)} atleta</div>
        </div>
        <div class='sidebar-stat'>
            <div class='sidebar-stat-label'>Akurasia Modelu</div>
            <div class='sidebar-stat-value'>{metrics['accuracy']*100:.1f}%</div>
        </div>
        <div class='sidebar-stat'>
            <div class='sidebar-stat-label'>CV Score</div>
            <div class='sidebar-stat-value'>{metrics['cv_mean']*100:.1f}%</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<p style='text-align:center; font-size:0.75rem; color:#94a3b8;'>DIT Computer Science<br>Trimester I, 2026</p>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — BERANDA
# ══════════════════════════════════════════════════════════════════════════════
if page == "Beranda":
    st.markdown("""
        <div class='hero-banner'>
            <span class='hero-emoji'>🥊</span>
            <h1 class='hero-title'>BuirenggoBoxFit</h1>
            <p class='hero-subtitle'>Sistema Klasifikasaun Prontidaun Fiziku Atleta Boxing</p>
            <span class='hero-badge'>Powered by Random Forest Machine Learning</span>
        </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Total Atleta", f"{len(df_full)}", "Dataset")
    with c2: st.metric("Akurasia Modelu", f"{metrics['accuracy']*100:.1f}%", "Random Forest")
    with c3: st.metric("Variavel Input", "10", "Fitur Fiziku")
    with c4: st.metric("Klase Output", "4", "Nivel Prontidaun")

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1.3, 1])

    with col_l:
        st.markdown('<div class="section-header">Kona-ba Sistema</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class='info-card'>
            <p style='color:#475569; line-height:1.7; font-size:1rem;'>
            <strong>BuirenggoBoxFit</strong> mak sistema klasifikasaun inteligente nebee dezenvolve
            atu ajuda treinador boxing sira halo avaliasaun prontidaun fiziku atleta nian ho metodu
            <strong>objetivu</strong>, <strong>bazeia ba dadus</strong>, no <strong>sistematiku</strong>.
            </p>
            <p style='color:#475569; line-height:1.7; font-size:1rem; margin-bottom:0;'>
            Uza algoritmu <strong>Random Forest</strong> — metodu <em>ensemble machine learning</em>
            avansadu — sistema nee analiza 10 parametru teste fitness fiziku no klasifika atleta
            iha nivel haat ho akurasia aas.
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-header">Nivel Klasifikasaun</div>', unsafe_allow_html=True)

        class_data = [
            ("Ekselente", "#16a34a", "Prontu kompete, kondisaun fiziku diak liu"),
            ("Diak",      "#2563eb", "Kondisaun fiziku diak, mellora area kiik"),
            ("Moderadu",  "#ea580c", "Kondisaun mediu, presiza treinamentu barak"),
            ("Fraku",     "#dc2626", "Kondisaun fraku, labele kompete agora"),
        ]
        for name, color, desc in class_data:
            st.markdown(f"""
                <div class='class-legend'>
                    <div class='class-dot' style='background:{color};'></div>
                    <div>
                        <p class='class-legend-title'>{name}</p>
                        <p class='class-legend-desc'>{desc}</p>
                    </div>
                </div>
            """, unsafe_allow_html=True)

    with col_r:
        st.markdown('<div class="section-header">Variavel Teste</div>', unsafe_allow_html=True)
        features_info = [
            ("VO2 Max",       "Kapasidade aerobiku (ml/kg/min)"),
            ("Push-up",       "Rezistensia liman-leten (reps/min)"),
            ("Sit-up",        "Rezistensia tali-fuan (reps/min)"),
            ("Shuttle Run",   "Agilidade 4x10m (segundus)"),
            ("Sprint 30m",    "Velosidade (segundus)"),
            ("Sit & Reach",   "Fleksibilidade (cm)"),
            ("Body Fat",      "Komposisaun isin (%)"),
            ("BMI",           "Body Mass Index"),
            ("Grip Strength", "Forsa liman (kg)"),
            ("Vertical Jump", "Forsa ain (cm)"),
        ]
        for k, v in features_info:
            st.markdown(f"""
                <div class='feature-item'>
                    <strong>{k}</strong><br>
                    <span>{v}</span>
                </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Distribuisaun Dataset</div>', unsafe_allow_html=True)

    dist = df_full['prontidaun_fiziku'].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.patch.set_facecolor('white')
    colors = [CLASS_COLORS.get(c, '#999') for c in dist.index]

    bars = axes[0].bar(dist.index, dist.values, color=colors, edgecolor='white', linewidth=2)
    axes[0].set_title('Distribuisaun Klase Prontidaun Fiziku', fontweight='bold',
                       fontsize=13, color='#0f1e3d', pad=15)
    axes[0].set_xlabel('Klase', fontweight='600', color='#475569')
    axes[0].set_ylabel('Numero Atleta', fontweight='600', color='#475569')
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    axes[0].grid(axis='y', alpha=0.3, linestyle='--')
    for i, v in enumerate(dist.values):
        axes[0].text(i, v + 0.8, str(v), ha='center', fontweight='bold',
                     color='#0f1e3d', fontsize=11)

    wedges, texts, autotexts = axes[1].pie(
        dist.values, labels=dist.index, colors=colors,
        autopct='%1.1f%%', startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 3},
        textprops={'fontweight': 'bold', 'fontsize': 10}
    )
    axes[1].set_title('Persentajen Klase', fontweight='bold',
                       fontsize=13, color='#0f1e3d', pad=15)
    for autotext in autotexts:
        autotext.set_color('white'); autotext.set_fontsize(11)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — KLASIFIKASAUN
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Klasifikasaun":
    st.markdown('<h1 class="page-title">Klasifikasaun Atleta</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Hatama dadus teste fiziku atleta atu hetan klasifikasaun prontidaun fiziku</p>',
                unsafe_allow_html=True)

    st.markdown('<div class="section-header">Dadus Atleta</div>', unsafe_allow_html=True)
    athlete_name = st.text_input("Naran Atleta", placeholder="Hatama naran atleta...")

    st.markdown('<div class="section-header">Data Teste Fiziku</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        vo2max        = st.number_input("VO2 Max (ml/kg/min)", 15.0, 80.0, 45.0, 0.5)
        pushup        = st.number_input("Push-up (reps/min)", 5, 100, 40)
        situp         = st.number_input("Sit-up (reps/min)", 5, 100, 40)
        shuttle_run   = st.number_input("Shuttle Run 4x10m (segundus)", 8.0, 18.0, 11.0, 0.1)
        sprint_30m    = st.number_input("Sprint 30m (segundus)", 3.0, 8.0, 4.5, 0.1)
    with col2:
        sit_reach     = st.number_input("Sit & Reach (cm)", 0.0, 60.0, 28.0, 0.5)
        body_fat      = st.number_input("Body Fat (%)", 3.0, 40.0, 14.0, 0.5)
        bmi           = st.number_input("BMI (kg/m2)", 15.0, 40.0, 23.0, 0.1)
        grip_strength = st.number_input("Grip Strength (kg)", 10.0, 90.0, 48.0, 0.5)
        vertical_jump = st.number_input("Vertical Jump (cm)", 10.0, 100.0, 55.0, 0.5)

    st.markdown("<br>", unsafe_allow_html=True)
    classify_btn = st.button("Klasifika Agora", use_container_width=True)

    if classify_btn:
        input_data = {
            'vo2max': vo2max, 'pushup': pushup, 'situp': situp,
            'shuttle_run': shuttle_run, 'sprint_30m': sprint_30m,
            'sit_reach': sit_reach, 'body_fat': body_fat, 'bmi': bmi,
            'grip_strength': grip_strength, 'vertical_jump': vertical_jump,
        }

        label, confidence, prob_dict = predict(rf_model, le, input_data)

        css_map = {
            'Ekselente': 'result-excellent',
            'Diak':      'result-good',
            'Moderadu':  'result-moderate',
            'Fraku':     'result-poor',
        }
        desc  = CLASS_DESC.get(label, '')
        css   = css_map.get(label, 'result-good')

        st.markdown(f"""
        <div class='result-card {css}'>
            <h1>{label.upper()}</h1>
            <p class='result-conf'>Konfidensja: {confidence:.1f}%</p>
            <p class='result-name'>{athlete_name if athlete_name else 'Atleta'}</p>
            <p class='result-desc'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

        c_prob, c_radar = st.columns(2)

        with c_prob:
            st.markdown('<div class="section-header">Probabilidade</div>', unsafe_allow_html=True)
            for cls in CLASS_ORDER[::-1]:
                prob  = prob_dict.get(cls, 0)
                color = CLASS_COLORS.get(cls, '#999')
                st.markdown(f"""
                    <div style='margin-bottom:0.8rem;'>
                        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.3rem;'>
                            <strong style='color:#0f1e3d;'>{cls}</strong>
                            <span style='color:{color}; font-weight:700;'>{prob:.1f}%</span>
                        </div>
                        <div style='background:#e2e8f0; border-radius:10px; height:12px; overflow:hidden;'>
                            <div style='background:linear-gradient(90deg, {color}, {color}cc); width:{prob}%; height:100%; border-radius:10px; transition:width 0.5s ease;'></div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)

        with c_radar:
            st.markdown('<div class="section-header">Profil Fiziku</div>', unsafe_allow_html=True)
            norms = {
                'vo2max': (20, 70), 'pushup': (10, 75), 'situp': (10, 72),
                'shuttle_run': (15, 9), 'sprint_30m': (6.5, 3.8),
                'sit_reach': (0, 52), 'body_fat': (28, 6),
                'bmi': (32, 20), 'grip_strength': (15, 72), 'vertical_jump': (18, 82),
            }
            vals_norm = []
            for feat in FEATURE_COLS:
                lo, hi = norms[feat]
                raw = input_data[feat]
                if lo < hi:
                    n = (raw - lo) / (hi - lo)
                else:
                    n = (lo - raw) / (lo - hi)
                vals_norm.append(max(0, min(1, n)))

            labels_short = ['VO2 Max', 'Push Up', 'Sit Up', 'Shuttle', 'Sprint',
                            'Sit&Rch', 'BodyFat', 'BMI', 'Grip', 'V.Jump']

            angles = np.linspace(0, 2 * np.pi, len(FEATURE_COLS), endpoint=False).tolist()
            vals_closed = [v*100 for v in vals_norm] + [vals_norm[0]*100]
            angles += angles[:1]

            color = CLASS_COLORS.get(label, '#3498db')
            fig2, ax2 = plt.subplots(figsize=(5, 5), subplot_kw=dict(polar=True))
            fig2.patch.set_facecolor('white')
            ax2.plot(angles, vals_closed, color=color, linewidth=2.5)
            ax2.fill(angles, vals_closed, color=color, alpha=0.25)
            ax2.set_xticks(angles[:-1])
            ax2.set_xticklabels(labels_short, fontsize=9, color='#0f1e3d', fontweight='bold')
            ax2.set_ylim(0, 100)
            ax2.set_yticks([20, 40, 60, 80])
            ax2.set_yticklabels(['20', '40', '60', '80'], fontsize=7, color='#94a3b8')
            ax2.grid(color='#cbd5e1', linestyle='--', alpha=0.6)
            ax2.spines['polar'].set_color('#cbd5e1')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        # Save to database
        save_result(athlete_name, input_data, label, confidence)
        st.success("Rezultadu salva ona ba database!")

        st.markdown('<div class="section-header">Rekomendasaun Treinamentu</div>',
                    unsafe_allow_html=True)
        recs = {
            'Ekselente': [
                "Mantein programa treinamentu atual no foka ba estratejia kompetisaun.",
                "Aumenta intensidade treinamentu taktiku no sparring.",
                "Monitora rekuperasaun atu prevene overtraining.",
            ],
            'Diak': [
                "Foka treinamentu iha area nebee skor kiik liu.",
                "Aumenta volume treinamentu kondisaun fiziku 15-20%.",
                "Kontinua programa atual no avalia fila-fila iha loron 30.",
            ],
            'Moderadu': [
                "Desinha programa treinamentu kondisaun fiziku intensivu.",
                "Prioriza VO2 Max no rezistensia kardiovaskular.",
                "Avalia fali iha loron 45-60 depois treinamentu intensivu.",
                "Konsidera adikional sesaun treinamentu 2-3 vezes semana.",
            ],
            'Fraku': [
                "Para kompetisaun sira ate kondisaun fiziku mellora.",
                "Dezenvolve programa treinamentu baziku durante fulan 2-3.",
                "Foka ba kondisaun aerobiku no forsa baziku uluk.",
                "Konsulta ho mediku desportu antes reasumir treinamentu intensivu.",
                "Avalia fali iha loron 60 depois hahu programa mellora.",
            ],
        }
        for rec in recs.get(label, []):
            st.markdown(f"<div class='rec-card'>{rec}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — DATASET
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Dataset":
    st.markdown('<h1 class="page-title">Dataset Atleta Boxing</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Dataset teste fitness fiziku atleta boxing 150 nian</p>',
                unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: st.metric("Total Rekordu", len(df_full))
    with c2: st.metric("Variavel Input", len(FEATURE_COLS))
    with c3: st.metric("Klase Output", 4)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">Filtra Dataset</div>', unsafe_allow_html=True)
    filter_class = st.multiselect(
        "Filtra ho Klase Prontidaun:",
        options=df_full['prontidaun_fiziku'].unique().tolist(),
        default=df_full['prontidaun_fiziku'].unique().tolist()
    )
    df_show = df_full[df_full['prontidaun_fiziku'].isin(filter_class)]
    st.dataframe(df_show, use_container_width=True, height=400)

    st.markdown('<div class="section-header">Estatistika Deskriptiva ba Klase</div>',
                unsafe_allow_html=True)
    st.dataframe(
        df_full.groupby('prontidaun_fiziku')[FEATURE_COLS].mean().round(2),
        use_container_width=True
    )

    csv = df_full.to_csv(index=False).encode('utf-8')
    st.download_button("Download Dataset CSV", csv, "boxing_dataset.csv", "text/csv")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — MODELU
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Modelu":
    st.markdown('<h1 class="page-title">Performance Modelu</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Avaliasaun modelu Random Forest klasifikasaun prontidaun fiziku atleta boxing</p>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Akurasia",   f"{metrics['accuracy']*100:.2f}%")
    with c2: st.metric("Preisizaun", f"{metrics['precision']*100:.2f}%")
    with c3: st.metric("Recall",     f"{metrics['recall']*100:.2f}%")
    with c4: st.metric("F1-Score",   f"{metrics['f1']*100:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_fi = st.columns(2)

    with col_cm:
        st.markdown('<div class="section-header">Konfusaun Matrix</div>', unsafe_allow_html=True)
        class_names = [le.inverse_transform([i])[0] for i in range(len(le.classes_))]
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4.5))
        fig_cm.patch.set_facecolor('white')
        sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names, ax=ax_cm,
                    linewidths=2, linecolor='white',
                    annot_kws={'fontweight':'bold','fontsize':12},
                    cbar_kws={'shrink': 0.7})
        ax_cm.set_xlabel('Predisaun', fontweight='600', color='#0f1e3d', fontsize=11)
        ax_cm.set_ylabel('Atual',     fontweight='600', color='#0f1e3d', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close()

    with col_fi:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        fi = metrics['feature_importance']
        fi_sorted = dict(sorted(fi.items(), key=lambda x: x[1]))
        labels_fi = [FEATURE_LABELS.get(k, k) for k in fi_sorted.keys()]
        fig_fi, ax_fi = plt.subplots(figsize=(5, 4.5))
        fig_fi.patch.set_facecolor('white')
        bars = ax_fi.barh(labels_fi, list(fi_sorted.values()),
                          color='#daa520', alpha=0.9, edgecolor='#b8860b', linewidth=1.5)
        ax_fi.set_xlabel('Importansia', fontweight='600', color='#0f1e3d')
        ax_fi.spines['top'].set_visible(False)
        ax_fi.spines['right'].set_visible(False)
        ax_fi.grid(axis='x', alpha=0.3, linestyle='--')
        for bar, val in zip(bars, fi_sorted.values()):
            ax_fi.text(val + 0.003, bar.get_y() + bar.get_height()/2,
                       f'{val:.3f}', va='center', fontsize=9,
                       fontweight='bold', color='#0f1e3d')
        plt.tight_layout()
        st.pyplot(fig_fi)
        plt.close()

    st.markdown('<div class="section-header">Relatoriu Klasifikasaun</div>', unsafe_allow_html=True)
    st.code(metrics['classification_report'], language='text')

    st.markdown('<div class="section-header">Validasaun Kruzada (5-Fold)</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: st.metric("CV Mean Accuracy", f"{metrics['cv_mean']*100:.2f}%")
    with c2: st.metric("CV Std Deviation", f"+/-{metrics['cv_std']*100:.2f}%")

    st.markdown('<div class="section-header">Konfigurasaun Modelu</div>', unsafe_allow_html=True)
    st.json({
        "n_estimators": 100, "max_depth": 10, "min_samples_split": 5,
        "min_samples_leaf": 2, "max_features": "sqrt",
        "class_weight": "balanced", "random_state": 42
    })

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — KONA-BA
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kona-ba":
    st.markdown('<h1 class="page-title">Kona-ba Peskiza</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Informasaun kompletu kona-ba peskiza, metodu, no autor</p>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class='info-card'>
        <h3>Titulu Peskiza</h3>
        <p style='color:#475569; font-style:italic; font-size:1.05rem; line-height:1.6;'>
            "Classification of Boxing Athlete Physical Readiness Using Random Forest Algorithm
            Based on Fitness Test Data"
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_info, col_method = st.columns(2)

    with col_info:
        st.markdown("""
        <div class='info-card'>
            <h3>Informasaun Peskiza</h3>
            <table style='width:100%; color:#475569;'>
                <tr><td style='padding:6px 0;'><strong>Estudante</strong></td><td>Joao Nuno Urle Pereira Boavida</td></tr>
                <tr><td style='padding:6px 0;'><strong>NIM</strong></td><td>2101020110073</td></tr>
                <tr><td style='padding:6px 0;'><strong>Instituisaun</strong></td><td>Dili Institute of Technology</td></tr>
                <tr><td style='padding:6px 0;'><strong>Programa</strong></td><td>Computer Science</td></tr>
                <tr><td style='padding:6px 0;'><strong>Eskola</strong></td><td>Siencia no Engineeria</td></tr>
                <tr><td style='padding:6px 0;'><strong>Unidade</strong></td><td>Research Methodology</td></tr>
                <tr><td style='padding:6px 0;'><strong>Trimester</strong></td><td>I, 2026</td></tr>
                <tr><td style='padding:6px 0;'><strong>Dosente</strong></td><td>Dr. Edio da Costa, BSc., MCs.</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    with col_method:
        st.markdown("""
        <div class='info-card'>
            <h3>Metodu Machine Learning</h3>
            <p style='color:#475569; line-height:1.7;'>
                <strong>Random Forest</strong> mak algoritmu <em>ensemble learning</em> nebee uza
                decision tree multiplu.
            </p>
            <ul style='color:#475569; line-height:1.8;'>
                <li><strong>N Trees:</strong> 100 decision trees</li>
                <li><strong>Max Depth:</strong> 10 levels</li>
                <li><strong>Max Features:</strong> sqrt(n) per split</li>
                <li><strong>Validasaun:</strong> 5-Fold Stratified CV</li>
                <li><strong>Train/Test:</strong> 80% / 20%</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    col_var, col_tek = st.columns(2)

    with col_var:
        st.markdown("""
        <div class='info-card'>
            <h3>Variavel Peskiza</h3>
            <p style='color:#0f1e3d; font-weight:700; margin-bottom:0.3rem;'>Input (10 variavel):</p>
            <p style='color:#475569; line-height:1.7;'>
                VO2 Max, Push-up, Sit-up, Shuttle Run, Sprint 30m,
                Sit & Reach, Body Fat %, BMI, Grip Strength, Vertical Jump
            </p>
            <p style='color:#0f1e3d; font-weight:700; margin-bottom:0.3rem;'>Output (4 klase):</p>
            <p style='color:#475569; line-height:1.7;'>
                Ekselente | Diak | Moderadu | Fraku
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_tek:
        st.markdown("""
        <div class='info-card'>
            <h3>Teknolojia Uza</h3>
            <ul style='color:#475569; line-height:1.8;'>
                <li><strong>Python 3.10+</strong> — Linguajen programasaun</li>
                <li><strong>scikit-learn</strong> — Random Forest model</li>
                <li><strong>Streamlit</strong> — Web interface</li>
                <li><strong>Pandas / NumPy</strong> — Data processing</li>
                <li><strong>Matplotlib / Seaborn</strong> — Visualizasaun</li>
                <li><strong>SQLite / MySQL</strong> — Database</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — EDA (Exploratory Data Analysis)
# ══════════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown('<h1 class="page-title">Explorasaun Dadus (EDA)</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Analiza vizual no estatistiku husi dataset atleta boxing 150 nian</p>', unsafe_allow_html=True)

    # ── Descriptive stats ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Estatistika Deskriptiva</div>', unsafe_allow_html=True)
    st.dataframe(df_full[FEATURE_COLS].describe().round(2), use_container_width=True)

    st.markdown('<div class="section-header">Estatistika Media ba Klase</div>', unsafe_allow_html=True)
    st.dataframe(
        df_full.groupby('prontidaun_fiziku')[FEATURE_COLS].mean().round(2),
        use_container_width=True
    )

    # ── Correlation heatmap ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Matrix (Heatmap)</div>', unsafe_allow_html=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 7))
    fig_corr.patch.set_facecolor('white')
    corr = df_full[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax_corr, linewidths=0.5, annot_kws={'size': 9},
                vmin=-1, vmax=1, center=0,
                xticklabels=[FEATURE_LABELS[c].split('(')[0].strip() for c in FEATURE_COLS],
                yticklabels=[FEATURE_LABELS[c].split('(')[0].strip() for c in FEATURE_COLS])
    ax_corr.set_title('Korelasaun antre Variavel Fiziku', fontweight='bold',
                      fontsize=13, color='#0f1e3d', pad=15)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    st.pyplot(fig_corr)
    plt.close()
    st.caption("Valor besik +1 = korelasaun pozitivu forte | besik -1 = korelasaun negativu forte | 0 = la iha korelasaun")

    # ── Box plots ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Box Plot — Distribuisaun tiha Klase</div>', unsafe_allow_html=True)
    key_feats = ['vo2max', 'pushup', 'situp', 'shuttle_run', 'sprint_30m',
                 'sit_reach', 'body_fat', 'bmi', 'grip_strength', 'vertical_jump']
    n_cols = 2
    n_rows = (len(key_feats) + 1) // n_cols
    fig_box, axes_box = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
    fig_box.patch.set_facecolor('white')
    palette = {c: CLASS_COLORS.get(c, '#999') for c in CLASS_ORDER}
    for i, feat in enumerate(key_feats):
        ax = axes_box[i // n_cols][i % n_cols]
        order = [c for c in CLASS_ORDER if c in df_full['prontidaun_fiziku'].unique()]
        sns.boxplot(data=df_full, x='prontidaun_fiziku', y=feat,
                    order=order, palette=palette, ax=ax, width=0.5,
                    linewidth=1.5, flierprops={'markerfacecolor': '#94a3b8', 'markersize': 4})
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontweight='bold',
                     fontsize=10, color='#0f1e3d')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8fafc')
    if len(key_feats) % n_cols:
        axes_box[-1][-1].set_visible(False)
    plt.suptitle('Distribuisaun Variavel Fiziku tiha Klase Prontidaun',
                 fontweight='bold', fontsize=13, color='#0f1e3d', y=1.01)
    plt.tight_layout()
    st.pyplot(fig_box)
    plt.close()

    # ── Histograms ───────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Histograma Distribuisaun Variavel</div>', unsafe_allow_html=True)
    fig_hist, axes_hist = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.2))
    fig_hist.patch.set_facecolor('white')
    for i, feat in enumerate(key_feats):
        ax = axes_hist[i // n_cols][i % n_cols]
        for cls in CLASS_ORDER:
            subset = df_full[df_full['prontidaun_fiziku'] == cls][feat]
            if len(subset):
                ax.hist(subset, bins=12, alpha=0.55,
                        color=CLASS_COLORS.get(cls, '#999'), label=cls, edgecolor='white')
        ax.set_title(FEATURE_LABELS.get(feat, feat), fontweight='bold',
                     fontsize=10, color='#0f1e3d')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_facecolor('#f8fafc')
        if i == 0:
            ax.legend(fontsize=8)
    if len(key_feats) % n_cols:
        axes_hist[-1][-1].set_visible(False)
    plt.suptitle('Histograma Distribuisaun tiha Klase',
                 fontweight='bold', fontsize=13, color='#0f1e3d', y=1.01)
    plt.tight_layout()
    st.pyplot(fig_hist)
    plt.close()

    # ── Scatter VO2 Max vs Push-up ────────────────────────────────────────────
    st.markdown('<div class="section-header">Scatter Plot — VO2 Max vs Push-up</div>', unsafe_allow_html=True)
    fig_sc, ax_sc = plt.subplots(figsize=(9, 5))
    fig_sc.patch.set_facecolor('white')
    for cls in CLASS_ORDER:
        sub = df_full[df_full['prontidaun_fiziku'] == cls]
        ax_sc.scatter(sub['vo2max'], sub['pushup'],
                      c=CLASS_COLORS.get(cls, '#999'), label=cls,
                      alpha=0.7, s=60, edgecolors='white', linewidth=0.5)
    ax_sc.set_xlabel('VO2 Max (ml/kg/min)', fontweight='600', color='#0f1e3d')
    ax_sc.set_ylabel('Push-up (reps/min)', fontweight='600', color='#0f1e3d')
    ax_sc.set_title('Relasaun VO2 Max ho Push-up tuir Klase',
                    fontweight='bold', fontsize=12, color='#0f1e3d')
    ax_sc.legend(title='Klase', title_fontsize=9)
    ax_sc.spines['top'].set_visible(False)
    ax_sc.spines['right'].set_visible(False)
    ax_sc.grid(alpha=0.3, linestyle='--')
    ax_sc.set_facecolor('#f8fafc')
    plt.tight_layout()
    st.pyplot(fig_sc)
    plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — KONTAGEN MANUAL
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Kontagen Manual":
    st.markdown('<h1 class="page-title">Kontagen Manual — Random Forest</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Demonstrasaun kalkulasaun manual Gini Impurity no Random Forest laho scikit-learn</p>', unsafe_allow_html=True)

    # ── Sample dataset ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 0 — Amostra Dadus (6 Atleta)</div>', unsafe_allow_html=True)
    st.markdown("Uza amostra ki'ik ho **6 atleta**, **2 feature** (VO2 Max, Push-up), no **2 klase** (Ekselente/Fraku):")
    st.dataframe(SAMPLE_DATA, use_container_width=True, hide_index=True)

    labels_sample = list(SAMPLE_DATA['Klase'])
    vo2_vals      = list(SAMPLE_DATA['VO2 Max'])
    push_vals     = list(SAMPLE_DATA['Push-up'])

    # ── Step 1: Gini root ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 1 — Gini Impurity Root Node</div>', unsafe_allow_html=True)
    st.markdown("**Formula:** $Gini(D) = 1 - \\sum p_i^2$")

    g_root, root_rows = step1_root_gini(labels_sample)
    df_root = pd.DataFrame(root_rows)
    st.dataframe(df_root, use_container_width=True, hide_index=True)

    n_tot = len(labels_sample)
    formula_parts = " + ".join([f"({r['Kontas']}/{n_tot})²" for r in root_rows])
    st.markdown(f"""
    ```
    Gini(Root) = 1 - [{formula_parts}]
               = 1 - [{' + '.join([str(r['pi²']) for r in root_rows])}]
               = 1 - {sum(r['pi²'] for r in root_rows):.4f}
               = {g_root}
    ```
    """)
    st.info(f"**Gini Impurity Root Node = {g_root}** (0.5 = impureza maksimál, klase distribui igual)")

    # ── Step 2: Best split ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 2 — Buka Split Di\'ak Liu (VO2 Max ≥ 50)</div>', unsafe_allow_html=True)
    threshold = 50.0
    sp = step2_split_analysis(vo2_vals, labels_sample, threshold)

    col_l2, col_r2 = st.columns(2)
    with col_l2:
        st.markdown(f"**Node Karuk (VO2 Max < {threshold}):**")
        df_left = SAMPLE_DATA.iloc[sp['left_idx']][['ID', 'VO2 Max', 'Klase']]
        st.dataframe(df_left, use_container_width=True, hide_index=True)
        st.markdown(f"Gini(Karuk) = **{sp['g_left']}**")
    with col_r2:
        st.markdown(f"**Node Loos (VO2 Max ≥ {threshold}):**")
        df_right = SAMPLE_DATA.iloc[sp['right_idx']][['ID', 'VO2 Max', 'Klase']]
        st.dataframe(df_right, use_container_width=True, hide_index=True)
        st.markdown(f"Gini(Loos) = **{sp['g_right']}**")

    st.markdown(f"""
    ```
    Gini_weighted = ({len(sp['left_idx'])}/{n_tot}) × {sp['g_left']}
                  + ({len(sp['right_idx'])}/{n_tot}) × {sp['g_right']}
                  = {sp['gini_weighted']}

    Information Gain = {g_root} - {sp['gini_weighted']} = {sp['info_gain']}
    ```
    """)
    st.success(f"**Information Gain = {sp['info_gain']}** — Split perfetu! Gini = 0 iha child nodes rua hotu.")

    # ── Step 3: Bootstrap ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 3 — Bootstrap Sampling</div>', unsafe_allow_html=True)
    st.markdown("Random Forest kria **bootstrap sample** diferente ba kada tree (hili ho repozisaun):")

    np.random.seed(42)
    boot_data = []
    for t in range(1, 4):
        idx = list(np.random.choice(6, size=6, replace=True))
        oob = [i for i in range(6) if i not in idx]
        boot_data.append({
            'Tree': f'Tree {t}',
            'Bootstrap Sample (index)': str([i+1 for i in idx]),
            'OOB (la uza)': str([i+1 for i in oob]) if oob else '-'
        })
    st.dataframe(pd.DataFrame(boot_data), use_container_width=True, hide_index=True)

    # ── Step 4: Random features ──────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 4 — Seleksaun Feature Randómu</div>', unsafe_allow_html=True)
    st.markdown(f"Ba tiha split, RF hili **√(jumlah feature)** = √10 ≈ **3 feature** randómu husi 10 feature total.")
    feat_ex = pd.DataFrame({
        'Tree': ['Tree 1', 'Tree 2', 'Tree 3'],
        'Features Hili (random)': [
            'VO2 Max, Push-up, Grip Strength',
            'Sit-up, Shuttle Run, Body Fat',
            'VO2 Max, Sprint 30m, Vertical Jump',
        ],
        'Split Di\'ak Liu': ['VO2 Max ≥ 50', 'Shuttle Run ≤ 11.5', 'VO2 Max ≥ 50'],
        'Gini Weighted': ['0.00', '0.12', '0.00'],
    })
    st.dataframe(feat_ex, use_container_width=True, hide_index=True)

    # ── Step 5: Voting ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Pasu 5 — Ensemble Voting (Predisaun Final)</div>', unsafe_allow_html=True)
    st.markdown("Ba atleta foun ho **VO2 Max = 57.0, Push-up = 55**:")
    vote_data = pd.DataFrame({
        'Tree': [f'Tree {i}' for i in range(1, 8)] + ['...', 'Tree 100'],
        'Predisaun': ['Ekselente', 'Ekselente', 'Ekselente', 'Ekselente',
                      'Ekselente', 'Ekselente', 'Ekselente', '...', 'Ekselente'],
        'Razaun': ['VO2 Max 57 ≥ 50', 'VO2 Max 57 ≥ 50', 'VO2 Max 57 ≥ 50',
                   'VO2 Max 57 ≥ 50', 'VO2 Max 57 ≥ 50', 'VO2 Max 57 ≥ 50',
                   'VO2 Max 57 ≥ 50', '...', 'VO2 Max 57 ≥ 50'],
    })
    st.dataframe(vote_data, use_container_width=True, hide_index=True)

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        fig_vote, ax_vote = plt.subplots(figsize=(5, 3.5))
        fig_vote.patch.set_facecolor('white')
        bars = ax_vote.bar(['Ekselente', 'Diak', 'Moderadu', 'Fraku'],
                           [87, 8, 3, 2],
                           color=['#16a34a', '#2563eb', '#ea580c', '#dc2626'],
                           edgecolor='white', linewidth=2)
        ax_vote.set_ylabel('Jumlah Votu', fontweight='600')
        ax_vote.set_title('Distribuisaun Votu (100 Trees)', fontweight='bold', color='#0f1e3d')
        ax_vote.spines['top'].set_visible(False)
        ax_vote.spines['right'].set_visible(False)
        for bar, val in zip(bars, [87, 8, 3, 2]):
            ax_vote.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         str(val), ha='center', fontweight='bold', fontsize=11)
        plt.tight_layout()
        st.pyplot(fig_vote)
        plt.close()
    with col_v2:
        st.markdown("""
        <div style='background:#16a34a; color:white; border-radius:16px; padding:2rem; text-align:center; margin-top:1rem;'>
            <div style='font-size:2.5rem; font-weight:900;'>EKSELENTE</div>
            <div style='font-size:1.3rem; margin-top:0.5rem;'>Konfidensja: 87%</div>
            <div style='font-size:0.9rem; margin-top:0.5rem; opacity:0.85;'>87 husi 100 trees vota Ekselente</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Live manual RF ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Teste Kontagen Manual — Input Data Foun</div>', unsafe_allow_html=True)
    st.markdown("Klasifika atleta foun uza **ManualRandomForest** implementasaun Python (laho scikit-learn):")

    c1m, c2m = st.columns(2)
    with c1m:
        v2_manual  = st.number_input("VO2 Max", 20.0, 70.0, 57.0, 0.5, key="m_vo2")
        pu_manual  = st.number_input("Push-up", 10, 75, 55, key="m_pu")
    with c2m:
        su_manual  = st.number_input("Sit-up", 10, 72, 50, key="m_su")
        sr_manual  = st.number_input("Shuttle Run (seg)", 9.0, 15.0, 10.5, 0.1, key="m_sr")

    if st.button("Kalkulasaun Manual", key="btn_manual"):
        from manual_calc import ManualRandomForest as MRF
        X_s = df_full[['vo2max', 'pushup', 'situp', 'shuttle_run']].copy()
        y_s = df_full['prontidaun_fiziku'].copy()
        mrf = MRF(n_trees=10, random_state=42)
        mrf.fit(X_s, y_s)
        row = pd.Series({'vo2max': v2_manual, 'pushup': pu_manual,
                         'situp': su_manual, 'shuttle_run': sr_manual})
        pred_manual = mrf.predict_one(row)
        votes = [t.predict_one(row) for t in mrf.trees if t.feature is not None]
        vote_counts = {c: votes.count(c) for c in set(votes)}
        st.markdown(f"""
        <div style='background:#0f1e3d; color:white; border-radius:14px; padding:1.5rem;'>
            <strong>Rezultadu ManualRandomForest (10 trees):</strong><br>
            Predisaun: <span style='color:#ffd700; font-size:1.4rem; font-weight:800;'>{pred_manual}</span><br>
            Votu: {vote_counts}
        </div>
        """, unsafe_allow_html=True)

    # ── Python code ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Kódigu Python — Implementasaun Manual</div>', unsafe_allow_html=True)
    st.code("""
# Kalkulasaun Gini Impurity — MANUAL (laho scikit-learn)
def gini_impurity(labels):
    n = len(labels)
    classes = set(labels)
    return 1.0 - sum((labels.count(c) / n) ** 2 for c in classes)

def gini_weighted(y_left, y_right):
    n = len(y_left) + len(y_right)
    return (len(y_left)/n)*gini_impurity(y_left) + (len(y_right)/n)*gini_impurity(y_right)

# Ezemplu:
labels = ['Ekselente','Ekselente','Fraku','Ekselente','Fraku','Fraku']
print(f"Gini Root = {gini_impurity(labels)}")  # Output: 0.5

y_left  = ['Fraku', 'Fraku', 'Fraku']   # VO2 Max < 50
y_right = ['Ekselente', 'Ekselente', 'Ekselente']  # VO2 Max >= 50
print(f"Gini Weighted = {gini_weighted(y_left, y_right)}")  # Output: 0.0
print(f"Information Gain = {0.5 - gini_weighted(y_left, y_right)}")  # Output: 0.5
""", language='python')

# ══════════════════════════════════════════════════════════════════════════════
# PAGE — DATABASE
# ══════════════════════════════════════════════════════════════════════════════
elif page == "Database":
    st.markdown('<h1 class="page-title">Database — Historial Avaliasaun</h1>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Rekordu avaliasaun atleta salva iha database (SQLite / MySQL)</p>', unsafe_allow_html=True)

    total, dist = get_stats()
    c1d, c2d, c3d = st.columns(3)
    with c1d: st.metric("Total Rekordu", int(total))
    with c2d:
        if not dist.empty and 'klasifikasaun' in dist.columns:
            top = dist.sort_values('n', ascending=False).iloc[0]
            st.metric("Klasifikasaun Barak Liu", top['klasifikasaun'])
        else:
            st.metric("Klasifikasaun Barak Liu", "-")
    with c3d: st.metric("Status Database", "Online")

    st.markdown('<div class="section-header">Historial Avaliasaun Atleta</div>', unsafe_allow_html=True)
    df_hist = get_history()
    if df_hist.empty:
        st.info("Seidauk iha rekordu. Halo klasifikasaun iha pajina 'Klasifikasaun' uluk.")
    else:
        st.dataframe(df_hist, use_container_width=True, hide_index=True)
        csv_db = df_hist.to_csv(index=False).encode('utf-8')
        st.download_button("Download Historial CSV", csv_db, "historial_avaliasaun.csv", "text/csv")

    # Distribution from DB
    if not df_hist.empty and 'klasifikasaun' in df_hist.columns:
        st.markdown('<div class="section-header">Distribuisaun Klasifikasaun (Database)</div>', unsafe_allow_html=True)
        dist_db = df_hist['klasifikasaun'].value_counts()
        fig_db, ax_db = plt.subplots(figsize=(7, 3.5))
        fig_db.patch.set_facecolor('white')
        colors_db = [CLASS_COLORS.get(c, '#999') for c in dist_db.index]
        bars_db = ax_db.bar(dist_db.index, dist_db.values, color=colors_db,
                            edgecolor='white', linewidth=2)
        for bar, v in zip(bars_db, dist_db.values):
            ax_db.text(bar.get_x() + bar.get_width()/2, v + 0.05, str(v),
                       ha='center', fontweight='bold')
        ax_db.spines['top'].set_visible(False)
        ax_db.spines['right'].set_visible(False)
        ax_db.set_facecolor('#f8fafc')
        plt.tight_layout()
        st.pyplot(fig_db)
        plt.close()

    # MySQL Schema
    st.markdown('<div class="section-header">MySQL Schema (Kódigu SQL)</div>', unsafe_allow_html=True)
    st.code(MYSQL_SCHEMA, language='sql')

    st.markdown('<div class="section-header">Python — Koneksaun MySQL</div>', unsafe_allow_html=True)
    st.code("""
import mysql.connector

# Koneksaun ba MySQL database
conn = mysql.connector.connect(
    host     = "localhost",
    user     = "root",
    password = "password",
    database = "buirenggoboxfit"
)
cursor = conn.cursor()

# Save klasifikasaun ba database
def save_to_mysql(atleta_naran, vo2max, pushup, klasifikasaun, konfidensja):
    sql = \"\"\"INSERT INTO avaliasaun
              (atleta_naran, vo2max, pushup, klasifikasaun, konfidensja)
              VALUES (%s, %s, %s, %s, %s)\"\"\"
    cursor.execute(sql, (atleta_naran, vo2max, pushup, klasifikasaun, konfidensja))
    conn.commit()

# Hetan historial
cursor.execute("SELECT * FROM avaliasaun ORDER BY data_avaliasaun DESC")
results = cursor.fetchall()
conn.close()
""", language='python')
