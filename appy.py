import streamlit as st
import pandas as pd
import joblib

# ─────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────
st.set_page_config(
    page_title="SocialSense · Addiction Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #080c14;
    --surface:   #0d1424;
    --card:      #111827;
    --border:    #1e2d45;
    --accent1:   #00e5ff;
    --accent2:   #7c3aed;
    --accent3:   #f472b6;
    --text:      #e2e8f0;
    --muted:     #64748b;
    --success:   #10b981;
    --warning:   #f59e0b;
    --danger:    #ef4444;
    --glow1:     rgba(0,229,255,0.15);
    --glow2:     rgba(124,58,237,0.15);
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { display: none; }
.block-container { padding: 0 !important; max-width: 100% !important; }

/* ── Animated noise overlay ── */
body::before {
    content: '';
    position: fixed; inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.03'/%3E%3C/svg%3E");
    pointer-events: none; z-index: 0; opacity: 0.4;
}

/* ── Hero section ── */
.hero {
    position: relative;
    padding: 3.5rem 2.5rem 2rem;
    background: linear-gradient(135deg, #0d1424 0%, #080c14 60%);
    border-bottom: 1px solid var(--border);
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute; top: -60px; right: -60px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, var(--glow1) 0%, transparent 70%);
    animation: pulse 6s ease-in-out infinite;
}
.hero::after {
    content: '';
    position: absolute; bottom: -80px; left: 30%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, var(--glow2) 0%, transparent 70%);
    animation: pulse 8s ease-in-out infinite reverse;
}
@keyframes pulse {
    0%,100% { transform: scale(1); opacity: 1; }
    50%      { transform: scale(1.15); opacity: 0.7; }
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-family: 'Syne', sans-serif; font-size: 0.7rem;
    font-weight: 700; letter-spacing: 0.2em; text-transform: uppercase;
    border: 1px solid var(--border); padding: 4px 12px; border-radius: 20px;
    margin-bottom: 1rem;
    background-clip: text;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2rem, 5vw, 3.5rem);
    font-weight: 800; line-height: 1.1;
    background: linear-gradient(135deg, #fff 30%, var(--accent1) 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 0.75rem;
}
.hero-sub {
    color: var(--muted); font-size: 1rem;
    max-width: 520px; line-height: 1.6; margin: 0;
}


/* ── Main layout ── */
.main-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 1.25rem;
    padding: 2rem 2.5rem;
    position: relative; z-index: 1;
}
@media (max-width: 900px) { .main-grid { grid-template-columns: 1fr; } }

/* ── Section cards ── */
.section-card {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 1.5rem;
    position: relative; overflow: hidden;
    transition: border-color 0.3s;
}
.section-card:hover { border-color: var(--accent1); }
.section-card::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--accent2), var(--accent1));
    opacity: 0; transition: opacity 0.3s;
}
.section-card:hover::before { opacity: 1; }

.card-icon {
    font-size: 1.6rem; margin-bottom: 0.5rem;
}
.card-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem; font-weight: 700;
    letter-spacing: 0.15em; text-transform: uppercase;
    color: var(--accent1); margin-bottom: 1.25rem;
    display: flex; align-items: center; gap: 0.5rem;
}
.card-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Streamlit widget overrides ── */
div[data-testid="stNumberInput"] > div > div > input,
div[data-testid="stTextInput"] > div > div > input {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    padding: 0.6rem 0.9rem !important;
    transition: border-color 0.2s !important;
}
div[data-testid="stNumberInput"] > div > div > input:focus,
div[data-testid="stTextInput"] > div > div > input:focus {
    border-color: var(--accent1) !important;
    box-shadow: 0 0 0 3px var(--glow1) !important;
}

div[data-testid="stSelectbox"] > div > div {
    background: var(--surface) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
}

/* Slider */
div[data-testid="stSlider"] > div > div > div {
    background: var(--accent1) !important;
}
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent1) !important;
    border: 2px solid white !important;
    box-shadow: 0 0 12px var(--glow1) !important;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label, .stTextInput label, .stSlider label {
    color: var(--muted) !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.02em !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Predict button ── */
.predict-wrap {
    padding: 0 2.5rem 1.5rem;
    position: relative; z-index: 1;
}
div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--accent2) 0%, var(--accent1) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.9rem 3rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important; font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px rgba(124,58,237,0.4) !important;
    width: 100% !important;
    text-transform: uppercase !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,229,255,0.5) !important;
}
div[data-testid="stButton"] > button:active {
    transform: translateY(0px) !important;
}

/* ── Results section ── */
.results-wrap {
    padding: 0 2.5rem 3rem;
    position: relative; z-index: 1;
}
.results-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 1.5rem; margin-top: 0;
}
@media (max-width: 700px) { .results-grid { grid-template-columns: 1fr; } }

.result-card {
    border-radius: 20px; padding: 2rem;
    position: relative; overflow: hidden;
    border: 1px solid;
}
.result-card.addiction {
    background: linear-gradient(135deg, #1a0a24 0%, #0d0218 100%);
    border-color: var(--accent2);
}
.result-card.mental {
    background: linear-gradient(135deg, #001a24 0%, #020d18 100%);
    border-color: var(--accent1);
}
.result-card::before {
    content: '';
    position: absolute; top: -40px; right: -40px;
    width: 150px; height: 150px;
    border-radius: 50%;
}
.result-card.addiction::before {
    background: radial-gradient(circle, rgba(124,58,237,0.25) 0%, transparent 70%);
}
.result-card.mental::before {
    background: radial-gradient(circle, rgba(0,229,255,0.2) 0%, transparent 70%);
}
.result-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem; font-weight: 700;
    letter-spacing: 0.18em; text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.result-card.addiction .result-label { color: var(--accent2); }
.result-card.mental .result-label { color: var(--accent1); }

.result-score {
    font-family: 'Syne', sans-serif;
    font-size: 4rem; font-weight: 800; line-height: 1;
    margin: 0.25rem 0 0.5rem;
}
.result-card.addiction .result-score { color: var(--accent3); }
.result-card.mental .result-score { color: var(--accent1); }

.result-desc { color: var(--muted); font-size: 0.85rem; line-height: 1.5; }

.score-bar-wrap { margin: 1rem 0 0.5rem; }
.score-bar-track {
    height: 6px; background: var(--border);
    border-radius: 99px; overflow: hidden;
}
.score-bar-fill {
    height: 100%; border-radius: 99px;
    transition: width 1s cubic-bezier(.4,0,.2,1);
}
.result-card.addiction .score-bar-fill {
    background: linear-gradient(90deg, var(--accent2), var(--accent3));
    box-shadow: 0 0 10px rgba(244,114,182,0.5);
}
.result-card.mental .score-bar-fill {
    background: linear-gradient(90deg, var(--accent1), #34d399);
    box-shadow: 0 0 10px rgba(0,229,255,0.4);
}
.score-range {
    display: flex; justify-content: space-between;
    font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem;
}

.verdict-badge {
    display: inline-block;
    border-radius: 8px; padding: 3px 10px;
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.05em; margin-top: 0.5rem;
}

/* ── Section divider ── */
.divider {
    border: none; border-top: 1px solid var(--border);
    margin: 0 2.5rem;
}

/* ── Footer ── */
.footer {
    text-align: center; padding: 2rem;
    color: var(--muted); font-size: 0.78rem;
    border-top: 1px solid var(--border);
    position: relative; z-index: 1;
}
.footer span {
    background: linear-gradient(90deg, var(--accent1), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text; font-weight: 700;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  LOAD MODELS & DATA
# ─────────────────────────────────────────
@st.cache_resource
def load_models():
    lr_add = joblib.load("addiction_model.pkl")
    lr_mh  = joblib.load("mental_health_model.pkl")
    return lr_add, lr_mh

@st.cache_data
def load_feature_columns():
    df = pd.read_csv("social-media.csv")
    df["Affects_Academic_Performance"] = df["Affects_Academic_Performance"].map({"Yes": 1, "No": 0})
    df_dummies = pd.get_dummies(df, drop_first=True)
    return df_dummies.drop(columns=["Addicted_Score", "Mental_Health_Score", "Student_ID"]).columns

lr_add, lr_mh = load_models()
feature_columns = load_feature_columns()

# ─────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">📱 AI-Powered Analysis Tool</div>
    <h1 class="hero-title">Social<br>Sense AI</h1>
    <p class="hero-sub">Predict student social media addiction & mental health impact using machine learning — powered by real behavioral data.</p>

</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
#  INPUT GRID
# ─────────────────────────────────────────
st.markdown('<div class="main-grid">', unsafe_allow_html=True)

# ── Col 1: Personal Info ──
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="section-card">
        <div class="card-icon">👤</div>
        <div class="card-title">Personal Info</div>
    </div>
    """, unsafe_allow_html=True)
    age      = st.number_input("Age", 15, 30, 20)
    gender   = st.selectbox("Gender", ["Male", "Female"])
    academic = st.selectbox("Academic Level", ["High School", "Undergraduate", "Graduate"])
    country  = st.text_input("Country", "India")
    relationship = st.selectbox("Relationship Status", ["Single", "In Relationship", "Complicated"])

with col2:
    st.markdown("""
    <div class="section-card">
        <div class="card-icon">📲</div>
        <div class="card-title">Usage Patterns</div>
    </div>
    """, unsafe_allow_html=True)
    usage    = st.number_input("Avg Daily Usage (hrs)", 0.0, 12.0, 4.5, step=0.5)
    platform = st.selectbox("Most Used Platform",
                            ["Instagram", "Facebook", "Twitter", "TikTok", "YouTube", "WhatsApp", "WeChat"])
    affects  = st.selectbox("Affects Academic Performance?", ["Yes", "No"])
    conflicts = st.slider("Conflicts Over Social Media", 0, 10, 2)

with col3:
    st.markdown("""
    <div class="section-card">
        <div class="card-icon">🧠</div>
        <div class="card-title">Wellbeing</div>
    </div>
    """, unsafe_allow_html=True)
    sleep    = st.number_input("Sleep Hours / Night", 0.0, 12.0, 7.0, step=0.5)
    mh_score = st.slider("Self-Perceived Mental Health", 1, 10, 6)
    st.markdown("""
    <div style="background:#0d1424;border:1px solid #1e2d45;border-radius:12px;padding:1rem;margin-top:0.5rem;">
        <p style="color:#64748b;font-size:0.75rem;margin:0;line-height:1.6;">
            💡 <strong style="color:#94a3b8">Tip:</strong> Fill all fields accurately for best prediction results. The model was trained on 1,000+ student profiles.
        </p>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-grid

# ─────────────────────────────────────────
#  PREDICT BUTTON
# ─────────────────────────────────────────
st.markdown('<div class="predict-wrap">', unsafe_allow_html=True)
predict_btn = st.button("⚡  Run Prediction Analysis")
st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  RESULTS
# ─────────────────────────────────────────
if predict_btn:
    input_data = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Academic_Level": academic,
        "Country": country,
        "Avg_Daily_Usage_Hours": usage,
        "Most_Used_Platform": platform,
        "Affects_Academic_Performance": 1 if affects == "Yes" else 0,
        "Sleep_Hours_Per_Night": sleep,
        "Mental_Health_Score": mh_score,
        "Relationship_Status": relationship,
        "Conflicts_Over_Social_Media": conflicts
    }])

    input_data = pd.get_dummies(input_data, drop_first=True)
    input_data = input_data.reindex(columns=feature_columns, fill_value=0)

    addiction_pred    = lr_add.predict(input_data)[0]
    mental_health_pred = lr_mh.predict(input_data)[0]

    # Clamp to 0–10 for bar %
    add_pct = min(max(addiction_pred / 10 * 100, 0), 100)
    mh_pct  = min(max(mental_health_pred / 10 * 100, 0), 100)

    # Verdict helpers
    def addiction_verdict(score):
        if score < 4:   return ("🟢 Low Risk",   "#10b981", "#052e16")
        elif score < 7: return ("🟡 Moderate",    "#f59e0b", "#1c1400")
        else:           return ("🔴 High Risk",   "#ef4444", "#1f0505")

    def mh_verdict(score):
        if score >= 7:  return ("🟢 Good",        "#10b981", "#052e16")
        elif score >= 4:return ("🟡 Fair",         "#f59e0b", "#1c1400")
        else:           return ("🔴 Concerning",   "#ef4444", "#1f0505")

    av_label, av_color, av_bg = addiction_verdict(addiction_pred)
    mv_label, mv_color, mv_bg = mh_verdict(mental_health_pred)

    st.markdown('<div class="results-wrap">', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="results-grid">

      <!-- Addiction Card -->
      <div class="result-card addiction">
        <div class="result-label">Addiction Score</div>
        <div class="result-score">{addiction_pred:.1f}</div>
        <div class="score-bar-wrap">
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{add_pct:.0f}%"></div>
          </div>
          <div class="score-range"><span>0 · None</span><span>5 · Moderate</span><span>10 · Severe</span></div>
        </div>
        <div class="verdict-badge" style="background:{av_bg};color:{av_color};border:1px solid {av_color}40">{av_label}</div>
        <p class="result-desc" style="margin-top:0.75rem">
          {"This student shows minimal signs of addictive social media use." if addiction_pred < 4 else
           "Moderate usage patterns detected. Some behavioural intervention may help." if addiction_pred < 7 else
           "High addiction indicators found. Professional guidance is recommended."}
        </p>
      </div>

      <!-- Mental Health Card -->
      <div class="result-card mental">
        <div class="result-label">Mental Health Score</div>
        <div class="result-score">{mental_health_pred:.1f}</div>
        <div class="score-bar-wrap">
          <div class="score-bar-track">
            <div class="score-bar-fill" style="width:{mh_pct:.0f}%"></div>
          </div>
          <div class="score-range"><span>0 · Poor</span><span>5 · Average</span><span>10 · Excellent</span></div>
        </div>
        <div class="verdict-badge" style="background:{mv_bg};color:{mv_color};border:1px solid {mv_color}40">{mv_label}</div>
        <p class="result-desc" style="margin-top:0.75rem">
          {"Excellent mental health indicators — keep up healthy habits!" if mental_health_pred >= 7 else
           "Satisfactory mental health. A few lifestyle tweaks could improve wellbeing." if mental_health_pred >= 4 else
           "Low mental health score detected. Consider speaking with a counsellor."}
        </p>
      </div>

    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────
st.markdown("""
<div class="footer">
    Built with <span>SocialSense AI</span> · Data Analysis Mini-Project · Logistic Regression Models
</div>
""", unsafe_allow_html=True)