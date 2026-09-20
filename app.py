import inspect
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Diabetes Risk AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

GREEN, ORANGE, RED, BLUE = "#16a34a", "#f59e0b", "#dc2626", "#2563eb"


def html(s: str) -> str:
    """Remove indentation and blank lines so Markdown never shows HTML as code."""
    return " ".join(line.strip() for line in s.strip().splitlines() if line.strip())


def show_chart(fig, key):
    """Works on both old and new Streamlit versions."""
    if "width" in inspect.signature(st.plotly_chart).parameters:
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)
# =========================================================
# LOAD MODEL FILES
# =========================================================
@st.cache_resource
def load_files():
    model = joblib.load("diabetes_model.pkl")
    encoders = joblib.load("encoders.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, encoders, scaler


try:
    model, encoders, scaler = load_files()
except FileNotFoundError as e:
    st.error(
        f"Missing file: {e.filename}. Keep diabetes_model.pkl, "
        "encoders.pkl and scaler.pkl in the same folder as app.py."
    )
    st.stop()

DEFAULT_FEATURES = [
    "age", "gender", "city", "bmi", "family_history_diabetes",
    "physical_activity_level", "diet_type", "smoking_status",
    "alcohol_consumption", "hours_sleep_per_night", "stress_level",
    "fasting_blood_sugar", "hba1c_level", "blood_pressure_systolic",
    "blood_pressure_diastolic", "waist_circumference_cm", "income_bracket",
]
# The exact columns (and order) the model was trained on
FEATURES = list(getattr(scaler, "feature_names_in_", DEFAULT_FEATURES))


def want(name: str) -> bool:
    """Show an input only if the model actually uses it."""
    return name in FEATURES


LABELS = {
    "age": "Age", "gender": "Gender", "city": "City", "bmi": "BMI",
    "family_history_diabetes": "Family History",
    "physical_activity_level": "Physical Activity", "diet_type": "Diet Type",
    "smoking_status": "Smoking", "alcohol_consumption": "Alcohol",
    "hours_sleep_per_night": "Sleep Hours", "stress_level": "Stress Level",
    "fasting_blood_sugar": "Fasting Blood Sugar", "hba1c_level": "HbA1c",
    "blood_pressure_systolic": "Systolic BP",
    "blood_pressure_diastolic": "Diastolic BP",
    "waist_circumference_cm": "Waist Size", "income_bracket": "Income Bracket",
}

# =========================================================
# COLOR THEMES  (copy a block to add your own)
# bg1-bg4 = page background, p1-p3 = main accent colours,
# s1-s3 = sidebar gradient, card = card colour
# =========================================================
THEMES = {
    "💜 Berry Violet": {
        "bg1": "#e0e7ff", "bg2": "#fce7f3", "bg3": "#f8fafc", "bg4": "#eef2ff",
        "p1": "#4f46e5", "p2": "#7c3aed", "p3": "#db2777",
        "s1": "#312e81", "s2": "#4f46e5", "s3": "#7c3aed",
        "title": "#312e81", "card": "rgba(255,255,255,0.94)",
        "text": "#0f172a", "muted": "#64748b", "glow": "rgba(124,58,237,0.32)",
    },
    "🌊 Ocean Teal": {
        "bg1": "#cffafe", "bg2": "#d1fae5", "bg3": "#f0fdfa", "bg4": "#ecfeff",
        "p1": "#0f766e", "p2": "#0891b2", "p3": "#2563eb",
        "s1": "#134e4a", "s2": "#0f766e", "s3": "#0891b2",
        "title": "#134e4a", "card": "rgba(255,255,255,0.94)",
        "text": "#0f172a", "muted": "#475569", "glow": "rgba(8,145,178,0.32)",
    },
    "🌅 Sunset Coral": {
        "bg1": "#ffedd5", "bg2": "#fee2e2", "bg3": "#fff7ed", "bg4": "#fef2f2",
        "p1": "#ea580c", "p2": "#e11d48", "p3": "#be185d",
        "s1": "#7c2d12", "s2": "#c2410c", "s3": "#e11d48",
        "title": "#7c2d12", "card": "rgba(255,255,255,0.94)",
        "text": "#1c1917", "muted": "#57534e", "glow": "rgba(225,29,72,0.30)",
    },
    "🌿 Forest Mint": {
        "bg1": "#dcfce7", "bg2": "#ecfccb", "bg3": "#f7fee7", "bg4": "#f0fdf4",
        "p1": "#166534", "p2": "#16a34a", "p3": "#65a30d",
        "s1": "#14532d", "s2": "#166534", "s3": "#16a34a",
        "title": "#14532d", "card": "rgba(255,255,255,0.94)",
        "text": "#052e16", "muted": "#3f6212", "glow": "rgba(22,163,74,0.30)",
    },
    "👑 Royal Navy & Gold": {
        "bg1": "#e0e7ff", "bg2": "#fef3c7", "bg3": "#f8fafc", "bg4": "#eff6ff",
        "p1": "#1e3a8a", "p2": "#1d4ed8", "p3": "#d97706",
        "s1": "#0f172a", "s2": "#1e3a8a", "s3": "#1d4ed8",
        "title": "#1e3a8a", "card": "rgba(255,255,255,0.96)",
        "text": "#0f172a", "muted": "#475569", "glow": "rgba(29,78,216,0.30)",
    },
    "🌸 Rose Blossom": {
        "bg1": "#fce7f3", "bg2": "#fae8ff", "bg3": "#fff1f2", "bg4": "#fdf2f8",
        "p1": "#be185d", "p2": "#db2777", "p3": "#a855f7",
        "s1": "#831843", "s2": "#be185d", "s3": "#c026d3",
        "title": "#831843", "card": "rgba(255,255,255,0.94)",
        "text": "#1f1230", "muted": "#6b5a72", "glow": "rgba(219,39,119,0.30)",
    },
    "🌙 Midnight Neon": {
        "bg1": "#1e1b4b", "bg2": "#164e63", "bg3": "#0f172a", "bg4": "#020617",
        "p1": "#6366f1", "p2": "#8b5cf6", "p3": "#06b6d4",
        "s1": "#020617", "s2": "#1e1b4b", "s3": "#312e81",
        "title": "#c7d2fe", "card": "rgba(30,41,59,0.90)",
        "text": "#e2e8f0", "muted": "#94a3b8", "glow": "rgba(99,102,241,0.40)",
    },
}
# The theme picker lives in the sidebar (key="theme"); we read its value here
theme_name = st.session_state.get("theme", next(iter(THEMES)))
t = THEMES.get(theme_name, next(iter(THEMES.values())))
VARS = f"""
:root {{
    --bg1: {t['bg1']}; --bg2: {t['bg2']}; --bg3: {t['bg3']}; --bg4: {t['bg4']};
    --p1: {t['p1']}; --p2: {t['p2']}; --p3: {t['p3']};
    --s1: {t['s1']}; --s2: {t['s2']}; --s3: {t['s3']};
    --title: {t['title']}; --card: {t['card']}; --text: {t['text']};
    --muted: {t['muted']}; --glow: {t['glow']};
}}
"""
# =========================================================
# CSS
# =========================================================
CSS = """

.stApp {
    background:
        radial-gradient(circle at top left, var(--bg1) 0%, transparent 35%),
        radial-gradient(circle at top right, var(--bg2) 0%, transparent 35%),
        linear-gradient(135deg, var(--bg3), var(--bg4));
    background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.5rem; padding-bottom: 3rem; max-width: 1250px; }
div[data-testid="stWidgetLabel"] p { color: var(--text) !important; font-weight: 600; }

@keyframes fadeup { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: none; } }
@keyframes grow { from { width: 0; } }

/* ---------- HERO ---------- */
.hero {
    background: linear-gradient(135deg, var(--p1), var(--p2), var(--p3));
    padding: 35px; border-radius: 25px; text-align: center; margin-bottom: 25px;
    box-shadow: 0 12px 35px var(--glow); animation: fadeup .6s ease both;
}
.hero h1, .hero p { color: #fff !important; }
.hero h1 { font-size: 44px; font-weight: 800; margin-bottom: 5px; }
.hero p { font-size: 18px; opacity: .95; }

/* ---------- SECTION HEADERS ---------- */
.section-card {
    background: var(--card); padding: 18px 25px; border-radius: 20px;
    margin: 22px 0 14px; border-left: 6px solid var(--p2);
    box-shadow: 0 8px 25px rgba(0,0,0,.08);
}
.section-title { font-size: 22px; font-weight: 750; color: var(--title); }

/* ---------- SNAPSHOT CARDS ---------- */
.snap {
    background: var(--card); border-radius: 20px; padding: 18px 10px; text-align: center;
    box-shadow: 0 8px 25px rgba(0,0,0,.08); transition: all .25s ease;
    animation: fadeup .6s ease both;
}
.snap:hover { transform: translateY(-5px); box-shadow: 0 14px 30px var(--glow); }
.snap-label { color: var(--muted); font-size: 14px; font-weight: 600; }
.snap-value { font-size: 32px; font-weight: 800; color: var(--text); margin: 6px 0 10px; }
.badge {
    display: inline-block; color: #fff; font-weight: 700; font-size: 13px;
    padding: 4px 14px; border-radius: 999px;
}

/* ---------- BUTTON ---------- */
.stButton > button {
    height: 54px; padding: 0 30px; border-radius: 14px; border: none;
    background: linear-gradient(90deg, var(--p1), var(--p2), var(--p3));
    color: #fff; font-size: 17px; font-weight: 700;
    box-shadow: 0 8px 20px var(--glow); transition: all .25s ease;
}
.stButton > button p { color: #fff !important; font-weight: 700; }
.stButton > button:hover {
    transform: translateY(-3px); box-shadow: 0 12px 26px var(--glow); color: #fff;
}

/* ---------- RESULT CARD ---------- */
.result-card {
    margin-top: 26px; padding: 40px 30px; border-radius: 28px; text-align: center;
    background: var(--card); color: var(--text); box-shadow: 0 10px 35px rgba(0,0,0,.12);
    animation: fadeup .6s ease both;
}
.result-card p { color: var(--text); font-size: 16px; margin: 10px 0; }
.result-label { font-size: 21px; color: var(--muted); }
.result-value { font-size: 52px; font-weight: 850; margin: 14px 0; }

/* ---------- PROBABILITY BARS ---------- */
.prob-wrap { background: var(--card); padding: 22px 25px; border-radius: 20px; box-shadow: 0 8px 25px rgba(0,0,0,.08); }
.prob-row { margin-bottom: 16px; }
.prob-head { display: flex; justify-content: space-between; font-weight: 700; color: var(--text); margin-bottom: 6px; }
.prob-track { background: rgba(148,163,184,.25); border-radius: 999px; height: 16px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 999px; min-width: 4px; animation: grow 1s ease; }

/* ---------- TIPS ---------- */
.tip {
    background: var(--card); border-radius: 14px; padding: 15px 20px; margin-bottom: 12px;
    border-left: 5px solid var(--p2); color: var(--text); font-size: 16px;
    box-shadow: 0 5px 18px rgba(0,0,0,.07); animation: fadeup .5s ease both;
    transition: transform .2s ease;
}
.tip:hover { transform: translateX(6px); }

/* ---------- TABS ---------- */
button[data-baseweb="tab"] p { font-size: 15px; font-weight: 600; color: var(--text); }
button[data-baseweb="tab"][aria-selected="true"] p { color: var(--p2) !important; }
div[data-baseweb="tab-highlight"] { background-color: var(--p2) !important; }
div[data-baseweb="tab-border"] { background-color: rgba(148,163,184,.35) !important; }

/* ---------- HEADINGS / CAPTIONS follow the theme ---------- */
.stApp h2, .stApp h3, .stApp h4 { color: var(--text) !important; }
div[data-testid="stCaptionContainer"] { color: var(--muted) !important; }

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--s1), var(--s2) 55%, var(--s3)); }
section[data-testid="stSidebar"] * { color: #fff !important; }
section[data-testid="stSidebar"] div[data-baseweb="select"] * { color: #0f172a !important; }
.side-title { font-size: 26px; font-weight: 800; margin: 10px 0 30px; }
.side-head { font-size: 21px; font-weight: 700; margin: 6px 0 12px; }
.side-list { font-size: 16px; line-height: 2; }
.side-line { border-top: 1px solid rgba(255,255,255,.2); margin: 26px 0; }
.side-note { background: rgba(255,255,255,.12); border-radius: 14px; padding: 16px; font-size: 15px; line-height: 1.6; }

/* ---------- FOOTER ---------- */
.footer { text-align: center; color: var(--muted); margin-top: 35px; padding: 20px; font-size: 13px; }
"""
st.markdown("<style>" + VARS + CSS + "</style>", unsafe_allow_html=True)


def section(title: str):
    st.markdown(
        html(f'<div class="section-card"><div class="section-title">{title}</div></div>'),
        unsafe_allow_html=True,
    )


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown('<div class="side-title">🩺 Diabetes Risk AI</div>', unsafe_allow_html=True)
    st.selectbox("🎨 Color Theme", list(THEMES.keys()), key="theme")
    st.markdown(
        html("""
        <div class="side-line"></div>
        <div class="side-head">🚀 How to use</div>
        <div class="side-list">
            1️⃣ Fill in your details<br>
            2️⃣ Watch the live health snapshot update<br>
            3️⃣ Press Analyze<br>
            4️⃣ Read your risk + tips
        </div>
        <div class="side-line"></div>
        <div class="side-head">⚙️ Technology</div>
        <div class="side-list">
            🐍 Python<br>
            🌲 Random Forest (scikit-learn)<br>
            📊 Pandas • Plotly<br>
            🎨 Streamlit + HTML/CSS
        </div>
        <div class="side-line"></div>
        <div class="side-note">
            This is a screening tool for education. It is <b>not</b> a medical diagnosis.
        </div>
        """),
        unsafe_allow_html=True,
    )

# =========================================================
# HERO
# =========================================================
st.markdown(
    html("""
    <div class="hero">
        <h1>🩺 Diabetes Risk AI</h1>
        <p>Machine Learning Based Diabetes Risk Screening</p>
        <p>Enter your health and lifestyle details to get an estimate</p>
    </div>
    """),
    unsafe_allow_html=True,
)

# =========================================================
# INPUTS (an input is shown only if the model uses it)
# =========================================================
values = {}

section("👤 Personal Information")
cols = st.columns(3)
with cols[0]:
    values["age"] = st.number_input("🎂 Age", min_value=1, max_value=120, value=25)
with cols[1]:
    if want("gender"):
        values["gender"] = st.selectbox("⚧ Gender", encoders["gender"].classes_)
with cols[2]:
    if want("city"):
        values["city"] = st.selectbox("📍 City", encoders["city"].classes_)

section("❤️ Health Information")
c1, c2, c3 = st.columns(3)
with c1:
    values["bmi"] = st.number_input(
        "⚖️ BMI", min_value=10.0, max_value=70.0, value=25.0, step=0.1,
        help="Body Mass Index = weight (kg) / height (m) squared.",
    )
with c2:
    values["family_history_diabetes"] = st.selectbox(
        "👨‍👩‍👧 Family History", encoders["family_history_diabetes"].classes_
    )
with c3:
    values["physical_activity_level"] = st.selectbox(
        "🏃 Physical Activity", encoders["physical_activity_level"].classes_
    )

c1, c2, c3 = st.columns(3)
with c1:
    values["diet_type"] = st.selectbox("🥗 Diet Type", encoders["diet_type"].classes_)
with c2:
    values["smoking_status"] = st.selectbox("🚬 Smoking Status", encoders["smoking_status"].classes_)
with c3:
    values["alcohol_consumption"] = st.selectbox(
        "🥤 Alcohol Consumption", encoders["alcohol_consumption"].classes_
    )

section("🌿 Lifestyle Information")
c1, c2, c3 = st.columns(3)
with c1:
    values["hours_sleep_per_night"] = st.number_input(
        "😴 Hours of Sleep", min_value=0.0, max_value=24.0, value=7.0, step=0.5
    )
with c2:
    values["stress_level"] = st.slider("🧘 Stress Level", min_value=1, max_value=10, value=5)
with c3:
    if want("income_bracket"):
        values["income_bracket"] = st.selectbox(
            "💰 Income Bracket", encoders["income_bracket"].classes_
        )

section("🧪 Medical Measurements")
c1, c2, c3 = st.columns(3)
with c1:
    values["fasting_blood_sugar"] = st.number_input(
        "🩸 Fasting Blood Sugar (mg/dL)", min_value=40, max_value=400, value=100
    )
with c2:
    values["hba1c_level"] = st.number_input(
        "🧪 HbA1c Level (%)", min_value=2.0, max_value=20.0, value=5.5, step=0.1,
        help="Your average blood sugar over the last 2-3 months.",
    )
with c3:
    values["waist_circumference_cm"] = st.number_input(
        "📏 Waist Circumference (cm)", min_value=30.0, max_value=200.0, value=80.0, step=0.1
    )

c1, c2 = st.columns(2)
with c1:
    values["blood_pressure_systolic"] = st.number_input(
        "💓 Systolic Blood Pressure", min_value=50, max_value=300, value=120
    )
with c2:
    values["blood_pressure_diastolic"] = st.number_input(
        "💓 Diastolic Blood Pressure", min_value=30, max_value=200, value=80
    )


# =========================================================
# LIVE HEALTH SNAPSHOT (updates every time an input changes)
# =========================================================
def bmi_status(v):
    if v < 18.5:
        return "Underweight", BLUE
    if v < 25:
        return "Healthy", GREEN
    if v < 30:
        return "Overweight", ORANGE
    return "Obese", RED


def bp_status(sys_, dia):
    if sys_ < 120 and dia < 80:
        return "Normal", GREEN
    if sys_ < 140 and dia < 90:
        return "Elevated", ORANGE
    return "High", RED


def sugar_status(v):
    if v < 100:
        return "Normal", GREEN
    if v < 126:
        return "Prediabetic range", ORANGE
    return "Diabetic range", RED


def a1c_status(v):
    if v < 5.7:
        return "Normal", GREEN
    if v < 6.5:
        return "Prediabetic range", ORANGE
    return "Diabetic range", RED


section("📈 Live Health Snapshot")

sys_, dia = values["blood_pressure_systolic"], values["blood_pressure_diastolic"]
snapshot = [
    ("⚖️ BMI", f"{values['bmi']:.1f}", *bmi_status(values["bmi"])),
    ("💓 Blood Pressure", f"{sys_}/{dia}", *bp_status(sys_, dia)),
    ("🩸 Fasting Sugar", f"{values['fasting_blood_sugar']}", *sugar_status(values["fasting_blood_sugar"])),
    ("🧪 HbA1c", f"{values['hba1c_level']:.1f}%", *a1c_status(values["hba1c_level"])),
]
for col, (label, value, badge, color) in zip(st.columns(4), snapshot):
    with col:
        st.markdown(
            html(
                f'<div class="snap"><div class="snap-label">{label}</div>'
                f'<div class="snap-value">{value}</div>'
                f'<span class="badge" style="background:{color};">{badge}</span></div>'
            ),
            unsafe_allow_html=True,
        )
# =========================================================
# SMART INPUT VALIDATION
# =========================================================
def validate_inputs(v):
    warnings = []

    if v["fasting_blood_sugar"] >= 126:
        warnings.append(
            "🩸 Fasting blood sugar is in the diabetes range. "
            "A healthcare professional should confirm this with appropriate testing."
        )
    elif v["fasting_blood_sugar"] >= 100:
        warnings.append(
            "🩸 Fasting blood sugar is in the prediabetes range."
        )

    if v["hba1c_level"] >= 6.5:
        warnings.append(
            "🧪 HbA1c is in the diabetes range. "
            "Please discuss the result with a healthcare professional."
        )
    elif v["hba1c_level"] >= 5.7:
        warnings.append(
            "🧪 HbA1c is in the prediabetes range."
        )

    if v["blood_pressure_systolic"] >= 140 or v["blood_pressure_diastolic"] >= 90:
        warnings.append(
            "💓 Blood pressure is high based on the entered reading."
        )

    if v["bmi"] < 18.5:
        warnings.append("⚖️ BMI is below the commonly used adult healthy range.")
    elif v["bmi"] >= 30:
        warnings.append("⚖️ BMI is in the obesity range.")

    if v["hours_sleep_per_night"] < 6:
        warnings.append("😴 You entered less than 6 hours of sleep.")

    if v["stress_level"] >= 8:
        warnings.append("🧘 Your entered stress level is high.")

    return warnings

# =========================================================
# ANALYZE BUTTON
# =========================================================
st.markdown("<br>", unsafe_allow_html=True)

RISK_STYLE = {
    "High": {
        "emoji": "🔴", "label": "HIGH RISK", "color": RED,
        "note": "Please talk to a doctor soon. This is a screening estimate, not a diagnosis.",
    },
    "Moderate": {
        "emoji": "🟠", "label": "MODERATE RISK", "color": ORANGE,
        "note": "Some factors need attention. Small lifestyle changes can make a big difference.",
    },
    "Low": {
        "emoji": "🟢", "label": "LOW RISK", "color": GREEN,
        "note": "Keep up the healthy habits! Regular check-ups are still a good idea.",
    },
}
# =========================================================
# ANALYZE BUTTON
# =========================================================

if st.button("🔍 Analyze Diabetes Risk"):

    input_data = pd.DataFrame([[
        values["age"],
        encoders["gender"].transform([values["gender"]])[0],
        encoders["city"].transform([values["city"]])[0],
        values["bmi"],
        encoders["family_history_diabetes"].transform(
            [values["family_history_diabetes"]
        ])[0],
        encoders["physical_activity_level"].transform(
            [values["physical_activity_level"]
        ])[0],
        encoders["diet_type"].transform(
            [values["diet_type"]
        ])[0],
        encoders["smoking_status"].transform(
            [values["smoking_status"]
        ])[0],
        encoders["alcohol_consumption"].transform(
            [values["alcohol_consumption"]
        ])[0],
        values["hours_sleep_per_night"],
        values["stress_level"],
        values["fasting_blood_sugar"],
        values["hba1c_level"],
        values["blood_pressure_systolic"],
        values["blood_pressure_diastolic"],
        values["waist_circumference_cm"],
        encoders["income_bracket"].transform(
            [values["income_bracket"]
        ])[0],
    ]], columns=FEATURES)

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probabilities = model.predict_proba(input_scaled)[0]

    st.session_state["result"] = {
        "prediction": prediction,
        "probs": probabilities,
        "classes": model.classes_,
        "inputs": values.copy(),
    }

    st.rerun()
#
validation_warnings = validate_inputs(values)

if validation_warnings:
    st.markdown("### ⚠️ Health Check")
    for warning in validation_warnings:
        st.warning(warning)
else:
    st.success("✅ No major warning flags detected from the entered values.")

# =========================================================
# TIPS
# =========================================================
def build_tips(v):
    tips = []

    fbs = v["fasting_blood_sugar"]
    if fbs >= 126:
        tips.append(("🩸", "Fasting sugar is 126 mg/dL or higher (diabetic range). Please get it checked by a doctor."))
    elif fbs >= 100:
        tips.append(("🩸", "Fasting sugar is 100–125 mg/dL (prediabetic range). Cut back on sugary foods and refined carbs."))

    a1c = v["hba1c_level"]
    if a1c >= 6.5:
        tips.append(("🧪", "HbA1c is 6.5% or higher (diabetic range). Please see a doctor for a proper test."))
    elif a1c >= 5.7:
        tips.append(("🧪", "HbA1c is 5.7–6.4% (prediabetic range). Diet and exercise can bring it down."))

    bmi = v["bmi"]
    if bmi >= 25:
        tips.append(("⚖️", "BMI is above the healthy range. Even a 5% weight loss can improve blood sugar control."))
    elif bmi < 18.5:
        tips.append(("⚖️", "BMI is below the healthy range. Eat balanced meals and talk to a doctor if unsure."))

    s, d = v["blood_pressure_systolic"], v["blood_pressure_diastolic"]
    if s >= 120 or d >= 80:
        tips.append(("💓", "Blood pressure is elevated. Reduce salt and keep track of it regularly."))

    sleep = v["hours_sleep_per_night"]
    if sleep < 6:
        tips.append(("😴", "You sleep less than 6 hours. Aim for 7–8 hours, poor sleep affects blood sugar."))
    elif sleep > 9:
        tips.append(("😴", "You sleep more than 9 hours. Try to keep a steady 7–8 hour routine."))

    if v["stress_level"] >= 7:
        tips.append(("🧘", "Stress is high. Try walking, breathing exercises or hobbies to unwind."))

    # NOTE: these checks use the words in your category labels.
    # Change the keyword lists if your dataset uses different words.
    act = str(v.get("physical_activity_level", "")).lower()
    if any(k in act for k in ["low", "sedentary", "none"]):
        tips.append(("🏃", "Try a 30-minute walk most days. Regular movement helps your body use sugar better."))

    smk = str(v.get("smoking_status", "")).lower()
    if any(k in smk for k in ["current", "yes", "regular", "daily", "heavy"]):
        tips.append(("🚬", "Smoking raises diabetes and heart risk. Quitting is one of the best steps you can take."))

    fh = str(v.get("family_history_diabetes", "")).lower()
    if fh in ("yes", "true", "1", "y"):
        tips.append(("👨‍👩‍👧", "Diabetes runs in your family, so get your blood sugar tested once a year."))

    if not tips:
        tips.append(("✅", "Your numbers look healthy. Keep up your habits!"))
    return tips


# =========================================================
# RESULTS
# =========================================================
res = st.session_state.get("result")

if res:
    style = RISK_STYLE.get(res["prediction"], RISK_STYLE["Low"])
    confidence = max(res["probs"]) * 100

    if res["inputs"] != values:
        st.info("You changed some inputs. Press **Analyze** again to update the result.")

    st.markdown(
        html(f"""
        <div class="result-card" style="border-top: 8px solid {style['color']};">
            <div class="result-label">🩸 Predicted Diabetes Risk</div>
            <div class="result-value" style="color: {style['color']};">
                {style['emoji']} {style['label']}
            </div>
            <p>Model confidence: <b>{confidence:.1f}%</b></p>
            <p>{style['note']}</p>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["📊 Risk Meter", "🌲 What Matters Most", "💡 Personal Tips"])

    # ---------- TAB 1: gauge + probability bars ----------
    with tab1:
        left, right = st.columns(2)

        weights = {"Low": 0, "Moderate": 50, "High": 100}
        score = sum(p * weights.get(c, 0) for c, p in zip(res["classes"], res["probs"]))

        with left:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score,
                number={"suffix": "/100", "font": {"size": 40}},
                title={"text": "Risk Score"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": style["color"]},
                    "steps": [
                        {"range": [0, 33], "color": "#bbf7d0"},
                        {"range": [33, 66], "color": "#fde68a"},
                        {"range": [66, 100], "color": "#fecaca"},
                    ],
                },
            ))
            fig.update_layout(
                height=320, margin=dict(l=25, r=25, t=70, b=10),
                paper_bgcolor="rgba(0,0,0,0)", font={"color": t["text"]},
            )
            show_chart(fig, key="gauge")

        with right:
            st.markdown("#### 📊 Prediction Probability")
            order = {"High": 0, "Moderate": 1, "Low": 2}
            pairs = sorted(zip(res["classes"], res["probs"]), key=lambda x: order.get(x[0], 9))
            rows = ""
            for cls, prob in pairs:
                color = RISK_STYLE.get(cls, RISK_STYLE["Low"])["color"]
                pct = prob * 100
                rows += (
                    f'<div class="prob-row">'
                    f'<div class="prob-head"><span>{cls}</span><span>{pct:.1f}%</span></div>'
                    f'<div class="prob-track">'
                    f'<div class="prob-fill" style="width:{pct:.1f}%; background:{color};"></div>'
                    f'</div></div>'
                )
            st.markdown(f'<div class="prob-wrap">{rows}</div>', unsafe_allow_html=True)

    # ---------- TAB 2: feature importance ----------
    with tab2:
        st.markdown("#### 🌲 Which factors does the model rely on most?")
        if hasattr(model, "feature_importances_") and len(model.feature_importances_) == len(FEATURES):
            imp = pd.DataFrame({
                "feature": [LABELS.get(f, f) for f in FEATURES],
                "importance": model.feature_importances_ * 100,
            }).sort_values("importance").tail(10)

            fig = go.Figure(go.Bar(
                x=imp["importance"], y=imp["feature"], orientation="h",
                marker=dict(
                    color=imp["importance"],
                    colorscale=[[0, t["p3"]], [1, t["p1"]]],
                ),
                text=[f"{x:.1f}%" for x in imp["importance"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=430, margin=dict(l=10, r=40, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Importance (%)", font={"color": t["text"]},
            )
            show_chart(fig, key="importance")
            st.caption("A longer bar means the model uses that factor more when deciding the risk level.")
        else:
            st.info("This model does not provide feature importance.")

    # ---------- TAB 3: personal tips ----------
    with tab3:
        st.markdown("#### 💡 Tips based on your numbers")
        for emoji, text in build_tips(res["inputs"]):
            st.markdown(html(f'<div class="tip">{emoji} {text}</div>'), unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    html("""
    <div class="footer">
        ⚠️ <b>Important:</b> This application is for educational and informational
        purposes only. It is not a medical diagnosis and does not replace
        professional medical advice.
        <br><br>
        💙 Built with Python • Scikit-learn • Random Forest • Streamlit
    </div>
    """),
    unsafe_allow_html=True,
)