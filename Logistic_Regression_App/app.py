import streamlit as st
import joblib
import numpy as np

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Breast Cancer Detection AI",
    page_icon="🩺",
    layout="wide"
)

st.markdown("""
<style>

/* Background */

.stApp{
    background: linear-gradient(to right,#0f2027,#203a43,#2c5364);
}

/* Main Title */

h1{
    color:white;
    text-align:center;
    font-size:48px;
}

/* Subtitle */

h3{
    color:#dfe6e9;
}

/* Sidebar */

section[data-testid="stSidebar"]{
    background:#111827;
}

/* Buttons */

.stButton>button{

    width:100%;

    height:55px;

    border-radius:15px;

    background:#00b894;

    color:white;

    font-size:22px;

    font-weight:bold;

    border:none;
}

.stButton>button:hover{

    background:#00cec9;
}

/* Metric Cards */

div[data-testid="metric-container"]{

    background:#1f2937;

    border-radius:15px;

    padding:20px;

    box-shadow:0px 0px 15px rgba(0,0,0,.3);

}

/* Number Inputs */

.stNumberInput{

    background:#1f2937;

    padding:8px;

    border-radius:12px;

}

</style>
""",unsafe_allow_html=True)
# ==========================
# LOAD MODEL
# ==========================

model = joblib.load("logistic_model.pkl")

# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🩺 Breast Cancer AI")

st.sidebar.markdown("---")

st.sidebar.success("✅ Logistic Regression Loaded")

st.sidebar.info("""
### Model Information

Algorithm:
- Logistic Regression

Dataset:
- Breast Cancer Wisconsin

Features:
- 30
""")

# ==========================
# HERO SECTION
# ==========================

st.title("🩺 Breast Cancer Detection AI")

st.markdown("""
### AI Powered Diagnosis using Logistic Regression

Analyze patient measurements and predict whether the tumor is:

🟢 **Benign**

or

🔴 **Malignant**
""")

st.divider()
st.markdown("""
<div style="
padding:30px;
border-radius:20px;
background:linear-gradient(90deg,#2563eb,#06b6d4);
color:white;
text-align:center;
margin-bottom:25px;
">

<h1>🩺 Breast Cancer Detection AI</h1>

<h3>AI Powered Diagnosis using Logistic Regression</h3>

<p>
Early prediction helps doctors make better clinical decisions.
</p>

</div>
""", unsafe_allow_html=True)

# ==========================
# DASHBOARD CARDS
# ==========================

c1, c2, c3 = st.columns(3)

with c1:
    st.metric("🤖 Model", "Logistic Regression")

with c2:
    st.metric("📊 Features", "30")

with c3:
    st.metric("🧬 Dataset", "Breast Cancer")

st.divider()


# =====================================
# FEATURE LIST
# =====================================

feature_names = [
    'mean radius',
    'mean texture',
    'mean perimeter',
    'mean area',
    'mean smoothness',
    'mean compactness',
    'mean concavity',
    'mean concave points',
    'mean symmetry',
    'mean fractal dimension',

    'radius error',
    'texture error',
    'perimeter error',
    'area error',
    'smoothness error',
    'compactness error',
    'concavity error',
    'concave points error',
    'symmetry error',
    'fractal dimension error',

    'worst radius',
    'worst texture',
    'worst perimeter',
    'worst area',
    'worst smoothness',
    'worst compactness',
    'worst concavity',
    'worst concave points',
    'worst symmetry',
    'worst fractal dimension'
]

st.subheader("📋 Patient Medical Measurements")

inputs = {}

# ---------------- Mean Features ----------------

st.markdown("## 📋 Mean Features")

col1, col2 = st.columns(2)

for i in range(10):

    if i % 2 == 0:
        with col1:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

# ---------------- Error Features ----------------

st.markdown("---")
st.markdown("## 📊 Error Features")

col1, col2 = st.columns(2)

for i in range(10,20):

    if i % 2 == 0:
        with col1:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

# ---------------- Worst Features ----------------

st.markdown("---")
st.markdown("## 🔬 Worst Features")

col1, col2 = st.columns(2)

for i in range(20,30):

    if i % 2 == 0:
        with col1:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

    else:
        with col2:
            inputs[feature_names[i]] = st.number_input(
                feature_names[i].title(),
                value=0.0,
                format="%.6f"
            )

st.divider()

if st.button("🔬 Analyze Patient", use_container_width=True):

    # Convert Dictionary → List → NumPy Array
    features = np.array([list(inputs.values())])

    # Prediction
    prediction = model.predict(features)[0]

    # Probability
    probability = model.predict_proba(features)[0]

    confidence = np.max(probability) * 100

    st.divider()

    st.subheader("🧠 AI Diagnosis")

    if prediction == 1:

     st.markdown("""
     <div style="
     background:#7f1d1d;
     padding:25px;
     border-radius:15px;
     color:white;
     text-align:center;
     ">

     <h1>🔴 MALIGNANT</h1>

     <h3>Consult an Oncologist Immediately</h3>
 
      </div>
     """, unsafe_allow_html=True)

    else:

     st.markdown("""
     <div style="
     background:#14532d;
     padding:25px;
     border-radius:15px;
     color:white;
     text-align:center;
     ">

     <h1>🟢 BENIGN</h1>

     <h3>No Immediate Cancer Indication</h3>

     </div>
     """, unsafe_allow_html=True)

    st.metric(
    "Prediction Confidence",
    f"{confidence:.2f}%"
    )

    st.progress(confidence/100)

    

    st.divider()

    st.subheader("💡 Recommendation")

    if prediction == 1:

        st.warning("""
    Immediate medical consultation is recommended.

    Further diagnostic tests should be performed.
    """)

    else:

        st.info("""
The tumor appears to be benign.

Regular medical checkups are still recommended.
""")
        
st.markdown("---")

st.markdown("""
<div style="text-align:center;color:lightgray;">

Developed with ❤️ using

<b>Python • Scikit-Learn • Streamlit</b>

</div>
""", unsafe_allow_html=True)