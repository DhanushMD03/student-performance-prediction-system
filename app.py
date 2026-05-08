import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# PAGE CONFIGURATION
# =========================================

st.set_page_config(
    page_title="Student Performance Prediction System",
    page_icon="🎓",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.stProgress > div > div > div > div {
    background-color: #00C853;
}

.result-box {
    padding: 20px;
    border-radius: 10px;
    text-align: center;
    font-size: 25px;
    font-weight: bold;
    margin-top: 10px;
    margin-bottom: 10px;
}

.pass-box {
    background-color: #0f5132;
    color: white;
}

.fail-box {
    background-color: #842029;
    color: white;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# TITLE
# =========================================

st.title("🎓 Student Performance Prediction System")

st.markdown("""
This AI system predicts whether a student is likely to PASS or FAIL
based on academic and personal performance factors.
""")

# =========================================
# LOAD MODEL AND SCALER
# =========================================

try:
    model = joblib.load("student_model.pkl")
    scaler = joblib.load("scaler.pkl")

    st.success("✅ Model and Scaler Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error Loading Model or Scaler: {e}")
    st.stop()

# =========================================
# LOAD DATASET
# =========================================

try:
    df = pd.read_csv("student-mat.csv", sep=";")

    df["Final_Result"] = df["G3"].apply(
        lambda x: 1 if x >= 10 else 0
    )

    st.success("✅ Dataset Loaded Successfully")

except Exception as e:
    st.error(f"❌ Error Loading Dataset: {e}")
    st.stop()

# =========================================
# DASHBOARD METRICS
# =========================================

total_students = len(df)

pass_students = len(df[df['Final_Result'] == 1])

fail_students = len(df[df['Final_Result'] == 0])

pass_rate = (pass_students / total_students) * 100

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", total_students)

col2.metric("Pass Students", pass_students)

col3.metric("Fail Students", fail_students)

col4.metric("Pass Rate", f"{pass_rate:.2f}%")

# =========================================
# SIDEBAR INPUTS
# =========================================

st.sidebar.header("Enter Student Details")

studytime = st.sidebar.slider("Study Time", 1, 4, 2)

failures = st.sidebar.slider("Failures", 0, 4, 0)

absences = st.sidebar.slider("Absences", 0, 100, 5)

G1 = st.sidebar.slider("G1 Marks", 0, 20, 10)

G2 = st.sidebar.slider("G2 Marks", 0, 20, 10)

Medu = st.sidebar.slider("Mother Education", 0, 4, 2)

Fedu = st.sidebar.slider("Father Education", 0, 4, 2)

traveltime = st.sidebar.slider("Travel Time", 1, 4, 1)

internet = st.sidebar.selectbox(
    "Internet Access",
    [0, 1]
)

higher = st.sidebar.selectbox(
    "Higher Education Interest",
    [0, 1]
)

# =========================================
# PREDICTION
# =========================================

if st.sidebar.button("Predict Performance"):

    try:

        new_student = [[
            studytime,
            failures,
            absences,
            G1,
            G2,
            Medu,
            Fedu,
            traveltime,
            internet,
            higher
        ]]

        # Scale input
        new_student_scaled = scaler.transform(new_student)

        # Predict
        prediction = model.predict(new_student_scaled)

        # Probability
        probability = model.predict_proba(new_student_scaled)

        pass_probability = probability[0][1] * 100
        fail_probability = probability[0][0] * 100

        # =========================================
        # VISUAL RESULT SECTION
        # =========================================

        st.header("🎯 Prediction Dashboard")

        result_col1, result_col2 = st.columns(2)

        # =========================================
        # RESULT CARD
        # =========================================

        with result_col1:

            if prediction[0] == 1:

                st.markdown(
                    f"""
                    <div class="result-box pass-box">
                    ✅ STUDENT IS LIKELY TO PASS
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="result-box fail-box">
                    ❌ STUDENT IS LIKELY TO FAIL
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # =========================================
        # RISK LEVEL CARD
        # =========================================

        with result_col2:

            if fail_probability > 70:
                st.error("🔴 HIGH RISK STUDENT")

            elif fail_probability > 40:
                st.warning("🟠 MEDIUM RISK STUDENT")

            else:
                st.success("🟢 LOW RISK STUDENT")

        # =========================================
        # PROGRESS BARS
        # =========================================

        st.subheader("📊 Prediction Confidence")

        st.write(f"✅ Pass Probability: {pass_probability:.2f}%")

        st.progress(int(pass_probability))

        st.write(f"❌ Fail Probability: {fail_probability:.2f}%")

        st.progress(int(fail_probability))

        # =========================================
        # GAUGE STYLE METRICS
        # =========================================

        metric1, metric2 = st.columns(2)

        metric1.metric(
            "PASS %",
            f"{pass_probability:.2f}%"
        )

        metric2.metric(
            "FAIL %",
            f"{fail_probability:.2f}%"
        )

        # =========================================
        # AI RECOMMENDATIONS
        # =========================================

        st.subheader("📌 AI Recommendations")

        recommendations = []

        if absences > 20:
            recommendations.append("Reduce student absences.")

        if studytime < 2:
            recommendations.append("Increase daily study time.")

        if G1 < 10 or G2 < 10:
            recommendations.append("Improve academic performance in internal exams.")

        if failures > 1:
            recommendations.append("Provide extra academic support.")

        if internet == 0:
            recommendations.append("Provide internet access for online learning.")

        if len(recommendations) == 0:
            st.success("✅ Student performance indicators are good.")

        else:
            for rec in recommendations:
                st.warning(rec)

        # =========================================
        # PERFORMANCE SCORE
        # =========================================

        st.subheader("🏆 Student Performance Score")

        performance_score = (
            (studytime * 10)
            + (G1 * 2)
            + (G2 * 2)
            - (failures * 5)
            - (absences * 0.3)
        )

        performance_score = max(0, min(100, performance_score))

        st.write(f"Performance Score: {performance_score:.2f}/100")

        st.progress(int(performance_score))

    except Exception as e:
        st.error(f"Prediction Error: {e}")

# =========================================
# VISUALIZATIONS
# =========================================

st.header("📊 Data Visualizations")

# =========================================
# CREATE COLUMNS FOR SMALLER GRAPHS
# =========================================

col1, col2 = st.columns(2)

# =========================================
# Study Time vs Final Grade
# =========================================

with col1:

    fig1, ax1 = plt.subplots(figsize=(4,2.5))

    sns.boxplot(
        x=df['studytime'],
        y=df['G3'],
        ax=ax1
    )

    ax1.set_title("Study Time vs Final Grade")

    st.pyplot(fig1, use_container_width=False)

# =========================================
# Absences vs Final Grade
# =========================================

with col2:

    fig2, ax2 = plt.subplots(figsize=(4,2.5))

    sns.scatterplot(
        x=df['absences'],
        y=df['G3'],
        ax=ax2
    )

    ax2.set_title("Absences vs Final Grade")

    st.pyplot(fig2, use_container_width=False)

# =========================================
# SECOND ROW OF GRAPHS
# =========================================

col3, col4 = st.columns(2)

# =========================================
# Pass vs Fail Distribution
# =========================================

with col3:

    fig3, ax3 = plt.subplots(figsize=(3,3))

    result_counts = df['Final_Result'].value_counts()

    ax3.pie(
        result_counts,
        labels=['Pass', 'Fail'],
        autopct='%1.1f%%'
    )

    ax3.set_title("Pass vs Fail Distribution")

    st.pyplot(fig3, use_container_width=False)

# =========================================
# FEATURE IMPORTANCE
# =========================================

with col4:

    st.subheader("📌 Feature Importance")

    importance_df = pd.DataFrame({
        'Feature': [
            'studytime',
            'failures',
            'absences',
            'G1',
            'G2',
            'Medu',
            'Fedu',
            'traveltime',
            'internet',
            'higher'
        ],
        'Importance': model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by='Importance',
        ascending=False
    )

    fig4, ax4 = plt.subplots(figsize=(5,2.5))

    sns.barplot(
        x='Importance',
        y='Feature',
        data=importance_df,
        ax=ax4
    )

    ax4.set_title("Feature Importance")

    st.pyplot(fig4, use_container_width=False)

# =========================================
# FOOTER
# =========================================

st.markdown("---")

st.markdown("""
### 🎯 Project Features

✅ Machine Learning Prediction  
✅ Student Risk Detection  
✅ Interactive Dashboard  
✅ Data Visualization  
✅ Feature Importance Analysis  
✅ AI Recommendations  
✅ Visual Prediction Dashboard  
✅ Performance Score Analysis  
✅ Random Forest Classifier  
""")