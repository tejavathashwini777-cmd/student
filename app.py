import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from model import train_models
from utils import load_data, get_insight
from sklearn.metrics import r2_score

# Page Config
st.set_page_config(page_title="Student Performance System", layout="wide")

# Title
st.title("🎓 Student Performance Prediction & Analytics System")

# =========================
# 📂 File Upload Section
# =========================
st.sidebar.header("📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

try:
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("✅ Uploaded File Loaded Successfully")
    else:
        data = load_data("students.csv")
        st.info("ℹ️ Using Default Dataset (students.csv)")

except FileNotFoundError:
    st.error("❌ No dataset found. Please upload a CSV file.")
    st.stop()

# Show Dataset
st.subheader("📊 Dataset Preview")
st.dataframe(data)

# =========================
# 🧠 Train Models
# =========================
lr, rf, X_test, y_test = train_models(data)

# Model Accuracy
lr_acc = r2_score(y_test, lr.predict(X_test))
rf_acc = r2_score(y_test, rf.predict(X_test))

st.subheader("📈 Model Performance")

col1, col2 = st.columns(2)
col1.metric("Linear Regression Accuracy", f"{lr_acc:.2f}")
col2.metric("Random Forest Accuracy", f"{rf_acc:.2f}")

# =========================
# 🎯 User Input Section
# =========================
st.sidebar.header("🎯 Enter Student Details")

study_hours = st.sidebar.slider("Study Hours", 0, 10, 5)
attendance = st.sidebar.slider("Attendance (%)", 0, 100, 70)
previous_marks = st.sidebar.slider("Previous Marks", 0, 100, 60)

input_data = np.array([[study_hours, attendance, previous_marks]])

# =========================
# 🔮 Prediction Section
# =========================
if st.sidebar.button("Predict"):

    lr_pred = lr.predict(input_data)[0]
    rf_pred = rf.predict(input_data)[0]

    st.subheader("📌 Prediction Results")

    col1, col2 = st.columns(2)
    col1.success(f"Linear Regression: {lr_pred:.2f}")
    col2.success(f"Random Forest: {rf_pred:.2f}")

    # Insight
    st.subheader("💡 Insight")
    st.info(get_insight(rf_pred))

# =========================
# 📊 Visualization Section
# =========================
st.subheader("📊 Data Visualization")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    sns.scatterplot(data=data, x="StudyHours", y="FinalMarks")
    ax.set_title("Study Hours vs Final Marks")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig)

# =========================
# ⚠️ Weak Students Detection
# =========================
st.subheader("⚠️ Weak Students (Predicted < 60)")

data["Predicted"] = rf.predict(data[["StudyHours", "Attendance", "PreviousMarks"]])
weak_students = data[data["Predicted"] < 60]

if not weak_students.empty:
    st.dataframe(weak_students)
else:
    st.success("🎉 No weak students found!")

# =========================
# 📥 Download Section
# =========================
st.subheader("📥 Download Report")

csv = data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Updated Dataset",
    data=csv,
    file_name="student_report.csv",
    mime="text/csv",
)

# =========================
# 🎉 Footer
# =========================
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")