import streamlit as st
import pandas as pd
import numpy as np
import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="Breast Cancer Diagnosis Prediction",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>

[data-testid="stSidebar"]{
    background-color:#dff0d8;
}

.metric-card{
    background-color:white;
    padding:18px;
    border-left:6px solid #34c759;
    border-radius:10px;
    box-shadow:0px 1px 5px rgba(0,0,0,0.1);
}

.metric-value{
    font-size:38px;
    font-weight:bold;
    color:#073b35;
}

.metric-title{
    font-size:16px;
}

</style>
""", unsafe_allow_html=True)

# ======================================================
# LOAD RESULTS
# ======================================================
try:
    results_df = pd.read_csv("results.csv")
except Exception:
    results_df = pd.DataFrame()

# ======================================================
# LOAD SCALER
# ======================================================
try:
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception:
    scaler = None

# ======================================================
# MODEL FILES
# ======================================================
model_files = {
    "Logistic Regression":
        "models/logistic_regression.pkl",

    "Decision Tree":
        "models/decision_tree.pkl",

    "KNN":
        "models/knn.pkl",

    "Naive Bayes":
        "models/naive_bayes.pkl",

    "SVM":
        "models/svm.pkl"
}

# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.title("Model Evaluation")

    selected_model = st.selectbox(
        "Select Classification Model",
        list(model_files.keys())
    )

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )

    st.markdown("---")

    st.subheader("Sample Test Data")

    sample_csv = pd.DataFrame(
        columns=[
            "radius_mean",
            "texture_mean",
            "perimeter_mean",
            "area_mean",
            "smoothness_mean",
            "compactness_mean",
            "concavity_mean",
            "concave points_mean",
            "symmetry_mean",
            "fractal_dimension_mean",
            "radius_se",
            "texture_se",
            "perimeter_se",
            "area_se",
            "smoothness_se",
            "compactness_se",
            "concavity_se",
            "concave points_se",
            "symmetry_se",
            "fractal_dimension_se",
            "radius_worst",
            "texture_worst",
            "perimeter_worst",
            "area_worst",
            "smoothness_worst",
            "compactness_worst",
            "concavity_worst",
            "concave points_worst",
            "symmetry_worst",
            "fractal_dimension_worst",
            "Diagnosis"
        ]
    )

    st.download_button(
        "Download Test Data",
        sample_csv.to_csv(index=False),
        file_name="sample_test_data.csv",
        mime="text/csv"
    )

    st.info(
        """
        Upload a CSV containing:

        • All 30 feature columns

        • Diagnosis column

        Diagnosis values should be:
        B (Benign) or M (Malignant)
        """
    )

# ======================================================
# HEADER
# ======================================================
st.title("🎗️ Breast Cancer Diagnosis Prediction")

st.caption(
    "Machine Learning Model Evaluation Dashboard"
)

st.write(
    """
    Compare five trained classification models
    using the Breast Cancer Wisconsin Diagnostic Dataset.
    """
)

# ======================================================
# SUMMARY CARDS
# ======================================================
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Classification Models</div>
        <div class="metric-value">5</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Input Features</div>
        <div class="metric-value">30</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="metric-card">
        <div class="metric-title">Problem Type</div>
        <div class="metric-value">Binary</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ======================================================
# MODEL COMPARISON TABLE
# ======================================================
st.subheader("Available Models")

if not results_df.empty:

    st.dataframe(
        results_df,
        width="stretch"
    )

else:

    st.warning(
        "results.csv not found. Run train_models.py first."
    )

st.markdown("---")

# ======================================================
# NO FILE UPLOADED
# ======================================================
if uploaded_file is None:
    st.info(
        "Upload a CSV file from the sidebar to begin evaluation."
    )

# ======================================================
# FILE UPLOADED
# ======================================================
if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset")

    st.dataframe(
        df.head(),
        width="stretch"
    )

    # ==========================================
    # Validate Diagnosis Column
    # ==========================================
    if "Diagnosis" not in df.columns:
        st.error(
            "Diagnosis column is missing from uploaded CSV."
        )

        st.stop()

    # ==========================================
    # Separate Features and Target
    # ==========================================
    X_test = df.drop(columns=["Diagnosis"])

    y_test = df["Diagnosis"].copy()

    # Remove ID column if it exists
    for col in ["id", "ID", "Id"]:

        if col in X_test.columns:
            X_test = X_test.drop(columns=[col])

    # ==========================================
    # Convert Diagnosis Labels
    # ==========================================
    y_test = (
        y_test
        .astype(str)
        .str.strip()
        .str.upper()
    )

    y_test = y_test.replace({
        "B": 0,
        "M": 1
    })

    y_test = pd.to_numeric(
        y_test,
        errors="coerce"
    )

    # Check invalid labels
    if y_test.isnull().any():
        st.error(
            "Diagnosis column contains invalid values."
        )

        st.write(
            "Unique values found:",
            df["Diagnosis"].unique()
        )

        st.stop()

    # ==========================================
    # Load Model
    # ==========================================
    try:

        with open(
                model_files[selected_model],
                "rb"
        ) as f:

            model = pickle.load(f)

    except Exception as e:

        st.error(
            f"Unable to load model.\n\n{e}"
        )

        st.stop()

    # ==========================================
    # Feature Scaling
    # ==========================================
    if selected_model in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        X_test_model = scaler.transform(X_test)

    else:

        X_test_model = X_test

    # ==========================================
    # Predictions
    # ==========================================
    y_pred = model.predict(X_test_model)

    y_prob = model.predict_proba(
        X_test_model
    )[:, 1]

    # ==========================================
    # METRICS
    # ==========================================
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    auc_score = roc_auc_score(
        y_test,
        y_prob
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    mcc = matthews_corrcoef(
        y_test,
        y_pred
    )

    # ==========================================
    # DISPLAY METRICS
    # ==========================================
    st.subheader(
        f"Evaluation Results : {selected_model}"
    )

    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)

    c1.metric(
        "Accuracy",
        f"{accuracy:.4f}"
    )

    c2.metric(
        "AUC Score",
        f"{auc_score:.4f}"
    )

    c3.metric(
        "Precision",
        f"{precision:.4f}"
    )

    c4.metric(
        "Recall",
        f"{recall:.4f}"
    )

    c5.metric(
        "F1 Score",
        f"{f1:.4f}"
    )

    c6.metric(
        "MCC",
        f"{mcc:.4f}"
    )

    st.markdown("---")

    # ==========================================
    # CONFUSION MATRIX
    # ==========================================
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    fig, ax = plt.subplots(
        figsize=(6, 4)
    )

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(selected_model)

    st.pyplot(fig)

    # ==========================================
    # CLASSIFICATION REPORT
    # ==========================================
    st.subheader(
        "Classification Report"
    )

    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df,
        width="stretch"
    )