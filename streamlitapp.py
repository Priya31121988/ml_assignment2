import streamlit as st
import pandas as pd
import numpy as np
import pickle

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

# ====================================================
# PAGE CONFIG
# ====================================================
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

# ====================================================
# CSS
# ====================================================
st.markdown("""
<style>

[data-testid="stSidebar"]{
    background-color:#dff0d8;
}

.metric-card{
    background:white;
    padding:15px;
    border-left:6px solid #34c759;
    border-radius:10px;
}

</style>
""", unsafe_allow_html=True)

# ====================================================
# LOAD SCALER
# ====================================================
try:
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except:
    scaler = None

# ====================================================
# MODEL PATHS
# ====================================================
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

# ====================================================
# SIDEBAR
# ====================================================
with st.sidebar:

    st.title("Breast Cancer ML Models")

    uploaded_file = st.file_uploader(
        "Upload Test CSV",
        type=["csv"]
    )

    selected_model = st.selectbox(
        "Select Model",
        list(model_files.keys())
    )
    st.subheader("Sample Test Dataset")

    st.link_button(
        "📥 Download test_data.csv",
        "https://raw.githubusercontent.com/Priya31121988/ml_assignment2/main/test_data.csv"
    )

    st.info(
        """
        Steps:

        1. Download test_data.csv
        2. Upload the file using the uploader above
        3. Select a model
        4. View evaluation metrics and predictions
        """
    )

# ====================================================
# HEADER
# ====================================================
st.title("🎗️ Breast Cancer Diagnosis Prediction")

st.write(
    """
    Compare Logistic Regression, Decision Tree,
    KNN, Naive Bayes and SVM on uploaded data.
    """
)

# ====================================================
# SUMMARY
# ====================================================
c1, c2, c3 = st.columns(3)

c1.metric(
    "Models",
    "5"
)

c2.metric(
    "Features",
    "30"
)

c3.metric(
    "Classification",
    "Binary"
)

st.divider()

# ====================================================
# UPLOAD CHECK
# ====================================================
if uploaded_file is None:

    st.info(
        "Upload a CSV file to begin model evaluation."
    )

    st.stop()

# ====================================================
# READ CSV
# ====================================================
df = pd.read_csv(uploaded_file)

st.subheader("Uploaded Dataset")

st.dataframe(
    df.head(),
    width="stretch"
)

# ====================================================
# VALIDATE TARGET
# ====================================================
if "Diagnosis" not in df.columns:

    st.error(
        "Diagnosis column not found."
    )

    st.stop()

# ====================================================
# FEATURES/TARGET
# ====================================================
X_test = df.drop(
    columns=["Diagnosis"]
)

for col in ["id", "ID", "Id"]:

    if col in X_test.columns:
        X_test = X_test.drop(
            columns=[col]
        )

y_test = (
    df["Diagnosis"]
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

if y_test.isnull().any():

    st.error(
        "Diagnosis contains invalid labels."
    )

    st.stop()

# ====================================================
# EVALUATE ALL MODELS
# ====================================================
comparison_results = []

for model_name, model_path in model_files.items():

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    if model_name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        X_eval = scaler.transform(X_test)

    else:

        X_eval = X_test

    y_pred = model.predict(
        X_eval
    )

    y_prob = model.predict_proba(
        X_eval
    )[:, 1]

    comparison_results.append({

        "Model":
            model_name,

        "Accuracy":
            round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                4
            ),

        "AUC Score":
            round(
                roc_auc_score(
                    y_test,
                    y_prob
                ),
                4
            ),

        "Precision":
            round(
                precision_score(
                    y_test,
                    y_pred
                ),
                4
            ),

        "Recall":
            round(
                recall_score(
                    y_test,
                    y_pred
                ),
                4
            ),

        "F1 Score":
            round(
                f1_score(
                    y_test,
                    y_pred
                ),
                4
            ),

        "MCC":
            round(
                matthews_corrcoef(
                    y_test,
                    y_pred
                ),
                4
            )
    })

# ====================================================
# COMPARISON TABLE
# ====================================================
comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

st.subheader(
    "All Models Comparison"
)

st.dataframe(
    comparison_df,
    width="stretch"
)

# ====================================================
# SELECTED MODEL DETAILS
# ====================================================
with open(
    model_files[selected_model],
    "rb"
) as f:

    model = pickle.load(f)

if selected_model in [
    "Logistic Regression",
    "KNN",
    "SVM"
]:

    X_eval = scaler.transform(X_test)

else:

    X_eval = X_test

y_pred = model.predict(X_eval)

y_prob = model.predict_proba(
    X_eval
)[:, 1]

# ====================================================
# METRICS
# ====================================================
accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
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

st.subheader(
    f"Selected Model: {selected_model}"
)

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric("Accuracy", f"{accuracy:.4f}")
c2.metric("AUC", f"{auc:.4f}")
c3.metric("Precision", f"{precision:.4f}")
c4.metric("Recall", f"{recall:.4f}")
c5.metric("F1 Score", f"{f1:.4f}")
c6.metric("MCC", f"{mcc:.4f}")

# ====================================================
# CONFUSION MATRIX
# ====================================================
st.subheader(
    "Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm, index=["Actual B", "Actual M"], columns=["Predicted B", "Predicted M"]
)

st.dataframe(
    cm_df, width="stretch"
)

# ====================================================
# CLASSIFICATION REPORT
# ====================================================
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