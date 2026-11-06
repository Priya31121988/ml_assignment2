import streamlit as st
import pandas as pd
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

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Breast Cancer Diagnosis Prediction",
    page_icon="🎗️",
    layout="wide"
)

# =====================================================
# CUSTOM CSS
# =====================================================
st.markdown("""
<style>

[data-testid="stSidebar"]{
    background-color:#dff0d8;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# MODEL PATHS
# =====================================================
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

# =====================================================
# LOAD SCALER
# =====================================================
try:
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
except Exception:
    scaler = None

# =====================================================
# SIDEBAR
# =====================================================
with st.sidebar:

    st.title("Model Evaluation")

    uploaded_file = st.file_uploader(
        "Upload Custom Test CSV (Optional)",
        type=["csv"]
    )

    st.markdown("---")

    st.subheader("Sample Test Dataset")

    st.link_button(
        "📥 Download test_data.csv",
        "https://raw.githubusercontent.com/Priya31121988/ml_assignment2/main/test_data.csv"
    )

    st.caption(
        "Download and upload test_data.csv to evaluate the models."
    )

    st.markdown("---")

    selected_model = st.selectbox(
        "Select Model",
        list(model_files.keys())
    )

# =====================================================
# HEADER
# =====================================================
st.title("🎗️ Breast Cancer Diagnosis Prediction")

st.write(
    """
    This application evaluates five trained machine learning models
    using the Breast Cancer Wisconsin Diagnostic Dataset.
    """
)

# =====================================================
# SUMMARY CARDS
# =====================================================
c1, c2, c3 = st.columns(3)

c1.metric("Models", "5")
c2.metric("Features", "30")
c3.metric("Problem Type", "Binary")

st.divider()

# =====================================================
# LOAD DATASET
# =====================================================
try:

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success(
            "Using uploaded dataset."
        )

    else:

        df = pd.read_csv("test_data.csv")

        st.info(
            "Using default test_data.csv."
        )

except Exception as e:

    st.error(
        f"Unable to load dataset.\n\n{e}"
    )

    st.stop()

# =====================================================
# DATASET PREVIEW
# =====================================================
st.subheader("Dataset Preview")

st.dataframe(
    df.head(),
    width="stretch"
)

st.write(
    f"Rows: {df.shape[0]}"
)

st.write(
    f"Columns: {df.shape[1]}"
)

# =====================================================
# VALIDATE TARGET
# =====================================================
if "Diagnosis" not in df.columns:

    st.error(
        "Diagnosis column not found."
    )

    st.stop()

# =====================================================
# FEATURES / TARGET
# =====================================================
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
    .map({
        "B": 0,
        "M": 1
    })
)

if y_test.isnull().any():

    st.error(
        "Invalid labels detected in Diagnosis column."
    )

    st.stop()

# =====================================================
# EVALUATE ALL MODELS
# =====================================================
comparison_results = []

for model_name, model_path in model_files.items():

    try:

        with open(model_path, "rb") as f:
            model = pickle.load(f)

    except Exception as e:

        st.error(
            f"Could not load {model_name}\n\n{e}"
        )

        continue

    if model_name in [
        "Logistic Regression",
        "KNN",
        "SVM"
    ]:

        X_eval = scaler.transform(X_test)

    else:

        X_eval = X_test

    y_pred_tmp = model.predict(
        X_eval
    )

    y_prob_tmp = model.predict_proba(
        X_eval
    )[:, 1]

    comparison_results.append({

        "Model":
            model_name,

        "Accuracy":
            round(
                accuracy_score(
                    y_test,
                    y_pred_tmp
                ),
                4
            ),

        "AUC Score":
            round(
                roc_auc_score(
                    y_test,
                    y_prob_tmp
                ),
                4
            ),

        "Precision":
            round(
                precision_score(
                    y_test,
                    y_pred_tmp
                ),
                4
            ),

        "Recall":
            round(
                recall_score(
                    y_test,
                    y_pred_tmp
                ),
                4
            ),

        "F1 Score":
            round(
                f1_score(
                    y_test,
                    y_pred_tmp
                ),
                4
            ),

        "MCC":
            round(
                matthews_corrcoef(
                    y_test,
                    y_pred_tmp
                ),
                4
            )
    })

# =====================================================
# MODEL COMPARISON TABLE
# =====================================================
comparison_df = pd.DataFrame(
    comparison_results
)

comparison_df = comparison_df.sort_values(
    by="Accuracy",
    ascending=False
)

st.subheader(
    "Model Comparison Table"
)

st.dataframe(
    comparison_df,
    width="stretch"
)

# =====================================================
# LOAD SELECTED MODEL
# =====================================================
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

# =====================================================
# PREDICTIONS
# =====================================================
y_pred = model.predict(
    X_eval
)

y_prob = model.predict_proba(
    X_eval
)[:, 1]

# =====================================================
# METRICS
# =====================================================
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

# =====================================================
# DISPLAY METRICS
# =====================================================
st.subheader(
    f"Selected Model: {selected_model}"
)

c1, c2, c3 = st.columns(3)
c4, c5, c6 = st.columns(3)

c1.metric(
    "Accuracy",
    f"{accuracy:.4f}"
)

c2.metric(
    "AUC Score",
    f"{auc:.4f}"
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

# =====================================================
# CONFUSION MATRIX
# =====================================================
st.subheader(
    "Confusion Matrix"
)

cm = confusion_matrix(
    y_test,
    y_pred
)

cm_df = pd.DataFrame(
    cm,
    index=["Actual B", "Actual M"],
    columns=[
        "Predicted B",
        "Predicted M"
    ]
)

st.dataframe(
    cm_df,
    width="stretch"
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================
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