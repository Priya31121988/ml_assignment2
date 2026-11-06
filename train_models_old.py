import os
import pickle
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

# =====================================================
# LOAD DATASET
# =====================================================
df = pd.read_csv("test_data.csv")

print("Dataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())

# =====================================================
# DATA PREPROCESSING
# =====================================================

# Drop ID column if present
for col in ['id', 'ID', 'Id']:
    if col in df.columns:
        df.drop(columns=[col], inplace=True)

# Target variable
target_column = "Diagnosis"

# Convert target values
df[target_column] = df[target_column].map({
    "B": 0,
    "M": 1
})

# Features and Target
X = df.drop(columns=[target_column])
y = df[target_column]

# =====================================================
# TRAIN TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# FEATURE SCALING
# =====================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =====================================================
# CREATE MODELS DIRECTORY
# =====================================================
os.makedirs("models", exist_ok=True)

# Save scaler
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# =====================================================
# MODELS
# =====================================================
models = {
    "Logistic Regression":
        LogisticRegression(max_iter=1000),

    "Decision Tree":
        DecisionTreeClassifier(random_state=42),

    "KNN":
        KNeighborsClassifier(n_neighbors=5),

    "Naive Bayes":
        GaussianNB(),

    "SVM":
        SVC(
            kernel='rbf',
            probability=True,
            random_state=42
        )
}

# =====================================================
# TRAIN & EVALUATE
# =====================================================
results = []

for name, model in models.items():

    print(f"\nTraining {name}...")

    # Models requiring scaling
    if name in ["Logistic Regression", "KNN", "SVM"]:

        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)

        y_prob = model.predict_proba(
            X_test_scaled
        )[:, 1]

    else:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        y_prob = model.predict_proba(
            X_test
        )[:, 1]

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, y_pred)

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

    # Save Results
    results.append({
        "Model": name,
        "Accuracy": round(accuracy, 4),
        "AUC Score": round(auc, 4),
        "Precision": round(precision, 4),
        "Recall": round(recall, 4),
        "F1 Score": round(f1, 4),
        "MCC": round(mcc, 4)
    })

    # Save Model
    filename = (
        name.lower()
        .replace(" ", "_")
        + ".pkl"
    )

    with open(
        f"models/{filename}",
        "wb"
    ) as f:
        pickle.dump(model, f)

# =====================================================
# RESULTS TABLE
# =====================================================
results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by="Accuracy",
    ascending=False
)

print("\n==============================")
print("MODEL COMPARISON")
print("==============================")
print(results_df)

# Save results for Streamlit
results_df.to_csv(
    "results.csv",
    index=False
)

print("\nResults saved to results.csv")
print("Models saved in models/ folder")