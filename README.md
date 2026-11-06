# Breast Cancer Diagnosis Classification Using Machine Learning

## a. Problem Statement

Breast cancer is a major health concern worldwide. Early and accurate diagnosis can significantly improve treatment outcomes. This project applies and compares five machine learning classification algorithms to predict whether a breast tumor is **Malignant (M)** or **Benign (B)** using diagnostic measurements from breast tissue samples.

---

## b. Dataset Description

**Dataset:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** UCI Machine Learning Repository

**Dataset Link:**  
https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

### Dataset Summary

- Number of Instances: 569
- Number of Features: 30
- Target Variable: Diagnosis
- Classes:
  - Malignant (M)
  - Benign (B)
- Missing Values: None

The dataset contains features such as radius, texture, perimeter, area, smoothness, compactness, concavity, symmetry, and fractal dimension.

---

## c. GitHub Repository Link

**Repository URL:**  
https://github.com/Priya31121988/ml_assignment2/

---

## d. Models Used

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

---

## Model Comparison Table

| ML Model Name | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC |
|--------------|----------|----------|----------|----------|----------|----------|
| Logistic Regression | 0.9824 | 0.9974 | 0.9951 | 0.9575 | 0.9760 | 0.9626 |
| Decision Tree | 0.9859 | 0.9850 | 0.9811 | 0.9811 | 0.9811 | 0.9699 |
| KNN | 0.9736 | 0.9949 | 0.9900 | 0.9387 | 0.9637 | 0.9438 |
| Naive Bayes | 0.9420 | 0.9894 | 0.9543 | 0.8868 | 0.9193 | 0.8756 |
| Support Vector Machine (SVM) | 0.9842 | 0.9974 | 1.0000 | 0.9575 | 0.9783 | 0.9664 |

---

## Model Performance Observations

| ML Model Name | Observation |
|---------------|-------------|
| Logistic Regression | Excellent baseline model with strong accuracy and AUC performance. |
| Decision Tree | Achieved the highest Accuracy, Recall, F1 Score, and MCC. |
| KNN | Good performance after feature scaling with competitive results. |
| Naive Bayes | Fast and efficient but achieved the lowest overall performance. |
| Support Vector Machine | Achieved perfect Precision (100%) and the highest AUC score. |
| **Overall Winner** | **Decision Tree Classifier** achieved the best overall balance across Accuracy, Recall, F1 Score, and MCC. |

---

## Streamlit Web Application

The deployed Streamlit application can be accessed using the following link:

🔗 **Streamlit App:**  
https://mlassignment2-breast-cancer-classification.streamlit.app/

---
## Conclusion

All five machine learning models performed well on the Breast Cancer Wisconsin Diagnostic Dataset, achieving accuracies above 94%.

The **Decision Tree Classifier** was the best-performing model overall, achieving:

- Accuracy: **98.59%**
- Recall: **98.11%**
- F1 Score: **98.11%**
- MCC: **0.9699**

Although **SVM** achieved perfect Precision (100%) and an excellent AUC Score (0.9974), the **Decision Tree Classifier** provided the most balanced performance across all evaluation metrics and was selected as the overall winner for this dataset.