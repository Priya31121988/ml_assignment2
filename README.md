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
| Support Vector Machine (SVM) | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| K-Nearest Neighbors (KNN) | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Decision Tree Classifier | 0.9298 | 0.9246 | 0.9048 | 0.9048 | 0.9048 | 0.8492 |

---

## Model Performance Observations

| ML Model Name | Observation |
|---------------|-------------|
| Support Vector Machine (SVM) | Achieved the highest Accuracy (97.37%) and MCC (0.9442), demonstrating the best overall classification performance. It also achieved perfect Precision (100%), indicating no false positive predictions. |
| Logistic Regression | Delivered strong and consistent performance with high AUC Score (0.9960) and Accuracy (96.49%), making it an effective baseline classifier. |
| K-Nearest Neighbors (KNN) | Produced competitive results with 95.61% Accuracy and balanced Precision and Recall after feature scaling. |
| Gaussian Naive Bayes | Achieved perfect Precision (100%) and a high AUC Score (0.9934), but lower Recall (83.33%), indicating more missed positive cases. |
| Decision Tree Classifier | Delivered the lowest overall performance among the evaluated models, with Accuracy of 92.98% and MCC of 0.8492. |
| **Overall Winner** | **Support Vector Machine (SVM)** achieved the best overall performance with the highest Accuracy, strongest MCC score, perfect Precision, and excellent AUC score. |

---

## Streamlit Web Application

The deployed Streamlit application can be accessed using the following link:

🔗 **Streamlit App:**
https://ml-assignment2-breast-cancer-diagnosis.streamlit.app/
---
## Conclusion

This project implemented and compared five machine learning classification algorithms on the Breast Cancer Wisconsin Diagnostic Dataset.

All models achieved strong predictive performance, with classification accuracies exceeding 92%. Among the evaluated algorithms, the **Support Vector Machine (SVM)** emerged as the best-performing model, achieving the highest Accuracy (**97.37%**), highest MCC (**0.9442**), perfect Precision (**100%**), and an excellent AUC Score (**0.9947**).

**Logistic Regression** and **KNN** also demonstrated strong performance and reliable classification results. **Gaussian Naive Bayes** achieved perfect Precision but exhibited lower Recall, while the **Decision Tree Classifier** produced the lowest overall performance among the evaluated models.

Based on all evaluation metrics, the **Support Vector Machine (SVM)** was selected as the overall winner for this dataset due to its superior accuracy, balanced predictive capability, and excellent ability to distinguish between malignant and benign tumors.

