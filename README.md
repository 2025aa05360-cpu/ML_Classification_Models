# Breast Cancer Classification - Machine Learning Assignment 2

## Problem Statement

The objective of this project is to develop and evaluate multiple machine learning classification models to predict whether a breast tumor is **malignant** (cancerous) or **benign** (non-cancerous) based on various cell nuclei characteristics extracted from fine needle aspirate (FNA) images. Early and accurate detection of malignant tumors is crucial for timely medical intervention and improved patient outcomes.

This assignment demonstrates end-to-end machine learning workflow including data preprocessing, model training, evaluation, comparison, and deployment through an interactive web application.

---

## Dataset Description

**Dataset Name:** Breast Cancer Wisconsin (Diagnostic) Dataset

**Source:** [Kaggle - UCI Breast Cancer Wisconsin Dataset](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data?resource=download)

**Description:**
- The dataset contains features computed from digitized images of fine needle aspirate (FNA) of breast masses
- Each sample describes characteristics of the cell nuclei present in the image
- **Target Variable:** Diagnosis (M = Malignant, B = Benign)
- **Total Instances:** 569
- **Total Features:** 30 numeric features (after removing ID and unnamed columns)
- **Feature Categories:** Mean, standard error, and "worst" (mean of three largest values) of 10 real-valued characteristics:
  - Radius, Texture, Perimeter, Area, Smoothness
  - Compactness, Concavity, Concave points, Symmetry, Fractal dimension
**Data Preprocessing:**
- Removed unnecessary columns: `id`, `Unnamed: 32`
- Encoded target variable: M=1 (Malignant), B=0 (Benign)
- Applied StandardScaler for feature normalization
- Train-Test Split: 80% training, 20% testing (random_state=42)

---

## Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name                 | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|-------------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression           | 0.9737   | 0.9974 | 0.9762    | 0.9535 | 0.9647 | 0.9439 |
| Decision Tree                 | 0.9474   | 0.9440 | 0.9302    | 0.9302 | 0.9302 | 0.8880 |
| K-Nearest Neighbor            | 0.9474   | 0.9820 | 0.9302    | 0.9302 | 0.9302 | 0.8880 |
| Naive Bayes (Gaussian)        | 0.9649   | 0.9974 | 0.9756    | 0.9302 | 0.9524 | 0.9253 |
| Random Forest (Ensemble)      | 0.9649   | 0.9953 | 0.9756    | 0.9302 | 0.9524 | 0.9253 |
| XGBoost (Ensemble)            | 0.9561   | 0.9908 | 0.9524    | 0.9302 | 0.9412 | 0.9064 |

---

## Observations on Model Performance

| ML Model Name               | Observation about Model Performance |
|-----------------------------|-------------------------------------|
| **Logistic Regression**     | Achieved highest accuracy (97.37%) with excellent AUC (0.9974) and best MCC (0.9439). Shows strong linear separability and balanced precision-recall performance. The superior performance indicates that cell nuclei features have predominantly linear relationships with malignancy, making logistic regression highly effective for this medical diagnosis task. |
| **Decision Tree**           | Moderate performance with 94.74% accuracy and lowest AUC (0.9440). Balanced metrics but prone to overfitting without ensemble methods. The lower AUC suggests difficulty in learning optimal probability thresholds from the 30-dimensional feature space, resulting in less reliable confidence scores for clinical decision-making. |
| **K-Nearest Neighbor**      | Achieved 94.74% accuracy with high AUC (0.9820). Better probability estimates than Decision Tree despite identical accuracy metrics. The strong AUC performance reflects well-clustered malignant and benign samples in the scaled feature space, allowing distance-based classification to capture neighborhood patterns effectively. |
| **Naive Bayes (Gaussian)**  | Strong performance with 96.49% accuracy and excellent AUC (0.9974). Simple probabilistic approach with high precision (0.9756). Despite assuming feature independence, the model performs exceptionally well, suggesting that individual cell nuclei measurements provide strong independent signals for cancer detection. |
| **Random Forest (Ensemble)**| Achieved 96.49% accuracy with very high AUC (0.9953). Ensemble approach significantly outperforms standalone Decision Tree. Multiple decision trees voting together reduce overfitting and capture complex non-linear interactions between cell features, critical for minimizing false negatives in cancer diagnosis. |
| **XGBoost (Ensemble)**      | Solid performance with 95.61% accuracy and high AUC (0.9908). Gradient boosting provides strong regularization and generalization. Sequential error correction in gradient boosting effectively handles class imbalance (37% malignant vs 63% benign), though slightly lower precision indicates potential for false positives that require clinical validation. |

---
