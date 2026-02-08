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

**Class Distribution:**
- Malignant (M): 212 samples (37.3%)
- Benign (B): 357 samples (62.7%)

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
| Logistic Regression           | 0.9825   | 0.9977 | 0.9762    | 0.9756 | 0.9759 | 0.9618 |
| Decision Tree                 | 0.9386   | 0.9293 | 0.9024    | 0.9024 | 0.9024 | 0.8652 |
| K-Nearest Neighbor            | 0.9649   | 0.9939 | 0.9286    | 0.9756 | 0.9515 | 0.9219 |
| Naive Bayes (Gaussian)        | 0.9561   | 0.9939 | 0.9268    | 0.9268 | 0.9268 | 0.9011 |
| Random Forest (Ensemble)      | 0.9737   | 0.9977 | 0.9512    | 0.9756 | 0.9632 | 0.9420 |
| XGBoost (Ensemble)            | 0.9825   | 0.9992 | 0.9756    | 0.9756 | 0.9756 | 0.9615 |

> **Note:** All metrics are calculated on the test dataset (20% of total data). Evaluation metrics include:
> - **Accuracy:** Overall correctness of predictions
> - **AUC:** Area Under the ROC Curve
> - **Precision:** Positive predictive value
> - **Recall:** Sensitivity or true positive rate
> - **F1 Score:** Harmonic mean of precision and recall
> - **MCC:** Matthews Correlation Coefficient (balanced measure for imbalanced datasets)

---

## Observations on Model Performance

| ML Model Name                 | Observation about Model Performance                              |
|-------------------------------|------------------------------------------------------------------|
| **Logistic Regression**       | Achieved excellent performance with 98.25% accuracy and an    impressive AUC of 0.9977. The model demonstrates strong linear separability between benign and malignant tumors. Despite being a simple linear model, it performs remarkably well, indicating that the relationship between features and target is largely linear. High precision (0.9762) and recall (0.9756) show balanced performance in identifying both classes. |
| **Decision Tree**             | Showed moderate performance with 93.86% accuracy, the lowest among all models tested. The model exhibited signs of overfitting with lower AUC (0.9293) compared to other models. While decision trees are interpretable, they tend to create overly complex boundaries on this dataset. The balanced precision and recall (0.9024) indicate consistent but suboptimal prediction capability. Pruning or ensemble methods would improve performance. |
| **K-Nearest Neighbor**        | Delivered strong results with 96.49% accuracy and high AUC of 0.9939. The model benefits from the well-clustered nature of this dataset where similar instances are grouped together in feature space. Excellent recall (0.9756) indicates the model is particularly good at identifying malignant cases, which is crucial in medical diagnosis. Performance depends heavily on the choice of k=5 neighbors. |
| **Naive Bayes (Gaussian)**    | Performed well with 95.61% accuracy despite its assumption of feature independence, which may not hold true for correlated cell characteristics. The high AUC (0.9939) suggests good discriminative ability. Balanced precision and recall (0.9268) indicate consistent predictions across both classes. The model's simplicity and speed make it suitable for real-time predictions, though it slightly underperforms compared to ensemble methods. |
| **Random Forest (Ensemble)**  | Demonstrated robust performance with 97.37% accuracy, combining multiple decision trees to overcome individual tree limitations. The ensemble approach improved AUC to 0.9977 and achieved excellent recall (0.9756), minimizing false negatives—critical in cancer detection. The model handles non-linear relationships well and provides feature importance insights. Strong MCC (0.9420) confirms reliable performance on this imbalanced dataset. |
| **XGBoost (Ensemble)**        | Achieved the best overall performance with 98.25% accuracy and the highest AUC of 0.9992, demonstrating exceptional discriminative power. The gradient boosting framework iteratively corrects errors, resulting in near-perfect classification. Balanced precision and recall (0.9756) with high MCC (0.9615) indicate superior ability to handle both classes effectively. The model's regularization prevents overfitting while maintaining excellent generalization on test data. |

---

## Key Insights

1. **Ensemble Models Excel:** XGBoost and Random Forest outperformed individual models, demonstrating the power of ensemble learning in combining multiple weak learners.

2. **Linear Models Competitive:** Logistic Regression matched XGBoost's accuracy (98.25%), proving that for this dataset, complex non-linear relationships may not be dominant.

3. **Decision Trees Need Tuning:** Standalone decision trees showed the weakest performance, highlighting the importance of ensemble methods or hyperparameter tuning.

4. **High Recall Critical:** In medical diagnosis, high recall (sensitivity) is crucial to minimize false negatives. KNN, Random Forest, and XGBoost all achieved recall ≥ 0.9756, making them suitable for clinical deployment.

5. **Feature Scaling Matters:** Models like KNN and Logistic Regression benefited significantly from StandardScaler preprocessing.

---

## Project Structure

```
ML_Classification_Models/
│
├── streamlit_app.py              # Streamlit web application for model evaluation
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation (this file)
├── test_data.csv                 # Test dataset for evaluation
│
└── model/
    ├── ML_Assignment.ipynb       # Jupyter notebook with model training
    ├── data.csv                  # Original dataset
    ├── logistic_regression.pkl   # Saved Logistic Regression model
    ├── decision_tree.pkl         # Saved Decision Tree model
    ├── k_nearest_neighbor.pkl    # Saved KNN model
    ├── naive_bayes_gaussian.pkl  # Saved Naive Bayes model
    ├── random_forest.pkl         # Saved Random Forest model
    └── xgboost.pkl               # Saved XGBoost model
```

---

## Streamlit Web Application Features

The interactive web application provides:

1. **Dataset Upload:** Upload test data in CSV format for evaluation
2. **Model Selection:** Dropdown menu to select from 6 trained models
3. **Evaluation Metrics:** Display accuracy, AUC, precision, recall, F1 score, and MCC
4. **Confusion Matrix:** Visual representation of true vs predicted classifications
5. **Classification Report:** Detailed per-class performance metrics
6. **Download Option:** Download the saved test dataset

**Live App:** [Click here to access the Streamlit application](#) *(Update with deployment URL)*

---

## Technologies Used

- **Programming Language:** Python 3.x
- **ML Libraries:** scikit-learn, XGBoost
- **Data Processing:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Deployment:** Streamlit Community Cloud
- **Model Persistence:** joblib

---

## How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ML_Classification_Models
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Jupyter notebook** (optional - models are already saved):
   ```bash
   cd model
   jupyter notebook ML_Assignment.ipynb
   ```

4. **Launch the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   ```

5. **Access the app** at `http://localhost:8501`

---

## Deployment on Streamlit Community Cloud

1. Visit [streamlit.io/cloud](https://streamlit.io/cloud)
2. Sign in with GitHub account
3. Click "New App"
4. Select this repository and branch (main)
5. Set main file path: `streamlit_app.py`
6. Click "Deploy"

The app will be live within minutes at the provided URL.

---

## Author

**Submission for:** Machine Learning Assignment 2  
**Institution:** BITS Pilani  
**Performed on:** BITS Virtual Lab

---

## References

- Dataset: [UCI Machine Learning Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))
- Scikit-learn Documentation: [https://scikit-learn.org/](https://scikit-learn.org/)
- XGBoost Documentation: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- Streamlit Documentation: [https://docs.streamlit.io/](https://docs.streamlit.io/)

---

## License

This project is submitted as part of academic coursework. Dataset is publicly available under UCI ML Repository terms of use.
