from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns

MODELS_DIR = Path("model")                 # where you saved .pkl models
DEFAULT_TEST_PATH = Path("test_data.csv")  # saved earlier in your notebook
POS_LABEL = 1                              # diagnosis encoded as M=1, B=0

def list_models(dir_path: Path) -> dict:
    dir_path.mkdir(exist_ok=True)
    models = {}
    for p in sorted(dir_path.glob("*.pkl")):
        name = p.stem.replace("_", " ").title()
        models[name] = p
    return models

def load_model(model_path: Path):
    return joblib.load(model_path)

def get_scores(model, X: np.ndarray) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        # scale to [0,1] for AUC comparability if needed
        scores = model.decision_function(X)
        # Min-max normalize to [0,1] when necessary
        s_min, s_max = np.min(scores), np.max(scores)
        return (scores - s_min) / (s_max - s_min + 1e-12)
    return None

def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None):
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POS_LABEL)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if y_score is not None:
        metrics["auc"] = float(roc_auc_score(y_true, y_score))
    else:
        metrics["auc"] = None
    report_text = classification_report(y_true, y_pred)
    return metrics, report_text

def main():
    # Custom CSS for colorful styling
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            text-align: center;
            margin-bottom: 20px;
        }
        .metric-box {
            background-color: #f0f2f6;
            padding: 15px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
        }
        .success-box {
            background-color: #d4edda;
            padding: 10px;
            border-radius: 5px;
            border-left: 4px solid #28a745;
            color: #155724;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Colorful header
    st.markdown('<div class="main-header"><h1>🩺 Breast Cancer Classification — Model Evaluation</h1></div>', unsafe_allow_html=True)
    st.markdown("### 📋 **Workflow Steps**")
    st.markdown("**Step 1:** 📥 Download test dataset (if required) → **Step 2:** 📤 Upload test dataset → **Step 3:** 🤖 Select the model → **Step 4:** 📊 View evaluation metrics")
    st.divider()

    # Download and Upload in same line with emojis
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📥 **Step 1: Download Dataset**")
        # Download saved test_data.csv (if present)
        if DEFAULT_TEST_PATH.exists():
            with DEFAULT_TEST_PATH.open("rb") as f:
                data_bytes = f.read()
            st.download_button(
                label="⬇️ Download saved test_data.csv",
                data=data_bytes,
                file_name="test_data.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("💡 Saved test_data.csv not found in workspace.")
    
    with col2:
        st.markdown("### 📤 **Step 2: Upload Dataset**")
        # Upload test CSV
        uploaded = st.file_uploader("Drag and drop your test CSV file here", type=["csv"], label_visibility="collapsed")

    st.divider()
    
    # Model selection with emoji
    st.markdown("### 🤖 **Step 3: Select Model**")
    models = list_models(MODELS_DIR)
    if not models:
        st.error("❌ No models found in model/. Train and save models first.")
        return

    model_name = st.selectbox("Choose a classification model", options=list(models.keys()), label_visibility="collapsed")
    model_path = models[model_name]
    model = load_model(model_path)
    st.markdown(f'<div class="success-box">✅ <b>Successfully loaded model:</b> {model_name}</div>', unsafe_allow_html=True)

    if uploaded is None:
        st.info("👆 Please upload a test dataset to continue")
        st.stop()

    df = pd.read_csv(uploaded)

    # Separate features and target (if present)
    if "target" in df.columns:
        y_true = df["target"].to_numpy()
        X = df.drop(columns=["target"]).to_numpy()
    else:
        y_true = None
        X = df.to_numpy()
        st.info("No 'target' column found. Metrics requiring ground truth will be unavailable.")

    # Predict
    y_pred = model.predict(X)
    y_score = get_scores(model, X)

    # Metrics - Side by side layout
    st.subheader("Model Performance Evaluation")
    if y_true is None:
        st.warning("Upload a CSV with a 'target' column to compute metrics.")
    else:
        metrics, report_text = evaluate(y_true, y_pred, y_score)
        
        # Create two columns for Evaluation Metrics and Confusion Matrix
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Evaluation Metrics")
            metrics_df = pd.DataFrame([
                {"Metric": "Accuracy", "Value": f"{metrics['accuracy']:.4f}"},
                {"Metric": "AUC", "Value": f"{metrics['auc']:.4f}" if metrics['auc'] is not None else "N/A"},
                {"Metric": "Precision", "Value": f"{metrics['precision']:.4f}"},
                {"Metric": "Recall", "Value": f"{metrics['recall']:.4f}"},
                {"Metric": "F1 Score", "Value": f"{metrics['f1']:.4f}"},
                {"Metric": "MCC", "Value": f"{metrics['mcc']:.4f}"}
            ])
            st.table(metrics_df)
        
        with col2:
            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            cm_df = pd.DataFrame(cm, 
                                index=['Actual: 0', 'Actual: 1'],
                                columns=['Pred: 0', 'Pred: 1'])
            st.table(cm_df)
            st.caption("0=Benign, 1=Malignant")
        
        # Classification Report - Full width below
        st.markdown("#### Classification Report")
        # Parse classification report into dataframe
        report_dict = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame({
            'Class': ['0 (Benign)', '1 (Malignant)', 'Accuracy', 'Macro Avg', 'Weighted Avg'],
            'Precision': [
                f"{report_dict['0']['precision']:.3f}",
                f"{report_dict['1']['precision']:.3f}",
                "-",
                f"{report_dict['macro avg']['precision']:.3f}",
                f"{report_dict['weighted avg']['precision']:.3f}"
            ],
            'Recall': [
                f"{report_dict['0']['recall']:.3f}",
                f"{report_dict['1']['recall']:.3f}",
                "-",
                f"{report_dict['macro avg']['recall']:.3f}",
                f"{report_dict['weighted avg']['recall']:.3f}"
            ],
            'F1-Score': [
                f"{report_dict['0']['f1-score']:.3f}",
                f"{report_dict['1']['f1-score']:.3f}",
                f"{report_dict['accuracy']:.3f}",
                f"{report_dict['macro avg']['f1-score']:.3f}",
                f"{report_dict['weighted avg']['f1-score']:.3f}"
            ]
        })
        st.table(report_df)
        
        if y_score is None:
            st.caption("⚠️ AUC not available for this model (no predict_proba/decision_function).")

    st.divider()
    st.caption(f"Models directory: {MODELS_DIR}")

if __name__ == "__main__":
    main()