from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, classification_report
)

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
    st.title("Breast Cancer Classification — Model Evaluation")
    st.caption("Select a trained model, download the saved test data, upload a test CSV, and view evaluation metrics plus the classification report.")

    # Model selection
    models = list_models(MODELS_DIR)
    if not models:
        st.warning(f"No models found in {MODELS_DIR}. Train and save models first.")
        return

    model_name = st.selectbox("Select a model", options=list(models.keys()))
    model_path = models[model_name]
    model = load_model(model_path)
    st.success(f"Loaded model: {model_name}")

    # Download saved test_data.csv (if present)
    if DEFAULT_TEST_PATH.exists():
        with DEFAULT_TEST_PATH.open("rb") as f:
            data_bytes = f.read()
        st.download_button(
            label="Download saved test_data.csv",
            data=data_bytes,
            file_name="test_data.csv",
            mime="text/csv",
        )
    else:
        st.info("Saved test_data.csv not found in workspace.")

    # Upload test CSV
    uploaded = st.file_uploader("Upload test CSV (columns: scaled features; optional 'target')", type=["csv"])

    # Evaluate either uploaded file or saved test_data.csv
    src_choice = st.radio(
        "Evaluation data source",
        options=["Uploaded CSV", "Saved test_data.csv"],
        index=0 if uploaded is not None else (1 if DEFAULT_TEST_PATH.exists() else 0),
        help="Use the uploaded file if provided; otherwise the saved test_data.csv."
    )

    df = None
    if src_choice == "Uploaded CSV" and uploaded is not None:
        df = pd.read_csv(uploaded)
    elif src_choice == "Saved test_data.csv" and DEFAULT_TEST_PATH.exists():
        df = pd.read_csv(DEFAULT_TEST_PATH)

    if df is None:
        st.stop()

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

    # Show predictions preview
    st.subheader("Predictions preview")
    preview = pd.DataFrame({"prediction": y_pred[:10]})
    st.dataframe(preview, use_container_width=True)

    # Metrics
    st.subheader("Evaluation Metrics")
    if y_true is None:
        st.warning("Upload a CSV with a 'target' column to compute metrics.")
    else:
        metrics, report_text = evaluate(y_true, y_pred, y_score)
        # Nicely formatted metrics table
        metrics_rows = [{"metric": k, "value": (v if v is not None else "N/A")} for k, v in metrics.items()]
        st.table(pd.DataFrame(metrics_rows))
        # Classification report
        st.subheader("Classification Report")
        st.text(report_text)
        if y_score is None:
            st.caption("AUC not available for this model (no predict_proba/decision_function).")

    st.divider()
    st.caption(f"Models directory: {MODELS_DIR} • Saved test: {DEFAULT_TEST_PATH}")

if __name__ == "__main__":
    main()