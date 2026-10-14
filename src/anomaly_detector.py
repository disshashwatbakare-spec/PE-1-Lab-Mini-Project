"""
anomaly_detector.py
===================
Unsupervised AI component: Isolation Forest trained ONLY on legitimate URLs
(no label leakage). At inference time, produces an anomaly score and a binary
flag indicating structural unusualness.

Rationale: Phishing pages often have novel structural signatures (zero-day
patterns) that a supervised model trained on historical examples may not
capture. An anomaly detector acts as a complementary second-opinion.

Evaluation: Although training is unsupervised, we evaluate alignment with
ground-truth labels on the test set (phishing URLs should score as anomalies).
"""

import pathlib
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report
)

ROOT         = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR   = ROOT / "models"
OUTPUTS_DIR  = ROOT / "outputs"
ISO_PKL      = MODELS_DIR / "isolation_forest.pkl"
ISO_META_PKL = MODELS_DIR / "isolation_forest_meta.pkl"


class AnomalyDetector:
    """
    Isolation Forest trained on legitimate URLs only.
    Higher anomaly_score -> more anomalous -> more likely phishing.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self.threshold: float | None = None  # learned from training set

    # ------------------------------------------------------------------
    def fit(self, train_df: pd.DataFrame, feature_cols: list[str]) -> None:
        """
        Train ONLY on legitimate rows (label == 0).
        Also calibrate an anomaly threshold using all training rows.
        """
        legit_df = train_df[train_df["label"] == 0]
        X_legit  = legit_df[feature_cols].values
        print(f"[anomaly] Training IsolationForest on {len(X_legit)} legitimate rows ...")
        self.model.fit(X_legit)

        # Calibrate: score_samples returns negative anomaly scores
        # (more negative = more anomalous). Flip sign for interpretability.
        all_scores = -self.model.score_samples(train_df[feature_cols].values)
        # Set threshold at 75th percentile of legitimate URL scores
        legit_scores = -self.model.score_samples(X_legit)
        self.threshold = float(np.percentile(legit_scores, 90))
        print(f"[anomaly] Anomaly score threshold: {self.threshold:.4f}")

    # ------------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores (positive float; higher = more anomalous)."""
        return -self.model.score_samples(X)

    def predict_df(self, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
        """
        Score each row.
        Returns DataFrame with 'anomaly_score' and 'anomaly_flag' (1 = anomalous).
        """
        X      = df[feature_cols].values
        scores = self.score_samples(X)
        flags  = (scores >= self.threshold).astype(int)
        return pd.DataFrame({
            "anomaly_score": np.round(scores, 6),
            "anomaly_flag" : flags,
        })

    def predict_sample(self, sample_vec: np.ndarray, feature_cols: list[str] | None = None) -> dict:
        """Score a single sample (1-D numpy array or list)."""
        x = np.array(sample_vec).reshape(1, -1)
        score = float(self.score_samples(x)[0])
        thresh = float(self.threshold) if self.threshold is not None else 0.5
        flag  = int(score >= thresh)
        return {
            "anomaly_score": round(score, 6),
            "anomaly_flag" : flag,
            "threshold"    : round(thresh, 4),
        }

    # ------------------------------------------------------------------
    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model,     ISO_PKL)
        joblib.dump(self.threshold, ISO_META_PKL)
        print(f"[anomaly] Saved IsolationForest -> {ISO_PKL}")

    def load(self) -> None:
        self.model     = joblib.load(ISO_PKL)
        self.threshold = joblib.load(ISO_META_PKL)
        print(f"[anomaly] Loaded IsolationForest (threshold={self.threshold:.4f})")


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def evaluate(detector: AnomalyDetector,
             test_df: pd.DataFrame,
             feature_cols: list[str]) -> dict:
    """
    Evaluate anomaly detector on labelled test set.
    anomaly_flag==1 should correspond to phishing (label==1).
    """
    result_df = detector.predict_df(test_df, feature_cols)
    y_true    = test_df["label"].values
    y_pred    = result_df["anomaly_flag"].values
    scores    = result_df["anomaly_score"].values

    metrics = {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    try:
        metrics["roc_auc"] = round(roc_auc_score(y_true, scores), 4)
    except Exception:
        metrics["roc_auc"] = None

    return metrics


# ---------------------------------------------------------------------------
# Training convenience function
# ---------------------------------------------------------------------------

def train_anomaly_detector(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list[str],
    save: bool = True,
) -> tuple[AnomalyDetector, dict]:
    detector = AnomalyDetector()
    detector.fit(train_df, feature_cols)

    metrics = evaluate(detector, test_df, feature_cols)
    print(f"[anomaly] Test evaluation:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v}")

    if save:
        detector.save()

    return detector, metrics


def load_anomaly_detector() -> AnomalyDetector:
    d = AnomalyDetector()
    d.load()
    return d


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data_loader import load_processed, run_pipeline

    if not (ROOT / "data" / "processed" / "train.csv").exists():
        run_pipeline()

    train_df, test_df, feature_cols, scaler = load_processed()

    # Sample for speed
    train_s = train_df.sample(min(10000, len(train_df)), random_state=42)
    test_s  = test_df.sample(min(2000,  len(test_df)),  random_state=42)

    detector, metrics = train_anomaly_detector(train_s, test_s, feature_cols)
    print("\n=== Smoke test passed ===")
    print(f"  Metrics: {metrics}")

    # Single sample
    demo_row = test_s.iloc[0][feature_cols].values
    print(f"  Single sample: {detector.predict_sample(demo_row)}")
