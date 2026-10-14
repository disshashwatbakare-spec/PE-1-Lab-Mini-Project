"""
classifiers.py
==============
Supervised ML component: trains and evaluates three classifiers
  1. Logistic Regression  (baseline)
  2. XGBoost              (expected best)
  3. MLPClassifier        (neural network paradigm, CPU-friendly)

Also performs feature selection comparison across:
  - SelectKBest (mutual_info_classif)
  - RFE (Recursive Feature Elimination)
  - Greedy forward selection (custom)

All results are saved to /outputs/classifier_results.csv.
Best model is persisted to /models/best_model.pkl.
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import joblib
import json
from itertools import product

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif, RFE
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import HistGradientBoostingClassifier as XGBClassifier
    HAS_XGBOOST = False

warnings.filterwarnings("ignore")

ROOT        = pathlib.Path(__file__).resolve().parents[1]
MODELS_DIR  = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

BEST_MODEL_PKL    = MODELS_DIR / "best_model.pkl"
BEST_FEATURES_PKL = MODELS_DIR / "best_features.pkl"
RESULTS_CSV       = OUTPUTS_DIR / "classifier_results.csv"


# ---------------------------------------------------------------------------
# Classifier catalogue
# ---------------------------------------------------------------------------

def get_classifiers() -> dict:
    if HAS_XGBOOST:
        xgb_inst = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    else:
        xgb_inst = XGBClassifier(
            max_iter=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )

    return {
        "LogisticRegression": LogisticRegression(
            max_iter=500, C=1.0, solver="lbfgs", random_state=42, n_jobs=-1
        ),
        "XGBoost": xgb_inst,
        "MLPClassifier": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            max_iter=200,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        ),
    }


# ---------------------------------------------------------------------------
# Feature selection methods
# ---------------------------------------------------------------------------

def select_kbest(X_train: np.ndarray, y_train: np.ndarray,
                 feature_cols: list[str], k: int = 25
                 ) -> tuple[np.ndarray, list[str]]:
    """SelectKBest with mutual information."""
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    selector.fit(X_train, y_train)
    mask = selector.get_support()
    return mask, [c for c, m in zip(feature_cols, mask) if m]


def select_rfe(X_train: np.ndarray, y_train: np.ndarray,
               feature_cols: list[str], n_features: int = 20
               ) -> tuple[np.ndarray, list[str]]:
    """RFE with LogisticRegression as estimator."""
    estimator = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)
    rfe = RFE(estimator, n_features_to_select=min(n_features, X_train.shape[1]),
              step=2)
    rfe.fit(X_train, y_train)
    mask = rfe.support_
    return mask, [c for c, m in zip(feature_cols, mask) if m]


def select_greedy(X_train: np.ndarray, y_train: np.ndarray,
                  feature_cols: list[str], max_features: int = 30,
                  cv: int = 3) -> tuple[np.ndarray, list[str]]:
    """
    Custom greedy forward feature selection.
    Ranks features by individual XGBoost importance, then adds them
    one at a time keeping only those that improve CV accuracy.
    """
    # Step 1: rank features by individual importance
    if HAS_XGBOOST:
        quick_clf = XGBClassifier(
            n_estimators=50, max_depth=4, random_state=42,
            eval_metric="logloss", n_jobs=-1,
        )
        quick_clf.fit(X_train, y_train)
        importances = quick_clf.feature_importances_
    else:
        from sklearn.ensemble import RandomForestClassifier
        quick_clf = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=42, n_jobs=-1)
        quick_clf.fit(X_train, y_train)
        importances = quick_clf.feature_importances_
    ranked_idx  = np.argsort(importances)[::-1]

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    selected_idx = []
    best_score   = 0.0

    clf = LogisticRegression(max_iter=300, solver="lbfgs", random_state=42)

    for idx in ranked_idx[:max_features]:
        candidate = selected_idx + [idx]
        X_cand    = X_train[:, candidate]
        scores    = cross_val_score(clf, X_cand, y_train, cv=skf,
                                    scoring="accuracy", n_jobs=-1)
        mean_score = scores.mean()
        if mean_score > best_score:
            best_score   = mean_score
            selected_idx = candidate

    mask = np.zeros(X_train.shape[1], dtype=bool)
    for i in selected_idx:
        mask[i] = True
    return mask, [c for c, m in zip(feature_cols, mask) if m]


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    out = {
        "Accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }
    if y_prob is not None:
        out["ROC-AUC"] = round(roc_auc_score(y_true, y_prob), 4)
    else:
        out["ROC-AUC"] = None
    return out


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def train_and_evaluate(
    train_df: pd.DataFrame,
    test_df:  pd.DataFrame,
    feature_cols: list[str],
) -> tuple[pd.DataFrame, object, list[str]]:
    """
    Train all classifiers × feature selection methods.
    Returns (results_df, best_model, best_feature_cols).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test  = test_df[feature_cols].values
    y_test  = test_df["label"].values

    # ---- Build feature selection masks ----
    print("[classifiers] Computing feature selection masks ...")
    masks = {
        "All Features": np.ones(X_train.shape[1], dtype=bool),
    }

    kb_mask, kb_cols = select_kbest(X_train, y_train, feature_cols)
    masks["SelectKBest"] = kb_mask
    print(f"  SelectKBest selected {kb_mask.sum()} features")

    rfe_mask, rfe_cols = select_rfe(X_train, y_train, feature_cols)
    masks["RFE"] = rfe_mask
    print(f"  RFE selected {rfe_mask.sum()} features")

    greedy_mask, greedy_cols = select_greedy(X_train, y_train, feature_cols)
    masks["Greedy"] = greedy_mask
    print(f"  Greedy selected {greedy_mask.sum()} features")

    classifiers = get_classifiers()
    rows  = []
    models = {}

    for fs_name, mask in masks.items():
        Xtr = X_train[:, mask]
        Xte = X_test[:, mask]
        selected_cols = [c for c, m in zip(feature_cols, mask) if m]
        print(f"\n[classifiers] Feature selection: {fs_name} ({mask.sum()} features)")

        for clf_name, clf in classifiers.items():
            print(f"  Training {clf_name} ...", end=" ", flush=True)
            clf_copy = clf.__class__(**clf.get_params())

            clf_copy.fit(Xtr, y_train)
            y_pred = clf_copy.predict(Xte)

            if hasattr(clf_copy, "predict_proba"):
                y_prob = clf_copy.predict_proba(Xte)[:, 1]
            else:
                y_prob = None

            metrics = compute_metrics(y_test, y_pred, y_prob)
            print(f"F1={metrics['F1']:.4f}  AUC={metrics.get('ROC-AUC','N/A')}")

            row = {
                "FeatureSelection": fs_name,
                "Classifier"      : clf_name,
                "NumFeatures"     : int(mask.sum()),
                **metrics,
            }
            rows.append(row)
            models[(fs_name, clf_name)] = (clf_copy, selected_cols)

    results_df = pd.DataFrame(rows)
    results_df.to_csv(RESULTS_CSV, index=False)
    print(f"\n[classifiers] Results saved -> {RESULTS_CSV}")

    # ---- Find best model (prioritize XGBoost when tied on top F1) ----
    max_f1 = results_df["F1"].max()
    top_candidates = results_df[results_df["F1"] == max_f1]
    xgb_cand = top_candidates[top_candidates["Classifier"] == "XGBoost"]
    if not xgb_cand.empty:
        best_row = xgb_cand.iloc[0]
    else:
        best_row = top_candidates.iloc[0]

    best_fs  = best_row["FeatureSelection"]
    best_clf = best_row["Classifier"]
    best_model, best_feats = models[(best_fs, best_clf)]

    print(f"\n[classifiers] Best: {best_clf} + {best_fs}  "
          f"F1={best_row['F1']:.4f}  AUC={best_row['ROC-AUC']}")

    joblib.dump(best_model, BEST_MODEL_PKL)
    joblib.dump(best_feats, BEST_FEATURES_PKL)
    print(f"[classifiers] Best model saved -> {BEST_MODEL_PKL}")

    return results_df, best_model, best_feats


def load_best_model():
    """Load persisted best model and its feature list."""
    model    = joblib.load(BEST_MODEL_PKL)
    features = joblib.load(BEST_FEATURES_PKL)
    return model, features


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data_loader import load_processed, run_pipeline

    if not (ROOT / "data" / "processed" / "train.csv").exists():
        print("[classifiers] Processed data not found - running data pipeline ...")
        run_pipeline()

    train_df, test_df, feature_cols, scaler = load_processed()

    # Use a 10% sample for smoke test speed
    n = min(10000, len(train_df))
    m = min(2000,  len(test_df))
    train_s = train_df.sample(n, random_state=42)
    test_s  = test_df.sample(m,  random_state=42)

    results, best_model, best_feats = train_and_evaluate(train_s, test_s, feature_cols)
    print("\n=== Smoke test passed ===")
    print(results[["FeatureSelection","Classifier","F1","ROC-AUC"]].to_string(index=False))
