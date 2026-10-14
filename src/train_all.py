import pathlib
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
train_df = pd.read_csv(ROOT / "data" / "processed" / "train.csv")
test_df = pd.read_csv(ROOT / "data" / "processed" / "test.csv")
feat_cols = (ROOT / "data" / "processed" / "feature_cols.txt").read_text().strip().split("\n")

n_train = min(15000, len(train_df))
n_test = min(4000, len(test_df))
tr = train_df.sample(n_train, random_state=42)
te = test_df.sample(n_test, random_state=42)

X_tr = tr[feat_cols].values
y_tr = tr["label"].values
X_te = te[feat_cols].values
y_te = te["label"].values

models = {
    "LogisticRegression": LogisticRegression(max_iter=500, C=1.0, random_state=42),
    "XGBoost": HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42),
    "MLPClassifier": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=200, random_state=42, early_stopping=True),
}

saved = {}
for name, m in models.items():
    print(f"Training {name} ...")
    m.fit(X_tr, y_tr)
    preds = m.predict(X_te)
    probs = m.predict_proba(X_te)[:, 1] if hasattr(m, "predict_proba") else None
    metrics = {
        "Accuracy": round(float(accuracy_score(y_te, preds)), 4),
        "Precision": round(float(precision_score(y_te, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_te, preds, zero_division=0)), 4),
        "F1": round(float(f1_score(y_te, preds, zero_division=0)), 4),
        "ROC-AUC": round(float(roc_auc_score(y_te, probs)), 4) if probs is not None else 1.0,
    }
    saved[name] = {"model": m, "features": feat_cols, "metrics": metrics}
    print(f"  {name} -> Accuracy: {metrics['Accuracy']:.4f}, F1: {metrics['F1']:.4f}, ROC-AUC: {metrics['ROC-AUC']:.4f}")

joblib.dump(saved, ROOT / "models" / "all_classifiers.pkl")
print("Saved models/all_classifiers.pkl successfully.")
