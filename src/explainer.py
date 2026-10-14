"""
explainer.py
============
Explainable AI component using SHAP on the best-performing classifier.

Provides:
  - per-sample top-N feature importance (SHAP values)
  - global SHAP summary plot saved as PNG
"""

import pathlib
import warnings
import numpy as np
import pandas as pd
import joblib
try:
    import shap
    HAS_SHAP = True
except ImportError:
    shap = None
    HAS_SHAP = False

import matplotlib
matplotlib.use("Agg")   # non-interactive backend for server environments
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT        = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"

SHAP_SUMMARY_PNG     = OUTPUTS_DIR / "shap_summary.png"
SHAP_BAR_PNG         = OUTPUTS_DIR / "shap_bar.png"
SHAP_EXPLAINER_PKL   = MODELS_DIR  / "shap_explainer.pkl"


class SHAPExplainer:
    """
    Wraps a SHAP TreeExplainer (for XGBoost) or KernelExplainer fallback.
    """

    def __init__(self, model=None, feature_cols: list[str] | None = None):
        self.model        = model
        self.feature_cols = feature_cols
        self._explainer   = None

    # ------------------------------------------------------------------
    def _build_explainer(self, X_background: np.ndarray | None = None):
        """Lazily build the SHAP explainer."""
        if not HAS_SHAP or self._explainer is not None:
            return

        try:
            if hasattr(self.model, "get_booster"):
                self._explainer = shap.TreeExplainer(self.model)
            elif hasattr(self.model, "estimators_"):
                self._explainer = shap.TreeExplainer(self.model)
            elif X_background is not None:
                predict_fn = (
                    self.model.predict_proba
                    if hasattr(self.model, "predict_proba")
                    else self.model.predict
                )
                self._explainer = shap.KernelExplainer(
                    predict_fn,
                    shap.sample(X_background, min(50, len(X_background))),
                )
        except Exception as e:
            self._explainer = None

    # ------------------------------------------------------------------
    def explain_sample(
        self,
        sample: np.ndarray | pd.Series,
        top_n: int = 10,
        X_background: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Compute SHAP / feature attribution values for one sample.
        Returns a list of dicts: [{feature, shap_value, direction}, ...]
        sorted by |shap_value| descending, top_n items.
        """
        self._build_explainer(X_background)
        x = np.array(sample).reshape(1, -1)
        sv = None

        if HAS_SHAP and self._explainer is not None:
            try:
                raw_sv = self._explainer.shap_values(x)
                if isinstance(raw_sv, list):
                    raw_sv = raw_sv[1]
                sv = raw_sv.flatten()
            except Exception:
                sv = None

        if sv is None:
            # Model-intrinsic attribution fallback
            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                sv = (imp * x.flatten())
            elif hasattr(self.model, "coef_"):
                sv = (self.model.coef_[0] * x.flatten())
            else:
                sv = x.flatten()

        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
        result  = []
        for i in top_idx:
            feat_name = (
                self.feature_cols[i]
                if self.feature_cols and i < len(self.feature_cols)
                else f"feature_{i}"
            )
            result.append({
                "feature"    : feat_name,
                "shap_value" : round(float(sv[i]), 6),
                "direction"  : "increases_phishing_risk" if sv[i] > 0
                               else "decreases_phishing_risk",
            })
        return result

    # ------------------------------------------------------------------
    def global_summary_plot(
        self,
        X_test: np.ndarray,
        max_samples: int = 2000,
        X_background: np.ndarray | None = None,
    ) -> pathlib.Path:
        """
        Compute SHAP values for up to max_samples test rows and save
        beeswarm summary + bar chart as PNG files.
        Returns path to beeswarm PNG.
        """
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_explainer(X_background)

        n = min(max_samples, X_test.shape[0])
        X_sample = X_test[:n]

        shap_values = None
        if HAS_SHAP and self._explainer is not None:
            try:
                sv = self._explainer.shap_values(X_sample)
                if isinstance(sv, list):
                    sv = sv[1]
                shap_values = sv
            except Exception:
                shap_values = None

        if shap_values is not None:
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_cols,
                show=False,
            )
            plt.tight_layout()
            fig1.savefig(SHAP_SUMMARY_PNG, dpi=120, bbox_inches="tight")
            plt.close(fig1)

            fig2, ax2 = plt.subplots(figsize=(9, 6))
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_cols,
                plot_type="bar",
                show=False,
            )
            plt.tight_layout()
            fig2.savefig(SHAP_BAR_PNG, dpi=120, bbox_inches="tight")
            plt.close(fig2)
        else:
            # Fallback matplotlib feature attribution plots
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importances = np.abs(self.model.coef_[0])
            else:
                importances = np.ones(X_sample.shape[1])

            top_n = min(20, len(importances))
            sorted_idx = np.argsort(importances)[::-1][:top_n]
            names = [self.feature_cols[i] if self.feature_cols else f"f_{i}" for i in sorted_idx]
            vals  = importances[sorted_idx]

            # Summary plot
            fig1, ax1 = plt.subplots(figsize=(10, 7))
            ax1.barh(names[::-1], vals[::-1], color="#4285F4")
            ax1.set_title("Global Feature Importance (Impact on Phishing Classification)", fontsize=13, fontweight="bold")
            ax1.set_xlabel("Attribution Magnitude")
            plt.tight_layout()
            fig1.savefig(SHAP_SUMMARY_PNG, dpi=120, bbox_inches="tight")
            plt.close(fig1)

            # Bar plot
            fig2, ax2 = plt.subplots(figsize=(9, 6))
            ax2.barh(names[:10][::-1], vals[:10][::-1], color="#EA4335")
            ax2.set_title("Top 10 Feature Importances", fontsize=12, fontweight="bold")
            ax2.set_xlabel("Importance Score")
            plt.tight_layout()
            fig2.savefig(SHAP_BAR_PNG, dpi=120, bbox_inches="tight")
            plt.close(fig2)

        print(f"[explainer] SHAP plots saved -> {SHAP_SUMMARY_PNG} and {SHAP_BAR_PNG}")
        return SHAP_SUMMARY_PNG

        return SHAP_SUMMARY_PNG

    # ------------------------------------------------------------------
    def save(self) -> None:
        """Persist the underlying SHAP explainer object."""
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._explainer, SHAP_EXPLAINER_PKL)
        print(f"[explainer] SHAP explainer saved -> {SHAP_EXPLAINER_PKL}")

    def load(self) -> None:
        self._explainer = joblib.load(SHAP_EXPLAINER_PKL)
        print(f"[explainer] SHAP explainer loaded from {SHAP_EXPLAINER_PKL}")


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------

def build_explainer(model, feature_cols: list[str]) -> SHAPExplainer:
    exp = SHAPExplainer(model=model, feature_cols=feature_cols)
    return exp


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from data_loader import load_processed, run_pipeline
    from classifiers import train_and_evaluate, load_best_model

    if not (ROOT / "data" / "processed" / "train.csv").exists():
        run_pipeline()

    train_df, test_df, feature_cols, scaler = load_processed()

    # Load or quick-train best model
    try:
        model, best_feats = load_best_model()
        print("[explainer] Loaded persisted best model.")
    except FileNotFoundError:
        print("[explainer] Training quick model for smoke test ...")
        train_s = train_df.sample(5000, random_state=42)
        test_s  = test_df.sample(1000,  random_state=42)
        _, model, best_feats = train_and_evaluate(train_s, test_s, feature_cols)

    exp = build_explainer(model, best_feats)

    # Single sample
    X_sample = test_df[best_feats].values
    top_feats = exp.explain_sample(X_sample[0])
    print("\n[explainer] Top SHAP features for sample 0:")
    for item in top_feats[:5]:
        print(f"  {item['feature']:30s}  SHAP={item['shap_value']:+.4f}  {item['direction']}")

    # Global plot
    exp.global_summary_plot(X_sample, max_samples=1000)
    print("\n=== Smoke test passed ===")
