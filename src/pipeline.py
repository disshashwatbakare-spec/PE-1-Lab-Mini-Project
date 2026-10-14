"""
pipeline.py
===========
Orchestrates the full PhishShield AI pipeline end-to-end.

Usage
-----
  from src.pipeline import PhishShieldPipeline
  pipe = PhishShieldPipeline()
  pipe.train()          # trains all models; call once
  result = pipe.predict(feature_vector_dict)
  # or
  result = pipe.predict_from_row(df_row)
"""

import pathlib
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(pathlib.Path(__file__).parent))

from data_loader import (
    load_raw, get_feature_cols, split_and_scale, save_splits,
    load_processed, RAW_CSV, SCALER_PKL
)
from rule_engine    import RuleEngine
from classifiers    import train_and_evaluate, load_best_model
from anomaly_detector import AnomalyDetector, train_anomaly_detector, load_anomaly_detector
from fusion         import FusionEngine
from explainer      import SHAPExplainer, build_explainer
from llm_explainer  import generate_report

TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
TEST_CSV  = ROOT / "data" / "processed" / "test.csv"


class PhishShieldPipeline:
    """
    Top-level controller that wires all AI modules together.

    Attributes loaded lazily after calling .train() or .load_all():
      rule_engine   : RuleEngine
      ml_model      : trained classifier
      ml_features   : list[str]  - features used by ml_model
      anomaly       : AnomalyDetector
      shap_exp      : SHAPExplainer
      scaler        : StandardScaler
      feature_cols  : list[str]  - all numeric feature names
    """

    def __init__(self):
        self.rule_engine  = RuleEngine()
        self.ml_model     = None
        self.ml_features  = None
        self.anomaly      = None
        self.shap_exp     = None
        self.scaler       = None
        self.feature_cols = None
        self._trained     = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        csv_path=None,
        skip_if_cached: bool = True,
    ) -> dict:
        """
        Full training pipeline.
        If skip_if_cached=True and processed splits exist, skips data loading.
        Returns dict with performance metrics.
        """
        csv_path = csv_path or RAW_CSV
        best_pkl = ROOT / "models" / "best_model.pkl"
        iso_pkl  = ROOT / "models" / "isolation_forest.pkl"
        res_csv  = ROOT / "outputs" / "classifier_results.csv"

        if skip_if_cached and best_pkl.exists() and iso_pkl.exists() and res_csv.exists() and TRAIN_CSV.exists():
            print("[pipeline] Cached models and results found - loading from disk ...")
            self.load_all()
            results_df = pd.read_csv(res_csv)
            best_row   = results_df.loc[results_df["F1"].idxmax()]
            return {
                "best_classifier"       : best_row["Classifier"],
                "best_feature_selection": best_row["FeatureSelection"],
                "best_f1"               : best_row["F1"],
                "best_roc_auc"          : best_row["ROC-AUC"],
            }

        # ---- Step 1: data ----
        if skip_if_cached and TRAIN_CSV.exists():
            print("[pipeline] Processed data found - loading from cache ...")
            train_df, test_df, self.feature_cols, self.scaler = load_processed()
        else:
            print("[pipeline] Running data pipeline ...")
            df_raw = load_raw(csv_path)
            train_df, test_df, self.feature_cols, self.scaler = split_and_scale(df_raw)
            save_splits(train_df, test_df, self.scaler, self.feature_cols)

        # Sample for CPU-friendly training if dataset is huge
        if len(train_df) > 15000:
            train_sub = train_df.sample(15000, random_state=42)
            test_sub  = test_df.sample(min(4000, len(test_df)), random_state=42)
        else:
            train_sub, test_sub = train_df, test_df

        # ---- Step 2: classifiers ----
        print("\n[pipeline] Training classifiers ...")
        results_df, self.ml_model, self.ml_features = train_and_evaluate(
            train_sub, test_sub, self.feature_cols
        )

        # ---- Step 3: anomaly detector ----
        print("\n[pipeline] Training anomaly detector ...")
        self.anomaly, anomaly_metrics = train_anomaly_detector(
            train_sub, test_sub, self.feature_cols, save=True
        )

        # ---- Step 4: SHAP explainer ----
        print("\n[pipeline] Building SHAP explainer ...")
        self.shap_exp = build_explainer(self.ml_model, self.ml_features)

        # Global SHAP plot on a sample
        X_test = test_df[self.ml_features].values
        try:
            self.shap_exp.global_summary_plot(X_test, max_samples=1500)
        except Exception as e:
            print(f"[pipeline] SHAP global plot skipped: {e}")

        self._trained = True

        # ---- Headline metrics ----
        best_row = results_df.loc[results_df["F1"].idxmax()]
        metrics = {
            "best_classifier"       : best_row["Classifier"],
            "best_feature_selection": best_row["FeatureSelection"],
            "best_f1"               : best_row["F1"],
            "best_roc_auc"          : best_row["ROC-AUC"],
            "anomaly_metrics"       : anomaly_metrics,
        }
        print("\n[pipeline] Training complete.")
        print(f"  Best classifier : {metrics['best_classifier']} + {metrics['best_feature_selection']}")
        print(f"  F1              : {metrics['best_f1']}")
        print(f"  ROC-AUC         : {metrics['best_roc_auc']}")

        return metrics

    # ------------------------------------------------------------------
    # Loading pre-trained models
    # ------------------------------------------------------------------

    def load_all(self) -> None:
        """Load all persisted models without retraining."""
        _, _, self.feature_cols, self.scaler = load_processed()
        self.ml_model, self.ml_features      = load_best_model()
        self.anomaly                         = load_anomaly_detector()
        self.shap_exp                        = build_explainer(self.ml_model, self.ml_features)
        self._trained = True
        print("[pipeline] All models loaded successfully.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _prepare_feature_vector(self, feature_dict: dict) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert a raw-feature dict to two arrays:
          - unscaled_vec : for rule engine (original space)
          - scaled_vec   : for ML model and anomaly detector
        """
        unscaled = np.array([
            feature_dict.get(c, 0.0) for c in self.feature_cols
        ], dtype=float)

        scaled = self.scaler.transform(unscaled.reshape(1, -1)).flatten()
        return unscaled, scaled

    def predict(
        self,
        feature_dict: dict,
        top_n_shap: int = 10,
        use_llm: bool = True,
        llm_model: str = "phi3",
        llm_host: str = "http://localhost:11434",
        llm_audience: str = "general",
    ) -> dict:
        """
        Full inference for a single URL's feature vector (unscaled dict).

        Parameters
        ----------
        feature_dict : dict mapping feature name -> raw (unscaled) value
        top_n_shap   : number of SHAP features to return
        use_llm      : whether to attempt Ollama call
        llm_model    : model tag name in Ollama
        llm_host     : Ollama API endpoint
        llm_audience : 'general' or 'technical'

        Returns
        -------
        Structured result dict with all component outputs.
        """
        if not self._trained:
            raise RuntimeError("Call .train() or .load_all() first.")

        unscaled_vec, scaled_vec = self._prepare_feature_vector(feature_dict)

        # ---- Rule engine (unscaled) ----
        rule_result = self.rule_engine.predict_sample(feature_dict)

        # ---- ML classifier (scaled) ----
        ml_feat_vec = np.array([
            scaled_vec[self.feature_cols.index(c)]
            if c in self.feature_cols else 0.0
            for c in self.ml_features
        ]).reshape(1, -1)

        ml_pred = self.ml_model.predict(ml_feat_vec)[0]
        if hasattr(self.ml_model, "predict_proba"):
            ml_conf = float(self.ml_model.predict_proba(ml_feat_vec)[0][ml_pred])
        else:
            ml_conf = 0.5
        ml_verdict = "phishing" if ml_pred == 1 else "legitimate"

        # ---- Anomaly detector (scaled) ----
        anomaly_result = self.anomaly.predict_sample(scaled_vec)

        # ---- Fusion ----
        fuser  = FusionEngine()
        m_name = getattr(self.ml_model, "__class__", type("")).__name__ or "Ensemble"
        fused  = fuser.fuse(rule_result, ml_verdict, ml_conf, anomaly_result, model_name=m_name)

        # ---- SHAP (scaled ML-feature vector) ----
        shap_features = []
        try:
            shap_features = self.shap_exp.explain_sample(ml_feat_vec[0], top_n=top_n_shap)
        except Exception as e:
            print(f"[pipeline] SHAP explain skipped: {e}")

        # ---- LLM report ----
        llm_out = generate_report(
            verdict      = fused["final_verdict"],
            confidence   = fused["final_confidence"],
            top_features = shap_features,
            rule_triggers= rule_result.get("triggered", []),
            use_llm      = use_llm,
            model_name   = llm_model,
            host         = llm_host,
            audience     = llm_audience,
        )

        return {
            "verdict"         : fused["final_verdict"],
            "confidence"      : fused["final_confidence"],
            "manual_review"   : fused["manual_review"],
            "summary"         : fused["summary"],
            "nl_report"       : llm_out["report"],
            "report_source"   : llm_out["source"],
            "report_model"    : llm_out.get("model", "template"),
            "report_error"    : llm_out.get("error"),
            "signals"         : fused["signals"],
            "shap_top_features": shap_features,
        }

    def predict_from_row(
        self,
        row: pd.Series,
        top_n_shap: int = 10,
        use_llm: bool = True,
    ) -> dict:
        """Convenience wrapper: accepts a pd.Series (unscaled)."""
        return self.predict(row.to_dict(), top_n_shap=top_n_shap, use_llm=use_llm)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    pipe = PhishShieldPipeline()

    # Quick training on sample data
    print("=== PhishShield Pipeline Smoke Test ===\n")
    pipe.train(skip_if_cached=True)

    # Grab a raw sample from the dataset
    raw_df = load_raw()
    sample = raw_df.sample(1, random_state=7).iloc[0]
    feature_cols = get_feature_cols(raw_df)
    feature_dict = {c: sample[c] for c in feature_cols}

    result = pipe.predict(feature_dict, use_llm=False)

    print(f"\n--- Prediction result ---")
    print(f"  Verdict        : {result['verdict']}")
    print(f"  Confidence     : {result['confidence']:.2%}")
    print(f"  Manual review  : {result['manual_review']}")
    print(f"  NL report      : {result['nl_report']}")
    print(f"  SHAP top feats : {[f['feature'] for f in result['shap_top_features'][:3]]}")
    print("\n=== Smoke test passed ===")
