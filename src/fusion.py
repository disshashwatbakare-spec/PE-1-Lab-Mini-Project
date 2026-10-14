"""
fusion.py
=========
Decision fusion layer: combines outputs from three AI components
  1. Rule Engine  (symbolic AI)
  2. Best ML Classifier (supervised)
  3. Isolation Forest Anomaly Detector (unsupervised)

Fusion logic
------------
* AGREE     : Rule engine and ML classifier agree -> high-confidence verdict
* DISAGREE  : They disagree -> use ML confidence weighted against rule score;
              raise "manual_review" flag
* ANOMALY   : If ML says legitimate but anomaly flag is raised -> lower
              confidence and flag for review

Final output dict
-----------------
{
  "final_verdict"      : "phishing" | "legitimate",
  "final_confidence"   : 0.0-1.0,
  "manual_review"      : bool,
  "signals"            : {
      "rule_engine"       : { verdict, confidence, explanation, triggered },
      "ml_classifier"     : { verdict, confidence },
      "anomaly_detector"  : { anomaly_score, anomaly_flag },
  },
  "summary"            : "Human-readable string"
}
"""

import pathlib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ROOT = pathlib.Path(__file__).resolve().parents[1]


class FusionEngine:
    """
    Combines rule engine, ML classifier, and anomaly detector verdicts.

    Parameters
    ----------
    ml_weight : float
        Weight given to ML confidence (0-1). Rule engine gets (1 - ml_weight).
    anomaly_confidence_penalty : float
        Confidence reduction applied when anomaly flag contradicts ML verdict.
    """

    def __init__(
        self,
        ml_weight: float = 0.65,
        anomaly_confidence_penalty: float = 0.15,
        manual_review_threshold: float = 0.55,
    ):
        self.ml_weight                  = ml_weight
        self.rule_weight                = 1.0 - ml_weight
        self.anomaly_confidence_penalty = anomaly_confidence_penalty
        self.review_threshold           = manual_review_threshold

    # ------------------------------------------------------------------
    def fuse(
        self,
        rule_result:    dict,
        ml_verdict:     str,
        ml_confidence:  float,
        anomaly_result: dict,
        model_name:     str = "Ensemble",
    ) -> dict:
        """
        Fuse verdicts from all three AI components for a single sample.

        Parameters
        ----------
        rule_result    : output of RuleEngine.predict_sample()
        ml_verdict     : "phishing" | "legitimate"
        ml_confidence  : probability [0, 1]
        anomaly_result : output of AnomalyDetector.predict_sample()
        model_name     : name of winning ML classifier
        """
        rule_verdict    = rule_result["verdict"]
        rule_confidence = rule_result["confidence"]
        rule_explanation= rule_result["explanation"]
        anomaly_flag    = anomaly_result["anomaly_flag"]
        anomaly_score   = anomaly_result["anomaly_score"]
        anomaly_thresh  = anomaly_result.get("threshold", 0.5)

        # ---- Step 1: check agreement ----
        agree = (rule_verdict == ml_verdict)
        manual_review = False

        if agree:
            # Weighted blend of confidences
            blended_conf = (
                self.ml_weight   * ml_confidence +
                self.rule_weight * rule_confidence
            )
            final_verdict = ml_verdict
        else:
            # Disagreement: trust ML more but reduce confidence
            blended_conf  = self.ml_weight * ml_confidence
            # Weighted verdict: pick whichever has higher weighted confidence
            ml_score   = self.ml_weight   * ml_confidence
            rule_score = self.rule_weight * rule_confidence

            if ml_score >= rule_score:
                final_verdict = ml_verdict
            else:
                final_verdict = rule_verdict
                blended_conf  = rule_score

            if blended_conf < self.review_threshold:
                manual_review = True

        # ---- Step 2: anomaly signal adjustment ----
        anomaly_contradicts = (
            anomaly_flag == 1 and final_verdict == "legitimate"
        )
        if anomaly_contradicts:
            blended_conf  = max(0.0, blended_conf - self.anomaly_confidence_penalty)
            manual_review = True

        blended_conf = round(float(np.clip(blended_conf, 0.0, 1.0)), 4)

        # ---- Step 3: build summary ----
        agreement_str = "agree" if agree else "DISAGREE"
        anomaly_str   = "flagged as anomalous" if anomaly_flag else "not anomalous"
        review_str    = " ⚠️ Manual review recommended." if manual_review else ""

        summary = (
            f"Final verdict: {final_verdict.upper()} "
            f"(confidence {blended_conf:.0%}). "
            f"Rule engine and ML classifier {agreement_str}. "
            f"Anomaly detector: {anomaly_str}.{review_str}"
        )

        return {
            "final_verdict"    : final_verdict,
            "final_confidence" : blended_conf,
            "manual_review"    : manual_review,
            "signals": {
                "rule_engine": {
                    "verdict"    : rule_verdict,
                    "confidence" : rule_confidence,
                    "explanation": rule_explanation,
                    "triggered"  : rule_result.get("triggered", []),
                },
                "ml_classifier": {
                    "verdict"   : ml_verdict,
                    "confidence": round(float(ml_confidence), 4),
                    "model_name": model_name,
                },
                "anomaly_detector": {
                    "anomaly_score": anomaly_score,
                    "anomaly_flag" : anomaly_flag,
                    "threshold"    : anomaly_thresh,
                },
            },
            "summary": summary,
        }

    # ------------------------------------------------------------------
    def fuse_batch(
        self,
        rule_results   : list[dict],
        ml_verdicts    : list[str],
        ml_confidences : list[float],
        anomaly_results: list[dict],
    ) -> list[dict]:
        """Vectorised fusion for a list of samples."""
        return [
            self.fuse(r, mv, mc, ar)
            for r, mv, mc, ar in zip(
                rule_results, ml_verdicts, ml_confidences, anomaly_results
            )
        ]


# ---------------------------------------------------------------------------
# Batch evaluation helper
# ---------------------------------------------------------------------------

def evaluate_fusion(
    fused_results: list[dict],
    y_true: np.ndarray,
) -> dict:
    """Compare fused verdicts against ground truth."""
    y_pred = np.array([
        1 if r["final_verdict"] == "phishing" else 0
        for r in fused_results
    ])
    return {
        "accuracy" : round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def compare_approaches(
    rule_preds:   np.ndarray,
    ml_preds:     np.ndarray,
    fused_preds:  np.ndarray,
    y_true:       np.ndarray,
) -> pd.DataFrame:
    """Print and return a comparison table: rule-only vs ML-only vs fused."""
    rows = []
    for name, preds in [("Rule-only", rule_preds),
                        ("ML-only",   ml_preds),
                        ("Fused",     fused_preds)]:
        rows.append({
            "Approach" : name,
            "Accuracy" : round(accuracy_score(y_true, preds), 4),
            "Precision": round(precision_score(y_true, preds, zero_division=0), 4),
            "Recall"   : round(recall_score(y_true, preds, zero_division=0), 4),
            "F1"       : round(f1_score(y_true, preds, zero_division=0), 4),
        })
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))

    # Build dummy inputs
    rule_result_phish = {
        "verdict": "phishing", "confidence": 0.7,
        "explanation": "No HTTPS; IP address used.", "triggered": ["No HTTPS"],
        "risk_score": 4.5,
    }
    rule_result_legit = {
        "verdict": "legitimate", "confidence": 0.1,
        "explanation": "No rules triggered.", "triggered": [],
        "risk_score": 0.5,
    }
    anomaly_phish = {"anomaly_score": 0.8, "anomaly_flag": 1}
    anomaly_legit = {"anomaly_score": 0.2, "anomaly_flag": 0}

    fe = FusionEngine()

    # Case 1: both agree phishing
    r1 = fe.fuse(rule_result_phish, "phishing", 0.91, anomaly_phish)
    print("Case 1 (agree-phishing):", r1["final_verdict"], r1["final_confidence"], r1["manual_review"])

    # Case 2: rule says phishing, ML says legitimate
    r2 = fe.fuse(rule_result_phish, "legitimate", 0.52, anomaly_phish)
    print("Case 2 (disagree):", r2["final_verdict"], r2["final_confidence"], r2["manual_review"])

    # Case 3: both legit but anomaly
    r3 = fe.fuse(rule_result_legit, "legitimate", 0.85, anomaly_phish)
    print("Case 3 (legit+anomaly):", r3["final_verdict"], r3["final_confidence"], r3["manual_review"])

    print("\n=== Smoke test passed ===")
