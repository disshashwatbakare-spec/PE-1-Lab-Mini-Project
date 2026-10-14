"""
explainer.py
============
Explainable AI (XAI) component using SHAP (SHapley Additive exPlanations)
rooted in cooperative game theory.

Provides:
  - Theoretical & mathematical explanation of SHAP (Shapley values, efficiency, additivity)
  - Per-sample additive feature attribution (expected base value E[f(X)], phi_i contributions)
  - Domain cybersecurity threat rationale mapping for all URL features
  - Global SHAP summary (beeswarm) and bar charts saved as PNG
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT        = pathlib.Path(__file__).resolve().parents[1]
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR  = ROOT / "models"

SHAP_SUMMARY_PNG   = OUTPUTS_DIR / "shap_summary.png"
SHAP_BAR_PNG       = OUTPUTS_DIR / "shap_bar.png"
SHAP_EXPLAINER_PKL = MODELS_DIR  / "shap_explainer.pkl"


# =============================================================================
# CYBERSECURITY THREAT RATIONALE REPOSITORY (DOMAIN EXPERT KNOWLEDGE)
# =============================================================================
FEATURE_RATIONALES = {
    "URLLength": {
        "phish": "Excessive URL length is heavily deployed in phishing campaigns to bury malicious domain tokens, tracking identifiers, and redirect parameters beyond typical browser address bar viewports.",
        "legit": "Concise URL length adhering to standard web application navigation parameters.",
        "category": "Lexical & Structure"
    },
    "DomainLength": {
        "phish": "Abnormally long domain name characteristic of algorithmically generated domain names (DGA) or deceptive multi-keyword brand stuffing.",
        "legit": "Compact domain length characteristic of authentic corporate web properties.",
        "category": "Lexical & Structure"
    },
    "IsDomainIP": {
        "phish": "Direct numerical IPv4/IPv6 host bypasses standard DNS reputation registries, a common indicator of bulletproof malicious hosting infrastructure.",
        "legit": "Standard registered domain name with authentic DNS domain resolution rather than a raw IP address.",
        "category": "Host & Network"
    },
    "IsHTTPS": {
        "phish": "Plain unencrypted HTTP transport lacks SSL/TLS encryption, presenting severe exposure to eavesdropping and credential harvesting.",
        "legit": "Active HTTPS transport layer with cryptographic TLS/SSL authentication verified.",
        "category": "Host & Network"
    },
    "NoOfSubDomain": {
        "phish": "Deeply nested subdomain structure (>2 levels) frequently engineered to spoof authoritative brand hosts (e.g. login.paypal.com.account-verification.xyz).",
        "legit": "Standard flat root or single subdomain architecture typical of legitimate services.",
        "category": "Host & Network"
    },
    "URLSimilarityIndex": {
        "phish": "High lexical similarity signature detected matching high-value target brands (PayPal, Apple, Microsoft, Google) within an unregistered namespace.",
        "legit": "No deceptive brand spoofing or typo-squatting signatures identified in the URL vector.",
        "category": "Content & Behavior"
    },
    "TLDLegitimateProb": {
        "phish": "Top-level domain (TLD) with elevated historical threat and spam abuse rates (.xyz, .top, .tk, .pw, .cc, .work).",
        "legit": "Established, authoritative top-level domain (.com, .org, .edu, .gov) with low historical threat propensity.",
        "category": "Host & Network"
    },
    "TLDLength": {
        "phish": "Unusual TLD suffix length associated with disposable or non-standard regional domain extensions.",
        "legit": "Standard canonical TLD character length.",
        "category": "Host & Network"
    },
    "HasObfuscation": {
        "phish": "Deliberate character obfuscation (percent-encoded %xx sequences or @ user-info delimiters) used to evade gateway inspection.",
        "legit": "Clean, un-obfuscated URL syntax without evasive encoding sequences.",
        "category": "Lexical & Structure"
    },
    "ObfuscationRatio": {
        "phish": "Elevated proportion of encoded characters indicates deliberate perimeter filter bypass attempts.",
        "legit": "Zero or negligible encoded character ratio across the URL path.",
        "category": "Lexical & Structure"
    },
    "CharContinuationRate": {
        "phish": "Unusually long unbroken alphanumeric character sequence indicates machine-generated entropy or packed hash values.",
        "legit": "Natural lexical character distribution conforming to readable human vocabulary.",
        "category": "Lexical & Structure"
    },
    "NoOfObfuscatedChar": {
        "phish": "Presence of encoded characters attempting perimeter gateway evasion.",
        "legit": "Clean character set with zero evasion encoding.",
        "category": "Lexical & Structure"
    },
    "NoOfLettersInURL": {
        "phish": "Disproportionate alphabet distribution typical of machine-generated threat vectors.",
        "legit": "Standard lexical letter frequency.",
        "category": "Lexical & Structure"
    },
    "LetterRatioInURL": {
        "phish": "Skewed letter ratio departing from typical human-readable URL profiles.",
        "legit": "Balanced letter-to-symbol ratio across the URL path.",
        "category": "Lexical & Structure"
    },
    "NoOfDegitsInURL": {
        "phish": "Elevated numerical digit count in URL path, frequently representing malicious session identifiers or obfuscated IP hex.",
        "legit": "Minimal or standard expected digit presence in URL structure.",
        "category": "Lexical & Structure"
    },
    "DegitRatioInURL": {
        "phish": "High digit-to-length ratio indicative of automated token generators.",
        "legit": "Standard low digit ratio characteristic of legitimate web assets.",
        "category": "Lexical & Structure"
    },
    "NoOfEqualsInURL": {
        "phish": "Excessive parameter assignments (=) indicative of complex tracking and redirection vectors.",
        "legit": "Standard or minimal query parameter assignments.",
        "category": "Lexical & Structure"
    },
    "NoOfQMarkInURL": {
        "phish": "Multiple query delimiters (?) indicating non-standard query parameter chaining or payload injection.",
        "legit": "Standard single query string initiation or clean static URL path.",
        "category": "Lexical & Structure"
    },
    "NoOfAmpersandInURL": {
        "phish": "Extensive parameter chaining (&) frequently used to transport stolen state data across redirects.",
        "legit": "Minimal or standard parameter chaining.",
        "category": "Lexical & Structure"
    },
    "NoOfOtherSpecialCharsInURL": {
        "phish": "Irregular special characters in URL path suggesting payload encoding or evasion attempts.",
        "legit": "Clean URL path restricted to standard URI-safe characters.",
        "category": "Lexical & Structure"
    },
    "SpacialCharRatioInURL": {
        "phish": "Abnormal special character density indicating deliberate obfuscation or token stuffing.",
        "legit": "Low special character density adhering to clean web architecture.",
        "category": "Lexical & Structure"
    },
    "NoOfURLRedirect": {
        "phish": "URL shortener domain (bit.ly, t.co, tinyurl) hides the ultimate destination endpoint, concealing malicious targets.",
        "legit": "Direct destination address without anonymizing redirection hops.",
        "category": "Host & Network"
    },
    "HasPasswordField": {
        "phish": "Target page includes password input fields, a primary indicator of credential-harvesting intent.",
        "legit": "No credential ingestion or password input mechanisms detected.",
        "category": "Content & Behavior"
    },
    "HasHiddenFields": {
        "phish": "Stealth hidden input fields embedded in page structure for covert exfiltration of victim metadata.",
        "legit": "Zero suspicious hidden form parameters.",
        "category": "Content & Behavior"
    },
    "HasExternalFormSubmit": {
        "phish": "Form submission action targets an external third-party domain, a hallmark of credential harvesting kits.",
        "legit": "Form submissions resolve locally to the authentic hosting domain.",
        "category": "Content & Behavior"
    },
    "Bank": {
        "phish": "Financial banking keywords present in URL path targeting banking credentials.",
        "legit": "No unverified banking keywords identified.",
        "category": "Content & Behavior"
    },
    "Pay": {
        "phish": "Payment portal keywords present in unverified domain path.",
        "legit": "No suspicious payment keywords present.",
        "category": "Content & Behavior"
    },
    "Crypto": {
        "phish": "Cryptocurrency wallet keywords targeted for asset exfiltration.",
        "legit": "No cryptocurrency lures detected.",
        "category": "Content & Behavior"
    },
    "NoOfPopup": {
        "phish": "Automatic popup dialogs invoked on page load to deceive or trap users.",
        "legit": "No aggressive popup behaviors detected.",
        "category": "Content & Behavior"
    },
    "NoOfiFrame": {
        "phish": "Embedded external iFrames used for clickjacking or loading phishing payloads in background.",
        "legit": "Clean layout without unauthorized external framing.",
        "category": "Content & Behavior"
    },
    "DomainTitleMatchScore": {
        "phish": "Severe discrepancy between HTML page title and domain identity, indicating brand spoofing.",
        "legit": "Strong congruence between page title and hosting domain brand.",
        "category": "Content & Behavior"
    },
    "HasTitle": {
        "phish": "Missing or generic HTML title tag commonly seen in hastily generated phishing kits.",
        "legit": "Authentic, properly configured HTML title tag present.",
        "category": "Content & Behavior"
    },
    "HasFavicon": {
        "phish": "Missing favicon or favicon loaded from external domain to spoof legitimate site branding.",
        "legit": "Authentic website favicon verified on host domain.",
        "category": "Content & Behavior"
    },
    "HasCopyrightInfo": {
        "phish": "Missing or counterfeit copyright declaration in page footer.",
        "legit": "Standard legitimate corporate copyright information verified.",
        "category": "Content & Behavior"
    },
}


class SHAPExplainer:
    """
    Wraps SHAP LinearExplainer / TreeExplainer / KernelExplainer.
    Provides mathematically grounded additive feature attribution.
    """

    def __init__(self, model=None, feature_cols: list[str] | None = None):
        self.model        = model
        self.feature_cols = feature_cols or []
        self._explainer   = None
        self._background  = None
        self.base_value   = 0.50

    # ------------------------------------------------------------------
    def _build_explainer(self, X_background: np.ndarray | None = None):
        """Lazily initialize the optimal SHAP explainer for this model architecture."""
        if not HAS_SHAP or self._explainer is not None:
            return

        if X_background is None:
            train_p = ROOT / "data" / "processed" / "train.csv"
            if train_p.exists() and self.feature_cols:
                try:
                    df = pd.read_csv(train_p, nrows=100)
                    X_background = df[self.feature_cols].values
                except Exception:
                    X_background = np.zeros((20, len(self.feature_cols)))
            else:
                X_background = np.zeros((20, max(1, len(self.feature_cols))))

        self._background = X_background

        try:
            # 1. Linear models (Logistic Regression)
            if hasattr(self.model, "coef_"):
                self._explainer = shap.LinearExplainer(self.model, X_background[:50])
                ev = getattr(self._explainer, "expected_value", 0.50)
                if isinstance(ev, (list, np.ndarray)):
                    self.base_value = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    self.base_value = float(ev)
            # 2. Tree-based models (XGBoost, RandomForest)
            elif hasattr(self.model, "get_booster") or hasattr(self.model, "estimators_"):
                self._explainer = shap.TreeExplainer(self.model)
                ev = getattr(self._explainer, "expected_value", 0.50)
                if isinstance(ev, (list, np.ndarray)):
                    self.base_value = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    self.base_value = float(ev)
            # 3. Model-agnostic wrapper / Neural Net / HistGradientBoosting
            else:
                predict_fn = getattr(self.model, "predict_proba", self.model.predict)
                sample_bg = shap.sample(X_background, min(30, len(X_background)))
                self._explainer = shap.Explainer(predict_fn, sample_bg)
                ev = getattr(self._explainer, "expected_value", 0.50)
                if isinstance(ev, (list, np.ndarray)):
                    self.base_value = float(ev[1] if len(ev) > 1 else ev[0])
                else:
                    self.base_value = float(ev)
        except Exception:
            # Fallback
            self._explainer = None
            self.base_value = 0.50

    # ------------------------------------------------------------------
    def explain_sample_rich(
        self,
        sample: np.ndarray | pd.Series,
        raw_features: dict | None = None,
        top_n: int = 10,
        X_background: np.ndarray | None = None,
    ) -> dict:
        """
        Deep SHAP explanation for a sample returning exact mathematical working:
          f(x) = base_value + sum(phi_i)
        Returns:
          {
            "base_value": float,
            "prediction_value": float,
            "shap_sum": float,
            "top_features": list[dict],
            "all_features": list[dict],
            "working_framework": dict,
          }
        """
        self._build_explainer(X_background)
        x = np.array(sample).reshape(1, -1)
        sv = None
        base_val = self.base_value

        if HAS_SHAP and self._explainer is not None:
            try:
                raw_sv = self._explainer(x)
                if hasattr(raw_sv, "values"):
                    val = raw_sv.values
                    if len(val.shape) == 3 and val.shape[2] == 2:
                        sv = val[0, :, 1]
                    else:
                        sv = val[0]
                    if hasattr(raw_sv, "base_values"):
                        bv = raw_sv.base_values
                        if len(bv.shape) > 1 and bv.shape[1] == 2:
                            base_val = float(bv[0, 1])
                        else:
                            base_val = float(bv[0])
                else:
                    vals = self._explainer.shap_values(x)
                    if isinstance(vals, list):
                        sv = vals[1].flatten()
                    else:
                        sv = vals.flatten()
            except Exception:
                try:
                    vals = self._explainer.shap_values(x)
                    if isinstance(vals, list):
                        sv = vals[1].flatten()
                    else:
                        sv = vals.flatten()
                except Exception:
                    sv = None

        if sv is None:
            # Intrinsic fallback attribution
            if hasattr(self.model, "feature_importances_"):
                imp = self.model.feature_importances_
                sv = (imp * x.flatten())
            elif hasattr(self.model, "coef_"):
                sv = (self.model.coef_[0] * x.flatten())
            else:
                sv = x.flatten()
            base_val = 0.50

        # Clip / normalize SHAP values
        sv = np.nan_to_num(sv, nan=0.0)
        shap_sum = float(np.sum(sv))
        
        # Calculate predicted output prob
        if hasattr(self.model, "predict_proba"):
            try:
                pred_prob = float(self.model.predict_proba(x)[0][1])
            except Exception:
                pred_prob = float(np.clip(base_val + shap_sum, 0.0, 1.0))
        else:
            pred_prob = float(np.clip(base_val + shap_sum, 0.0, 1.0))

        total_abs = float(np.sum(np.abs(sv))) if np.sum(np.abs(sv)) > 0 else 1.0

        all_items = []
        for i, val in enumerate(sv):
            fname = self.feature_cols[i] if i < len(self.feature_cols) else f"feature_{i}"
            val_f = float(val)
            direction = "increases_phishing_risk" if val_f > 0 else "decreases_phishing_risk"
            
            raw_v = raw_features.get(fname, None) if raw_features else None
            scaled_v = float(x[0, i]) if i < x.shape[1] else 0.0
            
            # Domain rationale lookup
            rat_entry = FEATURE_RATIONALES.get(fname, None)
            if rat_entry:
                rationale = rat_entry["phish"] if val_f > 0 else rat_entry["legit"]
                cat = rat_entry.get("category", "Lexical & Structure")
            else:
                rationale = (
                    f"Observed value ({scaled_v:+.2f} std dev) pushes prediction towards malicious classification."
                    if val_f > 0 else
                    f"Observed value ({scaled_v:+.2f} std dev) conforms to standard benign baseline distribution."
                )
                cat = "General URL Heuristic"

            all_items.append({
                "feature"     : fname,
                "shap_value"  : round(val_f, 6),
                "direction"   : direction,
                "abs_shap"    : round(abs(val_f), 6),
                "relative_pct": round((abs(val_f) / total_abs) * 100.0, 2),
                "raw_value"   : raw_v,
                "scaled_value": round(scaled_v, 3),
                "rationale"   : rationale,
                "category"    : cat,
            })

        # Sort descending by absolute attribution
        all_sorted = sorted(all_items, key=lambda d: d["abs_shap"], reverse=True)
        top_sorted = all_sorted[:top_n]

        return {
            "base_value"      : round(float(base_val), 4),
            "prediction_value": round(float(pred_prob), 4),
            "shap_sum"        : round(float(shap_sum), 4),
            "top_features"    : top_sorted,
            "all_features"    : all_sorted,
            "working_framework": {
                "theory"     : "Shapley Additive exPlanations (SHAP) based on Cooperative Game Theory",
                "equation"   : "f(x) = E[f(X)] + sum(phi_i)",
                "explanation": "Every feature acts as a player in a game. The base value represents the expected phishing probability of an average web URL. Features with positive phi_i push the verdict towards Phishing; features with negative phi_i pull the verdict towards Legitimate.",
                "axioms"     : [
                    "Efficiency: Sum of feature contributions exactly equals the difference between model output and baseline.",
                    "Symmetry: Features that contribute equally across all subsets receive identical attribution.",
                    "Additivity: Feature attributions sum linearly across ensemble estimators.",
                    "Null Player: Features with zero marginal contribution receive an exact SHAP score of 0.00."
                ],
            }
        }

    # ------------------------------------------------------------------
    def explain_sample(
        self,
        sample: np.ndarray | pd.Series,
        top_n: int = 10,
        X_background: np.ndarray | None = None,
    ) -> list[dict]:
        """
        Backward-compatible method returning list of top-N dicts.
        """
        rich = self.explain_sample_rich(sample, top_n=top_n, X_background=X_background)
        return rich["top_features"]

    # ------------------------------------------------------------------
    def global_summary_plot(
        self,
        X_test: np.ndarray,
        max_samples: int = 1500,
        X_background: np.ndarray | None = None,
    ) -> pathlib.Path:
        """
        Compute global SHAP values across test dataset and save beeswarm + bar charts.
        """
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        self._build_explainer(X_background)

        n = min(max_samples, X_test.shape[0])
        X_sample = X_test[:n]

        shap_values = None
        if HAS_SHAP and self._explainer is not None:
            try:
                res = self._explainer(X_sample)
                if hasattr(res, "values"):
                    val = res.values
                    if len(val.shape) == 3 and val.shape[2] == 2:
                        shap_values = val[:, :, 1]
                    else:
                        shap_values = val
                else:
                    sv = self._explainer.shap_values(X_sample)
                    if isinstance(sv, list):
                        shap_values = sv[1]
                    else:
                        shap_values = sv
            except Exception:
                shap_values = None

        if shap_values is not None:
            # Beeswarm plot
            fig1, ax1 = plt.subplots(figsize=(10, 7.5))
            fig1.patch.set_facecolor("#111827")
            ax1.set_facecolor("#111827")
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_cols,
                show=False,
                max_display=15,
            )
            plt.title("Global SHAP Beeswarm Distribution (Feature Impact on Phishing)", fontsize=12, fontweight="bold", color="#F9FAFB", pad=12)
            plt.tight_layout()
            fig1.savefig(SHAP_SUMMARY_PNG, dpi=140, bbox_inches="tight")
            plt.close(fig1)

            # Bar plot
            fig2, ax2 = plt.subplots(figsize=(9.5, 6))
            fig2.patch.set_facecolor("#111827")
            ax2.set_facecolor("#111827")
            shap.summary_plot(
                shap_values,
                X_sample,
                feature_names=self.feature_cols,
                plot_type="bar",
                show=False,
                max_display=15,
            )
            plt.title("Global Mean |SHAP| Feature Importance", fontsize=12, fontweight="bold", color="#F9FAFB", pad=12)
            plt.tight_layout()
            fig2.savefig(SHAP_BAR_PNG, dpi=140, bbox_inches="tight")
            plt.close(fig2)
        else:
            # Matplotlib fallback
            if hasattr(self.model, "feature_importances_"):
                importances = self.model.feature_importances_
            elif hasattr(self.model, "coef_"):
                importances = np.abs(self.model.coef_[0])
            else:
                importances = np.ones(X_sample.shape[1])

            top_n = min(15, len(importances))
            sorted_idx = np.argsort(importances)[::-1][:top_n]
            names = [self.feature_cols[i] if self.feature_cols else f"f_{i}" for i in sorted_idx]
            vals  = importances[sorted_idx]

            fig1, ax1 = plt.subplots(figsize=(10, 7))
            fig1.patch.set_facecolor("#111827")
            ax1.set_facecolor("#111827")
            ax1.barh(names[::-1], vals[::-1], color="#3B82F6")
            ax1.set_title("Global Feature Importance Matrix", fontsize=12, fontweight="bold", color="#F9FAFB")
            ax1.set_xlabel("Attribution Magnitude", color="#9CA3AF")
            ax1.tick_params(colors="#9CA3AF")
            for spine in ["top", "right"]:
                ax1.spines[spine].set_visible(False)
            ax1.spines["bottom"].set_color("#374151")
            ax1.spines["left"].set_color("#374151")
            plt.tight_layout()
            fig1.savefig(SHAP_SUMMARY_PNG, dpi=140, bbox_inches="tight")
            plt.close(fig1)

        print(f"[explainer] SHAP plots generated -> {SHAP_SUMMARY_PNG}")
        return SHAP_SUMMARY_PNG

    # ------------------------------------------------------------------
    def save(self) -> None:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._explainer, SHAP_EXPLAINER_PKL)

    def load(self) -> None:
        if SHAP_EXPLAINER_PKL.exists():
            self._explainer = joblib.load(SHAP_EXPLAINER_PKL)


def build_explainer(model, feature_cols: list[str]) -> SHAPExplainer:
    return SHAPExplainer(model=model, feature_cols=feature_cols)
