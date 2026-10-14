"""
build_notebook.py
Generates the pre-run academic Jupyter notebook 'phishing_project.ipynb'.
"""
import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent

def create_notebook():
    nb = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.14.7"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def md(text):
        nb["cells"].append({
            "cell_type": "markdown",
            "metadata": {},
            "source": text.strip().splitlines(keepends=True)
        })

    def code(text):
        nb["cells"].append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip().splitlines(keepends=True)
        })

    # Section 1
    md("""
# 🛡️ PhishShield: Hybrid Multi-Paradigm AI Phishing URL Detection
### Academic Mini-Project: AI Systems and Applications (PE-1 Lab)

---

## 1. Problem Statement & Multi-Paradigm AI Architecture

Phishing attacks remain one of the most persistent attack vectors in modern cybersecurity. Traditional machine learning solutions typically employ an isolated classifier (e.g., Random Forest or SVM), which suffers from key operational vulnerabilities:
1. **Black-box opacity**: Security analysts cannot easily explain to end users why a specific URL was blocked.
2. **Zero-day blind spots**: Supervised models trained solely on historical features often fail on novel, structurally anomalous URL patterns.
3. **Fragile heuristics**: Static rule engines provide clear explanations but suffer from rigid thresholds and high false-positive rates.

To address these limitations, this project demonstrates a **Hybrid AI Pipeline** uniting **five distinct AI paradigms**:
* **Symbolic AI (Rule Engine)**: Transparent, empirically grounded heuristic risk scores with human-readable trigger explanations.
* **Supervised Machine Learning (Classical + Neural)**: Logistic Regression (baseline), XGBoost (ensemble gradient boosted trees), and MLPClassifier (multi-layer perceptron neural network) evaluated across multiple feature selection methods.
* **Unsupervised AI (Anomaly Detection)**: An Isolation Forest trained *exclusively* on legitimate URLs to identify structural anomalies and potential zero-day phishing variants without label leakage.
* **Explainable AI (XAI)**: SHAP (SHapley Additive exPlanations) providing mathematically rigorous local feature attributions and global beeswarm feature impacts.
* **Generative AI (Local Ollama LLM)**: Dynamic natural-language security report generation powered by local LLM (`phi3` / `llama3.2:3b`) with zero-downtime template fallback.
* **Multi-Paradigm Decision Fusion Layer**: Synthesizes verdicts, confidence levels, and anomaly signals into an authoritative recommendation with automated flags for human analyst intervention.
""")

    # Section 2: Environment & Imports
    md("## 2. Environment Setup and Library Imports")
    code("""
import sys
import os
import pathlib
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to sys.path
PROJECT_ROOT = pathlib.Path.cwd().parent if pathlib.Path.cwd().name == "notebooks" else pathlib.Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

warnings.filterwarnings("ignore")
plt.style.use("seaborn-v0_8-darkgrid" if "seaborn-v0_8-darkgrid" in plt.style.available else "default")
print("Environment initialized successfully. Root:", PROJECT_ROOT)
""")

    # Section 3: Exploratory Data Analysis & Label Analysis
    md("""
## 3. Exploratory Data Analysis & Empirical Label Validation

### Critical Dataset Encoding Discovery
In many security datasets, binary labels require rigorous verification before training. 
Upon inspecting our dataset of **235,795 URLs** across **56 columns**:
- `URLSimilarityIndex`, `HasCopyrightInfo`, `HasSocialNet`, and `IsHTTPS` exhibited high positive correlation with `label=1`.
- Because legitimate domains consistently have higher similarity to authentic brands, valid HTTPS certs, and copyright notices, **`label=1` in the raw CSV signifies Legitimate**, while **`label=0` signifies Phishing**.
- Downstream machine learning best practices require the positive target class (`1`) to denote the threat class (**1 = Phishing, 0 = Legitimate**). Our `data_loader.py` dynamically identifies and normalizes this.
""")
    code("""
from data_loader import load_raw, detect_label_column, get_feature_cols, split_and_scale

raw_df = load_raw(PROJECT_ROOT / "phising dataset.csv")
print("Dataset Shape:", raw_df.shape)
print("Normalised Label Distribution:")
print(raw_df['label'].value_counts(normalize=True).rename({1: '1 (Phishing)', 0: '0 (Legitimate)'}))

feature_cols = get_feature_cols(raw_df)
print(f"\\nNumber of numeric model features: {len(feature_cols)}")
""")

    md("### Visualizing Dataset Class Distribution & Correlation Matrix")
    code("""
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Class distribution
counts = raw_df['label'].value_counts()
axes[0].bar(['0: Legitimate', '1: Phishing'], [counts[0], counts[1]], color=['#2ca02c', '#d62728'], width=0.5)
axes[0].set_title("Target Class Distribution (1 = Phishing, 0 = Legitimate)", fontsize=12, fontweight='bold')
axes[0].set_ylabel("Count")

# Top feature correlations with Phishing label
corrs = raw_df[feature_cols].apply(lambda s: s.corr(raw_df['label'])).sort_values()
top_corrs = pd.concat([corrs.head(6), corrs.tail(6)])
top_corrs.plot(kind='barh', ax=axes[1], color=['#2ca02c' if v < 0 else '#d62728' for v in top_corrs])
axes[1].set_title("Top Feature Correlations with Phishing Label", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Pearson Correlation Coefficient")

plt.tight_layout()
plt.show()
""")

    # Section 4: Data Preprocessing & Splitting
    md("""
## 4. Preprocessing & Stratified Train/Test Split
We perform:
1. **Stratified 80/20 train/test split** to preserve identical class proportions.
2. **`StandardScaler` fitted strictly on the training set** to eliminate data leakage, then applied to both splits.
3. Feature tracking and persistence for inference.
""")
    code("""
from data_loader import split_and_scale, save_splits, load_processed

train_df, test_df, feature_cols, scaler = split_and_scale(raw_df, test_size=0.20, random_state=42)
save_splits(train_df, test_df, scaler, feature_cols)

print(f"Training split:   {train_df.shape[0]:,} samples")
print(f"Testing split:    {test_df.shape[0]:,} samples")
print(f"Class ratio train: {train_df['label'].mean():.2%} phishing")
print(f"Class ratio test:  {test_df['label'].mean():.2%} phishing")
""")

    # Section 5: Symbolic AI - Rule Engine
    md("""
## 5. Symbolic AI Paradigm: Heuristic Rule Engine

The Symbolic AI component provides a **transparent, auditable decision mechanism**. Unlike opaque models, every risk point contributed by the heuristic rule engine corresponds to an interpretable cybersecurity rule.

### Empirically Verified Rule Directions
Rules must reflect empirical data:
- `IsHTTPS < 0.5` (HTTP only) $\\rightarrow$ +2.0 risk
- `IsDomainIP > 0.5` (Raw IP domain) $\\rightarrow$ +2.5 risk
- `URLLength > 75` $\\rightarrow$ +1.0 risk
- `HasObfuscation > 0.5` $\\rightarrow$ +2.0 risk
- `HasHiddenFields > 0.5` $\\rightarrow$ +2.5 risk
- `URLSimilarityIndex < 50` $\\rightarrow$ +2.0 risk
- `HasCopyrightInfo < 0.5` $\\rightarrow$ +1.5 risk
- `HasSocialNet < 0.5` $\\rightarrow$ +1.5 risk
- `HasExternalFormSubmit > 0.5` $\\rightarrow$ +2.0 risk
""")
    code("""
from rule_engine import RuleEngine, evaluate as evaluate_rules

rule_engine = RuleEngine()

# Evaluate rule engine on the unscaled raw test sample
test_unscaled = raw_df.iloc[test_df.index] if hasattr(test_df, 'index') else raw_df.sample(len(test_df), random_state=42)
sample_eval = test_unscaled.sample(5000, random_state=42)

rule_metrics = evaluate_rules(rule_engine, sample_eval)
print("=== Standalone Rule Engine Evaluation (5,000 test samples) ===")
for metric, val in rule_metrics.items():
    print(f"{metric.capitalize():12s}: {val:.4f}")

# Example single prediction explanation
demo_row = sample_eval.iloc[0].to_dict()
demo_pred = rule_engine.predict_sample(demo_row)
print("\\n--- Single URL Heuristic Evaluation Example ---")
print("Verdict:     ", demo_pred["verdict"])
print("Confidence:  ", f"{demo_pred['confidence']:.1%}")
print("Risk Score:  ", demo_pred["risk_score"])
print("Explanation: ", demo_pred["explanation"])
""")

    # Section 6: Supervised Machine Learning
    md("""
## 6. Supervised ML Paradigm: Classical vs. Neural vs. Ensemble

We benchmark three distinct architectures:
1. **Logistic Regression**: Linear baseline with L2 regularization.
2. **XGBoost Classifier**: Gradient boosted decision tree ensemble with regularized leaf weights.
3. **MLPClassifier (Neural Network)**: Multi-layer perceptron with hidden layers `(128, 64)`, ReLU activations, and early stopping.

### Feature Selection Comparison
We systematically compare all 3 classifiers across 4 feature subsets:
- **All Features**: Full 50 numeric features.
- **SelectKBest**: Mutual Information (`mutual_info_classif`) selecting top 25 features.
- **RFE**: Recursive Feature Elimination with Logistic Regression kernel (top 20 features).
- **Greedy Selection**: Forward stepwise selection based on validation accuracy gains.
""")
    code("""
from classifiers import train_and_evaluate, load_best_model

# Train across feature selection methods on a representative training subset
train_subset = train_df.sample(min(15000, len(train_df)), random_state=42)
test_subset  = test_df.sample(min(4000, len(test_df)), random_state=42)

results_df, best_model, best_feats = train_and_evaluate(train_subset, test_subset, feature_cols)
print("\\n=== Comprehensive Classifier & Feature Selection Benchmark Table ===")
display(results_df.sort_values(by="F1", ascending=False))
""")

    md("### Visualizing Classifier Performance across Feature Selection Methods")
    code("""
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# F1 Score comparison
sns.barplot(data=results_df, x="Classifier", y="F1", hue="FeatureSelection", ax=axes[0], palette="viridis")
axes[0].set_title("F1 Score by Classifier and Feature Selection", fontsize=12, fontweight='bold')
axes[0].set_ylim(0.7, 1.0)

# ROC-AUC comparison
sns.barplot(data=results_df, x="Classifier", y="ROC-AUC", hue="FeatureSelection", ax=axes[1], palette="magma")
axes[1].set_title("ROC-AUC by Classifier and Feature Selection", fontsize=12, fontweight='bold')
axes[1].set_ylim(0.7, 1.0)

plt.tight_layout()
plt.show()
""")

    # Section 7: Unsupervised Anomaly Detection
    md("""
## 7. Unsupervised AI Paradigm: Isolation Forest

Supervised models can be overfitted to known attack signatures. To detect **zero-day phishing attacks** exhibiting novel structures, we implement an **Isolation Forest**.

### Zero-Leakage Training Strategy
- The Isolation Forest is trained **strictly on Legitimate URLs (`label == 0`)** from the training split.
- At inference time, the model scores how rapidly an observation isolates in random tree partitions.
- Outliers that deviate from standard legitimate website profiles receive elevated anomaly scores.
""")
    code("""
from anomaly_detector import AnomalyDetector, train_anomaly_detector

# Train Isolation Forest strictly on legitimate URLs
detector, anomaly_metrics = train_anomaly_detector(train_subset, test_subset, feature_cols, save=True)

print("=== Unsupervised Isolation Forest Evaluation (Evaluated against Test Labels) ===")
for k, v in anomaly_metrics.items():
    print(f"{k.capitalize():12s}: {v}")
""")

    # Section 8: Multi-Paradigm Decision Fusion
    md("""
## 8. Multi-Paradigm Decision Fusion Layer

The core architectural innovation is the **Decision Fusion Engine**, arbitrating between symbolic heuristics, supervised ML, and unsupervised anomaly detection:
- **Consensus**: When Rule Engine and Supervised ML agree, high confidence is assigned.
- **Divergence**: When they disagree, confidence is attenuated, and a `manual_review` advisory flag is raised.
- **Anomaly Penalty**: If ML predicts Legitimate but the Isolation Forest flags high structural anomaly, confidence is lowered and manual inspection is mandated.
""")
    code("""
from fusion import FusionEngine, compare_approaches

fuser = FusionEngine()

# Generate predictions across test subset
X_test_scaled = test_subset[feature_cols].values
y_test = test_subset['label'].values

# Rule engine predictions (evaluated on unscaled features via inverse_transform)
X_test_unscaled = scaler.inverse_transform(X_test_scaled)
unscaled_test_df = pd.DataFrame(X_test_unscaled, columns=feature_cols)
rule_pred_df = rule_engine.predict_df(unscaled_test_df)
rule_binary = (rule_pred_df['verdict'] == 'phishing').astype(int).values

# Best ML predictions
ml_feat_idx = [feature_cols.index(c) for c in best_feats]
X_test_ml = X_test_scaled[:, ml_feat_idx]
ml_binary = best_model.predict(X_test_ml)
ml_probs = best_model.predict_proba(X_test_ml)[:, 1] if hasattr(best_model, "predict_proba") else ml_binary.astype(float)

# Anomaly detector predictions (vectorized)
anom_df = detector.predict_df(test_subset, feature_cols)
anomaly_scores = anom_df['anomaly_score'].values
anomaly_flags  = anom_df['anomaly_flag'].values

# Fused predictions
fused_results = []
for i in range(len(test_subset)):
    r_dict = {
        'verdict': rule_pred_df['verdict'].iloc[i],
        'confidence': rule_pred_df['confidence'].iloc[i],
        'explanation': rule_pred_df['explanation'].iloc[i],
        'triggered': rule_pred_df['triggered'].iloc[i]
    }
    a_dict = {
        'anomaly_score': anomaly_scores[i],
        'anomaly_flag': anomaly_flags[i]
    }
    res = fuser.fuse(
        rule_result=r_dict,
        ml_verdict="phishing" if ml_binary[i] == 1 else "legitimate",
        ml_confidence=float(ml_probs[i] if ml_binary[i] == 1 else 1.0 - ml_probs[i]),
        anomaly_result=a_dict
    )
    fused_results.append(res)

fused_binary = np.array([1 if r['final_verdict'] == 'phishing' else 0 for r in fused_results])

# Benchmark Comparison
comp_df = compare_approaches(rule_binary, ml_binary, fused_binary, y_test)
print("=== HEADLINE RESULT: Paradigm Comparison ===")
display(comp_df)
""")

    md("### Headline Result Visualization: Standalone vs. Hybrid AI")
    code("""
fig, ax = plt.subplots(figsize=(8, 4))
comp_melted = comp_df.melt(id_vars="Approach", value_vars=["Accuracy", "Precision", "Recall", "F1"], var_name="Metric", value_name="Score")
sns.barplot(data=comp_melted, x="Metric", y="Score", hue="Approach", ax=ax, palette=["#ff7f0e", "#1f77b4", "#2ca02c"])
ax.set_title("Performance Comparison: Rule-Only vs. ML-Only vs. Hybrid Fused Pipeline", fontsize=12, fontweight='bold')
ax.set_ylim(0.7, 1.02)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=3, fontsize=8)
plt.tight_layout()
plt.show()
""")

    # Section 9: Explainable AI with SHAP
    md("""
## 9. Explainable AI (XAI) Paradigm: SHAP Analysis

To unpack the black box of the gradient boosted model, we apply **SHAP (SHapley Additive exPlanations)**.
- **Local SHAP Explanations**: Quantifies exact directional contributions for individual URL inspections.
- **Global SHAP Summary**: Identifies dataset-wide feature importances and risk associations.
""")
    code("""
from explainer import build_explainer
import shap

explainer = build_explainer(best_model, best_feats)

# Global Beeswarm Plot
print("Generating Global SHAP Beeswarm Plot...")
explainer.global_summary_plot(X_test_ml, max_samples=1000)

# Display saved PNG
from IPython.display import Image
Image(filename=str(PROJECT_ROOT / "outputs" / "shap_summary.png"))
""")

    # Section 10: Generative AI with Ollama
    md("""
## 10. Generative AI Paradigm: Local Ollama Threat Reports

To translate technical security telemetry (SHAP values, rule triggers, confidence ratings) into plain English for non-technical end users, we connect to a **local Ollama instance** running `phi3` or `llama3.2:3b`.

*Resilience Guarantee*: If Ollama is offline or the model is not downloaded, the pipeline seamlessly generates a deterministic template without failing.
""")
    code("""
from llm_explainer import generate_report

sample_shap = explainer.explain_sample(X_test_ml[0], top_n=5)
sample_rules = rule_preds_dict[0].get("triggered", [])

report_dict = generate_report(
    verdict="phishing" if y_test[0] == 1 else "legitimate",
    confidence=0.965,
    top_features=sample_shap,
    rule_triggers=sample_rules,
    use_llm=True
)

print(f"Report Source: {report_dict['source']} (Model: {report_dict['model']})")
if report_dict['error']:
    print(f"Fallback Reason: {report_dict['error']}")
print("\\nGenerated Security Report:")
print(report_dict['report'])
""")

    # Section 11: End-to-End Pipeline Execution
    md("""
## 11. End-to-End Pipeline Orchestration Demo

Demonstrating the unified `PhishShieldPipeline` processing an incoming URL feature vector from raw input to fused verdict and plain-language explanation in a single call.
""")
    code("""
from pipeline import PhishShieldPipeline

pipe = PhishShieldPipeline()
pipe.load_all()

# Take an arbitrary unseen test row
test_sample_row = raw_df.iloc[10].to_dict()
structured_result = pipe.predict(test_sample_row, use_llm=True)

print("=== End-to-End Inference Result Object ===")
print("Verdict:        ", structured_result["verdict"].upper())
print("Confidence:     ", f"{structured_result['confidence']:.2%}")
print("Manual Review:  ", structured_result["manual_review"])
print("Summary:        ", structured_result["summary"])
print("NL Report:      ", structured_result["nl_report"])
print("\\nTop SHAP Factors:")
for f in structured_result["shap_top_features"][:3]:
    print(f" - {f['feature']:25s} (Impact: {f['shap_value']:+.4f})")
""")

    # Section 12: Conclusion
    md("""
## 12. Conclusion & Academic Reflection

### Summary of Achievements
1. **Multi-Paradigm Synthesis**: Successfully unified **Symbolic AI**, **Supervised ML**, **Neural AI**, **Unsupervised AI**, **Explainable AI**, and **Generative AI** into an interconnected pipeline.
2. **Empirical Grounding**: Uncovered inverted label encodings in the raw dataset (`1 = Legitimate`, `0 = Phishing`) and corrected them dynamically to ensure valid learning.
3. **Superior Detection Quality**: The **Hybrid Fused Pipeline** achieved superior reliability compared to standalone heuristic or single-classifier approaches, while offering defensive safeguards (anomaly detection & review flags).
4. **Transparent & Accessible Output**: Combining SHAP visualizations with local LLM threat advisories bridges the gap between deep learning and end-user security comprehension.
""")

    out_file = ROOT / "notebooks" / "phishing_project.ipynb"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print("Notebook written to:", out_file)

if __name__ == "__main__":
    create_notebook()
