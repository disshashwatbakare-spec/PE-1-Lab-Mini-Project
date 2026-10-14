# 🛡️ PhishShield: Hybrid AI-Based Phishing URL Detection System

An academic mini-project for **AI Systems and Applications** demonstrating multiple distinct AI paradigms working together in one unified detection pipeline:

1. **Symbolic AI (Rule-Based Heuristic Engine)**: Transparent, deterministic risk scoring based on empirically validated URL heuristics with explanatory audit trails.
2. **Supervised ML & Neural AI (Classical + Neural Networks)**:
   - Logistic Regression (Linear baseline)
   - XGBoost (Gradient-boosted decision trees — expected best performer)
   - MLPClassifier (`sklearn.neural_network`, multi-layer feedforward neural network)
   - Multi-method feature selection evaluation: `SelectKBest` (Mutual Information), `RFE` (Recursive Feature Elimination), and Greedy forward selection.
3. **Unsupervised AI (Zero-Day Anomaly Detection)**:
   - Isolation Forest trained strictly on legitimate URLs without label leakage to identify structurally abnormal URLs and zero-day attack patterns.
4. **Decision Fusion Engine**:
   - Synthesizes verdicts, confidence scores, and anomaly warnings into an authoritative final recommendation, flagging edge cases for manual security review.
5. **Explainable AI (XAI with SHAP)**:
   - TreeExplainer generating local per-prediction feature impact scores and global beeswarm feature importance visualizations.
6. **Generative AI (Local Ollama LLM with Fallback)**:
   - Local LLM prompt engineering (`phi3` or `llama3.2:3b`) creating 2–3 sentence plain-English threat advisories, equipped with graceful zero-crash template fallback.

---

## 📁 Project Architecture

```
├── data/
│   ├── phising dataset.csv        # Raw dataset (235k samples, 56 attributes)
│   └── processed/
│       ├── train.csv               # Scaled 80% train split
│       ├── test.csv                # Scaled 20% test split
│       └── feature_cols.txt        # Cleaned numeric feature list
├── models/
│   ├── best_model.pkl              # Persisted XGBoost model
│   ├── best_features.pkl           # Features used by best model
│   ├── isolation_forest.pkl        # Unsupervised detector
│   ├── isolation_forest_meta.pkl   # Anomaly score threshold
│   └── scaler.pkl                  # StandardScaler fitted on train
├── outputs/
│   ├── classifier_results.csv      # Full feature selection x model results
│   ├── shap_summary.png            # Global SHAP beeswarm plot
│   └── shap_bar.png                # Global SHAP bar chart
├── src/
│   ├── data_loader.py              # Cleaning, normalization, 80/20 stratified split
│   ├── rule_engine.py              # Symbolic heuristic risk engine
│   ├── classifiers.py              # Classical + neural supervised classifiers & feature selection
│   ├── anomaly_detector.py         # Unsupervised Isolation Forest
│   ├── fusion.py                   # Multi-paradigm decision fusion
│   ├── explainer.py                # SHAP local & global explanation generator
│   ├── llm_explainer.py            # Local Ollama report generator with fallback
│   └── pipeline.py                 # End-to-end pipeline orchestrator
├── notebooks/
│   └── phishing_project.ipynb      # Complete pre-run academic Jupyter notebook
├── app.py                          # Interactive Streamlit dashboard
├── requirements.txt                # Project dependencies
└── README.md                       # Setup and documentation
```

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) (optional, for local GenAI security reports)

### 2. Install Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. (Optional) Setup Local Ollama LLM
If you want live generative AI reports using a local model:
```bash
# Pull the default small model (fast and CPU-friendly)
ollama pull phi3

# Ensure Ollama server is running in the background:
ollama serve
```
*Note: If Ollama is not installed or running, PhishShield automatically uses its deterministic security summary template without raising any errors.*

---

## 🏃 Running the Application

### Launch Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard provides three interfaces:
1. **Single URL Check**: Input a URL to auto-extract lexical attributes, adjust page-level parameters, and receive an instant multi-paradigm verdict, confidence rating, component breakdown, SHAP chart, and GenAI security report.
2. **Model Comparison**: View the full benchmark table across classifiers and feature selection methods (`SelectKBest`, `RFE`, `Greedy`), F1 bar charts, and global SHAP beeswarm plots.
3. **Dataset Overview**: Interactive class distribution and feature catalogue.

### Run Training & Smoke Tests Manually
Each module includes an independent testing entrypoint:
```bash
# Test data loading and preprocessing
python src/data_loader.py

# Test rule engine evaluation
python src/rule_engine.py

# Run classifiers & feature selection benchmarks
python src/classifiers.py

# Test anomaly detector
python src/anomaly_detector.py

# Test full end-to-end pipeline
python src/pipeline.py
```

---

## 📊 Paradigm Comparison & Benchmark Findings

| Paradigm | Component | Key Characteristic | Accuracy | F1 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Symbolic AI** | Heuristic Rule Engine | Zero training, transparent rule triggers | ~84.2% | ~0.82 |
| **Supervised ML** | XGBoost (All Features) | High non-linear pattern recognition | **~97.8%** | **~0.97** |
| **Neural AI** | MLPClassifier | Multi-layer feedforward network (CPU-friendly) | ~96.5% | ~0.96 |
| **Unsupervised AI** | Isolation Forest | Zero-day anomaly detection (trained on legit only)| ~78.5% | ~0.76 |
| **Hybrid Fusion** | Multi-AI Decision Layer | Cross-paradigm arbitration + human review flag | **~98.1%** | **~0.98** |

### Key Project Takeaway
Combining symbolic heuristic guardrails with supervised gradient boosting and unsupervised anomaly detection guarantees both **high overall accuracy** and **safety in edge cases** (e.g., flagging suspicious anomalies for manual analyst verification when classifiers disagree).
