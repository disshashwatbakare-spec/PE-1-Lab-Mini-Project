"""
generate_documentation.py
=========================
Generates a comprehensive, highly detailed professional Microsoft Word (.docx)
document for the PhishShield project:
  - Project Overview & Architecture
  - Technology Stack
  - All 5 AI Paradigms & Algorithms (Symbolic, Supervised, Unsupervised, SHAP XAI, Ollama GenAI)
  - Decision Fusion Engine & Weighted Consensus
  - Complete Tab-by-Tab Breakdown of the Streamlit SaaS Dashboard & Displayed Contents
  - REST API Backend Integration
  - Deployment & Execution Guide
"""

import os
import pathlib
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUTPUT_FILE = "PhishShield_Project_Documentation.docx"

def set_cell_background(cell, hex_color):
    """Set background color of a table cell in docx."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tc_pr.append(shd)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    """Set padding for a cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    tc_mar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tc_mar.append(node)
    tc_pr.append(tc_mar)

def create_styled_document():
    doc = docx.Document()

    # Page Margins (1 inch all around)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Palette Constants
    COLOR_PRIMARY = RGBColor(99, 102, 241)     # Indigo/Violet #6366F1
    COLOR_DARK = RGBColor(15, 23, 42)          # Deep Slate #0F172A
    COLOR_CYAN = RGBColor(6, 182, 212)         # Cyan Teal #06B6D4
    COLOR_MUTED = RGBColor(100, 116, 139)      # Muted Slate #64748B
    HEX_HEADER_BG = "0F172A"                   # Table Header Dark
    HEX_ZEBRA_BG = "F8FAFC"                    # Light tint row

    # -------------------------------------------------------------------------
    # DOCUMENT COVER / TITLE BLOCK
    # -------------------------------------------------------------------------
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(20)
    title_p.paragraph_format.space_after = Pt(4)
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_title = title_p.add_run("PhishShield Enterprise AI")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(28)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_PRIMARY

    sub_p = doc.add_paragraph()
    sub_p.paragraph_format.space_after = Pt(16)
    sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_sub = sub_p.add_run("Autonomous Hybrid URL Threat Intelligence Engine\nMulti-Paradigm AI Detection, Explainability (XAI) & Generative SOC Reporting")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(13)
    run_sub.font.italic = True
    run_sub.font.color.rgb = COLOR_DARK

    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(30)
    meta_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_meta = meta_p.add_run("Course: AI Systems & Applications (PE-1 Lab) | Semester V | MIT-WPU\nDocument Version: 1.2 Enterprise Edition | Academic Year: 2024–2026")
    run_meta.font.name = "Arial"
    run_meta.font.size = Pt(9.5)
    run_meta.font.color.rgb = COLOR_MUTED

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # -------------------------------------------------------------------------
    # SECTION 1: EXECUTIVE SUMMARY & PROJECT OVERVIEW
    # -------------------------------------------------------------------------
    h1 = doc.add_heading("1. Executive Summary & Project Overview", level=1)
    h1.paragraph_format.space_before = Pt(16)
    h1.paragraph_format.space_after = Pt(8)

    p = doc.add_paragraph(
        "PhishShield is an autonomous, production-grade cybersecurity intelligence platform developed to detect, classify, "
        "and explain malicious phishing URLs in real time. Modern phishing threats evolve rapidly through domain generation algorithms (DGAs), "
        "typosquatting, deceptive subdomains, and zero-day evasion techniques that easily bypass static blacklists and traditional single-classifier "
        "machine learning models. To overcome these critical vulnerabilities, PhishShield implements a Hybrid Multi-Paradigm AI Architecture that "
        "harmonizes five distinct computational intelligence methodologies into a unified, high-assurance decision pipeline."
    )
    p.paragraph_format.line_spacing = 1.15
    p.paragraph_format.space_after = Pt(8)

    p2 = doc.add_paragraph(
        "Unlike academic prototypes that rely on manual feature inputs or single-dimensional classifiers, PhishShield operates with zero user burden: "
        "the user inputs a single raw URL string, from which 30 lexical, host, and behavioral security attributes are derived dynamically. The engine "
        "subsequently executes symbolic heuristic verification, supervised ensemble prediction, unsupervised isolation anomaly detection, SHAP (SHapley "
        "Additive exPlanations) attribution, and local on-device Generative AI incident reporting via Ollama—all completing in under one second with "
        "complete data privacy (zero cloud telemetry)."
    )
    p2.paragraph_format.line_spacing = 1.15
    p2.paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------------------
    # SECTION 2: END-TO-END TECHNOLOGY STACK
    # -------------------------------------------------------------------------
    h2 = doc.add_heading("2. End-to-End Technology Stack", level=1)
    h2.paragraph_format.space_before = Pt(16)
    h2.paragraph_format.space_after = Pt(8)

    p_tech = doc.add_paragraph(
        "The project is engineered using a robust modern Python data science and web architecture. The table below outlines the core layers, "
        "libraries, and frameworks powering the system:"
    )
    p_tech.paragraph_format.space_after = Pt(8)

    table_tech = doc.add_table(rows=1, cols=3)
    table_tech.alignment = WD_TABLE_ALIGNMENT.CENTER
    table_tech.autofit = False

    headers_tech = ["System Layer", "Technologies & Libraries", "Architectural Role & Functionality"]
    hdr_cells = table_tech.rows[0].cells
    for i, title in enumerate(headers_tech):
        hdr_cells[i].text = title
        set_cell_background(hdr_cells[i], HEX_HEADER_BG)
        set_cell_margins(hdr_cells[i], top=120, bottom=120, left=150, right=150)
        p = hdr_cells[i].paragraphs[0]
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = RGBColor(255, 255, 255)
        p.runs[0].font.size = Pt(9.5)

    tech_data = [
        ("Core Language & Runtime", "Python 3.10 / 3.11 / 3.14", "Execution runtime for feature extraction, ML pipelines, and API services."),
        ("Supervised Machine Learning", "Scikit-Learn, XGBoost, Joblib", "XGBoost, Multi-Layer Perceptron (Neural Net), Logistic Regression, and model serialization."),
        ("Unsupervised Anomaly Detection", "Scikit-Learn (IsolationForest)", "Detects out-of-distribution structural outliers without relying on supervised labels."),
        ("Explainable AI (XAI)", "SHAP (SHapley Additive exPlanations)", "KernelExplainer & TreeExplainer computing exact Shapley feature impact values."),
        ("Local Generative AI", "Ollama, LLaMA-3 / Phi-3 / Mistral", "On-device local Large Language Model generating natural-language incident briefings via HTTP REST."),
        ("Feature Engineering", "urllib.parse, Regex, NumPy, Pandas", "Dynamic parsing of raw URLs into 30 structural, host, and content-mimicking attributes."),
        ("Web Presentation Dashboard", "Streamlit, Vanilla CSS, Matplotlib, Seaborn", "Commercial SaaS dark-mode frontend with glassmorphism, responsive charts, and sidebar navigation."),
        ("REST API Backend", "FastAPI, Uvicorn, Pydantic", "Headless REST microservice enabling programmatic URL scanning and enterprise SIEM integration."),
        ("Data Preprocessing & Scaling", "StandardScaler, SelectKBest, Mutual Info", "Normalization of numeric features fit strictly on the training partition to prevent data leakage.")
    ]

    for row_idx, (layer, tech, role) in enumerate(tech_data):
        row = table_tech.add_row()
        cells = row.cells
        cells[0].text = layer
        cells[1].text = tech
        cells[2].text = role
        bg_col = HEX_ZEBRA_BG if row_idx % 2 == 1 else "FFFFFF"
        for c in cells:
            set_cell_background(c, bg_col)
            set_cell_margins(c, top=80, bottom=80, left=120, right=120)
            p = c.paragraphs[0]
            p.runs[0].font.size = Pt(8.5)
            p.runs[0].font.name = "Arial"

    doc.add_paragraph().paragraph_format.space_after = Pt(10)

    # -------------------------------------------------------------------------
    # SECTION 3: THE FIVE DISTINCT AI PARADIGMS & ALGORITHMS
    # -------------------------------------------------------------------------
    h3 = doc.add_heading("3. Multi-Paradigm AI Architecture & Algorithms", level=1)
    h3.paragraph_format.space_before = Pt(16)
    h3.paragraph_format.space_after = Pt(8)

    p_ai_intro = doc.add_paragraph(
        "A central requirement of the AI Systems & Applications course is demonstrating multiple distinct artificial intelligence paradigms "
        "cooperating within a cohesive pipeline rather than relying on a standalone classifier. PhishShield integrates five distinct paradigms:"
    )
    p_ai_intro.paragraph_format.space_after = Pt(8)

    # 3.1 Symbolic AI
    h3_1 = doc.add_heading("3.1 Paradigm 1: Symbolic AI (Empirically Weighted Rule Engine)", level=2)
    h3_1.paragraph_format.space_before = Pt(10)
    h3_1.paragraph_format.space_after = Pt(4)
    p_sym = doc.add_paragraph(
        "The Symbolic AI layer (implemented in src/rule_engine.py) models domain expertise using 15 transparent, deterministic heuristic rules. "
        "Unlike black-box models, this layer produces human-verifiable logic. All rule directions and weights were empirically calibrated against "
        "dataset correlation coefficients to prevent counter-intuitive assumptions.\n"
        "Key Heuristics Include:\n"
        "• Lack of HTTPS (IsHTTPS == 0): Strong positive correlation with phishing risk (Weight: +0.65).\n"
        "• IP-Based Domain (IsDomainIP == 1): Detects direct IPv4 hostnames used to bypass DNS domain reputation filters (Weight: +0.80).\n"
        "• Hex Obfuscation & Special Characters (HasObfuscation == 1, '@' character present): Exploited to deceive browser URL bars (Weight: +0.70).\n"
        "• Hidden Form Fields (HasHiddenFields == 1): Identifies clandestine credential harvesting inputs in login forms (Weight: +0.60).\n"
        "• Brand Spoofing & Low Similarity Index (URLSimilarityIndex < 50): Detects typosquatting targeting PayPal, Amazon, Google, etc. (Weight: +0.85).\n"
        "• Form Redirection & Shorteners (NoOfURLRedirect > 0): Detects multi-hop evasion gateways (Weight: +0.55).\n"
        "When aggregate weighted rule score exceeds the empirical threshold, the rule engine flags the sample and lists all triggered violations."
    )
    p_sym.paragraph_format.line_spacing = 1.15
    p_sym.paragraph_format.space_after = Pt(10)

    # 3.2 Supervised ML
    h3_2 = doc.add_heading("3.2 Paradigm 2: Supervised Machine Learning (Ensemble Classifiers)", level=2)
    h3_2.paragraph_format.space_before = Pt(10)
    h3_2.paragraph_format.space_after = Pt(4)
    p_sup = doc.add_paragraph(
        "The supervised learning component (src/classifiers.py) benchmarks three distinct model families across three feature selection regimes "
        "(9 total configurations):\n"
        "1. XGBoost (Extreme Gradient Boosting): Non-linear decision tree ensemble utilizing regularized gradient objective functions for rapid, high-accuracy inference.\n"
        "2. Multi-Layer Perceptron (MLP Neural Network): Feed-forward deep architecture with hidden ReLU layers and backpropagation to capture complex feature interactions.\n"
        "3. Logistic Regression (L2 Regularized): Generalized linear baseline providing probabilistic log-odds calibration.\n\n"
        "Feature Selection Strategies Evaluated:\n"
        "• Full 30 Attributes: Evaluates all engineered features.\n"
        "• SelectKBest via Mutual Information: Selects top 15 features maximizing mutual information (I(X; Y)) with the phishing label.\n"
        "• SelectKBest via Chi-Square (χ²): Selects top 15 features with highest statistical dependence on class distribution.\n\n"
        "The model achieving the highest held-out F1-Score is automatically serialized as models/best_model.pkl and deployed into production."
    )
    p_sup.paragraph_format.line_spacing = 1.15
    p_sup.paragraph_format.space_after = Pt(10)

    # 3.3 Unsupervised ML
    h3_3 = doc.add_heading("3.3 Paradigm 3: Unsupervised Machine Learning (Isolation Forest Anomaly Detection)", level=2)
    h3_3.paragraph_format.space_before = Pt(10)
    h3_3.paragraph_format.space_after = Pt(4)
    p_unsup = doc.add_paragraph(
        "Supervised models fail when confronting zero-day phishing attacks whose structural patterns do not resemble previously labeled samples. "
        "To mitigate this, PhishShield implements an unsupervised Isolation Forest (src/anomaly_detector.py) trained exclusively on legitimate/benign URL vectors.\n"
        "• Mathematical Principle: Isolation Forest isolates structural anomalies by randomly partitioning feature space. Because anomalous instances require "
        "fewer recursive splits to isolate, their average tree path length h(x) is significantly shorter than normal clusters.\n"
        "• Thresholding: Samples yielding an anomaly score s(x) > 0.50 are flagged as structural outliers. If the supervised classifier marks a URL as legitimate "
        "but the Isolation Forest flags it as an outlier, the system triggers a manual review alert."
    )
    p_unsup.paragraph_format.line_spacing = 1.15
    p_unsup.paragraph_format.space_after = Pt(10)

    # 3.4 Explainable AI (SHAP)
    h3_4 = doc.add_heading("3.4 Paradigm 4: Explainable AI (SHAP Feature Attribution)", level=2)
    h3_4.paragraph_format.space_before = Pt(10)
    h3_4.paragraph_format.space_after = Pt(4)
    p_shap = doc.add_paragraph(
        "Security analysts cannot rely on opaque black-box predictions. The Explainable AI module (src/explainer.py) computes exact Shapley Additive "
        "exPlanations (SHAP) grounded in cooperative game theory:\n"
        "• Local Attribution: Quantifies the positive or negative contribution (ϕ_i) of every single URL feature for the current prediction. "
        "Features pushing the prediction toward phishing are rendered in crimson (+ϕ), while features supporting legitimate safety are rendered in emerald (-ϕ).\n"
        "• Global Attribution: Summarizes feature importance across the entire corpus via beeswarm distributions, exposing the dominant threat indicators globally."
    )
    p_shap.paragraph_format.line_spacing = 1.15
    p_shap.paragraph_format.space_after = Pt(10)

    # 3.5 Generative AI (Local Ollama LLM)
    h3_5 = doc.add_heading("3.5 Paradigm 5: Generative AI (Local On-Device Ollama LLM Reporting)", level=2)
    h3_5.paragraph_format.space_before = Pt(10)
    h3_5.paragraph_format.space_after = Pt(4)
    p_gen = doc.add_paragraph(
        "The generative AI layer (src/llm_explainer.py) synthesizes numerical ML probabilities, triggered heuristic rules, and top SHAP features "
        "into a structured, natural-language Security Operations Center (SOC) Incident Briefing.\n"
        "• 100% On-Device Privacy: Connects to a locally running Ollama instance (http://localhost:11434) hosting open-weight models (LLaMA-3, Phi-3, Mistral). "
        "No URL data or telemetry is transmitted to external cloud providers.\n"
        "• Dual Transport Architecture: Supports both the official ollama Python library and standard-library HTTP REST fallback, ensuring seamless operation.\n"
        "• Deterministic Resilience: If Ollama is offline or uninstalled, the system automatically falls back to an internal deterministic safety brief generator, "
        "guaranteeing zero application downtime.\n"
        "• Structured Output Schema: Generates three distinct sections: (1) Executive Threat Summary, (2) Detected Attack Vectors, and (3) Recommended Action Checklist."
    )
    p_gen.paragraph_format.line_spacing = 1.15
    p_gen.paragraph_format.space_after = Pt(10)

    # 3.6 Decision Fusion Engine
    h3_6 = doc.add_heading("3.6 Decision Fusion & Conflict Resolution Layer", level=2)
    h3_6.paragraph_format.space_before = Pt(10)
    h3_6.paragraph_format.space_after = Pt(4)
    p_fuse = doc.add_paragraph(
        "The Decision Fusion Engine (src/fusion.py) integrates outputs from the Rule Engine (Symbolic), Classifier (Supervised), and Anomaly Detector (Unsupervised):\n"
        "1. Full Consensus: If Rule Engine and Supervised ML agree on 'phishing' or 'legitimate', confidence is boosted (0.85 - 0.99) and final verdict is locked.\n"
        "2. Disagreement & Weighting: If they disagree, a weighted composite confidence score is calculated: Confidence = (0.65 × ML_Confidence) + (0.35 × Rule_Confidence). "
        "The manual_review flag is set to True to alert human SOC operators.\n"
        "3. Anomaly Penalty: If the classifier clears a URL as legitimate but the Isolation Forest flags an anomaly, confidence is penalized by -0.15 and flagged for review."
    )
    p_fuse.paragraph_format.line_spacing = 1.15
    p_fuse.paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------------------
    # SECTION 4: DETAILED EXPLANATION OF APP PAGES & TABS
    # -------------------------------------------------------------------------
    h4 = doc.add_heading("4. Commercial SaaS Dashboard: Tab-by-Tab Breakdown", level=1)
    h4.paragraph_format.space_before = Pt(16)
    h4.paragraph_format.space_after = Pt(8)

    p_ui_intro = doc.add_paragraph(
        "The user interface has been completely restructured from a cluttered prototype into an enterprise cybersecurity SaaS dashboard. "
        "Navigation is handled via a modern sidebar menu, driving four dedicated, single-responsibility views. Below is a detailed explanation of each view:"
    )
    p_ui_intro.paragraph_format.space_after = Pt(10)

    # 4.1 Tab 1: URL Threat Scanner
    h4_1 = doc.add_heading("4.1 View 1: 🔍 URL Threat Scanner (Primary Main View)", level=2)
    h4_1.paragraph_format.space_before = Pt(10)
    h4_1.paragraph_format.space_after = Pt(4)
    p_tab1 = doc.add_paragraph(
        "Role: Serves as the primary operational workspace for entering and analyzing target URLs with zero input friction.\n\n"
        "Displayed Components & Contents:\n"
        "1. Centered SaaS Header & Trust Strip: Displays brand identity ('PhishShield Enterprise AI'), subtitle, and active paradigm pills.\n"
        "2. Single Raw-URL Input Field: Generously sized text input box with JetBrains Mono typography for pasting any target URL. Eliminates all manual feature inputs.\n"
        "3. Action Buttons: Bold 'Scan URL Vector with 5 AI Paradigms' (primary) and 'Reset Viewport' (secondary).\n"
        "4. High-Impact Verdict Hero Banner:\n"
        "   • Crimson Alert Banner (Phishing): Prominent warning icon (🚨), 'MALICIOUS PHISHING DETECTED' title, target URL display, and immediate safety warning.\n"
        "   • Emerald Clean Banner (Legitimate): Verified badge (✅), 'VERIFIED BENIGN / LEGITIMATE' title, and baseline clearance notice.\n"
        "5. Four Key Performance Metric Cards:\n"
        "   • Threat Probability (%): Fused multi-paradigm confidence score.\n"
        "   • Manual SOC Review: Shows 'FLAGGED (REVIEW)' if paradigms disagree or anomalies exist; otherwise 'CLEAR (AUTOMATED)'.\n"
        "   • AI Paradigm Consensus: States overall engine agreement ('MALICIOUS' vs. 'BENIGN').\n"
        "   • LLM Report Engine: Displays the active model name (e.g. 'LLAMA3') or 'SAFETY HEURISTIC' if offline.\n"
        "6. Autonomous AI Incident Briefing Card (Ollama):\n"
        "   • Luminous gradient card with a pulsing active status chip ('● OLLAMA ACTIVE · LLAMA3' or amber fallback notice).\n"
        "   • Structured markdown content with 'Executive Threat Summary', 'Primary Attack Indicators', and 'Recommended Security Actions'.\n"
        "   • Interactive Toolbar: 'Re-Generate Brief with Ollama' button to re-prompt on demand and 'Inspect Structured Prompt' expander.\n"
        "7. Multi-Paradigm Signal Decomposition (3 Equal-Width Cards):\n"
        "   • Rule Engine Card: Shows symbolic verdict, confidence, and triggered rule tags (e.g., 'No HTTPS', 'Brand Typosquatting').\n"
        "   • Supervised ML Ensemble Card: Displays ensemble model verdict and calibrated probability.\n"
        "   • Isolation Forest Card: Displays unsupervised outlier status and exact isolation score.\n"
        "8. Derived 30-Features Data Expander: Collapsible table listing all 30 dynamically computed lexical, host, and content attributes for forensic inspection."
    )
    p_tab1.paragraph_format.line_spacing = 1.15
    p_tab1.paragraph_format.space_after = Pt(12)

    # 4.2 Tab 2: Visual Analytics & XAI
    h4_2 = doc.add_heading("4.2 View 2: 📊 Visual Analytics & Explainability (XAI Workspace)", level=2)
    h4_2.paragraph_format.space_before = Pt(10)
    h4_2.paragraph_format.space_after = Pt(4)
    p_tab2 = doc.add_paragraph(
        "Role: Centralizes all charts, plots, and visual explainability diagrams into one dedicated visual workspace, eliminating clutter from the main scanner.\n\n"
        "Displayed Components & Contents:\n"
        "1. Local Target Feature Attribution (SHAP Bar Chart):\n"
        "   • Renders an interactive horizontal bar chart for the most recently scanned URL.\n"
        "   • Each bar represents a specific URL feature: Crimson bars (+ϕ) indicate features elevating phishing risk, while Emerald bars (-ϕ) indicate features supporting safety.\n"
        "   • X-axis indicates the exact SHAP impact magnitude, sorted by absolute influence.\n"
        "2. Global Dataset Feature Importance (SHAP Beeswarm Plot):\n"
        "   • High-resolution beeswarm plot (loaded from outputs/shap_summary.png) depicting global feature importances across the entire test dataset.\n"
        "   • Visualizes how high vs. low feature values drive model decisions across thousands of URLs.\n"
        "3. Multi-Paradigm Consensus Architecture Breakdown:\n"
        "   • Four structured glassmorphic cards explaining the mechanics of Symbolic Rules, Supervised Ensembles, Isolation Outlier Forests, and Local LLM Generation."
    )
    p_tab2.paragraph_format.line_spacing = 1.15
    p_tab2.paragraph_format.space_after = Pt(12)

    # 4.3 Tab 3: Model Benchmarks & Performance
    h4_3 = doc.add_heading("4.3 View 3: ⚡ Model Benchmarks & Empirical Performance", level=2)
    h4_3.paragraph_format.space_before = Pt(10)
    h4_3.paragraph_format.space_after = Pt(4)
    p_tab3 = doc.add_paragraph(
        "Role: Enterprise evaluation suite presenting empirical ML validation and comparative classifier metrics.\n\n"
        "Displayed Components & Contents:\n"
        "1. Empirical Model Evaluation Matrix (Leaderboard Table):\n"
        "   • Interactive data table comparing all 9 experimental configurations.\n"
        "   • Metrics displayed per row: Classifier Architecture, Feature Selection Strategy, F1-Score, ROC-AUC, Accuracy, Precision, Recall, and Inference Latency.\n"
        "2. Mean F1-Score Comparative Bar Chart:\n"
        "   • Multi-color bar chart comparing average F1 performance across XGBoost, MLP, and Logistic Regression with labeled numeric scores on top of each bar.\n"
        "3. ROC-AUC Cross-Strategy Heatmap:\n"
        "   • High-contrast Seaborn heatmap displaying ROC-AUC performance across Feature Selection rows (Full 30, Mutual Info, Chi-Square) vs. Classifier columns.\n"
        "4. Pipeline Validation Methodology Card:\n"
        "   • Summary card detailing the 80/20 stratified split, cross-validation parameters, and automated model persistence into models/best_model.pkl."
    )
    p_tab3.paragraph_format.line_spacing = 1.15
    p_tab3.paragraph_format.space_after = Pt(12)

    # 4.4 Tab 4: Dataset & Features
    h4_4 = doc.add_heading("4.4 View 4: 🗄️ Dataset & Categorized Feature Directory", level=2)
    h4_4.paragraph_format.space_before = Pt(10)
    h4_4.paragraph_format.space_after = Pt(4)
    p_tab4 = doc.add_paragraph(
        "Role: Transparent data governance view outlining dataset properties and detailed documentation for all 30 engineered features.\n\n"
        "Displayed Components & Contents:\n"
        "1. Corpus KPI Metric Counters:\n"
        "   • Total Corpus Samples (e.g., 10,000+ URLs).\n"
        "   • Malicious Phishing Count (label = 1).\n"
        "   • Benign Legitimate Count (label = 0).\n"
        "   • Total Engineered Feature Count (30 continuous & binary attributes).\n"
        "2. Corpus Class Balance Donut Chart:\n"
        "   • Modern dark-mode donut chart illustrating class distribution percentages with clean slice borders.\n"
        "3. Categorized 30-Feature Directory (Comprehensive Table):\n"
        "   • Categorized table dividing all 30 features into three functional cybersecurity domains:\n"
        "     a) Lexical & Structure: URLLength, DomainLength, LetterRatio, DegitRatio, SpecialCharRatio, HasObfuscation, ObfuscationRatio, CharContinuationRate.\n"
        "     b) Host & Network: IsDomainIP, NoOfSubDomain, IsHTTPS, TLDLegitimateProb, TLDLength, NoOfURLRedirect.\n"
        "     c) Content & Behavior: HasHiddenFields, HasPasswordField, HasExternalFormSubmit, NoOfPopup, NoOfiFrame, URLSimilarityIndex, Bank, Pay, Crypto, DomainTitleMatchScore, HasTitle, HasFavicon, HasCopyrightInfo.\n"
        "4. Data Pipeline Configuration Card: Details StandardScaler fit rules (strictly on training partition) and missing value remediation."
    )
    p_tab4.paragraph_format.line_spacing = 1.15
    p_tab4.paragraph_format.space_after = Pt(12)

    # 4.5 Sidebar Navigation Panel
    h4_5 = doc.add_heading("4.5 Sidebar: Primary SaaS Navigation & Telemetry Console", level=2)
    h4_5.paragraph_format.space_before = Pt(10)
    h4_5.paragraph_format.space_after = Pt(4)
    p_side = doc.add_paragraph(
        "Role: Replaces clunky configuration dropdowns with a sleek, distraction-free SaaS navigation drawer.\n\n"
        "Displayed Components:\n"
        "• Brand Lockup: Shield emblem, 'PhishShield', and 'v1.2 SaaS' badge.\n"
        "• Platform Navigation Radio Menu: Seamlessly switches views (🔍 Main Threat Scanner, 📊 Visual Analytics & XAI, ⚡ Model Benchmarks, 🗄️ Dataset & Features).\n"
        "• Local Ollama Telemetry Card: Live ping indicator (Green '● Ollama Local: Online' or Amber '○ Offline'), showing local port 11434 status.\n"
        "• Academic Attribution: PE-1 Lab · MIT-WPU footer."
    )
    p_side.paragraph_format.line_spacing = 1.15
    p_side.paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------------------
    # SECTION 5: HEADLESS REST API BACKEND (SERVER.PY)
    # -------------------------------------------------------------------------
    h5 = doc.add_heading("5. Headless REST API Backend (server.py)", level=1)
    h5.paragraph_format.space_before = Pt(16)
    h5.paragraph_format.space_after = Pt(8)

    p_api = doc.add_paragraph(
        "In addition to the interactive Streamlit dashboard, PhishShield includes a high-performance headless REST microservice (server.py) "
        "built with FastAPI and Uvicorn. This enables seamless integration into enterprise Security Orchestration, Automation, and Response (SOAR) "
        "playbooks, email security gateways, and SIEM pipelines.\n\n"
        "Primary Endpoints:\n"
        "• GET /health: Returns service health status and loaded model states.\n"
        "• GET /info: Returns pipeline metadata, active classifier architecture, and feature names.\n"
        "• POST /predict: Accepts a JSON payload containing either a raw URL string or a dictionary of features. Returns the full multi-paradigm "
        "verdict, calibrated confidence score, manual review flag, component signal breakdown, SHAP top features, and the Ollama incident briefing."
    )
    p_api.paragraph_format.line_spacing = 1.15
    p_api.paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------------------
    # SECTION 6: EXECUTION & USER GUIDE
    # -------------------------------------------------------------------------
    h6 = doc.add_heading("6. System Execution & Operational Guide", level=1)
    h6.paragraph_format.space_before = Pt(16)
    h6.paragraph_format.space_after = Pt(8)

    p_run = doc.add_paragraph(
        "To run PhishShield locally, execute the following commands in your PowerShell or terminal:\n\n"
        "1. Launch Local Ollama LLM Service (Terminal 1):\n"
        "   ollama run llama3\n"
        "   (Starts the local LLM server listening on http://localhost:11434 for zero-telemetry threat briefings).\n\n"
        "2. Launch Interactive SaaS Dashboard (Terminal 2):\n"
        "   python -m streamlit run app.py\n"
        "   (Opens the commercial SaaS interface at http://localhost:8501 or http://localhost:8502).\n\n"
        "3. Launch Headless REST API Server (Optional - Terminal 3):\n"
        "   python server.py\n"
        "   (Starts the FastAPI REST server at http://localhost:8000 with Swagger UI documentation at http://localhost:8000/docs)."
    )
    p_run.paragraph_format.line_spacing = 1.15
    p_run.paragraph_format.space_after = Pt(14)

    # -------------------------------------------------------------------------
    # SECTION 7: SUMMARY & CONCLUSION
    # -------------------------------------------------------------------------
    h7 = doc.add_heading("7. Conclusion & Academic Contributions", level=1)
    h7.paragraph_format.space_before = Pt(16)
    h7.paragraph_format.space_after = Pt(8)

    p_concl = doc.add_paragraph(
        "PhishShield fulfills all learning outcomes for the AI Systems & Applications course by demonstrating the practical fusion of symbolic heuristics, "
        "supervised learning, unsupervised anomaly detection, explainable attribution (SHAP), and generative natural-language intelligence. "
        "By packaging these advanced capabilities inside a sleek, commercial-grade SaaS dashboard with dynamic raw-URL feature derivation, the project "
        "bridges the gap between academic theory and real-world cybersecurity engineering."
    )
    p_concl.paragraph_format.line_spacing = 1.15
    p_concl.paragraph_format.space_after = Pt(14)

    # Save Document
    doc.save(OUTPUT_FILE)
    print(f"[SUCCESS] Document created and saved to {os.path.abspath(OUTPUT_FILE)}")

if __name__ == "__main__":
    create_styled_document()
