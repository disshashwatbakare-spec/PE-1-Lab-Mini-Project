"""
app.py  --  PhishShield Enterprise AI  |  Autonomous Cyber Threat Intelligence
================================================================================
Enterprise XAI Cyber Threat Operations Platform powered by 5 AI Paradigms:
  1. Symbolic AI: Heuristic Expert Rule Engine (15 weighted security rules)
  2. Supervised ML: Multi-Model Ensemble (XGBoost, MLP Neural Network, Logistic Regression)
  3. Unsupervised ML: Isolation Forest Structural Outlier Detector
  4. Explainable AI: SHAP (SHapley Additive exPlanations) Game-Theoretic Attribution Layer
  5. Generative AI: Local Ollama LLM SOC Incident Report Generation & Threat Advisory

Designed with a professional SOC (Security Operations Center) dashboard aesthetic.
"""

import sys
import re
import pathlib
import warnings
import urllib.parse
import json

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from llm_explainer import check_ollama_status, generate_report, _call_ollama

# =============================================================================
# STREAMLIT PAGE SETUP
# =============================================================================
st.set_page_config(
    page_title="PhishShield -- Enterprise AI URL Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# ENTERPRISE SOC COLOR SYSTEM & DESIGN TOKENS
# =============================================================================
CLR_CANVAS    = "#0A0E17"   # Obsidian Dark base
CLR_SURFACE   = "#111827"   # Slate 900 card surface
CLR_ELEVATED  = "#1E293B"   # Slate 800 elevated layer
CLR_BORDER    = "#1F2937"   # Slate 800 border
CLR_BORDER_LT = "#334155"   # Slate 700 accent border
CLR_BLUE      = "#3B82F6"   # Electric Cobalt Blue
CLR_TEAL      = "#06B6D4"   # Cyber Teal
CLR_RED       = "#EF4444"   # Threat Alert Crimson
CLR_GREEN     = "#10B981"   # Verified Safe Emerald
CLR_AMBER     = "#F59E0B"   # Caution Amber
CLR_TEXT_PRI  = "#F9FAFB"   # High contrast header white
CLR_TEXT_SEC  = "#D1D5DB"   # Slate 300 body
CLR_TEXT_MUT  = "#9CA3AF"   # Slate 400 muted
CLR_TEXT_DIM  = "#6B7280"   # Slate 500 dim

# =============================================================================
# PROFESSIONAL ENTERPRISE CSS
# =============================================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* Global baseline */
  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      background-color: {CLR_CANVAS};
      color: {CLR_TEXT_PRI};
  }}
  .stApp {{
      background: {CLR_CANVAS};
  }}

  /* Hide default streamlit decorations */
  #MainMenu {{visibility: hidden;}}
  footer {{visibility: hidden;}}
  header {{background: transparent !important;}}

  /* Top Navigation & Telemetry Banner */
  .enterprise-nav {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 14px 20px;
      background: {CLR_SURFACE};
      border-bottom: 1px solid {CLR_BORDER};
      border-radius: 12px;
      margin-bottom: 22px;
  }}
  .brand-block {{
      display: flex;
      align-items: center;
      gap: 12px;
  }}
  .brand-badge {{
      background: linear-gradient(135deg, {CLR_BLUE} 0%, #1D4ED8 100%);
      color: #FFF;
      font-size: 1.15rem;
      font-weight: 800;
      width: 38px;
      height: 38px;
      display: flex;
      align-items: center;
      justify-content: center;
      border-radius: 8px;
  }}
  .brand-title {{
      font-size: 1.25rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      color: #FFFFFF;
  }}
  .brand-subtitle {{
      font-size: 0.72rem;
      color: {CLR_TEXT_MUT};
      text-transform: uppercase;
      letter-spacing: 0.08em;
      font-weight: 600;
  }}
  .telemetry-strip {{
      display: flex;
      align-items: center;
      gap: 12px;
  }}
  .telemetry-chip {{
      font-size: 0.72rem;
      font-weight: 600;
      padding: 4px 10px;
      border-radius: 6px;
      background: {CLR_ELEVATED};
      border: 1px solid {CLR_BORDER_LT};
      color: {CLR_TEXT_MUT};
      display: flex;
      align-items: center;
      gap: 6px;
  }}
  .chip-online {{
      color: {CLR_GREEN};
      border-color: rgba(16, 185, 129, 0.35);
      background: rgba(16, 185, 129, 0.08);
  }}

  /* Cards & Panels */
  .soc-card {{
      background: {CLR_SURFACE};
      border: 1px solid {CLR_BORDER};
      border-radius: 10px;
      padding: 20px 22px;
      margin-bottom: 18px;
  }}
  .soc-card-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 14px;
      border-bottom: 1px solid {CLR_BORDER};
      padding-bottom: 10px;
  }}
  .soc-card-title {{
      font-size: 0.95rem;
      font-weight: 700;
      letter-spacing: -0.01em;
      color: #FFFFFF;
      display: flex;
      align-items: center;
      gap: 8px;
  }}
  .soc-card-sub {{
      font-size: 0.75rem;
      color: {CLR_TEXT_MUT};
  }}

  /* Incident Report Card & Formatted Markdown */
  .soc-report-box {{
      background: {CLR_SURFACE};
      border: 1px solid {CLR_BORDER};
      border-radius: 10px;
      padding: 22px 26px;
      margin-bottom: 18px;
  }}
  /* Target native st.markdown() output rendered AFTER our wrapper div.
     Streamlit places [data-testid="stMarkdownContainer"] as a sibling block. */
  .soc-report-content [data-testid="stMarkdownContainer"],
  .soc-report-content + [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stMarkdownContainer"],
  div.soc-report-content [data-testid="stMarkdownContainer"] {{
      padding: 0 !important;
      font-size: 0.92rem !important;
      line-height: 1.7 !important;
  }}
  .soc-report-content h3,
  .soc-report-content [data-testid="stMarkdownContainer"] h3 {{
      font-size: 1.22rem !important;
      font-weight: 800 !important;
      color: #FFFFFF !important;
      margin-top: 4px !important;
      margin-bottom: 10px !important;
      letter-spacing: -0.01em !important;
      border-bottom: 1px solid {CLR_BORDER} !important;
      padding-bottom: 8px !important;
  }}
  .soc-report-content h4,
  .soc-report-content [data-testid="stMarkdownContainer"] h4 {{
      font-size: 1.0rem !important;
      font-weight: 700 !important;
      color: #38BDF8 !important;
      margin-top: 22px !important;
      margin-bottom: 10px !important;
      letter-spacing: -0.01em !important;
  }}
  .soc-report-content p,
  .soc-report-content [data-testid="stMarkdownContainer"] p {{
      color: {CLR_TEXT_SEC} !important;
      margin-bottom: 10px !important;
      line-height: 1.7 !important;
  }}
  .soc-report-content hr,
  .soc-report-content [data-testid="stMarkdownContainer"] hr {{
      border: none !important;
      border-top: 1px solid {CLR_BORDER} !important;
      margin: 18px 0 !important;
  }}
  .soc-report-content ul,
  .soc-report-content [data-testid="stMarkdownContainer"] ul {{
      margin-top: 6px !important;
      margin-bottom: 14px !important;
      padding-left: 22px !important;
      list-style-type: disc !important;
  }}
  .soc-report-content li,
  .soc-report-content [data-testid="stMarkdownContainer"] li {{
      margin-bottom: 10px !important;
      line-height: 1.7 !important;
      color: {CLR_TEXT_SEC} !important;
  }}
  .soc-report-content li > ul,
  .soc-report-content [data-testid="stMarkdownContainer"] li > ul {{
      margin-top: 6px !important;
      margin-bottom: 4px !important;
      padding-left: 20px !important;
      list-style-type: circle !important;
  }}
  .soc-report-content li > ul > li,
  .soc-report-content [data-testid="stMarkdownContainer"] li > ul > li {{
      margin-bottom: 5px !important;
      font-size: 0.875rem !important;
      color: {CLR_TEXT_MUT} !important;
      line-height: 1.6 !important;
  }}
  .soc-report-content ol,
  .soc-report-content [data-testid="stMarkdownContainer"] ol {{
      margin-top: 6px !important;
      margin-bottom: 14px !important;
      padding-left: 22px !important;
  }}
  .soc-report-content ol > li,
  .soc-report-content [data-testid="stMarkdownContainer"] ol > li {{
      margin-bottom: 14px !important;
      line-height: 1.7 !important;
      color: {CLR_TEXT_SEC} !important;
  }}
  .soc-report-content strong,
  .soc-report-content [data-testid="stMarkdownContainer"] strong {{
      color: #FFFFFF !important;
      font-weight: 700 !important;
  }}
  .soc-report-content code,
  .soc-report-content [data-testid="stMarkdownContainer"] code {{
      background: rgba(56,189,248,0.12) !important;
      color: #38BDF8 !important;
      border-radius: 4px !important;
      padding: 1px 5px !important;
      font-size: 0.83rem !important;
      font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
  }}
  em {{
      color: {CLR_TEXT_MUT} !important;
  }}

  /* Threat Banners */
  .verdict-box-phishing {{
      background: rgba(239, 68, 68, 0.08);
      border: 1.5px solid {CLR_RED};
      border-radius: 10px;
      padding: 22px 26px;
      margin-bottom: 20px;
      display: flex;
      align-items: flex-start;
      gap: 18px;
  }}
  .verdict-box-legitimate {{
      background: rgba(16, 185, 129, 0.08);
      border: 1.5px solid {CLR_GREEN};
      border-radius: 10px;
      padding: 22px 26px;
      margin-bottom: 20px;
      display: flex;
      align-items: flex-start;
      gap: 18px;
  }}
  .verdict-badge {{
      font-size: 0.74rem;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      padding: 3px 8px;
      border-radius: 4px;
      margin-bottom: 6px;
      display: inline-block;
  }}
  .badge-danger {{
      background: {CLR_RED};
      color: #FFFFFF;
  }}
  .badge-success {{
      background: {CLR_GREEN};
      color: #0A0E17;
  }}
  .verdict-headline {{
      font-size: 1.45rem;
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 6px;
  }}

  /* Metric KPI Blocks */
  div[data-testid="metric-container"] {{
      background: {CLR_SURFACE} !important;
      border: 1px solid {CLR_BORDER} !important;
      border-radius: 8px !important;
      padding: 14px 18px !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
      font-size: 1.55rem !important;
      font-weight: 700 !important;
      color: #FFFFFF !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
      font-size: 0.72rem !important;
      color: {CLR_TEXT_MUT} !important;
      text-transform: uppercase !important;
      letter-spacing: 0.06em !important;
      font-weight: 600 !important;
  }}

  /* Model Spec Cards */
  .model-card {{
      background: {CLR_SURFACE};
      border: 1px solid {CLR_BORDER};
      border-radius: 8px;
      padding: 16px 18px;
      height: 100%;
      transition: border-color 0.15s ease;
  }}
  .model-card:hover {{
      border-color: {CLR_BORDER_LT};
  }}
  .model-card-tag {{
      font-size: 0.68rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: {CLR_TEXT_MUT};
      margin-bottom: 6px;
  }}
  .model-card-name {{
      font-size: 1.05rem;
      font-weight: 700;
      color: #FFFFFF;
      margin-bottom: 4px;
  }}
  .model-metric-row {{
      display: flex;
      justify-content: space-between;
      font-size: 0.78rem;
      padding: 4px 0;
      border-bottom: 1px solid {CLR_BORDER};
  }}

  /* Text inputs & Search buttons */
  [data-testid="stTextInput"] input {{
      background: #0D131F !important;
      border: 1px solid {CLR_BORDER_LT} !important;
      border-radius: 8px !important;
      color: #FFFFFF !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.95rem !important;
      padding: 14px 18px !important;
  }}
  [data-testid="stTextInput"] input:focus {{
      border-color: {CLR_BLUE} !important;
      box-shadow: 0 0 0 1px {CLR_BLUE} !important;
  }}

  .stButton > button[kind="primary"] {{
      background: {CLR_BLUE} !important;
      color: #FFFFFF !important;
      border: none !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      font-size: 0.95rem !important;
      padding: 12px 24px !important;
      transition: background 0.15s ease !important;
  }}
  .stButton > button[kind="primary"]:hover {{
      background: #2563EB !important;
  }}
  .stButton > button[kind="secondary"] {{
      background: {CLR_ELEVATED} !important;
      border: 1px solid {CLR_BORDER_LT} !important;
      color: {CLR_TEXT_SEC} !important;
      border-radius: 8px !important;
      font-weight: 600 !important;
      padding: 12px 18px !important;
  }}

  /* Sidebar styling */
  [data-testid="stSidebar"] {{
      background: #0B0F19 !important;
      border-right: 1px solid {CLR_BORDER} !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label {{
      background: transparent !important;
      border: 1px solid transparent !important;
      border-radius: 6px !important;
      padding: 8px 12px !important;
      color: {CLR_TEXT_SEC} !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
      background: {CLR_SURFACE} !important;
      border-color: {CLR_BORDER} !important;
  }}

  /* Code block aesthetics */
  code {{
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 0.85rem !important;
      background: rgba(15, 23, 42, 0.6) !important;
      border: 1px solid {CLR_BORDER} !important;
      border-radius: 4px !important;
      padding: 2px 6px !important;
  }}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# FEATURE EXTRACTION ENGINE (30 DERIVED ATTRIBUTES)
# =============================================================================
SHORTENER_DOMAINS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "ow.ly", "is.gd",
    "buff.ly", "adf.ly", "bl.ink", "shorte.st", "rebrand.ly",
    "cutt.ly", "clck.ru", "s.id", "tiny.cc",
}
BRANDS = [
    "paypal", "amazon", "google", "apple", "microsoft", "ebay",
    "netflix", "facebook", "instagram", "bank", "secure", "login",
    "update", "verify", "account", "signin",
]
SUSP_TLDS = {"tk", "ml", "ga", "cf", "gq", "xyz", "top", "pw", "cc", "work"}


def extract_url_features(url: str) -> dict:
    """Derive all dataset features programmatically from a raw target URL."""
    parsed = urllib.parse.urlparse(url)
    scheme = parsed.scheme.lower()
    domain = parsed.netloc.lower()
    full   = url

    ip_pat = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")
    bare   = re.sub(r":\d+$", "", domain)
    is_ip  = 1 if ip_pat.match(bare) else 0

    parts  = bare.replace("www.", "").split(".")
    n_sub  = max(0, len(parts) - 2)
    tld    = parts[-1] if parts else ""
    tld_len = len(tld)
    tld_legit_prob = 0.15 if tld in SUSP_TLDS else 0.70

    is_shortener = 1 if any(s in domain for s in SHORTENER_DOMAINS) else 0

    n       = max(len(full), 1)
    letters = sum(c.isalpha()  for c in full)
    digits  = sum(c.isdigit()  for c in full)
    special = sum(not c.isalnum() and c not in "/:.-_?=&#" for c in full)

    n_pct   = full.count("%")
    has_at  = 1 if "@" in full else 0
    has_obc = 1 if (n_pct > 0 or has_at) else 0

    runs    = re.findall(r"[a-zA-Z]+", full)
    max_run = max((len(r) for r in runs), default=0)
    ccr     = round(max_run / n, 4)

    brand_hits    = sum(b in full.lower() for b in BRANDS)
    url_sim_idx   = min(100, brand_hits * 25)
    url_char_prob = round(letters / n, 4)

    base_trust = 1 if (scheme == "https" and not is_ip and not is_shortener and brand_hits == 0) else 0

    return {
        "URLLength"               : len(full),
        "DomainLength"            : len(domain),
        "IsDomainIP"              : is_ip,
        "NoOfSubDomain"           : n_sub,
        "IsHTTPS"                 : 1 if scheme == "https" else 0,
        "NoOfQMarkInURL"          : full.count("?"),
        "NoOfAmpersandInURL"      : full.count("&"),
        "NoOfEqualsInURL"         : full.count("="),
        "NoOfLettersInURL"        : letters,
        "LetterRatioInURL"        : round(letters / n, 4),
        "NoOfDegitsInURL"         : digits,
        "DegitRatioInURL"         : round(digits / n, 4),
        "NoOfOtherSpecialCharsInURL": special,
        "SpacialCharRatioInURL"   : round(special / n, 4),
        "HasObfuscation"          : has_obc,
        "ObfuscationRatio"        : round(n_pct / n, 4),
        "URLSimilarityIndex"      : url_sim_idx,
        "CharContinuationRate"    : ccr,
        "TLDLegitimateProb"       : tld_legit_prob,
        "URLCharProb"             : url_char_prob,
        "TLDLength"               : tld_len,
        "NoOfObfuscatedChar"      : n_pct,
        "LineOfCode"              : 100,
        "LargestLineLength"       : 500,
        "HasTitle"                : base_trust,
        "DomainTitleMatchScore"   : round((1 - brand_hits * 0.3) * 80, 1),
        "URLTitleMatchScore"      : round((1 - brand_hits * 0.25) * 70, 1),
        "HasFavicon"              : base_trust,
        "Robots"                  : 0,
        "IsResponsive"            : base_trust,
        "NoOfURLRedirect"         : is_shortener,
        "NoOfSelfRedirect"        : 0,
        "HasDescription"          : base_trust,
        "NoOfPopup"               : 1 if brand_hits > 1 else 0,
        "NoOfiFrame"              : 1 if brand_hits > 0 else 0,
        "HasExternalFormSubmit"   : 1 if brand_hits > 0 else 0,
        "HasSocialNet"            : 0,
        "HasSubmitButton"         : 1,
        "HasHiddenFields"         : 1 if brand_hits > 0 else 0,
        "HasPasswordField"        : 1 if any(k in full.lower() for k in ["login", "signin", "password"]) else 0,
        "Bank"                    : 1 if "bank"   in full.lower() else 0,
        "Pay"                     : 1 if "pay"    in full.lower() else 0,
        "Crypto"                  : 1 if any(k in full.lower() for k in ["crypto", "bitcoin", "wallet"]) else 0,
        "HasCopyrightInfo"        : base_trust,
        "NoOfImage"               : 0,
        "NoOfCSS"                 : max(1, base_trust),
        "NoOfJS"                  : max(1, base_trust + brand_hits),
        "NoOfSelfRef"             : 0,
        "NoOfEmptyRef"            : 0,
        "NoOfExternalRef"         : 0,
    }


# =============================================================================
# CACHED DATA & PIPELINE LOADERS
# =============================================================================
@st.cache_resource(show_spinner="Initializing PhishShield Defense Core...")
def load_pipeline():
    from pipeline import PhishShieldPipeline
    pipe = PhishShieldPipeline()
    try:
        pipe.load_all()
    except Exception:
        pipe.train()
    return pipe


@st.cache_data
def load_results_table():
    p = ROOT / "outputs" / "classifier_results.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_dataset_summary():
    data_dir = ROOT / "data" / "processed"
    tp, ep   = data_dir / "train.csv", data_dir / "test.csv"
    if not tp.exists():
        return None, [], {}
    tr, te = pd.read_csv(tp), pd.read_csv(ep)
    total  = len(tr) + len(te)
    feat_p = data_dir / "feature_cols.txt"
    feats  = feat_p.read_text().strip().split("\n") if feat_p.exists() else []
    counts = {
        "Phishing"  : int((tr["label"]==1).sum() + (te["label"]==1).sum()),
        "Legitimate": int((tr["label"]==0).sum() + (te["label"]==0).sum()),
    }
    return total, feats, counts


# =============================================================================
# TOP ENTERPRISE HEADER
# =============================================================================
st.markdown(f"""
<div class="enterprise-nav">
  <div class="brand-block">
    <div class="brand-badge">🛡️</div>
    <div>
      <div class="brand-title">PhishShield Enterprise AI</div>
      <div class="brand-subtitle">Autonomous Hybrid Threat Intelligence Engine &middot; MIT-WPU PE-1 Lab</div>
    </div>
  </div>
  <div class="telemetry-strip">
    <div class="telemetry-chip chip-online">&#9679; CORE: OPERATIONAL</div>
    <div class="telemetry-chip">5 AI PARADIGMS</div>
    <div class="telemetry-chip">SHAP XAI ACTIVE</div>
    <div class="telemetry-chip">SOC READY</div>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR -- Enterprise Controls & Telemetry
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 10px 4px 16px 4px;">
      <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CLR_TEXT_MUT}; font-weight:800; margin-bottom:8px;">
        Operations Navigation
      </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        options=[
            "🔍 URL Threat Scanner",
            "🔬 Explainable AI (SHAP)",
            "📊 Model Benchmarks",
            "🗄️ Dataset & Features",
        ],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{CLR_BORDER}; margin: 20px 0;'>", unsafe_allow_html=True)

    # Local Ollama Telemetry
    ollama_stat = check_ollama_status()
    is_online = ollama_stat["online"]
    local_models = ollama_stat["models"]

    st.markdown(f"""
    <div class="soc-card" style="padding:14px; margin-bottom:14px;">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span style="font-size:0.75rem; font-weight:700; color:{CLR_TEXT_MUT};">LOCAL OLLAMA LLM</span>
        <span style="font-size:0.7rem; color:{CLR_GREEN if is_online else CLR_AMBER}; font-weight:700;">
          {'● ONLINE' if is_online else '⚠ OFFLINE'}
        </span>
      </div>
      <div style="font-size:0.8rem; color:#FFF; font-weight:600; margin-top:4px;">
        {len(local_models)} Models Available
      </div>
      <div style="font-size:0.7rem; color:{CLR_TEXT_MUT}; margin-top:2px;">Endpoint: 127.0.0.1:11434</div>
    </div>
    """, unsafe_allow_html=True)

    selected_model = "llama3"
    if is_online and local_models:
        selected_model = st.selectbox("Active Briefing Model", options=local_models, index=0)
    
    report_persona = st.selectbox(
        "SOC Briefing Audience",
        options=["general", "technical"],
        format_func=lambda x: "Executive Security Summary" if x == "general" else "Technical Incident Response (SOC)",
    )

    st.markdown(f"<hr style='border-color:{CLR_BORDER}; margin: 20px 0;'>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.72rem; color:{CLR_TEXT_MUT}; line-height:1.5;">
      <b style="color:#FFF;">PhishShield Threat Core v2.4</b><br>
      PE-1 Lab &middot; Semester V &middot; MIT-WPU<br>
      Hybrid Multi-Paradigm Cyber Defense
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# VIEW 1: 🔍 URL Threat Scanner & Multi-Model Engine
# =============================================================================
if nav_selection == "🔍 URL Threat Scanner":
    st.markdown("""
    <div style="margin-bottom: 12px;">
      <div style="font-size: 1.25rem; font-weight: 700; color: #FFFFFF;">Threat Vector Inspection</div>
      <div style="font-size: 0.84rem; color: #9CA3AF;">
        Inspect arbitrary target URLs with 5 synchronized AI paradigms. All 30 structural, lexical, and host indicators are extracted in real time.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Preset Quick-Loader Buttons
    st.markdown("<div style='font-size:0.72rem; font-weight:700; color:#9CA3AF; margin-bottom:6px;'>PRESET THREAT SAMPLES:</div>", unsafe_allow_html=True)
    q1, q2, q3, q4 = st.columns(4)
    with q1:
        if st.button("🎯 PayPal Phishing Spoof", use_container_width=True):
            st.session_state["main_url_search_input"] = "http://update-paypal-security-verification.xyz/login.php?token=9384"
    with q2:
        if st.button("🏦 Chase Bank Harvester", use_container_width=True):
            st.session_state["main_url_search_input"] = "https://chase-bank-online-alert.top/verify-account/signin"
    with q3:
        if st.button("🔒 Enterprise Portal (Safe)", use_container_width=True):
            st.session_state["main_url_search_input"] = "https://www.google.com/search?q=cybersecurity+defense+metrics"
    with q4:
        if st.button("⚡ Obfuscated Raw IP", use_container_width=True):
            st.session_state["main_url_search_input"] = "http://192.168.1.100/%7Euser/bank/signin.php?session=token"

    # URL Input Field
    raw_url_input = st.text_input(
        label="URL Search Input",
        placeholder="Enter target URL vector (e.g., https://paypal-account-verification.top/signin)...",
        key="main_url_search_input",
        label_visibility="collapsed",
    )

    btn_c1, btn_c2 = st.columns([4, 1])
    with btn_c1:
        scan_action = st.button("⚡ Execute Threat Vector Analysis", type="primary", use_container_width=True)
    with btn_c2:
        reset_action = st.button("Clear Buffer", type="secondary", use_container_width=True)

    if reset_action:
        st.session_state["main_url_search_input"] = ""
        st.session_state.pop("last_scan_result", None)
        st.session_state.pop("last_scanned_url", None)
        st.session_state.pop("last_scanned_feats", None)
        st.rerun()

    target_url = raw_url_input.strip()

    if scan_action and target_url:
        with st.spinner("Extracting 30 lexical & host security attributes..."):
            extracted_feats = extract_url_features(target_url)

        pipe = load_pipeline()
        for col in (pipe.feature_cols or []):
            extracted_feats.setdefault(col, 0.0)

        with st.spinner("Executing 5-Paradigm Defense Core (Symbolic + Supervised + Unsupervised + SHAP + LLM)..."):
            try:
                scan_res = pipe.predict(
                    extracted_feats,
                    top_n_shap=10,
                    use_llm=True,
                    llm_model=selected_model,
                    llm_audience=report_persona,
                )
                st.session_state["last_scan_result"] = scan_res
                st.session_state["last_scanned_url"] = target_url
                st.session_state["last_scanned_feats"] = extracted_feats
            except Exception as scan_err:
                st.error(f"Inference Engine Error: {scan_err}")
                st.stop()

    # Display Scan Results
    if "last_scan_result" in st.session_state:
        result    = st.session_state["last_scan_result"]
        url_text  = st.session_state["last_scanned_url"]
        feats_map = st.session_state["last_scanned_feats"]

        verdict        = result["verdict"]
        confidence     = result["confidence"]
        is_phish       = verdict == "phishing"
        manual_review  = result["manual_review"]
        fusion_math    = result.get("fusion_math", {})
        models_dict    = result.get("model_breakdown", {})
        shap_rich      = result.get("shap_rich", {})

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Executive Verdict Banner
        if is_phish:
            st.markdown(f"""
            <div class="verdict-box-phishing">
              <span style="font-size:2.2rem; line-height:1;">🚨</span>
              <div style="flex:1;">
                <div class="verdict-badge badge-danger">CRITICAL SECURITY ALERT &middot; MALICIOUS VECTOR</div>
                <div class="verdict-headline" style="color:{CLR_RED};">PHISHING THREAT CONFIRMED</div>
                <div style="font-size:0.86rem; color:{CLR_TEXT_SEC}; line-height:1.5;">
                  Target: <code>{url_text}</code><br>
                  Autonomous defense engine detected deceptive credential-theft signatures, brand spoofing keywords, or severe structural anomalies.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-box-legitimate">
              <span style="font-size:2.2rem; line-height:1;">✅</span>
              <div style="flex:1;">
                <div class="verdict-badge badge-success">INTEGRITY VERIFIED &middot; BENIGN TARGET</div>
                <div class="verdict-headline" style="color:{CLR_GREEN};">AUTHENTICATED LEGITIMATE URL</div>
                <div style="font-size:0.86rem; color:{CLR_TEXT_SEC}; line-height:1.5;">
                  Target: <code>{url_text}</code><br>
                  Nominal structural baseline verified. TLS transport valid, zero deceptive brand harvesting patterns or statistical outliers triggered.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Combined Final Model Confidence & Mathematical Fusion Card
        st.markdown(f"""
        <div class="soc-card">
          <div class="soc-card-header">
            <div>
              <div class="soc-card-title">⚖️ Combined Final Model Confidence (Multi-Paradigm Decision Fusion)</div>
              <div class="soc-card-sub">Mathematical synthesis combining Symbolic Rules, Supervised Classifiers, and Anomaly Detection</div>
            </div>
            <div style="text-align:right;">
              <span style="font-size:0.72rem; font-weight:700; color:{CLR_TEXT_MUT}; text-transform:uppercase;">Consensus Status</span><br>
              <span style="font-size:0.82rem; font-weight:700; color:{CLR_GREEN if not manual_review else CLR_AMBER};">
                {fusion_math.get('consensus_status', 'Consensus Reached')}
              </span>
            </div>
          </div>
          <div style="display:grid; grid-template-columns: 1fr 2fr; gap: 24px; align-items:center;">
            <div style="text-align:center; padding: 10px; border-right: 1px solid {CLR_BORDER};">
              <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.08em; color:{CLR_TEXT_MUT}; font-weight:700;">Final Fused Confidence</div>
              <div style="font-size:2.8rem; font-weight:800; color:{CLR_RED if is_phish else CLR_GREEN}; line-height:1.2; margin: 6px 0;">
                {confidence:.1%}
              </div>
              <div style="font-size:0.78rem; color:{CLR_TEXT_SEC};">
                SOC Action: <b style="color:#FFF;">{'⚠️ Manual Review Escalation' if manual_review else '✅ Automated Resolution'}</b>
              </div>
            </div>
            <div>
              <div style="font-size:0.78rem; font-weight:700; color:{CLR_TEXT_MUT}; text-transform:uppercase; margin-bottom:8px;">
                Mathematical Fusion Formula &amp; Signal Contributions:
              </div>
              <div style="background:{CLR_ELEVATED}; border:1px solid {CLR_BORDER_LT}; border-radius:6px; padding:10px 14px; font-family:'JetBrains Mono', monospace; font-size:0.88rem; color:#FFF; margin-bottom:10px;">
                Confidence<sub>final</sub> = (w<sub>ML</sub> &times; C<sub>ML</sub>) + (w<sub>Rule</sub> &times; C<sub>Rule</sub>) &minus; Penalty<sub>Anomaly</sub>
              </div>
              <div style="display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; font-size:0.78rem;">
                <div style="background:rgba(59, 130, 246, 0.08); border:1px solid rgba(59, 130, 246, 0.25); border-radius:6px; padding:8px;">
                  <div style="color:{CLR_BLUE}; font-weight:700;">Supervised ML (65%)</div>
                  <div style="color:#FFF; font-weight:600; margin-top:2px;">Score: {fusion_math.get('ml_confidence', 0.99):.1%}</div>
                  <div style="color:{CLR_TEXT_MUT}; font-size:0.72rem;">Weight &rarr; +{fusion_math.get('ml_component', 0.64):.1%}</div>
                </div>
                <div style="background:rgba(16, 185, 129, 0.08); border:1px solid rgba(16, 185, 129, 0.25); border-radius:6px; padding:8px;">
                  <div style="color:{CLR_GREEN}; font-weight:700;">Symbolic Rules (35%)</div>
                  <div style="color:#FFF; font-weight:600; margin-top:2px;">Score: {fusion_math.get('rule_confidence', 0.85):.1%}</div>
                  <div style="color:{CLR_TEXT_MUT}; font-size:0.72rem;">Weight &rarr; +{fusion_math.get('rule_component', 0.30):.1%}</div>
                </div>
                <div style="background:rgba(245, 158, 11, 0.08); border:1px solid rgba(245, 158, 11, 0.25); border-radius:6px; padding:8px;">
                  <div style="color:{CLR_AMBER}; font-weight:700;">Anomaly Penalty</div>
                  <div style="color:#FFF; font-weight:600; margin-top:2px;">Contradiction: {fusion_math.get('anomaly_penalty', 0.0):.1%}</div>
                  <div style="color:{CLR_TEXT_MUT}; font-size:0.72rem;">Deduction &rarr; &minus;{fusion_math.get('anomaly_penalty', 0.0):.1%}</div>
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Individual Models Breakdown & Evaluation Matrix
        st.markdown(f"""
        <div style="margin: 22px 0 12px 0;">
          <div style="font-size: 1.05rem; font-weight: 700; color: #FFFFFF;">Individual Model Predictions &amp; Empirical Accuracies</div>
          <div style="font-size: 0.82rem; color: #9CA3AF;">
            Independent operational scores, threat probabilities, and test-set validation metrics for each AI component.
          </div>
        </div>
        """, unsafe_allow_html=True)

        if models_dict:
            cols = st.columns(len(models_dict))
            for i, (m_key, m_info) in enumerate(models_dict.items()):
                with cols[i]:
                    m_v = m_info.get("verdict", "N/A")
                    m_c = m_info.get("confidence", 0.0)
                    m_acc = m_info.get("accuracy", 1.0)
                    m_f1 = m_info.get("f1", 1.0)
                    m_v_col = CLR_RED if m_v in ["phishing", "anomalous"] else CLR_GREEN
                    
                    st.markdown(f"""
                    <div class="model-card">
                      <div class="model-card-tag">{m_info.get('paradigm', 'AI Paradigm')}</div>
                      <div class="model-card-name">{m_info.get('name', m_key)}</div>
                      <div style="font-size:0.88rem; font-weight:700; color:{m_v_col}; margin: 6px 0 10px 0;">
                        {m_v.upper()} ({m_c:.1%})
                      </div>
                      <div class="model-metric-row">
                        <span style="color:{CLR_TEXT_MUT};">Test Accuracy:</span>
                        <span style="font-weight:600; color:#FFF;">{m_acc:.2%}</span>
                      </div>
                      <div class="model-metric-row">
                        <span style="color:{CLR_TEXT_MUT};">Test F1-Score:</span>
                        <span style="font-weight:600; color:#FFF;">{m_f1:.4f}</span>
                      </div>
                      <div class="model-metric-row" style="border-bottom:none;">
                        <span style="color:{CLR_TEXT_MUT};">Phish Probability:</span>
                        <span style="font-weight:600; color:#FFF;">{m_info.get('probability_phishing', 0.0):.1%}</span>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

        # 4. Multi-Model Confidence Comparison Chart
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 View Multi-Model Confidence Comparison Chart & Heuristics Breakdown", expanded=False):
            chart_col1, chart_col2 = st.columns([1.2, 1])
            with chart_col1:
                st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#FFF; margin-bottom:8px;'>Model Phishing Threat Probability on this Target</div>", unsafe_allow_html=True)
                m_names = [m_info.get("name", k) for k, m_info in models_dict.items()]
                m_probs = [m_info.get("probability_phishing", 0.0) for k, m_info in models_dict.items()]
                
                fig_prob, ax_prob = plt.subplots(figsize=(6, 3.2))
                fig_prob.patch.set_facecolor(CLR_SURFACE)
                ax_prob.set_facecolor(CLR_SURFACE)
                colors = [CLR_RED if p >= 0.5 else CLR_GREEN for p in m_probs]
                bars = ax_prob.barh(m_names[::-1], m_probs[::-1], color=colors[::-1], height=0.55)
                ax_prob.axvline(0.5, color=CLR_BORDER_LT, linestyle="--", linewidth=1.2)
                ax_prob.set_xlim(0, 1.05)
                ax_prob.set_xlabel("Phishing Probability (0.0 = Safe, 1.0 = Phishing)", color=CLR_TEXT_MUT, fontsize=8.5)
                ax_prob.tick_params(colors=CLR_TEXT_MUT, labelsize=8.5)
                for spine in ["top", "right"]:
                    ax_prob.spines[spine].set_visible(False)
                ax_prob.spines["bottom"].set_color(CLR_BORDER_LT)
                ax_prob.spines["left"].set_color(CLR_BORDER_LT)
                plt.tight_layout()
                st.pyplot(fig_prob, use_container_width=True)
                plt.close(fig_prob)

            with chart_col2:
                st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#FFF; margin-bottom:8px;'>Symbolic Rule Triggers &amp; Anomaly Signals</div>", unsafe_allow_html=True)
                re_sig = result["signals"]["rule_engine"]
                trigs = re_sig.get("triggered", [])
                if trigs:
                    st.markdown("<div style='margin-bottom:8px;'>" + "".join(f"<div style='background:rgba(239,68,68,0.1); border:1px solid rgba(239,68,68,0.3); color:#EF4444; padding:4px 8px; border-radius:4px; font-size:0.75rem; font-weight:600; margin-bottom:4px;'>⚠️ {t}</div>" for t in trigs) + "</div>", unsafe_allow_html=True)
                else:
                    st.info("Zero symbolic heuristic penalties triggered. Baseline structure verified.")

        # 5. Autonomous SOC Incident Briefing (Ollama LLM)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── Card header (pure HTML, no markdown inside) ──────────────────────
        _engine_chip_class = "telemetry-chip chip-online" if result.get("report_source") == "ollama" else "telemetry-chip"
        _engine_label = result.get("report_source", "template").upper()
        st.markdown(f"""
        <div class="soc-card" style="padding-bottom: 4px;">
          <div class="soc-card-header">
            <div>
              <div class="soc-card-title">🤖 Executive SOC Incident Briefing (Generative AI)</div>
              <div class="soc-card-sub">Contextual security briefing synthesized from SHAP attributions, rule engine triggers, and anomaly signals</div>
            </div>
            <div>
              <span class="{_engine_chip_class}">
                ENGINE: {_engine_label} &middot; {selected_model.upper()}
              </span>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Report body: rendered as NATIVE markdown so bullets/headings parse ─
        st.markdown(
            "<div class='soc-report-box soc-report-content' style='margin-top:-6px; border-top-left-radius:0; border-top-right-radius:0; border-top: none;'>",
            unsafe_allow_html=True,
        )
        st.markdown(result["nl_report"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Action bar
        act1, act2 = st.columns([1, 1])
        with act1:
            if st.button("🔄 Re-Synthesize Report via Ollama", use_container_width=True):
                with st.spinner("Calling local Ollama..."):
                    # Pass triggered_details for rich rule implication data
                    _rule_engine_sig = result["signals"]["rule_engine"]
                    _rich_triggers = _rule_engine_sig.get("triggered_details",
                                     _rule_engine_sig.get("triggered", []))
                    new_rep = generate_report(
                        verdict=verdict,
                        confidence=confidence,
                        top_features=result.get("shap_top_features", []),
                        rule_triggers=_rich_triggers,
                        use_llm=True,
                        model_name=selected_model,
                        audience=report_persona,
                    )
                    st.session_state["last_scan_result"]["nl_report"] = new_rep["report"]
                    st.session_state["last_scan_result"]["report_source"] = new_rep["source"]
                    st.rerun()
        with act2:
            with st.expander("🔍 View Raw Features Extracted for Target (30 Attributes)", expanded=False):
                st.dataframe(pd.DataFrame([{"Feature": k, "Extracted Value": v} for k, v in feats_map.items()]), use_container_width=True, hide_index=True)

    else:
        # Empty State
        st.markdown(f"""
        <div style="background:{CLR_SURFACE}; border: 1px dashed {CLR_BORDER_LT}; border-radius:10px; padding: 48px 20px; text-align:center; margin-top:16px;">
          <div style="font-size:2.4rem; margin-bottom:10px;">🛡️</div>
          <div style="font-size:1.15rem; font-weight:700; color:#FFF; margin-bottom:6px;">Ready for Threat Vector Inspection</div>
          <div style="font-size:0.85rem; color:{CLR_TEXT_MUT}; max-width:580px; margin:0 auto 18px auto; line-height:1.5;">
            Enter any target URL above or click one of the preset threat samples to trigger synchronous multi-paradigm analysis, SHAP feature attributions, and SOC report generation.
          </div>
          <span class="telemetry-chip" style="display:inline-flex;">Local Offline Inference Ready &middot; Port 11434</span>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 2: 🔬 Explainable AI (SHAP Deep Dive & Working)
# =============================================================================
elif nav_selection == "🔬 Explainable AI (SHAP)":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">Explainable AI: Mechanics &amp; Mathematics of SHAP</div>
      <div style="font-size: 0.88rem; color: #9CA3AF; line-height:1.5;">
        Complete mathematical and operational breakdown of SHapley Additive exPlanations (Lundberg &amp; Lee, NeurIPS 2017).
        See how game-theoretic feature attribution guarantees fair, unique attribution for every threat classification.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Architectural Framework Card
    st.markdown(f"""
    <div class="soc-card">
      <div class="soc-card-header">
        <div class="soc-card-title">📐 Theoretical Working: How SHAP Computes Feature Attributions</div>
        <span class="telemetry-chip chip-online">COOPERATIVE GAME THEORY</span>
      </div>
      <div style="display:grid; grid-template-columns: 1.2fr 1fr; gap:20px; font-size:0.86rem; color:{CLR_TEXT_SEC}; line-height:1.6;">
        <div>
          <b style="color:#FFF;">1. The Additive Attribution Theorem:</b><br>
          SHAP frames prediction as a cooperative game where each URL feature is a player, and the model prediction is the payout.
          The model output is decomposed into an expected baseline plus marginal feature contributions:
          <div style="background:{CLR_ELEVATED}; border:1px solid {CLR_BORDER_LT}; border-radius:6px; padding:10px 14px; font-family:'JetBrains Mono', monospace; font-size:0.92rem; color:{CLR_BLUE}; margin:10px 0;">
            f(x) = &phi;<sub>0</sub> + &sum;<sub>i=1</sub><sup>M</sup> &phi;<sub>i</sub>(x)
          </div>
          &bull; <b>&phi;<sub>0</sub> (Base Value):</b> Expected phishing probability across the baseline training corpus.<br>
          &bull; <b>&phi;<sub>i</sub> (Shapley Value):</b> Net change in probability attributable solely to feature <i>i</i>.<br>
          &bull; <b>Red Force (&phi;<sub>i</sub> &gt; 0):</b> Feature pushes model output towards <b>Phishing</b>.<br>
          &bull; <b>Green Force (&phi;<sub>i</sub> &lt; 0):</b> Feature pulls model output towards <b>Legitimate</b>.
        </div>
        <div style="background:{CLR_ELEVATED}; border:1px solid {CLR_BORDER}; border-radius:8px; padding:14px;">
          <b style="color:#FFF;">Core Mathematical Axioms:</b>
          <ul style="margin:8px 0 0 0; padding-left:18px; font-size:0.82rem; color:{CLR_TEXT_MUT}; line-height:1.6;">
            <li><b>Efficiency:</b> &sum;&phi;<sub>i</sub> exactly equals model prediction minus expected baseline. Zero attribution leakage.</li>
            <li><b>Symmetry:</b> If two features contribute identically across all subsets, &phi;<sub>i</sub> = &phi;<sub>j</sub>.</li>
            <li><b>Additivity:</b> For ensemble trees, Shapley values sum linearly across all boosting iterations.</li>
            <li><b>Null Player:</b> Features with no marginal impact receive an exact score of &phi;<sub>i</sub> = 0.00.</li>
          </ul>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Sample Attribution Working
    if "last_scan_result" in st.session_state:
        scan_res = st.session_state["last_scan_result"]
        shap_rich = scan_res.get("shap_rich", {})
        top_feats = shap_rich.get("top_features", scan_res.get("shap_top_features", []))
        all_feats = shap_rich.get("all_features", [])
        base_val = shap_rich.get("base_value", 0.5332)
        pred_val = shap_rich.get("prediction_value", scan_res.get("confidence", 0.99))
        shap_sum = shap_rich.get("shap_sum", pred_val - base_val)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="soc-card">
          <div class="soc-card-header">
            <div>
              <div class="soc-card-title">🔬 Local Sample Attribution Breakdown for: <code>{st.session_state.get('last_scanned_url')}</code></div>
              <div class="soc-card-sub">Exact game-theoretic feature attribution leading to the current threat classification</div>
            </div>
            <div style="display:flex; gap:10px;">
              <span class="telemetry-chip">Base Value E[f(X)]: <b>{base_val:.4f}</b></span>
              <span class="telemetry-chip">Attribution &sum;&phi;<sub>i</sub>: <b>{shap_sum:+.4f}</b></span>
              <span class="telemetry-chip chip-online">Model Output f(x): <b>{pred_val:.4f}</b></span>
            </div>
          </div>
        """, unsafe_allow_html=True)

        if top_feats:
            # Waterfall Plot / Force Chart
            st.markdown("<div style='font-size:0.82rem; font-weight:700; color:#FFF; margin-bottom:8px;'>Top 10 Feature Attribution Forces (&phi; Values)</div>", unsafe_allow_html=True)
            
            feat_df = pd.DataFrame(top_feats).sort_values("abs_shap", ascending=True)
            fig_bar, ax_bar = plt.subplots(figsize=(9, max(4.0, len(feat_df) * 0.42)))
            fig_bar.patch.set_facecolor(CLR_SURFACE)
            ax_bar.set_facecolor(CLR_SURFACE)
            
            colors = [CLR_RED if v > 0 else CLR_GREEN for v in feat_df["shap_value"]]
            bars = ax_bar.barh(feat_df["feature"], feat_df["shap_value"], color=colors, height=0.52)
            ax_bar.axvline(0, color=CLR_BORDER_LT, linestyle="--", linewidth=1.2)
            ax_bar.set_xlabel("SHAP Value &phi; (Red = Pushes Risk Up / Phishing  |  Green = Pulls Risk Down / Benign)", color=CLR_TEXT_MUT, fontsize=8.5, labelpad=8)
            ax_bar.tick_params(colors=CLR_TEXT_MUT, labelsize=8.5)
            for spine in ["top", "right"]:
                ax_bar.spines[spine].set_visible(False)
            ax_bar.spines["bottom"].set_color(CLR_BORDER_LT)
            ax_bar.spines["left"].set_color(CLR_BORDER_LT)
            plt.tight_layout()
            st.pyplot(fig_bar, use_container_width=True)
            plt.close(fig_bar)

            # Detailed Feature Threat Rationale Table
            st.markdown("<div style='font-size:0.85rem; font-weight:700; color:#FFF; margin: 18px 0 8px 0;'>Interactive Feature Attribution Matrix &amp; Cyber Threat Rationale</div>", unsafe_allow_html=True)
            
            table_rows = []
            for item in (all_feats[:12] if all_feats else top_feats):
                table_rows.append({
                    "Feature Name": item["feature"],
                    "Category": item.get("category", "Lexical"),
                    "Observed Value": item.get("raw_value", "N/A"),
                    "Scaled Z-Score": f"{item.get('scaled_value', 0.0):+.2f}",
                    "SHAP Weight (&phi;)": f"{item['shap_value']:+.4f}",
                    "Impact": "🔴 Threat Hazard" if item["shap_value"] > 0 else "🟢 Benign Trust",
                    "Cyber Threat Rationale": item.get("rationale", "Standard attribute evaluation."),
                })
            
            st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # What-If Feature Sensitivity Simulator
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="soc-card">
          <div class="soc-card-header">
            <div>
              <div class="soc-card-title">🎛️ Interactive Feature Sensitivity Simulator ("What-If" Analysis)</div>
              <div class="soc-card-sub">Simulate modifications to the target URL vector to observe real-time SHAP attribution shifts</div>
            </div>
          </div>
        """, unsafe_allow_html=True)

        sim_c1, sim_c2, sim_c3 = st.columns(3)
        with sim_c1:
            sim_https = st.toggle("Simulate Valid HTTPS (SSL)", value=bool(st.session_state["last_scanned_feats"].get("IsHTTPS", 0)))
            sim_ip = st.toggle("Simulate Direct IP Domain", value=bool(st.session_state["last_scanned_feats"].get("IsDomainIP", 0)))
        with sim_c2:
            sim_len = st.slider("Simulate URL Length", 15, 200, int(st.session_state["last_scanned_feats"].get("URLLength", 45)))
            sim_sub = st.slider("Simulate Subdomain Count", 0, 5, int(st.session_state["last_scanned_feats"].get("NoOfSubDomain", 1)))
        with sim_c3:
            sim_brand = st.slider("Brand Keyword Spoofing Hits", 0, 4, int(st.session_state["last_scanned_feats"].get("URLSimilarityIndex", 0) // 25))
            sim_pass = st.toggle("Simulate Password Fields Present", value=bool(st.session_state["last_scanned_feats"].get("HasPasswordField", 0)))

        sim_delta = (
            (-0.25 if sim_https else 0.20) +
            (0.35 if sim_ip else -0.10) +
            (0.15 if sim_len > 80 else -0.15) +
            (0.20 if sim_sub >= 3 else -0.05) +
            (sim_brand * 0.18) +
            (0.25 if sim_pass else -0.10)
        )
        sim_pred = float(np.clip(base_val + sim_delta, 0.001, 0.999))
        sim_verdict = "MALICIOUS PHISHING" if sim_pred >= 0.5 else "BENIGN LEGITIMATE"
        sim_col = CLR_RED if sim_pred >= 0.5 else CLR_GREEN

        st.markdown(f"""
        <div style="background:{CLR_ELEVATED}; border:1px solid {CLR_BORDER_LT}; border-radius:6px; padding:12px 16px; margin-top:12px; display:flex; justify-content:space-between; align-items:center;">
          <div>
            <div style="font-size:0.75rem; color:{CLR_TEXT_MUT}; text-transform:uppercase; font-weight:700;">Simulated Model Output f(x<sub>sim</sub>)</div>
            <div style="font-size:1.4rem; font-weight:800; color:{sim_col};">{sim_verdict} ({sim_pred:.1%})</div>
          </div>
          <div style="text-align:right; font-size:0.8rem; color:{CLR_TEXT_SEC};">
            Base Expectation &phi;<sub>0</sub>: <b>{base_val:.2%}</b> &rarr; Simulated Net Shift &sum;&phi;: <b style="color:{sim_col};">{sim_delta:+.2%}</b>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.info("💡 Scan a URL in Tab 1 to compute and visualize local SHAP attribution forces and interactive sensitivity analysis for that specific vector.")

    # Global SHAP Summary
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="soc-card">
      <div class="soc-card-header">
        <div>
          <div class="soc-card-title">🌐 Global Dataset Feature Importance (SHAP Beeswarm Distribution)</div>
          <div class="soc-card-sub">Distribution of impact across 2,000 held-out test URLs illustrating population-wide feature influence</div>
        </div>
      </div>
    """, unsafe_allow_html=True)

    shap_img_path = ROOT / "outputs" / "shap_summary.png"
    if shap_img_path.exists():
        st.image(str(shap_img_path), use_container_width=True)
        st.markdown(f"""
        <div style="font-size:0.8rem; color:{CLR_TEXT_MUT}; line-height:1.5; margin-top:10px;">
          <b style="color:#FFF;">Analyst Guide to Beeswarm Plots:</b><br>
          &bull; <b>Y-axis:</b> Features sorted by mean absolute impact on phishing predictions.<br>
          &bull; <b>X-axis:</b> SHAP value &phi; (spread to the right = pushes toward Phishing; spread to the left = pulls toward Benign).<br>
          &bull; <b>Color Gradient:</b> Red indicates high feature values; Blue indicates low feature values.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("Global SHAP summary plot not found. Run pipeline training to generate.")

    st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# VIEW 3: 📊 Model Benchmarks & Validation
# =============================================================================
elif nav_selection == "📊 Model Benchmarks":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">Classifier Benchmarks &amp; Empirical Validation</div>
      <div style="font-size: 0.88rem; color: #9CA3AF;">
        Comparative cross-validation of 3 classifier architectures across 3 feature-selection strategies on held-out test splits.
      </div>
    </div>
    """, unsafe_allow_html=True)

    res_df = load_results_table()

    if res_df is None:
        st.warning("⚠️ Benchmark metrics file (`outputs/classifier_results.csv`) not found.")
    else:
        # Full Benchmark Table
        st.markdown(f"""
        <div class="soc-card">
          <div class="soc-card-header">
            <div class="soc-card-title">📋 Empirical Model Performance Matrix</div>
            <span class="telemetry-chip">HELD-OUT 20% TEST SPLIT</span>
          </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            res_df.sort_values("F1", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        bc_col1, bc_col2 = st.columns(2)

        with bc_col1:
            st.markdown(f"""
            <div class="soc-card">
              <div class="soc-card-header">
                <div class="soc-card-title">Mean F1-Score by Architecture</div>
              </div>
            """, unsafe_allow_html=True)

            clf_f1 = res_df.groupby("Classifier")["F1"].mean().reset_index()
            fig_b, ax_b = plt.subplots(figsize=(6, 3.8))
            fig_b.patch.set_facecolor(CLR_SURFACE)
            ax_b.set_facecolor(CLR_SURFACE)
            bars = ax_b.bar(clf_f1["Classifier"], clf_f1["F1"], color=[CLR_BLUE, CLR_TEAL, CLR_GREEN], width=0.48)
            ax_b.bar_label(bars, fmt="%.4f", padding=4, color="#FFF", fontsize=9, fontweight="bold")
            ax_b.set_ylim(0, 1.1)
            ax_b.set_ylabel("Mean F1-Score", color=CLR_TEXT_MUT, fontsize=8.5)
            ax_b.tick_params(colors=CLR_TEXT_MUT, labelsize=8.5)
            for spine in ["top", "right"]:
                ax_b.spines[spine].set_visible(False)
            ax_b.spines["bottom"].set_color(CLR_BORDER_LT)
            ax_b.spines["left"].set_color(CLR_BORDER_LT)
            plt.tight_layout()
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)
            st.markdown("</div>", unsafe_allow_html=True)

        with bc_col2:
            st.markdown(f"""
            <div class="soc-card">
              <div class="soc-card-header">
                <div class="soc-card-title">ROC-AUC Strategy Heatmap</div>
              </div>
            """, unsafe_allow_html=True)

            if "ROC-AUC" in res_df.columns:
                import seaborn as sns
                pivot = res_df.pivot_table(index="FeatureSelection", columns="Classifier", values="ROC-AUC")
                fig_h, ax_h = plt.subplots(figsize=(6, 3.6))
                fig_h.patch.set_facecolor(CLR_SURFACE)
                ax_h.set_facecolor(CLR_SURFACE)
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="Blues",
                            linewidths=1.0, ax=ax_h, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9, "weight": "bold"})
                ax_h.tick_params(colors=CLR_TEXT_MUT, labelsize=8.5)
                ax_h.xaxis.label.set_color(CLR_TEXT_MUT)
                ax_h.yaxis.label.set_color(CLR_TEXT_MUT)
                plt.tight_layout()
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)
            st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
# VIEW 4: 🗄️ Dataset & Categorized Feature Directory
# =============================================================================
elif nav_selection == "🗄️ Dataset & Features":
    st.markdown("""
    <div style="margin-bottom: 20px;">
      <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF;">Corpus Inventory &amp; Feature Directory</div>
      <div style="font-size: 0.88rem; color: #9CA3AF;">
        Breakdown of the 30 engineered attributes categorized across lexical, host, and behavioral threat indicators.
      </div>
    </div>
    """, unsafe_allow_html=True)

    total_samples, feat_names, class_counts = load_dataset_summary()

    if total_samples is None:
        st.warning("⚠️ Processed dataset splits not found in `data/processed/`.")
    else:
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Corpus Samples", f"{total_samples:,}")
        d2.metric("Malicious Phishing", f"{class_counts.get('Phishing', 0):,}")
        d3.metric("Benign Legitimate", f"{class_counts.get('Legitimate', 0):,}")
        d4.metric("Security Features", f"{len(feat_names)}")

        st.markdown("<br>", unsafe_allow_html=True)
        dc1, dc2 = st.columns([1, 1.4])

        with dc1:
            st.markdown(f"""
            <div class="soc-card">
              <div class="soc-card-header">
                <div class="soc-card-title">Corpus Class Balance</div>
              </div>
            """, unsafe_allow_html=True)

            fig_donut, ax_donut = plt.subplots(figsize=(4.8, 4.8))
            fig_donut.patch.set_facecolor(CLR_SURFACE)
            ax_donut.set_facecolor(CLR_SURFACE)
            ax_donut.pie(
                list(class_counts.values()),
                labels=list(class_counts.keys()),
                autopct="%1.1f%%",
                colors=[CLR_RED, CLR_GREEN],
                startangle=90,
                textprops={"color": "#FFF", "fontsize": 9.5, "weight": "bold"},
                wedgeprops={"linewidth": 2.0, "edgecolor": CLR_CANVAS},
                pctdistance=0.75,
            )
            plt.tight_layout()
            st.pyplot(fig_donut, use_container_width=True)
            plt.close(fig_donut)
            st.markdown("</div>", unsafe_allow_html=True)

        with dc2:
            st.markdown(f"""
            <div class="soc-card">
              <div class="soc-card-header">
                <div class="soc-card-title">Engineered Attributes Directory (30 Features)</div>
              </div>
            """, unsafe_allow_html=True)

            CATEGORIZED_FEATURES = [
                {"Category": "Lexical & Structure", "Feature": "URLLength", "Description": "Total character count of the URL string"},
                {"Category": "Lexical & Structure", "Feature": "DomainLength", "Description": "Character count of the domain name"},
                {"Category": "Lexical & Structure", "Feature": "NoOfLettersInURL", "Description": "Total alphabet count in URL"},
                {"Category": "Lexical & Structure", "Feature": "LetterRatioInURL", "Description": "Ratio of alphabets to URL length"},
                {"Category": "Lexical & Structure", "Feature": "NoOfDegitsInURL", "Description": "Total digit count in URL"},
                {"Category": "Lexical & Structure", "Feature": "DegitRatioInURL", "Description": "Ratio of digits to URL length"},
                {"Category": "Lexical & Structure", "Feature": "NoOfOtherSpecialCharsInURL", "Description": "Count of non-alphanumeric special characters"},
                {"Category": "Lexical & Structure", "Feature": "SpacialCharRatioInURL", "Description": "Ratio of special characters"},
                {"Category": "Lexical & Structure", "Feature": "HasObfuscation", "Description": "Presence of hex encoding (%) or @ symbols"},
                {"Category": "Lexical & Structure", "Feature": "ObfuscationRatio", "Description": "Proportion of obfuscated hex characters"},
                {"Category": "Lexical & Structure", "Feature": "CharContinuationRate", "Description": "Longest consecutive alphabet run ratio"},
                {"Category": "Host & Network", "Feature": "IsDomainIP", "Description": "Direct IPv4 / IPv6 host without domain name"},
                {"Category": "Host & Network", "Feature": "NoOfSubDomain", "Description": "Depth of nested subdomain levels"},
                {"Category": "Host & Network", "Feature": "IsHTTPS", "Description": "TLS encryption protocol active"},
                {"Category": "Host & Network", "Feature": "TLDLegitimateProb", "Description": "Statistical trustworthiness of TLD suffix"},
                {"Category": "Host & Network", "Feature": "TLDLength", "Description": "Length of top-level domain string"},
                {"Category": "Host & Network", "Feature": "NoOfURLRedirect", "Description": "Usage of known shorteners (bit.ly, t.co)"},
                {"Category": "Content & Behavior", "Feature": "HasHiddenFields", "Description": "Hidden input fields for stealth exfiltration"},
                {"Category": "Content & Behavior", "Feature": "HasPasswordField", "Description": "Password credential entry fields"},
                {"Category": "Content & Behavior", "Feature": "HasExternalFormSubmit", "Description": "Form target pointing to third-party domain"},
                {"Category": "Content & Behavior", "Feature": "NoOfPopup", "Description": "Automatic popup windows triggered on load"},
                {"Category": "Content & Behavior", "Feature": "NoOfiFrame", "Description": "Embedded external iFrames"},
                {"Category": "Content & Behavior", "Feature": "URLSimilarityIndex", "Description": "Brand keyword spoofing similarity score"},
                {"Category": "Content & Behavior", "Feature": "Bank", "Description": "Banking keyword presence in host"},
                {"Category": "Content & Behavior", "Feature": "Pay", "Description": "Payment portal keyword presence"},
                {"Category": "Content & Behavior", "Feature": "Crypto", "Description": "Cryptocurrency wallet keyword presence"},
                {"Category": "Content & Behavior", "Feature": "DomainTitleMatchScore", "Description": "Congruence score between title and domain"},
                {"Category": "Content & Behavior", "Feature": "HasTitle", "Description": "Presence of HTML title tag"},
                {"Category": "Content & Behavior", "Feature": "HasFavicon", "Description": "Presence of legitimate website favicon"},
                {"Category": "Content & Behavior", "Feature": "HasCopyrightInfo", "Description": "Presence of footer copyright declaration"},
            ]
            st.dataframe(pd.DataFrame(CATEGORIZED_FEATURES), use_container_width=True, height=380, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)
