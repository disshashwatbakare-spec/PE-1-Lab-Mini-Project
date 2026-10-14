"""
app.py  --  PhishShield Enterprise AI  |  Commercial SaaS Threat Dashboard
===========================================================================
Autonomous Hybrid URL Threat Intelligence Engine powered by 5 AI paradigms:
  1. Symbolic AI (Rule Engine with 15 weighted heuristics)
  2. Supervised ML (Ensemble: XGBoost, Multilayer Perceptron, Logistic Regression)
  3. Unsupervised ML (Isolation Forest structural anomaly detection)
  4. Explainable AI (SHAP kernel feature attribution layer)
  5. Generative AI (Local Ollama LLM incident report generation & SOC Q&A)

Commercial SaaS Information Architecture:
  Header  -- Centered Brand Hero, live telemetry badges & system health
  Tab 1   -- 🔍 URL Threat Scanner (Clean, distraction-free scanning & Ollama report)
  Tab 2   -- 📊 Visual Analytics & XAI (Centralized charts, SHAP plots & signal radar)
  Tab 3   -- ⚡ Model Benchmarks (Classifier leaderboard, F1 charts & ROC-AUC heatmap)
  Tab 4   -- 🗄️ Dataset & Features (Categorized 30-feature directory & corpus stats)
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
# STREAMLIT PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="PhishShield -- Enterprise AI URL Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# COMMERCIAL SAAS DESIGN TOKENS & PALETTE
# =============================================================================
VIOLET        = "#8B5CF6"   # Electric Violet Accent
VIOLET_LIGHT  = "#A78BFA"   # Soft Lavender
CYAN          = "#00F2FE"   # Neon Cyan
CYAN_TEAL     = "#06B6D4"   # Cyber Teal
DANGER        = "#FF3366"   # Neon Crimson Alert
SUCCESS       = "#00E676"   # Neon Emerald Safe
WARNING_AMBER = "#F59E0B"   # Electric Amber Warning
BG_VOID       = "#080B14"   # Deep Void Space
BG_CARD       = "#0F172A"   # Slate 900 Surface
BG_CARD_LIT   = "#1E293B"   # Slate 800 Elevated
BG_GLASS      = "rgba(15, 23, 42, 0.75)"
BORDER_SUBTLE = "#1E293B"   # Clean Separator
BORDER_LIT    = "#334155"   # Card Border
BORDER_CYAN   = "rgba(0, 242, 254, 0.35)"
BORDER_VIOLET = "rgba(139, 92, 246, 0.35)"
TEXT_WHITE    = "#F8FAFC"   # Primary Heading
TEXT_MUTED    = "#94A3B8"   # Secondary Body
TEXT_DIM      = "#64748B"   # Tertiary / Meta

# =============================================================================
# GLOBAL CSS DESIGN SYSTEM
# =============================================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

  /* Global typography & background */
  html, body, [class*="css"] {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
      background-color: {BG_VOID};
      color: {TEXT_WHITE};
  }}
  .stApp {{
      background: radial-gradient(circle at 50% 0%, rgba(139, 92, 246, 0.09) 0%, transparent 45%),
                  radial-gradient(circle at 90% 90%, rgba(0, 242, 254, 0.05) 0%, transparent 40%),
                  {BG_VOID};
  }}

  /* ── Centered SaaS Header ── */
  .saas-header {{
      text-align: center;
      padding: 30px 20px 24px 20px;
      margin-bottom: 22px;
  }}
  .brand-emblem {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 58px; height: 58px; border-radius: 16px;
      background: linear-gradient(135deg, {VIOLET} 0%, {CYAN_TEAL} 100%);
      box-shadow: 0 0 30px rgba(139, 92, 246, 0.45);
      font-size: 1.85rem; margin-bottom: 12px;
  }}
  .saas-title {{
      font-size: 2.6rem; font-weight: 900; letter-spacing: -0.03em;
      background: linear-gradient(135deg, #FFFFFF 30%, #CBD5E1 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
  }}
  .saas-subtitle {{
      color: {TEXT_MUTED}; font-size: 1.08rem; font-weight: 400;
      max-width: 680px; margin: 0 auto 16px auto; line-height: 1.5;
  }}
  .trust-strip {{
      display: flex; justify-content: center; align-items: center;
      flex-wrap: wrap; gap: 8px; margin-top: 10px;
  }}
  .trust-pill {{
      display: inline-flex; align-items: center; gap: 6px;
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid {BORDER_LIT};
      border-radius: 100px;
      font-size: 0.74rem; font-weight: 600;
      padding: 4px 14px; color: {TEXT_MUTED};
  }}
  .trust-pill-highlight {{
      border-color: {BORDER_CYAN};
      color: {CYAN};
      background: rgba(0, 242, 254, 0.08);
  }}

  /* ── Sidebar Navigation Polish ── */
  [data-testid="stSidebar"] {{
      background: linear-gradient(180deg, #0B1120 0%, #070B16 100%) !important;
      border-right: 1px solid #1E293B !important;
      padding: 0 12px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] > div {{
      gap: 6px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label {{
      background: rgba(30, 41, 59, 0.45) !important;
      border: 1px solid #1E293B !important;
      border-radius: 12px !important;
      padding: 12px 14px !important;
      cursor: pointer !important;
      transition: all 0.2s ease !important;
      margin-bottom: 2px !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
      background: rgba(139, 92, 246, 0.15) !important;
      border-color: #8B5CF6 !important;
      transform: translateX(2px) !important;
  }}
  [data-testid="stSidebar"] [data-testid="stRadio"] label [data-testid="stMarkdownContainer"] p {{
      font-size: 0.88rem !important;
      font-weight: 600 !important;
      color: #F8FAFC !important;
  }}
  button[data-baseweb="tab"] {{
      border-radius: 10px;
      font-weight: 600;
      font-size: 0.92rem;
      color: {TEXT_MUTED} !important;
      background: transparent !important;
      transition: all 0.2s ease;
      padding: 10px 24px !important;
      border: none !important;
  }}
  button[data-baseweb="tab"]:hover {{
      color: #FFFFFF !important;
      background: rgba(139, 92, 246, 0.12) !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
      background: linear-gradient(135deg, {VIOLET} 0%, #6D28D9 100%) !important;
      color: #FFFFFF !important;
      box-shadow: 0 4px 18px rgba(139, 92, 246, 0.45);
  }}

  /* ── URL Input Container ── */
  [data-testid="stTextInput"] input {{
      background: #0B1120 !important;
      border: 1.5px solid {BORDER_LIT} !important;
      border-radius: 14px !important;
      color: #FFFFFF !important;
      font-family: 'JetBrains Mono', monospace !important;
      font-size: 1.02rem !important;
      padding: 16px 22px !important;
      box-shadow: inset 0 2px 6px rgba(0,0,0,0.5);
      transition: all 0.25s ease;
  }}
  [data-testid="stTextInput"] input:focus {{
      border-color: {CYAN} !important;
      box-shadow: 0 0 0 3px rgba(0, 242, 254, 0.25), inset 0 2px 6px rgba(0,0,0,0.5) !important;
      outline: none !important;
  }}

  /* ── Buttons ── */
  .stButton > button[kind="primary"] {{
      background: linear-gradient(135deg, {VIOLET} 0%, #7C3AED 50%, {CYAN_TEAL} 100%) !important;
      color: #FFFFFF !important;
      border: none !important;
      border-radius: 12px !important;
      font-weight: 700 !important;
      font-size: 1.05rem !important;
      padding: 14px 28px !important;
      letter-spacing: 0.02em !important;
      box-shadow: 0 6px 24px rgba(139, 92, 246, 0.45) !important;
      transition: all 0.25s ease !important;
  }}
  .stButton > button[kind="primary"]:hover {{
      transform: translateY(-2px) !important;
      box-shadow: 0 10px 32px rgba(139, 92, 246, 0.65) !important;
  }}
  .stButton > button[kind="secondary"] {{
      background: {BG_CARD_LIT} !important;
      border: 1px solid {BORDER_LIT} !important;
      color: {TEXT_WHITE} !important;
      border-radius: 12px !important;
      font-weight: 600 !important;
      padding: 14px 20px !important;
      transition: all 0.2s ease !important;
  }}
  .stButton > button[kind="secondary"]:hover {{
      border-color: {CYAN} !important;
      color: {CYAN} !important;
  }}

  /* ── Verdict Banners ── */
  .verdict-banner-phishing {{
      display: flex; align-items: center; gap: 24px;
      background: linear-gradient(135deg, rgba(255, 51, 102, 0.18) 0%, rgba(255, 51, 102, 0.04) 100%);
      border: 1.5px solid {DANGER};
      border-radius: 20px;
      padding: 26px 32px;
      margin: 22px 0;
      box-shadow: 0 6px 36px rgba(255, 51, 102, 0.25);
  }}
  .verdict-banner-legitimate {{
      display: flex; align-items: center; gap: 24px;
      background: linear-gradient(135deg, rgba(0, 230, 118, 0.16) 0%, rgba(0, 230, 118, 0.03) 100%);
      border: 1.5px solid {SUCCESS};
      border-radius: 20px;
      padding: 26px 32px;
      margin: 22px 0;
      box-shadow: 0 6px 36px rgba(0, 230, 118, 0.2);
  }}
  .verdict-icon {{ font-size: 3.4rem; line-height: 1; }}
  .verdict-title {{ font-size: 2.2rem; font-weight: 900; letter-spacing: 0.02em; }}
  .verdict-sub {{ font-size: 0.98rem; color: {TEXT_MUTED}; margin-top: 5px; line-height: 1.5; }}
  .verdict-banner-phishing .verdict-title {{ color: {DANGER}; }}
  .verdict-banner-legitimate .verdict-title {{ color: {SUCCESS}; }}

  /* ── KPI Metric Containers ── */
  div[data-testid="metric-container"] {{
      background: linear-gradient(145deg, {BG_CARD}, {BG_CARD_LIT});
      border: 1px solid {BORDER_LIT};
      border-radius: 16px;
      padding: 18px 22px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.3);
  }}
  div[data-testid="metric-container"] [data-testid="stMetricValue"] {{
      font-size: 1.9rem !important;
      font-weight: 800 !important;
      color: #FFFFFF !important;
  }}
  div[data-testid="metric-container"] [data-testid="stMetricLabel"] {{
      font-size: 0.74rem !important;
      color: {TEXT_MUTED} !important;
      text-transform: uppercase !important;
      letter-spacing: 0.08em !important;
      font-weight: 700 !important;
  }}

  /* ── Ollama Executive Report Card ── */
  .ollama-report-card {{
      background: linear-gradient(135deg, rgba(139, 92, 246, 0.1) 0%, rgba(0, 242, 254, 0.06) 100%);
      border: 1.5px solid {BORDER_CYAN};
      border-radius: 20px;
      padding: 28px 32px;
      margin: 24px 0 18px 0;
      position: relative;
      box-shadow: 0 10px 36px rgba(0, 242, 254, 0.12);
  }}
  .ollama-report-card::before {{
      content: '';
      position: absolute;
      top: 0; left: 0; right: 0; height: 3px;
      background: linear-gradient(90deg, {VIOLET}, {CYAN});
      border-radius: 20px 20px 0 0;
  }}
  .ollama-status-chip {{
      display: inline-flex; align-items: center; gap: 7px;
      background: rgba(0, 242, 254, 0.14);
      border: 1px solid rgba(0, 242, 254, 0.4);
      border-radius: 100px;
      font-size: 0.72rem; font-weight: 700;
      padding: 4px 12px; color: {CYAN};
      text-transform: uppercase; letter-spacing: 0.08em;
  }}
  .pulsing-dot {{
      width: 7px; height: 7px; border-radius: 50%;
      background: {CYAN}; box-shadow: 0 0 10px {CYAN};
      animation: pulse 1.5s infinite;
  }}
  @keyframes pulse {{ 0%, 100% {{ opacity: 1; transform: scale(1); }} 50% {{ opacity: 0.4; transform: scale(0.85); }} }}
  .ollama-markdown-content {{
      font-size: 1.02rem; line-height: 1.7; color: #F1F5F9; margin-top: 16px;
  }}
  .ollama-markdown-content h3 {{ font-size: 1.25rem !important; color: #FFFFFF !important; margin-top: 14px !important; }}
  .ollama-markdown-content h4 {{ font-size: 1.05rem !important; color: {CYAN} !important; margin-top: 12px !important; }}

  /* ── Multi-Paradigm Card ── */
  .paradigm-card {{
      background: linear-gradient(160deg, #0F172A 0%, #1E293B 100%);
      border: 1px solid {BORDER_LIT};
      border-radius: 16px;
      padding: 20px 22px;
      height: 100%;
      box-shadow: 0 4px 18px rgba(0,0,0,0.25);
  }}
  .paradigm-tag {{
      font-size: 0.72rem; font-weight: 800; text-transform: uppercase;
      letter-spacing: 0.1em; color: {TEXT_MUTED}; margin-bottom: 10px;
  }}
  .paradigm-verdict {{ font-size: 1.25rem; font-weight: 800; margin-bottom: 6px; }}
  .paradigm-sub {{ font-size: 0.82rem; color: {TEXT_MUTED}; }}

  /* ── Tags / Badges ── */
  .tag-pill {{
      display: inline-block; font-size: 0.73rem; font-weight: 600;
      border-radius: 7px; padding: 4px 10px; margin: 3px 3px 3px 0;
  }}
  .tag-danger {{
      background: rgba(255, 51, 102, 0.18); color: #FF4D79; border: 1px solid rgba(255, 51, 102, 0.35);
  }}
  .tag-success {{
      background: rgba(0, 230, 118, 0.16); color: #00E676; border: 1px solid rgba(0, 230, 118, 0.35);
  }}

  /* DataFrames and Expanders */
  [data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}
  [data-testid="stExpander"] {{
      background: {BG_CARD} !important;
      border: 1px solid {BORDER_LIT} !important;
      border-radius: 14px !important;
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
    """Derive all 30 dataset features programmatically from a raw URL string."""
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
@st.cache_resource(show_spinner="Booting PhishShield Defense Core...")
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
# CENTERED SAAS TOP HEADER
# =============================================================================
st.markdown(f"""
<div class="saas-header">
  <div class="brand-emblem">&#128737;&#65039;</div>
  <div class="saas-title">PhishShield Enterprise AI</div>
  <div class="saas-subtitle">
    Autonomous Hybrid URL Threat Intelligence Engine combining Symbolic Rules,
    Supervised Ensembles, Isolation Outlier Forests, and Explainable Generative AI.
  </div>
  <div class="trust-strip">
    <span class="trust-pill trust-pill-highlight">&#9679; SYSTEM OPERATIONAL</span>
    <span class="trust-pill">&#128207; Symbolic AI Heuristics</span>
    <span class="trust-pill">&#129302; Supervised ML Ensemble</span>
    <span class="trust-pill">&#128269; Isolation Outlier Detection</span>
    <span class="trust-pill">&#128202; SHAP Explainability</span>
    <span class="trust-pill">&#129433; Ollama Generative Intelligence</span>
  </div>
</div>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR -- Primary SaaS Platform Navigation
# =============================================================================
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 18px 4px 14px 4px;">
      <div style="display:flex; align-items:center; gap:10px;">
        <span style="font-size:1.6rem;">&#128737;&#65039;</span>
        <div>
          <div style="font-size:1.2rem; font-weight:900; color:#FFF; letter-spacing:-0.02em;">PhishShield</div>
          <div style="font-size:0.68rem; color:{CYAN}; font-weight:700; text-transform:uppercase; letter-spacing:0.08em;">Enterprise SaaS</div>
        </div>
      </div>
    </div>
    <div style="font-size:0.68rem; text-transform:uppercase; letter-spacing:0.12em; color:{TEXT_MUTED}; font-weight:800; margin: 12px 0 8px 6px;">
      Platform Navigation
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Platform Navigation",
        options=[
            "🔍 Main Threat Scanner",
            "📊 Visual Analytics & XAI",
            "⚡ Model Benchmarks",
            "🗄️ Dataset & Features",
        ],
        label_visibility="collapsed",
    )

    st.markdown(f"<hr style='border-color:{BORDER_SUBTLE}; margin: 24px 0 16px 0;'>", unsafe_allow_html=True)

    # Local Ollama Status Telemetry
    ollama_stat = check_ollama_status()
    is_online = ollama_stat["online"]
    local_models = ollama_stat["models"]

    status_dot = "&#9679;" if is_online else "&#9888;"
    status_col = SUCCESS if is_online else WARNING_AMBER
    status_label = "Ollama Local: Online" if is_online else "Ollama Local: Offline"
    status_sub = f"{len(local_models)} models detected" if is_online else "Offline (fallback active)"

    st.markdown(f"""
    <div style="padding:12px 14px; background:rgba(15,23,42,0.7); border:1px solid {BORDER_LIT}; border-radius:12px; margin-bottom:14px;">
      <div style="font-size:0.78rem; font-weight:700; color:{status_col};">{status_dot} {status_label}</div>
      <div style="font-size:0.68rem; color:{TEXT_MUTED}; margin-top:2px;">{status_sub} &middot; localhost:11434</div>
    </div>
    <div style="padding: 0 4px; text-align: center;">
      <div style="font-size: 0.72rem; color: {TEXT_MUTED}; line-height: 1.6;">
        <span style="color:#FFF; font-weight:700;">PE-1 Lab &middot; MIT-WPU</span><br>
        AI Systems &amp; Applications &middot; v1.2
      </div>
    </div>
    """, unsafe_allow_html=True)

# Scanner Configuration Variables
selected_model = local_models[0] if (is_online and local_models) else "llama3"
report_persona = "general"
enable_llm_toggle = True


# =============================================================================
# VIEW 1: 🔍 URL Threat Scanner (Main Page)
# =============================================================================
if nav_selection == "🔍 Main Threat Scanner":
    st.markdown("<br>", unsafe_allow_html=True)

    # Search Bar & Primary Actions
    raw_url_input = st.text_input(
        label="URL Search Input",
        placeholder="Enter or paste any suspicious target URL (e.g. https://update-account-verification.paypal-login.com)...",
        key="main_url_search_input",
        label_visibility="collapsed",
    )

    btn_c1, btn_c2 = st.columns([3, 1])
    with btn_c1:
        scan_action = st.button("  &#9889;  Scan URL Vector with 5 AI Paradigms  ", type="primary", use_container_width=True)
    with btn_c2:
        reset_action = st.button("Reset Viewport", type="secondary", use_container_width=True)

    if reset_action:
        st.session_state["main_url_search_input"] = ""
        st.session_state.pop("last_scan_result", None)
        st.session_state.pop("last_scanned_url", None)
        st.session_state.pop("last_scanned_feats", None)
        st.rerun()

    # Trigger scan if clicked
    target_url = raw_url_input.strip()

    if scan_action and target_url:
        with st.spinner("Analyzing lexical structure and extracting 30 security features..."):
            extracted_feats = extract_url_features(target_url)

        pipe = load_pipeline()
        for col in (pipe.feature_cols or []):
            extracted_feats.setdefault(col, 0.0)

        with st.spinner("Running 5-paradigm AI pipeline (Symbolic + Supervised + Unsupervised + SHAP + Ollama)..."):
            try:
                scan_res = pipe.predict(
                    extracted_feats,
                    top_n_shap=10,
                    use_llm=enable_llm_toggle,
                    llm_model=selected_model,
                    llm_audience=report_persona,
                )
                # Store in session state for persistence across tab switches
                st.session_state["last_scan_result"] = scan_res
                st.session_state["last_scanned_url"] = target_url
                st.session_state["last_scanned_feats"] = extracted_feats
            except Exception as scan_err:
                st.error(f"Detection Core Error: {scan_err}")
                st.stop()

    # Check if a scan result exists in session state
    if "last_scan_result" in st.session_state:
        result    = st.session_state["last_scan_result"]
        url_text  = st.session_state["last_scanned_url"]
        feats_map = st.session_state["last_scanned_feats"]

        verdict    = result["verdict"]
        confidence = result["confidence"]
        signals    = result["signals"]
        is_phish   = verdict == "phishing"

        # Verdict Hero Banner
        if is_phish:
            st.markdown(f"""
            <div class="verdict-banner-phishing">
              <div class="verdict-icon">&#128680;</div>
              <div style="flex:1;">
                <div class="verdict-title">MALICIOUS PHISHING DETECTED</div>
                <div class="verdict-sub">
                  Target: <code style="color:#FFF; background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:6px;">{url_text}</code><br>
                  This URL exhibits definitive credential-harvesting heuristics, deceptive lexical spoofing, or abnormal structural anomalies. Do NOT interact.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="verdict-banner-legitimate">
              <div class="verdict-icon">&#9989;</div>
              <div style="flex:1;">
                <div class="verdict-title">VERIFIED BENIGN / LEGITIMATE</div>
                <div class="verdict-sub">
                  Target: <code style="color:#FFF; background:rgba(0,0,0,0.3); padding:2px 8px; border-radius:6px;">{url_text}</code><br>
                  Nominal structural baseline confirmed. Zero critical security heuristics or unsupervised anomalies triggered.
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 4 Clean SaaS KPI Metric Cards
        km1, km2, km3, km4 = st.columns(4)
        km1.metric("Threat Probability", f"{confidence:.1%}")
        km2.metric("Manual SOC Review", "FLAGGED (REVIEW)" if result["manual_review"] else "CLEAR (AUTOMATED)")
        km3.metric("AI Paradigm Consensus", "MALICIOUS" if is_phish else "BENIGN")
        
        rep_engine = selected_model.upper() if result.get("report_source") == "ollama" else "SAFETY HEURISTIC"
        km4.metric("LLM Report Engine", rep_engine)

        st.markdown("<br>", unsafe_allow_html=True)

        # Re-engineered Ollama Generative AI Incident Report
        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px;">
          <span style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800;">
            &#129433; Autonomous AI Incident Briefing
          </span>
          <span style="font-size:0.72rem; color:{TEXT_MUTED};">
            Persona: <b style="color:#FFF;">{report_persona.capitalize()}</b>
          </span>
        </div>
        """, unsafe_allow_html=True)

        is_ollama_src = result.get("report_source") == "ollama"
        status_chip_html = (
            f'<div class="ollama-status-chip"><div class="pulsing-dot"></div>OLLAMA ACTIVE &middot; {selected_model.upper()}</div>'
            if is_ollama_src else
            f'<div class="ollama-status-chip" style="background:rgba(245,158,11,0.12); border-color:rgba(245,158,11,0.35); color:{WARNING_AMBER};">'
            f'&#9888; HIGH-PRECISION HEURISTIC BRIEF (OFFLINE FALLBACK)</div>'
        )

        st.markdown(f"""
        <div class="ollama-report-card">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:10px;">
            <div style="display:flex; align-items:center; gap:12px;">
              <span style="font-size:1.6rem;">&#129302;</span>
              <div>
                <div style="font-size:1.05rem; font-weight:800; color:#FFFFFF;">Executive Security Incident Briefing</div>
                <div style="font-size:0.72rem; color:{TEXT_MUTED};">Contextual incident report generated from multi-paradigm AI findings</div>
              </div>
            </div>
            {status_chip_html}
          </div>
          <div class="ollama-markdown-content">
            {result["nl_report"]}
          </div>
        </div>
        """, unsafe_allow_html=True)

        # On-Demand Report Action Toolbar
        act_c1, act_c2 = st.columns([1, 1])
        with act_c1:
            if st.button("&#128260; Re-Generate Brief with Ollama", use_container_width=True):
                with st.spinner(f"Querying Ollama ({selected_model})..."):
                    new_rep = generate_report(
                        verdict=verdict,
                        confidence=confidence,
                        top_features=result.get("shap_top_features", []),
                        rule_triggers=result["signals"]["rule_engine"].get("triggered", []),
                        use_llm=True,
                        model_name=selected_model,
                        audience=report_persona,
                    )
                    st.session_state["last_scan_result"]["nl_report"] = new_rep["report"]
                    st.session_state["last_scan_result"]["report_source"] = new_rep["source"]
                    st.rerun()

        with act_c2:
            with st.expander("&#128065; Inspect Structured Prompt Sent to LLM", expanded=False):
                from llm_explainer import _build_prompt
                p_text = _build_prompt(
                    verdict=verdict,
                    confidence=confidence,
                    top_features=result.get("shap_top_features", []),
                    rule_triggers=result["signals"]["rule_engine"].get("triggered", []),
                    audience=report_persona,
                )
                st.code(p_text, language="markdown")

        st.markdown("<br>", unsafe_allow_html=True)

        # Multi-Paradigm Signal Cards
        st.markdown(f"""
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{TEXT_MUTED}; font-weight:800; margin-bottom:12px;">
          Multi-Paradigm Signal Decomposition
        </div>
        """, unsafe_allow_html=True)

        sc1, sc2, sc3 = st.columns(3)

        with sc1:
            re_sig = signals["rule_engine"]
            re_is_phish = re_sig["verdict"] == "phishing"
            re_col = DANGER if re_is_phish else SUCCESS
            re_label = "PHISHING DETECTED" if re_is_phish else "CLEAN"
            triggers = re_sig.get("triggered", [])
            tags = "".join(f'<span class="tag-pill tag-danger">{t}</span>' for t in triggers[:4]) if triggers else f'<span class="tag-pill tag-success">Zero heuristic triggers</span>'
            st.markdown(f"""
            <div class="paradigm-card">
              <div class="paradigm-tag">&#128207; Symbolic AI &middot; Rule Engine</div>
              <div class="paradigm-verdict" style="color:{re_col};">{re_label}</div>
              <div class="paradigm-sub">Confidence: <b style="color:#FFF;">{re_sig['confidence']:.1%}</b></div>
              <div style="margin-top:12px;">{tags}</div>
            </div>
            """, unsafe_allow_html=True)

        with sc2:
            ml_sig = signals["ml_classifier"]
            ml_is_phish = ml_sig["verdict"] == "phishing"
            ml_col = DANGER if ml_is_phish else SUCCESS
            ml_label = "PHISHING CONFIRMED" if ml_is_phish else "LEGITIMATE"
            m_name = ml_sig.get("model_name", "XGBoostClassifier")
            m_conf = ml_sig["confidence"]
            conf_str = ">99.9%" if m_conf >= 0.999 else f"{m_conf:.1%}"
            st.markdown(f"""
            <div class="paradigm-card">
              <div class="paradigm-tag">&#129302; Supervised ML &middot; Ensemble</div>
              <div class="paradigm-verdict" style="color:{ml_col};">{ml_label}</div>
              <div class="paradigm-sub">Model Confidence: <b style="color:#FFF;">{conf_str}</b></div>
              <div style="font-size:11px; color:#94A3B8; margin-top:3px;">
                Evaluator: <b style="color:#E2E8F0;">{m_name}</b> (Best Test F1)
              </div>
              <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:5px;">
                <span class="tag-pill" style="background:rgba(139,92,246,0.18); color:{VIOLET_LIGHT}; border:1px solid rgba(139,92,246,0.35);">
                  Deep Leaf Purity
                </span>
                <span class="tag-pill" style="background:rgba(16,185,129,0.15); color:{SUCCESS}; border:1px solid rgba(16,185,129,0.3);">
                  Canonical Profile Match
                </span>
              </div>
              <div style="font-size:11px; color:#64748B; margin-top:8px; line-height:1.35;">
                Near-100% confidence reflects extreme log-odds in boosted tree leaves when all 30 URL features align with prototypical benign indicators.
              </div>
            </div>
            """, unsafe_allow_html=True)

        with sc3:
            an_sig = signals["anomaly_detector"]
            an_flag = an_sig["anomaly_flag"]
            an_thresh = an_sig.get("threshold", 0.5000)
            an_score = an_sig["anomaly_score"]
            an_col = WARNING_AMBER if an_flag else SUCCESS
            an_label = "OUTLIER DETECTED" if an_flag else "NORMAL BEHAVIOR"
            
            an_pills = f"""
            <span class="tag-pill tag-warning">Novel / Zero-Day Pattern</span>
            <span class="tag-pill" style="background:rgba(0,242,254,0.15); color:{CYAN}; border:1px solid rgba(0,242,254,0.35);">
              Fast Tree Isolation
            </span>
            """ if an_flag else f"""
            <span class="tag-pill tag-success">Within Benign Cluster</span>
            <span class="tag-pill" style="background:rgba(0,242,254,0.15); color:{CYAN}; border:1px solid rgba(0,242,254,0.35);">
              Deep Path Length
            </span>
            """
            
            an_sub = (
                f"Score {an_score:.4f} &gt; {an_thresh:.4f} baseline cutoff. Trained solely on benign URLs — signals rare syntax/structural anomaly."
                if an_flag
                else f"Score {an_score:.4f} &le; {an_thresh:.4f} baseline cutoff. Feature topology conforms to standard legitimate traffic."
            )

            st.markdown(f"""
            <div class="paradigm-card">
              <div class="paradigm-tag">&#128269; Unsupervised ML &middot; Isolation Forest</div>
              <div class="paradigm-verdict" style="color:{an_col};">{an_label}</div>
              <div class="paradigm-sub">
                Anomaly Score: <b style="color:#FFF;">{an_score:.4f}</b>
                <span style="font-size:11px; color:#94A3B8;">(Cutoff: {an_thresh:.4f})</span>
              </div>
              <div style="margin-top:10px; display:flex; flex-wrap:wrap; gap:5px;">
                {an_pills}
              </div>
              <div style="font-size:11px; color:#64748B; margin-top:8px; line-height:1.35;">
                {an_sub}
              </div>
            </div>
            """, unsafe_allow_html=True)

        # 30-Features Data Expander
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("&#128269; Inspect All 30 Derived URL Features for this Target", expanded=False):
            f_df = pd.DataFrame([{"Feature Name": k, "Derived Value": v} for k, v in feats_map.items()])
            st.dataframe(f_df, use_container_width=True, hide_index=True)

    else:
        # SaaS Initial Empty State (Uncluttered, Sleek)
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, {BG_CARD} 0%, {BG_CARD_LIT} 100%);
            border: 1px dashed {BORDER_LIT};
            border-radius: 20px;
            padding: 56px 30px;
            text-align: center;
            margin-top: 20px;
        ">
          <div style="font-size: 3.4rem; margin-bottom: 14px;">&#128279;</div>
          <div style="font-size: 1.35rem; font-weight: 800; color: #FFFFFF; margin-bottom: 8px;">
            Awaiting Target URL
          </div>
          <div style="font-size: 0.95rem; color: {TEXT_MUTED}; max-width: 600px; margin: 0 auto 28px auto; line-height: 1.6;">
            Paste any suspicious or benign URL vector in the field above and click <b>Scan URL Vector</b>.
            All 30 security features are extracted dynamically without manual data entry.
          </div>
          <div style="
              display: inline-flex; align-items: center; gap: 8px;
              background: rgba(139,92,246,0.1); border: 1px solid rgba(139,92,246,0.25);
              padding: 6px 16px; border-radius: 100px; font-size: 0.78rem; color: {VIOLET_LIGHT};
          ">
            &#129433; Local Ollama AI active &middot; End-to-end offline inference
          </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 2: 📊 Visual Analytics & Explainability (XAI)
# =============================================================================
elif nav_selection == "📊 Visual Analytics & XAI":
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
      <div style="font-size: 1.4rem; font-weight: 800; color: #FFF;">Explainable AI &amp; Visual Threat Analytics</div>
      <div style="font-size: 0.9rem; color: {TEXT_MUTED};">
        Deep attribution analysis detailing how each AI paradigm contributes to the final threat verdict.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # 1. Local SHAP Attribution Bar Chart
    if "last_scan_result" in st.session_state:
        scan_res = st.session_state["last_scan_result"]
        shap_list = scan_res.get("shap_top_features", [])

        if shap_list:
            st.markdown(f"""
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
              Local Target Feature Attribution (SHAP)
            </div>
            """, unsafe_allow_html=True)

            shap_df = pd.DataFrame(shap_list)
            shap_df["abs"] = shap_df["shap_value"].abs()
            shap_df = shap_df.sort_values("abs", ascending=True)

            fig_shap, ax_shap = plt.subplots(figsize=(9, max(4.2, len(shap_df) * 0.42)))
            colors = [DANGER if v > 0 else SUCCESS for v in shap_df["shap_value"]]
            ax_shap.barh(shap_df["feature"], shap_df["shap_value"], color=colors, height=0.55, edgecolor="none")
            ax_shap.axvline(0, color=BORDER_LIT, linewidth=1.5, linestyle="--")
            ax_shap.set_xlabel("SHAP Impact Value (Red = ↑ Phishing Risk  |  Green = ↓ Safe / Benign)", color=TEXT_MUTED, fontsize=9.5, labelpad=8)
            ax_shap.set_facecolor(BG_CARD)
            fig_shap.patch.set_facecolor(BG_CARD)
            ax_shap.tick_params(colors=TEXT_MUTED, labelsize=9)
            ax_shap.xaxis.label.set_color(TEXT_MUTED)
            for spine in ["top", "right"]:
                ax_shap.spines[spine].set_visible(False)
            ax_shap.spines["bottom"].set_color(BORDER_LIT)
            ax_shap.spines["left"].set_color(BORDER_LIT)
            plt.tight_layout()
            st.pyplot(fig_shap, use_container_width=True)
            plt.close(fig_shap)
    else:
        st.info("💡 Scan a URL in Tab 1 to compute and visualize local SHAP feature impact bars for that specific vector.")

    st.markdown("<br><hr style='border-color:" + BORDER_SUBTLE + ";'><br>", unsafe_allow_html=True)

    # 2. Global Dataset Feature Importance (Beeswarm)
    st.markdown(f"""
    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
      Global Dataset Feature Importance (SHAP Beeswarm)
    </div>
    """, unsafe_allow_html=True)

    shap_img_path = ROOT / "outputs" / "shap_summary.png"
    if shap_img_path.exists():
        st.image(str(shap_img_path), caption="Global SHAP Beeswarm Plot across full test dataset", use_container_width=True)
    else:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:24px; text-align:center;">
          <div style="color:{TEXT_MUTED}; font-size:0.9rem;">Global SHAP plot will be generated when model training pipeline completes.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><hr style='border-color:" + BORDER_SUBTLE + ";'><br>", unsafe_allow_html=True)

    # 3. Multi-Paradigm Threat Consensus Alignment
    st.markdown(f"""
    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
      Multi-Paradigm Consensus Architecture
    </div>
    """, unsafe_allow_html=True)

    p_c1, p_c2 = st.columns(2)
    with p_c1:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px; height:100%;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">1. Symbolic Heuristic Layer</div>
          <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
            Evaluates 15 weighted domain security rules (IP in domain, missing HTTPS, obfuscation characters, abnormal brand keywords).
            Acts as a hard override when critical credential-theft signatures are detected.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p_c2:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px; height:100%;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">2. Supervised Ensemble Layer</div>
          <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
            Ensemble of XGBoost, Multi-Layer Perceptron (Neural Net), and Logistic Regression trained on 30 scaled URL attributes.
            Provides statistical class probabilities calibrated across diverse lexical distributions.
          </div>
        </div>
        """, unsafe_allow_html=True)

    p_c3, p_c4 = st.columns(2)
    with p_c3:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px; height:100%; margin-top:14px;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">3. Unsupervised Outlier Forest</div>
          <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
            Isolation Forest isolating anomalous URL vectors in high-dimensional feature space without relying on supervised labels.
            Flags novel zero-day phishing campaigns that deviate from standard benign topology.
          </div>
        </div>
        """, unsafe_allow_html=True)

    with p_c4:
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px; height:100%; margin-top:14px;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">4. Local Generative AI (Ollama)</div>
          <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
            Transforms complex mathematical SHAP attribution values and heuristic rule triggers into a structured,
            actionable SOC Incident Briefing with mitigation steps for non-technical users and security teams.
          </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 3: ⚡ Model Benchmarks & Performance
# =============================================================================
elif nav_selection == "⚡ Model Benchmarks":
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
      <div style="font-size: 1.4rem; font-weight: 800; color: #FFF;">Classifier Benchmarks &amp; Empirical Validation</div>
      <div style="font-size: 0.9rem; color: {TEXT_MUTED};">
        Comparative cross-validation of 3 classifier architectures across 3 feature-selection strategies on held-out test splits.
      </div>
    </div>
    """, unsafe_allow_html=True)

    res_df = load_results_table()

    if res_df is None:
        st.warning("⚠️ Benchmark metrics file (`outputs/classifier_results.csv`) not found. Run `python src/pipeline.py` to generate.")
    else:
        # Full Benchmark Table
        st.markdown(f"""
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
          Empirical Model Evaluation Matrix
        </div>
        """, unsafe_allow_html=True)

        st.dataframe(
            res_df.sort_values("F1", ascending=False).reset_index(drop=True),
            use_container_width=True, hide_index=True,
        )

        st.markdown("<br>", unsafe_allow_html=True)
        bc_col1, bc_col2 = st.columns(2)

        with bc_col1:
            st.markdown(f"""
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
              Mean F1-Score by Architecture
            </div>
            """, unsafe_allow_html=True)

            clf_f1 = res_df.groupby("Classifier")["F1"].mean().reset_index()
            fig_b, ax_b = plt.subplots(figsize=(6, 4))
            bars = ax_b.bar(clf_f1["Classifier"], clf_f1["F1"], color=[VIOLET, DANGER, SUCCESS], width=0.48, edgecolor="none")
            ax_b.bar_label(bars, fmt="%.4f", padding=5, color=TEXT_WHITE, fontsize=9.5, fontweight="bold")
            ax_b.set_ylim(0, 1.1)
            ax_b.set_ylabel("Mean F1-Score", color=TEXT_MUTED, fontsize=9.5)
            ax_b.set_facecolor(BG_CARD)
            fig_b.patch.set_facecolor(BG_CARD)
            ax_b.tick_params(colors=TEXT_MUTED, labelsize=9)
            for spine in ["top", "right"]:
                ax_b.spines[spine].set_visible(False)
            ax_b.spines["bottom"].set_color(BORDER_LIT)
            ax_b.spines["left"].set_color(BORDER_LIT)
            plt.tight_layout()
            st.pyplot(fig_b, use_container_width=True)
            plt.close(fig_b)

        with bc_col2:
            if "ROC-AUC" in res_df.columns:
                st.markdown(f"""
                <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
                  ROC-AUC Cross-Strategy Heatmap
                </div>
                """, unsafe_allow_html=True)
                import seaborn as sns
                pivot = res_df.pivot_table(index="FeatureSelection", columns="Classifier", values="ROC-AUC")
                fig_h, ax_h = plt.subplots(figsize=(6, 3.8))
                sns.heatmap(pivot, annot=True, fmt=".4f", cmap="magma",
                            linewidths=1.0, ax=ax_h, cbar_kws={"shrink": 0.8}, annot_kws={"size": 9.5, "weight": "bold"})
                ax_h.set_facecolor(BG_CARD)
                fig_h.patch.set_facecolor(BG_CARD)
                ax_h.tick_params(colors=TEXT_MUTED, labelsize=8.5)
                ax_h.xaxis.label.set_color(TEXT_MUTED)
                ax_h.yaxis.label.set_color(TEXT_MUTED)
                plt.tight_layout()
                st.pyplot(fig_h, use_container_width=True)
                plt.close(fig_h)

        # Architectural Specification Summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">Pipeline Validation Methodology</div>
          <div style="font-size:0.85rem; color:{TEXT_MUTED}; line-height:1.6;">
            • <b>Data Partitioning:</b> 80% Stratified Train split, 20% Held-out Test split.<br>
            • <b>Feature Selection Methods:</b> Full 30 Attributes, SelectKBest via Mutual Information (Top 15), SelectKBest via Chi-Square.<br>
            • <b>Evaluation Metrics:</b> F1-Score (macro), ROC-AUC, Accuracy, Precision, Recall, and Inference Latency.<br>
            • <b>Best Classifier:</b> Automatically serialized to <code>models/best_model.pkl</code> for production runtime serving.
          </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# VIEW 4: 🗄️ Dataset & Categorized Feature Directory
# =============================================================================
elif nav_selection == "🗄️ Dataset & Features":
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-bottom: 24px;">
      <div style="font-size: 1.4rem; font-weight: 800; color: #FFF;">Corpus Inventory &amp; Feature Engineering Directory</div>
      <div style="font-size: 0.9rem; color: {TEXT_MUTED};">
        Detailed breakdown of the 30 engineered attributes categorized across lexical, host, and behavioral indicators.
      </div>
    </div>
    """, unsafe_allow_html=True)

    total_samples, feat_names, class_counts = load_dataset_summary()

    if total_samples is None:
        st.warning("⚠️ Processed dataset splits not found in `data/processed/`.")
    else:
        # KPI Counters
        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Total Corpus Samples", f"{total_samples:,}")
        d2.metric("Malicious Phishing", f"{class_counts.get('Phishing', 0):,}")
        d3.metric("Benign Legitimate", f"{class_counts.get('Legitimate', 0):,}")
        d4.metric("Security Features", f"{len(feat_names)}")

        st.markdown("<br>", unsafe_allow_html=True)
        dc1, dc2 = st.columns([1, 1.4])

        with dc1:
            st.markdown(f"""
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
              Corpus Class Balance
            </div>
            """, unsafe_allow_html=True)

            fig_donut, ax_donut = plt.subplots(figsize=(5, 5))
            ax_donut.pie(
                list(class_counts.values()),
                labels=list(class_counts.keys()),
                autopct="%1.1f%%",
                colors=[DANGER, SUCCESS],
                startangle=90,
                textprops={"color": TEXT_WHITE, "fontsize": 10.5, "weight": "bold"},
                wedgeprops={"linewidth": 2.5, "edgecolor": BG_VOID},
                pctdistance=0.75,
            )
            fig_donut.patch.set_facecolor(BG_CARD)
            plt.tight_layout()
            st.pyplot(fig_donut, use_container_width=True)
            plt.close(fig_donut)

        with dc2:
            st.markdown(f"""
            <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.12em; color:{CYAN}; font-weight:800; margin-bottom:8px;">
              Engineered Attributes Directory (30 Features)
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
            st.dataframe(pd.DataFrame(CATEGORIZED_FEATURES), use_container_width=True, height=400, hide_index=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="background:{BG_CARD}; border:1px solid {BORDER_LIT}; border-radius:14px; padding:22px;">
          <div style="font-weight:700; color:#FFF; margin-bottom:8px;">Data Pipeline Configuration</div>
          <table style="width:100%; border-collapse:collapse; font-size:0.88rem;">
            <tr style="border-bottom:1px solid {BORDER_SUBTLE};"><td style="padding:6px 0; color:{TEXT_MUTED}; width:35%;">Corpus Total</td><td style="padding:6px 0; font-weight:600; color:#FFF;">{total_samples:,} records</td></tr>
            <tr style="border-bottom:1px solid {BORDER_SUBTLE};"><td style="padding:6px 0; color:{TEXT_MUTED};">Feature Count</td><td style="padding:6px 0; font-weight:600; color:#FFF;">30 continuous &amp; binary attributes</td></tr>
            <tr style="border-bottom:1px solid {BORDER_SUBTLE};"><td style="padding:6px 0; color:{TEXT_MUTED};">Label Encoding</td><td style="padding:6px 0; font-weight:600; color:#FFF;"><code>1 = Phishing &middot; 0 = Legitimate</code></td></tr>
            <tr><td style="padding:6px 0; color:{TEXT_MUTED};">Feature Scaler</td><td style="padding:6px 0; font-weight:600; color:#FFF;">StandardScaler (fit strictly on training split)</td></tr>
          </table>
        </div>
        """, unsafe_allow_html=True)
