"""
llm_explainer.py
================
Generative AI component: uses a locally running Ollama model to produce
a plain-English security report for each URL prediction.

Falls back gracefully to a deterministic template if Ollama is unavailable,
so the pipeline NEVER crashes due to LLM connectivity issues.

CONFIGURATION
-------------
Change OLLAMA_MODEL to any model you have pulled locally, e.g.:
  ollama pull phi3
  ollama pull llama3.2:3b
"""

import pathlib
import textwrap
import time

# ---------------------------------------------------------------------------
# Configuration - change model name here
# ---------------------------------------------------------------------------
OLLAMA_MODEL    = "phi3"           # or "llama3.2:3b", "mistral", etc.
OLLAMA_TIMEOUT  = 30               # seconds before giving up on Ollama
MAX_RETRIES     = 1                # number of retry attempts
# ---------------------------------------------------------------------------

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _template_report(
    verdict:        str,
    confidence:     float,
    top_features:   list[dict],
    rule_triggers:  list[str],
) -> str:
    """Deterministic executive security report when Ollama is unavailable."""
    pct         = round(confidence * 100, 1)
    is_phish    = verdict.lower() == "phishing"
    verdict_str = "MALICIOUS PHISHING THREAT" if is_phish else "VERIFIED BENIGN / LEGITIMATE"
    
    feat_bullets = "\n".join(
        f"• **{f['feature']}**: Feature attribution score {f['shap_value']:+.3f} "
        f"({'elevates risk' if f['shap_value'] > 0 else 'supports safety'})"
        for f in (top_features or [])[:4]
    ) or "• No anomalous feature distortions identified."

    rule_bullets = "\n".join(f"• {r}" for r in rule_triggers[:4]) if rule_triggers else "• None triggered (standard behavioral baseline)."

    if is_phish:
        action_text = (
            "1. **Isolate Vector:** Block this domain at the firewall / DNS resolver immediately.\n"
            "2. **User Safety:** Do not click, input authentication tokens, or submit personal credentials.\n"
            "3. **SOC Action:** Purge inbound email gateways for matching URLs and inspect access logs."
        )
    else:
        action_text = (
            "1. **Standard Hygiene:** Domain exhibits nominal lexical and structural properties.\n"
            "2. **Continuous Monitoring:** Exercise standard organizational browser caution."
        )

    return (
        f"### 🛡️ Executive Threat Summary\n"
        f"The multi-paradigm AI pipeline identified this target as **{verdict_str}** with **{pct}% confidence**.\n\n"
        f"#### ⚠️ Primary Attack Indicators & Feature Shifts\n"
        f"{feat_bullets}\n\n"
        f"#### 📏 Heuristic Rule Violations\n"
        f"{rule_bullets}\n\n"
        f"#### 📋 Recommended Security Actions\n"
        f"{action_text}"
    )


def check_ollama_status(host: str = "http://localhost:11434") -> dict:
    """Check if Ollama server is reachable and query available local models."""
    import urllib.request
    import json
    clean_host = host.rstrip("/")
    try:
        req = urllib.request.Request(f"{clean_host}/api/tags", headers={"User-Agent": "PhishShield/1.0"})
        with urllib.request.urlopen(req, timeout=2.5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = [m.get("name") for m in data.get("models", [])]
            return {"online": True, "models": models, "host": clean_host, "error": None}
    except Exception as exc:
        return {"online": False, "models": [], "host": clean_host, "error": str(exc)}


def _call_ollama(prompt: str, model: str = OLLAMA_MODEL, host: str = "http://localhost:11434", timeout: int = OLLAMA_TIMEOUT) -> str:
    """
    Call the local Ollama API via python package or standard library HTTP REST.
    """
    clean_host = host.rstrip("/")
    # Attempt 1: standard library HTTP request (zero dependencies required)
    import json
    import urllib.request
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 240, "temperature": 0.25}
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{clean_host}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json", "User-Agent": "PhishShield/1.0"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            res_data = json.loads(resp.read().decode("utf-8"))
            text = res_data.get("response", "").strip()
            if text:
                return text
    except Exception as http_err:
        # Attempt 2: fallback to ollama package if installed
        try:
            import ollama
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 240, "temperature": 0.25},
            )
            return response["message"]["content"].strip()
        except Exception:
            raise http_err


def _build_prompt(
    verdict:       str,
    confidence:    float,
    top_features:  list[dict],
    rule_triggers: list[str],
    audience:      str = "general",
) -> str:
    """Construct the structured prompt sent to the LLM."""
    pct       = round(confidence * 100, 1)
    feat_list = "\n".join(
        f"  - {f['feature']}: impact {f['shap_value']:+.4f} "
        f"({'increases phishing risk' if f['shap_value'] > 0 else 'supports legitimate safety'})"
        for f in (top_features or [])[:5]
    )
    rule_list = ("; ".join(rule_triggers) if rule_triggers else "None")

    audience_directive = (
        "Write as a Senior SOC Analyst detailing technical vectors, protocol anomalies, and SIEM containment."
        if audience == "technical"
        else "Write a clear, structured incident advisory accessible to non-technical business users."
    )

    return textwrap.dedent(f"""
        You are PhishShield AI, an autonomous cybersecurity incident response intelligence engine.
        A target URL was inspected with the following forensic findings:

        Verdict: {verdict.upper()}
        Confidence Score: {pct}%

        Top AI-identified Risk Factors (SHAP feature attribution):
        {feat_list}

        Rule-Based Heuristic Triggers:
        {rule_list}

        Tone / Directive:
        {audience_directive}

        CRITICAL: Format your response using markdown with the following three distinct sections:
        ### 🛡️ Incident Summary
        (A 2-3 sentence executive synopsis stating verdict, confidence, and primary danger level)

        #### ⚠️ Detected Attack Vectors
        (3 bullet points highlighting why this domain is suspicious or clean based on the features above)

        #### 📋 Recommended Security Actions
        (2-3 numbered concrete actionable safety or containment steps)
    """).strip()


def generate_report(
    verdict:       str,
    confidence:    float,
    top_features:  list[dict],
    rule_triggers: list[str],
    use_llm:       bool = True,
    model_name:    str = OLLAMA_MODEL,
    host:          str = "http://localhost:11434",
    audience:      str = "general",
) -> dict:
    """
    Generate a natural-language security report.

    Returns
    -------
    dict with keys:
      "report"  : str  - the report text
      "source"  : "ollama" | "template"  - which generator was used
      "model"   : str  - model name (or "template")
      "error"   : str | None
    """
    if not use_llm:
        return {
            "report": _template_report(verdict, confidence, top_features, rule_triggers),
            "source": "template",
            "model" : "template",
            "error" : None,
        }

    prompt = _build_prompt(verdict, confidence, top_features, rule_triggers, audience=audience)
    error_msg = None

    for attempt in range(MAX_RETRIES + 1):
        try:
            text = _call_ollama(prompt, model=model_name, host=host)
            return {
                "report": text,
                "source": "ollama",
                "model" : model_name,
                "error" : None,
            }
        except Exception as exc:
            error_msg = str(exc)
            if attempt < MAX_RETRIES:
                time.sleep(0.5)
            continue

    # All attempts failed -> graceful fallback
    fallback_text = _template_report(verdict, confidence, top_features, rule_triggers)
    return {
        "report": fallback_text,
        "source": "template",
        "model" : "template",
        "error" : error_msg,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo_features = [
        {"feature": "IsHTTPS",        "shap_value": -0.72, "direction": "decreases_phishing_risk"},
        {"feature": "HasHiddenFields","shap_value": +0.55, "direction": "increases_phishing_risk"},
        {"feature": "URLSimilarityIndex","shap_value": -0.48, "direction": "decreases_phishing_risk"},
    ]
    demo_triggers = ["No HTTPS", "Hidden form fields detected", "Low URL similarity"]

    result = generate_report(
        verdict      ="phishing",
        confidence   = 0.87,
        top_features = demo_features,
        rule_triggers= demo_triggers,
    )

    print(f"[llm_explainer] Source: {result['source']}")
    if result["error"]:
        print(f"[llm_explainer] Fallback reason: {result['error']}")
    print(f"\nReport:\n{result['report']}")
    print("\n=== Smoke test passed ===")
