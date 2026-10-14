"""
llm_explainer.py
================
Generative AI component: uses a locally running Ollama model to produce
a high-detail, structured executive security incident briefing.

Falls back gracefully to an enriched deterministic template if Ollama is
unavailable, guaranteeing consistent, readable SOC reporting.
"""

import pathlib
import textwrap
import time
import json
import urllib.request

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OLLAMA_MODEL    = "llama3"         # or "phi3", "llama3.2:3b", "mistral"
OLLAMA_TIMEOUT  = 25               # seconds before fallback
MAX_RETRIES     = 1

ROOT = pathlib.Path(__file__).resolve().parents[1]


# =============================================================================
# DEFAULT THREAT IMPLICATIONS FOR COMMON SECURITY FEATURES
# =============================================================================
FALLBACK_INDICATOR_RATIONALES = {
    "URLSimilarityIndex": {
        "cat": "Brand & Identity Spoofing",
        "phish": "Domain string exhibits severe lexical divergence or keyword collision mimicking trusted brands (PayPal, Apple, Google, Microsoft) within an unregistered namespace.",
        "legit": "Authentic lexical brand namespace verified. No typosquatting or trademark spoofing detected.",
    },
    "IsDomainIP": {
        "cat": "Host & Infrastructure",
        "phish": "Direct numerical IPv4/IPv6 host detected without standard DNS domain resolution, heavily correlated with disposable bulletproof hosting setups.",
        "legit": "Properly registered hostname with standard DNS resolving infrastructure.",
    },
    "IsHTTPS": {
        "cat": "Transport Security",
        "phish": "Cleartext HTTP protocol without SSL/TLS encryption. Exposes all authentication tokens and input data to adversary-in-the-middle (AitM) interception.",
        "legit": "Encrypted HTTPS transport protocol active with verified TLS cryptographic session.",
    },
    "URLLength": {
        "cat": "Lexical Complexity",
        "phish": "Abnormally long URL string (>75 characters), commonly engineered to bury redirect targets and encode tracking payloads beyond standard address bar viewports.",
        "legit": "Concise URL length conforming to standard web application navigation parameters.",
    },
    "NoOfSubDomain": {
        "cat": "Domain Hierarchy",
        "phish": "Deeply nested subdomain structure (>2 levels) deployed to masquerade as reputable brand hosts (e.g. login.paypal.com.attacker.xyz).",
        "legit": "Flat domain hierarchy adhering to legitimate corporate architecture.",
    },
    "HasObfuscation": {
        "cat": "Evasion & Encoding",
        "phish": "Hex/percent-encoding (%xx sequences) or delimiter injection (@) detected to deceive perimeter gateway inspection engines.",
        "legit": "Clean, un-obfuscated URL syntax without perimeter evasion encoding.",
    },
    "HasPasswordField": {
        "cat": "Credential Harvesting",
        "phish": "Target page embeds password input elements on an unverified domain, presenting immediate credential theft risk.",
        "legit": "No credential harvesting or password input mechanisms detected.",
    },
    "HasHiddenFields": {
        "cat": "Stealth Exfiltration",
        "phish": "Hidden form fields embedded in DOM structure for covert exfiltration of victim metadata.",
        "legit": "Zero suspicious hidden form parameters.",
    },
    "HasExternalFormSubmit": {
        "cat": "Data Exfiltration",
        "phish": "HTML form submission action resolves to an external third-party domain, a hallmark of credential harvesting kits.",
        "legit": "Form submissions resolve internally to the authentic hosting domain.",
    },
    "TLDLegitimateProb": {
        "cat": "TLD Reputation",
        "phish": "Top-level domain (TLD) suffix has statistically elevated historical abuse and phishing spam rates (.xyz, .top, .tk, .pw, .cc).",
        "legit": "Established authoritative TLD (.com, .org, .edu, .gov) with low historical threat density.",
    },
}

FALLBACK_RULE_IMPLICATIONS = {
    "Low URL brand similarity index (<80)": "Severe lexical divergence indicating deceptive brand spoofing or typosquatting.",
    "Raw IP address used as domain instead of hostname": "Direct numerical IP addressing bypasses domain registrar validation, typical of bulletproof infrastructure.",
    "Insecure HTTP protocol (no SSL/TLS certificate)": "Unencrypted HTTP communication exposes credentials to network eavesdropping and interception.",
    "URL obfuscation detected (% hex or @ symbols)": "Encoding evasion tactics designed to bypass email perimeter filters and static regex scanners.",
    "No social network integration links": "Missing expected corporate social presence, characteristic of single-purpose static phishing clones.",
    "No legitimate copyright notice on page": "Absence of corporate copyright statements and legal notices, typical of ephemeral phishing sites.",
    "Low domain-to-page title match score (<30)": "Significant discrepancy between the domain identity and the HTML page title, indicating brand masquerading.",
    "Missing website meta description": "Omission of standard HTML OpenGraph and description metadata commonly skipped by phishing builders.",
    "Suspicious density of special characters in URL (>6%)": "Heavy parameter chaining and symbol clustering used to inject obfuscated payload variables.",
    "Abnormal digit ratio in URL (>2%)": "Excessive numerical sequences indicative of machine-generated session tokens or encoded IP hex.",
    "Excessively long URL (>40 characters)": "Lengthy URL structure designed to conceal malicious destination parameters.",
    "Multiple non-standard special characters (>2)": "Unconventional delimiter characters embedded to break lexical security scanners.",
    "Missing standard interactive submit elements": "Non-standard form construction that may rely on stealth JavaScript keyloggers.",
    "Site lacks responsive design layout": "Lack of responsive viewport layout, common in static phishing templates.",
}


def _template_report(
    verdict:        str,
    confidence:     float,
    top_features:   list[dict],
    rule_triggers:  list,
) -> str:
    """
    Deterministic executive security report with pristine markdown formatting,
    hierarchical bullet points, deep indicator breakdowns, and rule context.
    Fully compatible with Streamlit native st.markdown() rendering.
    """
    pct         = round(confidence * 100, 1)
    is_phish    = verdict.lower() == "phishing"
    verdict_str = "MALICIOUS PHISHING THREAT" if is_phish else "AUTHENTICATED BENIGN / LEGITIMATE"
    threat_tier = "CRITICAL / HIGH SEVERITY" if is_phish else "LOW RISK / NOMINAL"
    verdict_icon = "\U0001f6a8" if is_phish else "\u2705"

    # -------------------------------------------------------------------------
    # 1. Executive Synopsis
    # -------------------------------------------------------------------------
    if is_phish:
        synopsis = (
            f"The multi-paradigm autonomous defense pipeline classified this target as "
            f"**{verdict_str}** with a **combined final confidence of {pct}%** "
            f"(Threat Tier: **{threat_tier}**).\n\n"
            f"> Cross-paradigm consensus confirmed significant lexical deception, credential "
            f"harvesting vectors, and infrastructure red flags. "
            f"**Immediate perimeter containment is strongly recommended.**"
        )
    else:
        synopsis = (
            f"The multi-paradigm autonomous defense pipeline verified this target as "
            f"**{verdict_str}** with a **combined final confidence of {pct}%** "
            f"(Threat Tier: **{threat_tier}**).\n\n"
            f"> The target conforms to standard legitimate web topology, exhibits valid "
            f"transport security, and triggered zero critical heuristic penalties. "
            f"No containment action required."
        )

    # -------------------------------------------------------------------------
    # 2. Primary Attack Indicators (SHAP Attributions)
    # -------------------------------------------------------------------------
    feat_lines = []
    for f in (top_features or [])[:6]:
        fname   = f.get("feature", "Unknown")
        sv      = f.get("shap_value", 0.0)
        raw_val = f.get("raw_value")
        scaled_val = f.get("scaled_value")

        direction_label = "\U0001f534 **Elevates Phishing Risk**" if sv > 0 else "\U0001f7e2 **Supports Legitimate Safety**"
        val_str  = f"`{raw_val}`" if raw_val is not None else "*not extracted*"
        z_str    = f" | Z-Score: `{scaled_val:+.2f} std dev`" if scaled_val is not None else ""

        cat = f.get("category") or FALLBACK_INDICATOR_RATIONALES.get(fname, {}).get("cat", "Lexical & Host Vector")

        rationale = f.get("rationale")
        if not rationale:
            r_entry = FALLBACK_INDICATOR_RATIONALES.get(fname)
            if r_entry:
                rationale = r_entry["phish"] if sv > 0 else r_entry["legit"]
            else:
                rationale = (
                    "Observed value significantly elevates threat probability toward malicious classification."
                    if sv > 0 else
                    "Observed value conforms to typical authentic web traffic baseline distributions."
                )

        feat_lines.append(
            f"- **`{fname}`** -- *{cat}*\n"
            f"  - **SHAP Attribution:** `{sv:+.4f}` -- {direction_label}\n"
            f"  - **Observed Value:** {val_str}{z_str}\n"
            f"  - **Forensic Rationale:** {rationale}"
        )

    feat_bullets = "\n\n".join(feat_lines) if feat_lines else (
        "- *No significant anomalous feature distortions identified in this sample.*"
    )

    # -------------------------------------------------------------------------
    # 3. Heuristic Rule Violations
    # -------------------------------------------------------------------------
    rule_lines = []
    if rule_triggers:
        for r in rule_triggers[:8]:
            if isinstance(r, dict):
                desc   = r.get("description", r.get("rule", "Heuristic Penalty"))
                sev    = r.get("severity", "HIGH")
                weight = r.get("weight", 2.0)
                imp    = r.get("implication",
                               FALLBACK_RULE_IMPLICATIONS.get(desc, "Structural threshold anomaly detected."))
                obs    = r.get("observed")
                obs_text = f" | Observed: `{obs}`" if obs is not None else ""
            else:
                desc   = str(r)
                weight = 2.5
                sev    = "HIGH" if any(k in desc.lower() for k in ["ip", "similarity", "https", "obfuscation"]) else "MEDIUM"
                imp    = FALLBACK_RULE_IMPLICATIONS.get(desc, "Observed structural parameter departs from verified safe baseline.")
                obs_text = ""

            sev_icon = "\U0001f534" if sev in ("CRITICAL", "HIGH") else "\U0001f7e1"
            rule_lines.append(
                f"- {sev_icon} **[{sev}]** {desc}{obs_text} -- *Penalty: `+{weight:.1f} pts`*\n"
                f"  - **Security Implication:** {imp}"
            )

    rule_bullets = "\n\n".join(rule_lines) if rule_lines else (
        "- *Zero heuristic rule violations triggered. Structural parameters conform to verified normal baseline.*"
    )

    # -------------------------------------------------------------------------
    # 4. Recommended Security Actions
    # -------------------------------------------------------------------------
    if is_phish:
        actions = (
            "1. **Perimeter Network Isolation** -- Block DNS resolution and drop ingress/egress TCP "
            "traffic to this domain across enterprise firewall and proxy egress filters.\n\n"
            "2. **Email Gateway Quarantine** -- Scan M365 / Google Workspace message trace logs for "
            "this domain string. Purge all matching inbound phishing lures from quarantine queues.\n\n"
            "3. **Identity & Credential Containment** -- Force immediate password resets and revoke "
            "active OAuth session tokens for any accounts that initiated outbound connections to this URL.\n\n"
            "4. **EDR Endpoint Sweep** -- Run CrowdStrike / Microsoft Defender telemetry sweeps to "
            "verify no malicious payload stages, credential dumpers, or persistence mechanisms were "
            "executed in browser memory or process space.\n\n"
            "5. **SIEM Correlation** -- Create correlation rules for this domain/IP across your SIEM "
            "(Splunk / Sentinel) to detect lateral spread or follow-on phishing campaigns."
        )
    else:
        actions = (
            "1. **Automated Clearance** -- Target domain meets all cryptographic, structural, and "
            "heuristic verification standards. No perimeter restriction is required.\n\n"
            "2. **Continuous Telemetry** -- Maintain standard organizational web hygiene. Monitor "
            "outbound egress logs via continuous SIEM ingestion for any behavioral anomalies."
        )

    return (
        f"### {verdict_icon} Executive Threat Summary\n\n"
        f"{synopsis}\n\n"
        f"---\n\n"
        f"#### \u26a0\ufe0f Primary Attack Indicators & SHAP Feature Attributions\n\n"
        f"{feat_bullets}\n\n"
        f"---\n\n"
        f"#### \U0001f4cf Heuristic Rule Violations\n\n"
        f"{rule_bullets}\n\n"
        f"---\n\n"
        f"#### \U0001f4cb Recommended Security Actions & Playbook\n\n"
        f"{actions}"
    )


def check_ollama_status(host: str = "http://localhost:11434") -> dict:
    """Check if Ollama server is reachable and query available local models."""
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
    Call the local Ollama API via HTTP REST.
    """
    clean_host = host.rstrip("/")
    try:
        payload = json.dumps({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": 360, "temperature": 0.20}
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
        try:
            import ollama
            response = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"num_predict": 360, "temperature": 0.20},
            )
            return response["message"]["content"].strip()
        except Exception:
            raise http_err


def _build_prompt(
    verdict:       str,
    confidence:    float,
    top_features:  list[dict],
    rule_triggers: list,
    audience:      str = "general",
) -> str:
    """Construct a high-structure prompt sent to the local LLM."""
    pct = round(confidence * 100, 1)
    
    feat_text_lines = []
    for f in (top_features or [])[:5]:
        fn = f.get("feature", "Feature")
        sv = f.get("shap_value", 0.0)
        direction = "ELEVATES PHISHING RISK" if sv > 0 else "SUPPORTS LEGITIMATE"
        val = f.get("raw_value", "Observed")
        rat = f.get("rationale", "")
        feat_text_lines.append(f"  - {fn}: SHAP Impact={sv:+.4f} ({direction}), Observed Value={val}. Context: {rat}")

    rule_text_lines = []
    if rule_triggers:
        for r in rule_triggers[:5]:
            if isinstance(r, dict):
                rule_text_lines.append(f"  - [{r.get('severity','HIGH')}] {r.get('description')}: {r.get('implication','')}")
            else:
                rule_text_lines.append(f"  - {r}")
    rule_text = "\n".join(rule_text_lines) if rule_text_lines else "  - None triggered."

    return textwrap.dedent(f"""
        You are PhishShield AI, an autonomous enterprise SOC incident briefing engine.
        A target URL vector was evaluated with the following verified forensic telemetry:

        VERDICT: {verdict.upper()}
        FINAL CONFIDENCE: {pct}%

        TOP FORENSIC INDICATORS (SHAP Feature Attributions):
        {chr(10).join(feat_text_lines)}

        HEURISTIC RULE VIOLATIONS:
        {rule_text}

        AUDIENCE / PERSONA:
        {'Technical Senior SOC Analyst (focus on protocol mechanics, threat taxonomy, and SIEM playbooks)' if audience == 'technical' else 'Executive Security Incident Summary (structured, professional, accessible to business leaders)'}

        CRITICAL FORMATTING INSTRUCTIONS:
        You MUST use clean Markdown with the exact section headings below. Ensure EVERY bullet point is on its own separate line with indented sub-bullet points for details. Do NOT merge bullet points together.

        ### 🛡️ Executive Threat Summary
        (A 2-3 sentence executive synopsis stating verdict, confidence, and primary danger level)

        ---

        #### ⚠️ Primary Attack Indicators & Feature Shifts (SHAP Attributions)
        * **[Feature Name]**: [Brief Summary]
          * **Attribution Impact:** [State whether it elevates risk or supports safety]
          * **Forensic Rationale:** [Clear explanation of attacker behavior or safe baseline]

        ---

        #### 📏 Heuristic Rule Violations
        * **[Severity] [Rule Name]**:
          * **Security Implication:** [Why this violation poses a hazard]

        ---

        #### 📋 Recommended Security Actions & Playbook
        1. **Perimeter Network Isolation:** [Action]
        2. **Email Gateway Quarantine:** [Action]
        3. **Identity & Session Containment:** [Action]
    """).strip()


def generate_report(
    verdict:       str,
    confidence:    float,
    top_features:  list[dict],
    rule_triggers: list,
    use_llm:       bool = True,
    model_name:    str = OLLAMA_MODEL,
    host:          str = "http://localhost:11434",
    audience:      str = "general",
) -> dict:
    """
    Generate an enriched natural-language security report.
    """
    if not use_llm:
        return {
            "report": _template_report(verdict, confidence, top_features, rule_triggers),
            "source": "template",
            "model" : "deterministic_rule_template",
            "error" : None,
        }

    prompt = _build_prompt(
        verdict=verdict,
        confidence=confidence,
        top_features=top_features,
        rule_triggers=rule_triggers,
        audience=audience,
    )

    try:
        report_text = _call_ollama(prompt, model=model_name, host=host, timeout=OLLAMA_TIMEOUT)
        return {
            "report": report_text,
            "source": "ollama",
            "model" : model_name,
            "error" : None,
        }
    except Exception as exc:
        print(f"[llm_explainer] Ollama call failed ({exc}); falling back to enriched template.")
        return {
            "report": _template_report(verdict, confidence, top_features, rule_triggers),
            "source": "template",
            "model" : "deterministic_rule_template",
            "error" : str(exc),
        }
