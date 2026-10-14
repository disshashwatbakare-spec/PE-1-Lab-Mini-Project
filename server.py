"""
server.py  —  PhishShield Dedicated REST API Backend Server
=============================================================
Provides a standalone REST API backend for the hybrid phishing detection system.
Endpoints:
  GET  /health            — API status & health
  GET  /metrics           — Model benchmark & comparison tables
  POST /predict           — Run hybrid detection on a feature dictionary
  POST /predict/url       — Auto-extract lexical features from a raw URL string & predict
"""

import sys
import pathlib
import urllib.parse
from typing import Dict, Any, Optional

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware

from pipeline import PhishShieldPipeline

# Initialize and cache pipeline
pipeline = PhishShieldPipeline()
pipeline.load_all()

# Lexical feature extractor from URL string
def extract_lexical_features(url: str) -> dict:
    parsed  = urllib.parse.urlparse(url)
    domain  = parsed.netloc or ""
    full    = url

    import re
    ip_pattern = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")
    is_ip = 1 if ip_pattern.match(domain.split(":")[0]) else 0
    parts = domain.replace("www.", "").split(".")
    n_sub = max(0, len(parts) - 2)

    return {
        "URLLength"             : len(full),
        "DomainLength"          : len(domain),
        "IsDomainIP"            : is_ip,
        "NoOfSubDomain"         : n_sub,
        "IsHTTPS"               : 1 if parsed.scheme == "https" else 0,
        "NoOfQMarkInURL"        : full.count("?"),
        "NoOfAmpersandInURL"    : full.count("&"),
        "NoOfEqualsInURL"       : full.count("="),
        "NoOfLettersInURL"      : sum(c.isalpha() for c in full),
        "LetterRatioInURL"      : round(sum(c.isalpha() for c in full) / max(len(full), 1), 4),
        "NoOfDegitsInURL"       : sum(c.isdigit() for c in full),
        "DegitRatioInURL"       : round(sum(c.isdigit() for c in full) / max(len(full), 1), 4),
        "NoOfOtherSpecialCharsInURL": sum(not c.isalnum() and c not in "/:.-_?" for c in full),
        "SpacialCharRatioInURL" : round(sum(not c.isalnum() for c in full) / max(len(full), 1), 4),
        "HasObfuscation"        : 1 if "%" in full or "@" in full else 0,
        "ObfuscationRatio"      : round(full.count("%") / max(len(full), 1), 4),
    }


async def health(request):
    return JSONResponse({
        "status": "online",
        "system": "PhishShield Hybrid AI Detection Engine",
        "paradigms": [
            "Symbolic AI (Rule Engine)",
            "Supervised Classical + Neural ML (XGBoost, MLPClassifier, Logistic Regression)",
            "Unsupervised AI (Isolation Forest Anomaly Detection)",
            "Decision Fusion Layer",
            "Explainable AI (SHAP)",
            "Generative AI (Local Ollama LLM)"
        ]
    })


async def metrics(request):
    res_path = ROOT / "outputs" / "classifier_results.csv"
    comp_path = ROOT / "outputs" / "paradigm_comparison.csv"
    
    data = {}
    if res_path.exists():
        data["classifier_benchmark"] = pd.read_csv(res_path).to_dict(orient="records")
    if comp_path.exists():
        data["paradigm_comparison"] = pd.read_csv(comp_path).to_dict(orient="records")
    return JSONResponse(data)


async def predict(request):
    try:
        payload = await request.json()
        feature_dict = payload.get("features", payload)
        use_llm = payload.get("use_llm", True)
        result = pipeline.predict(feature_dict, use_llm=use_llm)
        return JSONResponse({"success": True, "result": result})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


async def predict_url(request):
    try:
        payload = await request.json()
        url = payload.get("url", "")
        if not url:
            return JSONResponse({"success": False, "error": "No 'url' provided in request"}, status_code=400)

        # Lexical extraction
        features = extract_lexical_features(url)
        # Apply any manual overrides provided in payload
        if "overrides" in payload and isinstance(payload["overrides"], dict):
            features.update(payload["overrides"])

        use_llm = payload.get("use_llm", True)
        result = pipeline.predict(features, use_llm=use_llm)
        return JSONResponse({
            "success": True,
            "url": url,
            "extracted_features": features,
            "result": result
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=400)


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/metrics", metrics, methods=["GET"]),
    Route("/predict", predict, methods=["POST"]),
    Route("/predict/url", predict_url, methods=["POST"]),
]

middleware = [
    Middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
]

app = Starlette(debug=True, routes=routes, middleware=middleware)

if __name__ == "__main__":
    print("[server] Starting PhishShield Backend REST API on http://localhost:8000 ...")
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
