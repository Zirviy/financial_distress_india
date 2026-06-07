"""
api.py
======
FastAPI endpoint for the Financial Distress Explainability Platform.

Endpoints:
  POST /predict          — prediction + SHAP explanation for one company
  POST /global-importance — population-level feature importance
  GET  /health           — service health check

Run:
  pip install fastapi uvicorn python-multipart
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from src.shap_explainer import DistressExplainer, PREDICTOR_NAMES

# ── Initialise app ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="Financial Distress Explainability API",
    description="LightGBM + SHAP explanations for Indian firm distress prediction",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load explainer once at startup ─────────────────────────────────────────────
MODEL_DIR  = Path("outputs/models")
OUTPUT_DIR = Path("outputs/shap")

try:
    EXPLAINER = DistressExplainer.load(
        model_dir=MODEL_DIR,
        output_dir=OUTPUT_DIR,
    )
    print("[api] Explainer loaded successfully.")
except Exception as e:
    EXPLAINER = None
    print(f"[api] WARNING: Could not load explainer: {e}")


# ── Request / Response schemas ─────────────────────────────────────────────────
class CompanyRatios(BaseModel):
    """
    Financial ratios for a single company.
    All fields are optional — missing values are filled with 0.
    """
    company_name: str = Field(default="Unknown Company",
                              description="Company display name")

    # Cash-flow ratios
    CFO_to_CL          : Optional[float] = None
    CFO_to_Debt        : Optional[float] = None
    CFO_to_Net_Income  : Optional[float] = None
    NCF_to_Debt        : Optional[float] = None
    NCF_to_TA          : Optional[float] = None

    # Liquidity
    Current_Ratio      : Optional[float] = None
    Quick_Ratio        : Optional[float] = None
    Cash_to_CL         : Optional[float] = None

    # Profitability
    ROA                : Optional[float] = None
    ROE                : Optional[float] = None
    ROCE               : Optional[float] = None
    EBITDA_Margin      : Optional[float] = None
    Operating_Profit_Margin: Optional[float] = None
    Net_Profit_Margin  : Optional[float] = None

    # Leverage
    Debt_to_Assets     : Optional[float] = None
    Debt_to_Equity     : Optional[float] = None
    LT_Debt_to_Assets  : Optional[float] = None
    TL_to_TA           : Optional[float] = None
    TA_to_Equity       : Optional[float] = None
    Net_Debt_to_EBITDA : Optional[float] = None

    # Turnover / efficiency
    Total_Asset_Turnover    : Optional[float] = None
    Inventory_Turnover      : Optional[float] = None
    Receivables_Turnover    : Optional[float] = None
    Payables_Turnover       : Optional[float] = None
    WC_Turnover             : Optional[float] = None

    # Working capital
    WC_to_TA               : Optional[float] = None
    Cash_Conversion_Cycle  : Optional[float] = None

    class Config:
        json_schema_extra = {
            "example": {
                "company_name"  : "Tata Steel Ltd",
                "ROA"           : 0.05,
                "Debt_to_Assets": 0.48,
                "Current_Ratio" : 1.2,
                "CFO_to_Debt"   : 0.18,
                "EBITDA_Margin" : 0.14,
                "TL_to_TA"      : 0.62,
            }
        }


class GlobalImportanceRequest(BaseModel):
    """Request body for global importance — pass a list of company ratios."""
    companies: list[dict]
    model_label: str = Field(
        default="model_A",
        description="One of: model_A, model_B, 4class"
    )


# ── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status"   : "ok",
        "models_loaded": EXPLAINER is not None,
        "model_A"  : EXPLAINER.model_A is not None if EXPLAINER else False,
        "model_B"  : EXPLAINER.model_B is not None if EXPLAINER else False,
        "model_4class": EXPLAINER.model_4class is not None if EXPLAINER else False,
    }


@app.post("/predict")
def predict_and_explain(company: CompanyRatios):
    """
    Full prediction + SHAP explanation for a single company.

    Returns:
    - predicted distress category
    - distress probability
    - top 5 factors increasing risk
    - top 5 factors decreasing risk
    - SHAP values per feature
    - paths to generated waterfall plots
    """
    if EXPLAINER is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Check server logs.")

    # Convert pydantic model to dict, drop None values and company_name
    raw = company.dict(exclude_none=True)
    name = raw.pop("company_name", "Company")
    data = {k: v for k, v in raw.items() if k in PREDICTOR_NAMES}

    try:
        result = EXPLAINER.explain_company(
            company_data=data,
            company_name=name,
            top_n=5,
            save_plots=True,
        )
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Explanation failed: {str(e)}")

    # Build clean API response
    pred = result["prediction"]
    response = {
        "company_name"        : name,
        "predicted_class"     : pred["predicted_class"],
        "display_label"       : pred["display_label"],
        "is_distressed"       : pred["is_distressed"],
        "distress_probability": pred["distress_probability"],
        "method"              : pred["method"],
        "top_factors"         : result["contributors"],
        "shap_values"         : result["shap_values"],
        "plots"               : result["plots"],
    }

    # Add model-specific probabilities if available
    if "model_A_prob" in pred:
        response["earnings_distress_probability"] = pred["model_A_prob"]
        response["cashflow_distress_probability"] = pred["model_B_prob"]

    return response


@app.get("/plots/waterfall/{filename}")
def get_waterfall_plot(filename: str):
    """Serve a generated waterfall plot by filename."""
    path = OUTPUT_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Plot not found")
    return FileResponse(str(path), media_type="image/png")


@app.post("/global-importance")
def global_importance(request: GlobalImportanceRequest):
    """
    Compute mean absolute SHAP values across a list of companies.
    Use this to show which features matter most across your portfolio.
    """
    if EXPLAINER is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    if not request.companies:
        raise HTTPException(status_code=400, detail="No companies provided.")

    # Build feature matrix
    rows = []
    for comp in request.companies:
        aligned = {f: comp.get(f, 0.0) for f in PREDICTOR_NAMES}
        rows.append(list(aligned.values()))
    X = np.array(rows, dtype=float)

    try:
        importance = EXPLAINER.get_global_importance(
            X, request.model_label)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "model_label"   : request.model_label,
        "n_companies"   : len(request.companies),
        "feature_ranking": importance,
    }