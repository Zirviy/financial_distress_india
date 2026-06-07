"""
DistressIQ — Financial Distress Prediction Platform
FastAPI Backend — Model A (Earnings Distress: AD+FD vs H+CD)

Architecture:
  File Upload → Claude API → 27 Ratios JSON
                           → Validation
                           → LightGBM Model A → Prediction
                           → SHAP TreeExplainer → Risk Drivers
                           → Panel Dataset Lookup → Demo Mode
"""

import json
import io
import os
import re
import base64
import warnings
from pathlib import Path
from difflib import SequenceMatcher
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
import anthropic
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

warnings.filterwarnings("ignore")

# ── Import reused SHAP module ─────────────────────────────────────────────────
from shap_explainer import DistressExplainer, PREDICTOR_NAMES, FEATURE_LABELS

# ── Constants ──────────────────────────────────────────────────────────────────
BASE_DIR     = Path(__file__).parent
MODEL_PATH   = BASE_DIR / "models" / "lgb_model_A.pkl"
PANEL_PATH   = BASE_DIR / "data"   / "panel_dataset.csv"
SHAP_DIR     = BASE_DIR / "outputs" / "shap"
STATIC_DIR   = BASE_DIR / "static"

THRESHOLD    = 0.53          # F1-optimal threshold on 2022 validation set
N_FEATURES   = 27

EARNINGS_DISTRESS_LABELS = {"Acct_Distress", "Full_Distress"}
LABEL_DISPLAY = {
    "Healthy":       "Healthy",
    "Acct_Distress": "Accounting Distress",
    "Cash_Distress": "Cash-Flow Distress",
    "Full_Distress": "Full Distress",
}

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="DistressIQ — Financial Distress Intelligence")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup: load heavy objects once ──────────────────────────────────────────
print("[init] Loading LightGBM model …")
try:
    lgb_model = joblib.load(MODEL_PATH)
    print(f"[init] ✓ Model loaded — best_iteration={lgb_model.best_iteration_}")
except Exception as exc:
    print(f"[init] ✗ Model load failed: {exc}")
    lgb_model = None

print("[init] Building SHAP TreeExplainer …")
try:
    explainer_wrapper = DistressExplainer(
        model_A=lgb_model,
        output_dir=str(SHAP_DIR),
        feature_names=PREDICTOR_NAMES,
    )
    print("[init] ✓ SHAP explainer ready")
except Exception as exc:
    print(f"[init] ✗ SHAP init failed: {exc}")
    explainer_wrapper = None

print("[init] Loading panel dataset for demo mode …")
try:
    df_panel = pd.read_csv(PANEL_PATH)
    # Pre-compute lower-case company names for fuzzy matching
    df_panel["_name_lower"] = df_panel["Company Name"].str.lower().str.strip()
    PANEL_NAMES = df_panel["_name_lower"].tolist()
    print(f"[init] ✓ Panel loaded — {len(df_panel):,} firm-years")
except Exception as exc:
    print(f"[init] ✗ Panel load failed: {exc}")
    df_panel   = None
    PANEL_NAMES = []

# Anthropic client (reads ANTHROPIC_API_KEY from env)
claude_client = anthropic.Anthropic()


# ── Claude prompt ─────────────────────────────────────────────────────────────
RATIO_SYSTEM = (
    "You are a specialist in Indian corporate financial statements "
    "(CMIE Prowess, Capitaline, BSE/NSE filings). "
    "Extract financial data and compute ratios precisely. "
    "Return ONLY valid JSON, no markdown, no explanation, no preamble."
)

RATIO_USER_TEMPLATE = """\
Given the following financial data, identify the company name and financial year, \
then compute all 27 ratios.

FINANCIAL DATA:
{financial_text}

INSTRUCTIONS: Use the MOST RECENT year if multiple years are present.

IDENTIFY THESE LINE ITEMS (common Indian accounting terminology):
- revenue          : "Total income", "Net revenue", "Revenue from operations", "Net sales"
- ebitda           : "PBDITA", "EBITDA", "Profit before depreciation interest tax amortization"
- ebit             : "PBIT", "EBIT", "Profit before interest and tax"
- operating_profit : "Operating profit (of non-financial companies)", "EBIT from operations"
- net_income       : "Profit after tax (PAT)", "Net profit", "Net income"
- cfo              : "Net cash flow from operating activities", "Cash from operations"
- ncf              : "Net increase (decrease) in cash and equivalents", "Net change in cash"
- total_assets     : "Total assets"
- current_assets   : "Current assets" (including short-term investments, loans & advances)
- cash             : "Cash and bank balance (short term)", "Cash and cash equivalents"
- inventory        : "Short-term inventories", "Inventories", "Stock-in-trade"
- accounts_receivable : "Short-term trade receivables", "Trade debtors", "Debtors"
- current_liabilities : "Current liabilities and provisions", "Total current liabilities"
- accounts_payable : "Short-term trade payables", "Trade creditors", "Creditors"
- total_debt       : "Debt", "Total borrowings" (long-term + short-term borrowings combined)
- lt_debt          : "Long-term borrowings excl. current portion", "Non-current borrowings"
- total_liabilities: "Total liabilities excluding Capital & Reserves"
- equity           : "Net worth", "Total equity", "Shareholders equity"
- cash_conversion_cycle : "Net working capital cycle (days)", "CCC in days" \
[use directly if provided as a number in days]

COMPUTE THESE EXACT 27 RATIOS:
Definitions:
  WC = current_assets - current_liabilities
  CE = total_assets - current_liabilities  (Capital Employed)

 1. CFO_to_CL               = cfo / current_liabilities
 2. CFO_to_Debt             = cfo / total_debt
 3. CFO_to_Net_Income       = cfo / net_income
 4. Cash_Conversion_Cycle   = cash_conversion_cycle  \
[use the pre-computed days value directly if available; \
else compute as (inventory/revenue*365) + (accounts_receivable/revenue*365) - \
(accounts_payable/revenue*365)]
 5. Cash_to_CL              = cash / current_liabilities
 6. Current_Ratio           = current_assets / current_liabilities
 7. Debt_to_Assets          = total_debt / total_assets
 8. Debt_to_Equity          = total_debt / equity
 9. EBITDA_Margin           = ebitda / revenue
10. Inventory_Turnover      = revenue / inventory
11. LT_Debt_to_Assets       = lt_debt / total_assets
12. NCF_to_Debt             = ncf / total_debt
13. NCF_to_TA               = ncf / total_assets
14. Net_Debt_to_EBITDA      = (total_debt - cash) / ebitda
15. Net_Profit_Margin       = net_income / revenue
16. Operating_Profit_Margin = operating_profit / revenue
17. Payables_Turnover       = revenue / accounts_payable
18. Quick_Ratio             = (current_assets - inventory) / current_liabilities
19. ROA                     = net_income / total_assets
20. ROCE                    = ebit / CE
21. ROE                     = net_income / equity
22. Receivables_Turnover    = revenue / accounts_receivable
23. TA_to_Equity            = total_assets / equity
24. TL_to_TA                = total_liabilities / total_assets
25. Total_Asset_Turnover    = revenue / total_assets
26. WC_Turnover             = revenue / WC
27. WC_to_TA                = WC / total_assets

RULES:
- Return null for any ratio with a zero denominator or missing input
- All ratio values must be plain decimal numbers (0.15, NOT 15%)
- Do not cap or winsorize any values
- Extract firm_name from the data (first column heading or company name line)
- Extract financial_year as "FY2022-23" or "2022" format

RETURN EXACTLY THIS JSON (no markdown fences, no extra text):
{{
  "firm_name": "string or null",
  "financial_year": "string or null",
  "CFO_to_CL": number_or_null,
  "CFO_to_Debt": number_or_null,
  "CFO_to_Net_Income": number_or_null,
  "Cash_Conversion_Cycle": number_or_null,
  "Cash_to_CL": number_or_null,
  "Current_Ratio": number_or_null,
  "Debt_to_Assets": number_or_null,
  "Debt_to_Equity": number_or_null,
  "EBITDA_Margin": number_or_null,
  "Inventory_Turnover": number_or_null,
  "LT_Debt_to_Assets": number_or_null,
  "NCF_to_Debt": number_or_null,
  "NCF_to_TA": number_or_null,
  "Net_Debt_to_EBITDA": number_or_null,
  "Net_Profit_Margin": number_or_null,
  "Operating_Profit_Margin": number_or_null,
  "Payables_Turnover": number_or_null,
  "Quick_Ratio": number_or_null,
  "ROA": number_or_null,
  "ROCE": number_or_null,
  "ROE": number_or_null,
  "Receivables_Turnover": number_or_null,
  "TA_to_Equity": number_or_null,
  "TL_to_TA": number_or_null,
  "Total_Asset_Turnover": number_or_null,
  "WC_Turnover": number_or_null,
  "WC_to_TA": number_or_null
}}"""


# ── Helpers: file parsing ─────────────────────────────────────────────────────
def excel_to_text(file_bytes: bytes) -> str:
    """Convert Excel workbook to a readable text block for Claude."""
    try:
        xl  = pd.ExcelFile(io.BytesIO(file_bytes))
        out = []
        for sheet in xl.sheet_names[:6]:
            df = xl.parse(sheet, header=None)
            out.append(f"\n=== Sheet: {sheet} ===\n")
            # Drop entirely-empty rows/columns to reduce noise
            df = df.dropna(how="all").dropna(axis=1, how="all")
            out.append(df.to_string(index=False, na_rep=""))
        return "\n".join(out)
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse Excel file: {exc}")


def csv_to_text(file_bytes: bytes) -> str:
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
        return df.to_string(index=False)
    except Exception as exc:
        raise HTTPException(400, f"Cannot parse CSV: {exc}")


def _clean_json(raw: str) -> str:
    """Strip markdown fences if Claude wraps the JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.split("\n")
        # Drop first line (```json or ```) and last line (```)
        raw = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    return raw.strip()


# ── Claude: extract ratios ─────────────────────────────────────────────────────
def call_claude_text(financial_text: str) -> dict:
    """Send financial text to Claude, get back 29-key JSON (firm_name + year + 27 ratios)."""
    prompt = RATIO_USER_TEMPLATE.format(financial_text=financial_text)
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=RATIO_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = _clean_json(response.content[0].text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Claude returned invalid JSON: {raw[:300]} — {exc}")


def call_claude_pdf(pdf_b64: str, filename: str) -> dict:
    """Send PDF as base64 document to Claude, get back ratios JSON."""
    prompt = RATIO_USER_TEMPLATE.format(
        financial_text=f"[PDF document: {filename}]"
    )
    # Re-phrase prompt for PDF context
    pdf_prompt = (
        "The attached PDF contains financial statements for an Indian company.\n\n"
        + RATIO_USER_TEMPLATE.replace(
            "Given the following financial data",
            "Using the attached PDF financial statements"
        ).replace("FINANCIAL DATA:\n{financial_text}\n\n", "")
    )
    response = claude_client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1200,
        system=RATIO_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": pdf_b64,
                    },
                },
                {"type": "text", "text": pdf_prompt},
            ],
        }],
    )
    raw = _clean_json(response.content[0].text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(500, f"Claude returned invalid JSON (PDF): {raw[:300]} — {exc}")


# ── Validation ─────────────────────────────────────────────────────────────────
def validate_and_build_feature_row(claude_json: dict) -> tuple[np.ndarray, dict, list[str]]:
    """
    Validate Claude's JSON, align to PREDICTOR_NAMES order.
    Returns:
        X           : (27,) numpy array in model feature order
        ratios_dict : {col: value} with nulls filled as 0.0
        missing     : list of cols that were null/missing
    """
    # Check all 27 keys are present
    missing_keys = [col for col in PREDICTOR_NAMES if col not in claude_json]
    if missing_keys:
        raise HTTPException(
            422,
            f"Claude response missing {len(missing_keys)} required ratio keys: "
            f"{missing_keys[:5]}{'…' if len(missing_keys) > 5 else ''}",
        )

    missing_vals = []
    row = {}
    for col in PREDICTOR_NAMES:
        val = claude_json.get(col)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            row[col] = 0.0
            missing_vals.append(col)
        else:
            row[col] = float(val)

    X = np.array([row[col] for col in PREDICTOR_NAMES], dtype=np.float64)
    return X, row, missing_vals


# ── Model prediction ────────────────────────────────────────────────────────────
def run_prediction(X: np.ndarray) -> tuple[float, bool, str, str, str]:
    """
    Run Model A on feature array X (shape: 27,).
    Returns (probability, predicted_distressed, category, category_detail, risk_level).
    """
    if lgb_model is None:
        raise HTTPException(500, "Model not loaded")

    X_2d = X.reshape(1, -1)          # (1, 27) — positional, no feature names
    prob  = float(lgb_model.predict_proba(X_2d)[0, 1])
    pred  = prob >= THRESHOLD

    if pred:
        category       = "Earnings Distressed"
        category_detail = "Accounting Distress or Full Distress"
        risk_level     = "HIGH" if prob >= 0.70 else "ELEVATED"
    else:
        category       = "Not Earnings Distressed"
        category_detail = "Healthy or Cash-Flow Distress"
        risk_level     = "LOW" if prob <= 0.30 else "MODERATE"

    return prob, pred, category, category_detail, risk_level


# ── SHAP explanation ────────────────────────────────────────────────────────────
def run_shap(X: np.ndarray, top_n: int = 5) -> dict:
    """
    Compute SHAP values for a single observation.
    Uses explainer_wrapper._explainer_A and get_top_contributors from shap_explainer.py.
    """
    if explainer_wrapper is None or explainer_wrapper._explainer_A is None:
        return {"error": "SHAP explainer not available"}

    try:
        X_2d = X.reshape(1, -1)
        sv, base = explainer_wrapper._get_shap_values_binary(
            explainer_wrapper._explainer_A, X_2d
        )
        contributors = explainer_wrapper.get_top_contributors(sv, X, top_n=top_n)

        return {
            "base_value":           round(float(base), 4),
            "top_risk_drivers":     contributors["increasing_risk"],
            "top_protective_drivers": contributors["decreasing_risk"],
            "all_shap_values": {
                PREDICTOR_NAMES[i]: round(float(sv[i]), 5)
                for i in range(len(PREDICTOR_NAMES))
            },
        }
    except Exception as exc:
        return {"error": str(exc)}


# ── Demo mode: panel dataset lookup ───────────────────────────────────────────
def _extract_year(financial_year_str: Optional[str]) -> Optional[int]:
    """Parse a year from strings like 'FY2022-23', '2022', '2021-22', etc."""
    if not financial_year_str:
        return None
    nums = re.findall(r'\b(20\d{2})\b', financial_year_str)
    if nums:
        return int(nums[0])   # Take the first 4-digit year found
    return None


def lookup_demo(
    firm_name: Optional[str],
    financial_year: Optional[str],
    predicted_distressed: bool,
) -> Optional[dict]:
    """
    Find company in panel_dataset.csv and return known outcome.

    Match strategy:
      1. Exact case-insensitive name match
      2. Fuzzy SequenceMatcher (threshold 0.80)
    """
    if df_panel is None or not firm_name:
        return None

    name_lower = firm_name.lower().strip()
    year       = _extract_year(financial_year)

    # Step 1 — exact match
    exact = df_panel[df_panel["_name_lower"] == name_lower]
    if exact.empty:
        # Step 2 — fuzzy
        best_score, best_idx = 0.0, -1
        for idx, panel_name in enumerate(PANEL_NAMES):
            score = SequenceMatcher(None, name_lower, panel_name).ratio()
            if score > best_score:
                best_score, best_idx = score, idx
        if best_score < 0.80:
            return None
        matched_df = df_panel.iloc[[best_idx]].copy()
        match_conf = round(best_score, 3)
    else:
        matched_df = exact
        match_conf = 1.0

    # Filter by year if available; otherwise take the latest year
    matched_company = matched_df["Company Name"].iloc[0]
    company_rows = df_panel[df_panel["Company Name"] == matched_company]

    if year is not None:
        year_rows = company_rows[company_rows["Year_t"] == year]
        row = year_rows.iloc[0] if not year_rows.empty else company_rows.sort_values("Year_t").iloc[-1]
    else:
        row = company_rows.sort_values("Year_t").iloc[-1]

    known_label      = row["Label"]
    known_distressed = known_label in EARNINGS_DISTRESS_LABELS
    year_t           = int(row["Year_t"])

    return {
        "found_in_dataset":      True,
        "matched_company":       matched_company,
        "match_confidence":      match_conf,
        "dataset_year_t":        year_t,
        "prediction_year":       year_t + 1,
        "known_label":           known_label,
        "known_label_display":   LABEL_DISPLAY.get(known_label, known_label),
        "known_distressed":      known_distressed,
        "prediction_matched":    (bool(predicted_distressed) == bool(known_distressed)),
    }


# ── Core analysis logic ────────────────────────────────────────────────────────
def analyse(
    claude_json: dict,
    file_type: str,
    firm_name_override: Optional[str] = None,
) -> dict:
    """
    Validate → Predict → SHAP → Demo lookup → Build response.
    """
    # 1. Extract metadata
    firm_name      = firm_name_override or claude_json.get("firm_name") or "Uploaded Firm"
    financial_year = claude_json.get("financial_year")

    # 2. Build prediction year string
    t_year = _extract_year(financial_year)
    if t_year:
        fy_display   = f"FY{t_year}-{str(t_year + 1)[-2:]}"
        pred_fy      = f"FY{t_year + 1}-{str(t_year + 2)[-2:]}"
    else:
        fy_display   = financial_year or "Unknown Year"
        pred_fy      = "Next Year"

    # 3. Validate ratios and build feature array
    X, ratio_row, missing_vals = validate_and_build_feature_row(claude_json)

    # 4. Predict
    prob, pred, category, cat_detail, risk_level = run_prediction(X)

    # 5. SHAP
    shap_result = run_shap(X, top_n=5)

    # 6. Demo mode
    demo = lookup_demo(firm_name, financial_year, pred)

    # 7. Build response
    return {
        "firm_name":           firm_name,
        "financial_year":      fy_display,
        "prediction_year":     pred_fy,
        "probability":         round(prob, 4),
        "probability_pct":     round(prob * 100, 1),
        "predicted_distressed": bool(pred),
        "category":            category,
        "category_detail":     cat_detail,
        "risk_level":          risk_level,
        "threshold_used":      THRESHOLD,
        "file_type":           file_type,
        "ratios_computed":     {k: round(v, 4) for k, v in ratio_row.items()},
        "missing_ratios":      missing_vals,
        "shap":                shap_result,
        "demo_mode":           demo,
        "model":               "LightGBM Model A — Earnings Distress (AUC 0.9042)",
    }


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = STATIC_DIR / "index.html"
    return html_path.read_text(encoding="utf-8")


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Accept Excel (.xlsx/.xls), CSV, or PDF.
    Returns full distress prediction with SHAP explanation.
    """
    filename    = (file.filename or "upload").lower()
    content     = await file.read()
    firm_name   = (
        file.filename
        .replace(".xlsx", "").replace(".xls", "")
        .replace(".csv",  "")
        .replace(".pdf",  "")
        .replace("_", " ")
        .title()
    ) if file.filename else "Uploaded Firm"

    if filename.endswith((".xlsx", ".xls")):
        financial_text = excel_to_text(content)
        file_type      = "Excel"
    elif filename.endswith(".csv"):
        financial_text = csv_to_text(content)
        file_type      = "CSV"
    elif filename.endswith(".pdf"):
        pdf_b64        = base64.standard_b64encode(content).decode()
        claude_json    = call_claude_pdf(pdf_b64, file.filename or "document.pdf")
        return JSONResponse(analyse(claude_json, "PDF", firm_name))
    else:
        raise HTTPException(400, "Unsupported file type. Upload .xlsx, .xls, .csv, or .pdf")

    claude_json = call_claude_text(financial_text)
    return JSONResponse(analyse(claude_json, file_type, firm_name))


@app.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  lgb_model is not None,
        "shap_ready":    explainer_wrapper is not None,
        "panel_loaded":  df_panel is not None,
    }


# Mount static files (CSS, images, etc.)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
