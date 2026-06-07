# DistressIQ — Financial Distress Prediction Platform

LightGBM Model A (Earnings Distress) · AUC 0.9042 · 73,255 Indian firm-years

## Architecture

```
File Upload (Excel / CSV / PDF)
        ↓
  Claude API  ←  27-ratio formula prompt
        ↓
  Ratio Validation (27 keys, exact match to PREDICTOR_NAMES)
        ↓
  LightGBM Model A  (binary, threshold 0.53)
        ↓
  SHAP TreeExplainer  (top risk drivers + protective factors)
        ↓
  Panel Dataset Lookup  (demo mode: known outcome t+1)
        ↓
  JSON Response → Frontend
```

## Project Structure

```
distressiq/
├── main.py               ← FastAPI backend
├── shap_explainer.py     ← SHAP module (reused as-is)
├── requirements.txt
├── models/
│   └── lgb_model_A.pkl   ← Trained LightGBM (27 positional features)
├── data/
│   └── panel_dataset.csv ← 73,255 firm-years for demo mode lookup
├── static/
│   └── index.html        ← Frontend (DM Serif + SHAP bars + demo mode)
└── outputs/
    └── shap/             ← Auto-created, SHAP plots written here
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY="sk-ant-..."

# 3. Run the server
uvicorn main:app --reload --port 8000

# 4. Open http://localhost:8000
```

## Key Technical Details

### Model Features (exact order, positional)
The model was saved with Column_0..26 feature names (sklearn-style positional).
Features are fed as a numpy array in this exact PREDICTOR_NAMES order:

```python
PREDICTOR_NAMES = [
  "CFO_to_CL", "CFO_to_Debt", "CFO_to_Net_Income",
  "Cash_Conversion_Cycle", "Cash_to_CL", "Current_Ratio",
  "Debt_to_Assets", "Debt_to_Equity", "EBITDA_Margin",
  "Inventory_Turnover", "LT_Debt_to_Assets", "NCF_to_Debt",
  "NCF_to_TA", "Net_Debt_to_EBITDA", "Net_Profit_Margin",
  "Operating_Profit_Margin", "Payables_Turnover", "Quick_Ratio",
  "ROA", "ROCE", "ROE", "Receivables_Turnover",
  "TA_to_Equity", "TL_to_TA", "Total_Asset_Turnover",
  "WC_Turnover", "WC_to_TA",
]
```

### Threshold
**0.53** — F1-maximising threshold on 2022 validation set.
- Probability ≥ 0.53 → Earnings Distressed
- Probability < 0.53 → Not Earnings Distressed

### Prediction Classes (Model A)
- **Positive (1)**: Accounting Distress + Full Distress
- **Negative (0)**: Healthy + Cash-Flow Distress

### t → t+1 Design
Panel dataset Year_t financial ratios predict distress status at Year_t+1.
Demo mode shows known panel label for uploaded company-year.

### Claude Prompt
- System: Indian financial statement specialist
- User: Financial text + 27 ratio formulas + exact JSON schema
- Response: strict JSON {firm_name, financial_year, CFO_to_CL, …}

### SHAP
Uses `shap.TreeExplainer` (exact for tree models).
- `top_risk_drivers`     → features with positive SHAP (push toward distress)
- `top_protective_drivers` → features with negative SHAP (push away from distress)

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | / | Serve frontend |
| POST | /predict | Upload file, get prediction + SHAP |
| GET | /health | Model/SHAP/panel status |

### POST /predict Response Schema
```json
{
  "firm_name":           "Waterbase Ltd.",
  "financial_year":      "FY2018-19",
  "prediction_year":     "FY2019-20",
  "probability":         0.1543,
  "probability_pct":     15.4,
  "predicted_distressed": false,
  "category":            "Not Earnings Distressed",
  "category_detail":     "Healthy or Cash-Flow Distress",
  "risk_level":          "LOW",
  "threshold_used":      0.53,
  "file_type":           "Excel",
  "ratios_computed":     { "CFO_to_CL": 0.5685, "..." : "..." },
  "missing_ratios":      [],
  "shap": {
    "base_value":              -1.0545,
    "top_risk_drivers":        [{ "feature": "...", "display_name": "...", "feature_value": 0.0, "shap_value": 0.0, "direction": "increases", "impact": "high" }],
    "top_protective_drivers":  [{ "..." }],
    "all_shap_values":         { "CFO_to_CL": -0.012, "..." }
  },
  "demo_mode": {
    "found_in_dataset":    true,
    "matched_company":     "Waterbase Ltd.",
    "match_confidence":    1.0,
    "dataset_year_t":      2018,
    "prediction_year":     2019,
    "known_label":         "Healthy",
    "known_label_display": "Healthy",
    "known_distressed":    false,
    "prediction_matched":  true
  }
}
```
