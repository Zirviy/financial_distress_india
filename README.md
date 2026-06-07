# Financial Distress Prediction for Indian Corporates
### LightGBM · Dual-Binary Architecture · SHAP Explainability · 73,255 Firm-Years

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://python.org)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.3-green)](https://lightgbm.readthedocs.io)
[![AUC](https://img.shields.io/badge/AUC-0.9042-brightgreen)](/)


---

## Why this project exists

Every major financial distress model in academic literature — Altman Z-Score (1968), Ohlson O-Score (1980), Zmijewski (1984) — was built on US and European data. When applied to Indian firms, they fail. Our validation confirms this: Altman Z-Score classifies **94.5% of healthy Indian firms as distressed**, and Springate flags **82% of the entire dataset as distressed** regardless of actual financial health.

The reason is structural. Indian listed companies operate under Ind AS accounting standards, with capital structures, interest rate environments, and sector compositions that differ fundamentally from Western markets. A model trained on US bankruptcy filings from the 1960s has no business being used by Indian credit analysts in 2025.

This project builds a **ground-up, India-specific financial distress prediction framework** using 73,255 firm-year observations from 18,342 Indian listed firms across 14 industry classes, spanning FY2015–FY2025.

---

## Core Contribution: A Cash-Flow-Aware Taxonomy

Existing work defines distress as a single threshold on the Interest Coverage Ratio (ICR < 1). This captures earnings failure but is blind to cash-flow stress. A firm can report profits while burning cash — and vice versa.

We introduce a **four-category dual-criterion taxonomy** combining the accrual ICR and a cash-based ICR:

| Category | ICR | Cash-ICR | Meaning |
|---|---|---|---|
| **Healthy** | ≥ 1 | ≥ 0 | Earnings and cash flow both sufficient |
| **Accounting Distress** | < 1 | ≥ 1 | Earnings fail to cover interest; cash flow still positive |
| **Cash Distress** | ≥ 1 | < 0 | Earnings look fine; operating cash flow is negative |
| **Full Distress** | < 1 | < 1 | Both earnings and cash flow simultaneously fail |

This distinction matters empirically. The transition matrix analysis (50,310 consecutive firm-year pairs) shows:

- Cash Distress has **19.6% year-to-year persistence** — it is a transient liquidity shock
- Full Distress has **44.5% persistence** — it is a structural condition
- 69.5% of Cash Distress firms return to Healthy within one year
- 20.9% of Accounting Distress firms deteriorate into Full Distress — it is an early warning state

A single ICR threshold cannot distinguish these pathways. A 4-class direct classifier is also insufficient — our experiments confirm balanced accuracy of 0.54, barely above the 0.25 random baseline for 4 classes. This motivates the **dual-binary architecture**.

---

## Architecture: Dual-Binary Prediction

Instead of predicting all four categories simultaneously, we train two independent binary models:

```
27 financial ratios (year t)
        │
        ├─── Model A: Earnings Distress ──► Predicts ICR < 1 at t+1
        │    (Positive = AD + FD)
        │
        └─── Model B: Cash-Flow Distress ──► Predicts Cash-ICR < 0 at t+1
             (Positive = CD + FD)

Model A output × Model B output → Taxonomy reconstruction:
  A=0, B=0 → Healthy
  A=1, B=0 → Accounting Distress
  A=0, B=1 → Cash Distress
  A=1, B=1 → Full Distress
```

Each model specialises on a different financial mechanism. SHAP analysis confirms they learn from different features — profitability and leverage dominate Model A; cash-flow ratios dominate Model B.

---

## Dataset

- **Source:** CMIE Prowess database
- **Firms:** 18,342 Indian listed non-financial companies
- **Observations:** 73,255 firm-year rows (after cleaning)
- **Predictor years:** 2018–2023 (labels at t+1: 2019–2024)
- **Industries:** 14 GICS industry classes
- **Predictors:** 27 financial ratios (profitability, liquidity, leverage, cash flow, efficiency)

**Excluded from predictors (label leakage prevention):**
- `Interest_Coverage` — defines the label directly
- `Cash_Interest_Coverage` — defines the label directly
- `DSCR` — structurally implied by ICR; 100% of AD and FD firms have DSCR < 1 by construction

**Excluded from predictors (design decision):**
- All 5 growth rate variables (revenue, EBITDA, asset, liability, working capital growth)
- Reserved for Phase 2 feature engineering

---

## Label Distribution (Test Year 2023)

| Category | Count | Share |
|---|---|---|
| Healthy | 8,963 | 74.3% |
| Accounting Distress | 994 | 8.2% |
| Cash Distress | 1,271 | 10.5% |
| Full Distress | 827 | 6.9% |

---

## Results

### Model A — Earnings Distress (Test Year 2023)

| Model | AUC | Balanced Acc | F1 | Precision | Recall | Type I | Type II |
|---|---|---|---|---|---|---|---|
| Dummy | 0.500 | 0.500 | 0.000 | — | — | 0.000 | 1.000 |
| Logistic Regression | 0.880 | 0.788 | 0.628 | 0.606 | 0.650 | 0.075 | 0.350 |
| Random Forest | 0.906 | 0.808 | 0.666 | 0.654 | 0.679 | 0.064 | 0.321 |
| **LightGBM** | **0.904** | **0.805** | **0.662** | **0.651** | **0.674** | 0.064 | 0.326 |

### Model B — Cash-Flow Distress (Test Year 2023)

| Model | AUC | Balanced Acc | F1 | Precision | Recall | Type I | Type II |
|---|---|---|---|---|---|---|---|
| Dummy | 0.500 | 0.500 | 0.000 | — | — | 0.000 | 1.000 |
| Logistic Regression | 0.724 | 0.667 | 0.425 | 0.342 | 0.562 | 0.228 | 0.438 |
| Random Forest | 0.772 | 0.676 | 0.446 | 0.384 | 0.531 | 0.180 | 0.469 |
| **LightGBM** | **0.757** | **0.681** | **0.443** | **0.357** | **0.583** | 0.221 | 0.417 |

### Direct 4-Class Model

| Model | Balanced Accuracy | Macro F1 |
|---|---|---|
| Dummy | 0.250 | 0.213 |
| Logistic Regression | 0.499 | 0.442 |
| Random Forest | 0.539 | 0.464 |
| LightGBM | 0.545 | 0.476 |

The 4-class model plateauing at ~0.55 (vs 0.25 random) confirms the dual-binary architecture is necessary — the categories are structurally too distinct for a joint classifier to handle.

### Legacy Benchmark Comparison (Test Year 2023)

| Benchmark | Healthy Flagged as Distressed | Methodology |
|---|---|---|
| Altman Z'' | 57.4% | Linear discriminant; RE proxy required |
| Altman Z' | 21.9% | Book-value variant; grey zone large |
| Ohlson O-Score | **94.5%** | Degenerate — GNP deflator unavailable |
| Zmijewski | 17.8% | Best legacy result; still 2.8× worse than LightGBM |
| Springate | 82.0% | No cash-flow variable |
| **LightGBM (ours)** | **6.4%** | 27 ratio, cash-flow aware |

---

## Top Predictors (SHAP — Model A)

| Rank | Feature | Direction |
|---|---|---|
| 1 | Operating Profit Margin | ↓ low margin → higher risk |
| 2 | ROCE | ↓ low return on capital → higher risk |
| 3 | Inventory Turnover | ↓ slow turnover → higher risk |
| 4 | TL to TA | ↑ high liabilities → higher risk |
| 5 | LT Debt to Assets | ↑ high long-term debt → higher risk |

Model B's top predictors are entirely different: CFO_to_Debt, CFO_to_CL, CFO_to_Net_Income — confirming the two models capture fundamentally different financial failure mechanisms.

---

## Temporal Split Design

```
2018──────2019──────2020──────2021 │ 2022 │ 2023
◄──────── TRAIN (48,814 rows) ────► │ VAL  │ TEST
                                    │ Tune │ Report
```

Random 80-20 splitting is explicitly avoided. The same firm appears in both train and test (different years) — this is expected and does not constitute leakage because labels are always t+1 and the split is strictly chronological.

---

## Repository Structure

```
financial_distress_india/
│
├── data/
│   ├── raw/
│   │   └── ratios_final.csv              ← CMIE Prowess source (not included)
│   └── processed/
│       └── panel_dataset.csv             ← 73,255 rows, 27 predictors + Label
│
├── src/
│   ├── build_panel.py                    ← Step 1: panel construction
│   ├── transition_matrix.py              ← Step 2: taxonomy validation
│   ├── taxonomy_validation.py            ← Step 3: KW tests, VIF, SHAP
│   ├── legacy_benchmarks.py              ← Step 4: Altman, Ohlson, Zmijewski
│   ├── baseline_models.py                ← Step 5: LR, RF baselines
│   ├── lightgbm_models.py                ← Step 6: LightGBM + Optuna
│   └── shap_explainer.py                 ← Step 7: SHAP explainability module
│
├── outputs/
│   ├── tables/                           ← all CSV result tables
│   ├── figures/                          ← all publication plots
│   └── models/                           ← saved .pkl files
│
├── demo/
│   ├── main.py                           ← FastAPI backend
│   ├── demo_companies.json               ← 10 demo firms with known outcomes
│   └── static/
│       └── index.html                    ← DistressIQ frontend
│
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/your-username/financial-distress-india
cd financial-distress-india

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

**requirements.txt**
```
pandas==2.2.2
numpy==1.26.4
scikit-learn==1.4.2
lightgbm==4.3.0
optuna==3.6.1
shap==0.45.0
matplotlib==3.8.4
seaborn==0.13.2
scipy==1.13.0
statsmodels==0.14.2
joblib==1.4.2
fastapi==0.111.0
uvicorn==0.29.0
anthropic==0.25.0
openpyxl==3.1.2
```

---

## Running the Pipeline

```bash
# Step 1 — Build panel dataset
python src/build_panel.py

# Step 2 — Transition matrix
python src/transition_matrix.py

# Step 3 — Taxonomy validation
python src/taxonomy_validation.py

# Step 4 — Legacy benchmarks
python src/legacy_benchmarks.py

# Step 5 — Baseline models
python src/baseline_models.py

# Step 6 — LightGBM (GPU optional, auto-detected)
python src/lightgbm_models.py

# Run demo platform
cd demo
uvicorn main:app --reload --port 8000
# Open http://localhost:8000
```

---

## Key Design Decisions

**Why not random 80-20?**  
Panel data with repeated firm observations requires temporal splitting. Random splitting allows a firm's 2022 data to train the model and its 2019 data to test it — the model has already seen the firm's financial trajectory, inflating test performance artificially.

**Why exclude DSCR?**  
DSCR = EBITDA / (Interest + Principal). When ICR < 1, EBITDA < Interest, so EBITDA < Interest + Principal always, making DSCR < 1 by construction. 100% of AD and FD firms have DSCR < 1 by definition, not by learning. Including it as a predictor when the label is ICR < 1 is structural leakage, not coincidence.

**Why dual-binary instead of 4-class?**  
Cash Distress has 19.6% year-to-year persistence. A model cannot reliably predict a forward label that 80% of current observations will exit within one year. Separating the two binary problems (earnings failure vs cash failure) lets each model specialise and generalise cleanly.

**Why LightGBM over XGBoost?**  
Native GPU support, leaf-wise growth (better for financial ratio non-linearity), Optuna-compatible, and the `is_unbalance` parameter handles the 73:27 class ratio without SMOTE. XGBoost is conceptually equivalent but slower on this dataset size.

---

## Limitations

- **Survivorship bias:** Firms that failed and were delisted before 2023 do not appear in the test set. The true distress rate in the Indian listed universe is likely higher than observed.
- **Sector scope:** Financial firms (banks, NBFCs, insurance) are excluded by design — ICR-based distress definitions are not meaningful for intermediaries.
- **Temporal scope:** The model is trained on 2018–2021. Post-COVID structural changes in Indian corporate balance sheets may reduce out-of-sample generalisability beyond 2024.
- **Growth features excluded:** Revenue, EBITDA, asset, liability, and working capital growth rates are excluded from the baseline. These are documented as Phase 2 additions.

---

## References

1. Altman, E. I. (1968). Financial Ratios, Discriminant Analysis and the Prediction of Corporate Bankruptcy. *Journal of Finance.*
2. Ohlson, J. A. (1980). Financial Ratios and the Probabilistic Prediction of Bankruptcy. *Journal of Accounting Research.*
3. Ke et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS.*
4. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS.*
5. Barboza, Kimura & Altman (2017). Machine Learning Models and Bankruptcy Prediction. *Expert Systems with Applications.*
6. Cheraghali & Molnár (2026). Predictors of Financial Distress: Differences Between Financial and Non-Financial SMEs. *Research in International Business and Finance.*

---


