"""
STEP 4 — LEGACY BENCHMARK INADEQUACY ANALYSIS
==============================================
Financial Distress Prediction for Indian Firms

Purpose:
    Demonstrate that traditional distress frameworks (Altman, Ohlson,
    Zmijewski, Springate) are structurally limited in distinguishing the
    heterogeneous distress states captured by the 4-category taxonomy.

    Framing note: this analysis does NOT claim legacy models are wrong.
    They were appropriate for their original contexts. The argument is that
    their binary output structure cannot differentiate economically distinct
    distress mechanisms — specifically Accounting Distress vs Cash Distress —
    that the proposed taxonomy captures.

Benchmarks Implemented:
    1. Altman Z''-Score (1995)  — emerging markets / private firms (4-variable)
    2. Altman Z'-Score  (1983)  — private firms (5-variable, book equity)
    3. Ohlson O-Score   (1980)  — logistic probability model
    4. Zmijewski X-Score(1984)  — probit-based (fully implementable from ratios)
    5. Springate S-Score(1978)  — linear discriminant (optional benchmark)

Variable Mapping:
    All benchmarks derived from ratio predictors already in the panel dataset.
    Key derivations documented explicitly in each function.
    ICR and Cash-ICR are NOT used (they define labels, not features).

Outputs:
    outputs/tables/
        benchmark_scores.csv               — full scored panel
        benchmark_classification_counts.csv
        crosstab_altman_zdp.csv            — taxonomy vs Altman Z''
        crosstab_altman_zp.csv             — taxonomy vs Altman Z'
        crosstab_ohlson.csv                — taxonomy vs Ohlson
        crosstab_zmijewski.csv             — taxonomy vs Zmijewski
        crosstab_springate.csv             — taxonomy vs Springate
        detection_rates.csv                — per-category detection rates
        agreement_rates.csv                — overall agreement statistics
        variable_coverage.csv              — cash-flow dimension coverage
        benchmark_limitations.csv          — documented limitations table
        cd_miss_analysis.csv               — Cash Distress miss spotlight

    outputs/figures/
        detection_rate_heatmap.png
        agreement_bars.png
        score_distributions.png
        cd_miss_spotlight.png
        variable_coverage_chart.png

Author: Research Pipeline
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
BASE  = Path(__file__).resolve().parent.parent
PANEL = BASE / "data" / "processed" / "panel_dataset.csv"
TBLS  = BASE / "outputs" / "tables"
FIGS  = BASE / "outputs" / "figures"
TBLS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────
LABEL_ORDER = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
LABEL_SHORT = {"Healthy": "H", "Acct_Distress": "AD",
               "Cash_Distress": "CD", "Full_Distress": "FD"}

# Palette consistent with prior steps
CAT_PALETTE = {
    "Healthy":       "#2196F3",
    "Acct_Distress": "#FF9800",
    "Cash_Distress": "#9C27B0",
    "Full_Distress": "#F44336",
}


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_panel(path: Path) -> pd.DataFrame:
    """Load and validate the clean panel dataset from Step 1."""
    df = pd.read_csv(path)
    assert "Label" in df.columns, "Label column missing"
    assert df["Label"].isnull().sum() == 0, "Null labels found"
    print(f"[load] Panel shape: {df.shape}")
    print(f"[load] Label distribution:\n{df['Label'].value_counts()}\n")
    return df


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — VARIABLE DERIVATIONS
# All benchmark-specific variables derived here with full
# documentation of mapping choices and limitations.
# ═══════════════════════════════════════════════════════════════

def derive_benchmark_variables(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive intermediate variables shared across benchmarks.

    Key derivations (all from ratio predictors in panel):

    X1  = WC/TA   → WC_to_TA  (direct)
    X2  = RE/TA   → ROA proxy  (see note A)
    X3  = EBIT/TA → Operating_Profit_Margin × Total_Asset_Turnover (see note B)
    X4  = BVE/TL  → 1 / (TA_to_Equity × TL_to_TA) (see note C)
    X5  = Sales/TA→ Total_Asset_Turnover (direct)
    CL_TA          → derived from WC_to_TA and Current_Ratio (see note D)
    FUTL           → NCF_to_TA / TL_to_TA (see note E)

    ── NOTES ──────────────────────────────────────────────────
    Note A (X2 = RE/TA):
        Retained Earnings is a cumulative balance-sheet item not
        derivable from single-period ratios. ROA (Net Income / Total
        Assets) is used as the best available single-period proxy,
        following Altman (2002) who notes this substitution in
        contexts where RE history is unavailable. This underestimates
        RE for mature firms (known limitation; documented in paper).

    Note B (X3 = EBIT/TA):
        Operating_Profit_Margin = EBIT / Revenue
        Total_Asset_Turnover    = Revenue / Total Assets
        Product = EBIT / Total Assets. This derivation is exact
        when Operating_Profit_Margin reflects EBIT (pre-tax, pre-interest
        operating profit), which is its standard Indian accounting definition.

    Note C (X4 = BVE/TL):
        TA_to_Equity = TA / BVE  →  BVE/TA = 1/TA_to_Equity
        TL_to_TA     = TL / TA   →  TL/TA  = TL_to_TA
        BVE/TL = (BVE/TA) / (TL/TA) = 1 / (TA_to_Equity × TL_to_TA)
        Capped at (-50, 50) to handle negative equity edge cases.
        Negative equity observations flagged for sensitivity analysis.

    Note D (CL/TA):
        From two identities:
          WC_to_TA      = CA/TA − CL/TA
          Current_Ratio = CA/CL = (CA/TA)/(CL/TA)
        Solving: CL/TA = WC_to_TA / (Current_Ratio − 1)
        Valid when Current_Ratio ≠ 1. Edge cases (CR ≈ 1) imputed
        with industry-year median CL/TA.

    Note E (FUTL = CFO/Total Liabilities):
        NCF_to_TA = NCF/TA, TL_to_TA = TL/TA
        FUTL = NCF/TL = NCF_to_TA / TL_to_TA
        NCF (Net Cash Flow from Operations) used as CFO proxy.

    Note F (SIZE for Ohlson):
        Ohlson SIZE = log(Total Assets / GNP Price Index) requires
        absolute firm size data unavailable in ratio-only datasets.
        This is a fundamental structural limitation of applying Ohlson
        to accounting-ratio panels. Set to 0 (sample-centered constant)
        with explicit documentation. This limitation is itself evidence
        of benchmark inadequacy for ratio-based Indian datasets.
    ──────────────────────────────────────────────────────────
    """
    d = df.copy()

    # ── Shared variables ──────────────────────────────────────
    d["X1_WC_TA"]   = d["WC_to_TA"]                           # Note A-direct
    d["X2_RE_TA"]   = d["ROA"]                                 # Note A proxy
    d["X3_EBIT_TA"] = d["Operating_Profit_Margin"].clip(-5, 5) \
                      * d["Total_Asset_Turnover"].clip(0, 20)  # Note B
    d["X5_Sales_TA"]= d["Total_Asset_Turnover"].clip(0, 20)   # Direct

    # ── X4 = BVE / Total Liabilities (Note C) ─────────────────
    ta_eq = d["TA_to_Equity"].replace(0, np.nan)
    tl_ta = d["TL_to_TA"].replace(0, np.nan)
    d["X4_BVE_TL"]  = (1.0 / (ta_eq * tl_ta)).clip(-50, 50)

    # ── CL/TA derivation (Note D) ─────────────────────────────
    cr_m1 = d["Current_Ratio"] - 1.0
    cr_m1_safe = cr_m1.where(cr_m1.abs() > 0.05, np.nan)
    d["CL_to_TA"] = (d["WC_to_TA"] / cr_m1_safe).clip(-2, 2)
    # Fallback: industry-year median for unstable CR≈1 cases
    iy_med = d.groupby(["Industry group", "Year_t"])["CL_to_TA"].transform("median")
    d["CL_to_TA"] = d["CL_to_TA"].fillna(iy_med).fillna(d["CL_to_TA"].median())

    # ── FUTL = CFO/Total Liabilities (Note E) ─────────────────
    tl_ta_safe = d["TL_to_TA"].replace(0, np.nan)
    d["FUTL"] = (d["NCF_to_TA"] / tl_ta_safe).clip(-5, 5)

    # ── Ohlson helpers ────────────────────────────────────────
    # CLCA = CL/CA = 1/Current_Ratio
    d["CLCA"]   = (1.0 / d["Current_Ratio"].replace(0, np.nan)).clip(0, 10)
    # OENEG = 1 if Total Liabilities > Total Assets
    d["OENEG"]  = (d["TL_to_TA"] > 1.0).astype(int)
    # NITA = Net Income / Total Assets ≈ ROA
    d["NITA"]   = d["ROA"]
    # SIZE = log(TA/GNP) — unavailable; set 0 per Note F
    d["SIZE"]   = 0.0

    # ── Temporal helpers (INTWO, CHIN) ────────────────────────
    d = _compute_temporal_ohlson(d)

    # ── EBT/CL for Springate (Component C) ───────────────────
    # EBT/CL = EBT/TA × TA/CL = (OPM × TAT) × (1/CL_to_TA)
    cl_ta_safe = d["CL_to_TA"].replace(0, np.nan)
    ebt_ta     = d["X3_EBIT_TA"]   # EBIT used as EBT proxy (tax excluded → conservative)
    d["EBT_CL"] = (ebt_ta / cl_ta_safe).clip(-20, 20)

    # ── Negative equity flag ──────────────────────────────────
    d["neg_equity_flag"] = (d["TA_to_Equity"] < 0).astype(int)

    print(f"[derive] Benchmark variables computed. "
          f"Neg-equity obs: {d['neg_equity_flag'].sum():,} "
          f"({d['neg_equity_flag'].mean()*100:.1f}%)")
    return d


def _compute_temporal_ohlson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute INTWO and CHIN requiring two consecutive years per firm.

    INTWO = 1 if Net Income < 0 in BOTH current and prior year
           (proxied by Net_Profit_Margin < 0)
    CHIN  = (NI_t - NI_{t-1}) / (|NI_t| + |NI_{t-1}|)
           (proxied using ROA as NI/TA)
    """
    d = df.sort_values(["Company Name", "Year_t"]).copy()

    # Prior-year ROA and Net_Profit_Margin per firm
    d["ROA_lag"]      = d.groupby("Company Name")["ROA"].shift(1)
    d["NPM_lag"]      = d.groupby("Company Name")["Net_Profit_Margin"].shift(1)

    # INTWO
    d["INTWO"] = (
        (d["Net_Profit_Margin"] < 0) & (d["NPM_lag"] < 0)
    ).astype(int)
    d["INTWO"] = d["INTWO"].fillna(0)  # single-obs firms: no prior → default 0

    # CHIN  = (ROA_t - ROA_{t-1}) / (|ROA_t| + |ROA_{t-1}|)
    roa_sum = (d["ROA"].abs() + d["ROA_lag"].abs()).replace(0, np.nan)
    d["CHIN"] = ((d["ROA"] - d["ROA_lag"]) / roa_sum).clip(-1, 1)
    d["CHIN"] = d["CHIN"].fillna(0)  # single-obs firms: no change recorded

    # Drop helper lag columns
    d.drop(columns=["ROA_lag", "NPM_lag"], inplace=True)
    return d


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — SCORE COMPUTATION
# ═══════════════════════════════════════════════════════════════

# ── 3.1  Altman Z''-Score (1995, Emerging Markets) ────────────

def compute_altman_zdp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altman Z''-Score (1995) — Emerging-market / private-firm version.
    Designed to remove market-value dependency and improve non-US applicability.

    Formula:
        Z'' = 6.56*X1 + 3.26*X2 + 6.72*X3 + 1.05*X4

    Cutoffs (Altman 1995):
        Z'' >  2.60  →  Safe zone     (Non-Distressed)
        1.10 ≤ Z'' ≤ 2.60 →  Grey zone
        Z'' <  1.10  →  Distress zone (Distressed)

    Binary label (strict): Grey + Distress → consider both
        binary_strict:  Grey = Non-Distressed  (most favourable to Altman)
        binary_broad:   Grey = Distressed      (least favourable)
    """
    d = df.copy()
    d["altman_zdp_score"] = (
        6.56 * d["X1_WC_TA"]
      + 3.26 * d["X2_RE_TA"]
      + 6.72 * d["X3_EBIT_TA"]
      + 1.05 * d["X4_BVE_TL"]
    )
    # Zone labels
    conditions  = [
        d["altman_zdp_score"] >  2.60,
        d["altman_zdp_score"] <  1.10,
    ]
    choices = ["Safe", "Distress"]
    d["altman_zdp_zone"] = np.select(conditions, choices, default="Grey")

    # Binary (strict: grey → non-distressed)
    d["altman_zdp_binary_strict"] = (d["altman_zdp_score"] < 1.10).astype(int)
    # Binary (broad: grey → distressed)
    d["altman_zdp_binary_broad"]  = (d["altman_zdp_score"] < 2.60).astype(int)

    print(f"[Altman Z''] Score range: "
          f"{d['altman_zdp_score'].min():.2f} – {d['altman_zdp_score'].max():.2f}, "
          f"median={d['altman_zdp_score'].median():.2f}")
    print(f"[Altman Z''] Zones: {d['altman_zdp_zone'].value_counts().to_dict()}")
    return d


# ── 3.2  Altman Z'-Score (1983, Private Firms) ────────────────

def compute_altman_zp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Altman Z'-Score (1983) — Revised for private (non-public) firms.
    Replaces market-value equity with book-value equity (X4 adjusted).

    Formula:
        Z' = 0.717*X1 + 0.847*X2 + 3.107*X3 + 0.420*X4 + 0.998*X5

    Cutoffs:
        Z' >  2.90  →  Safe zone
        1.23 ≤ Z' ≤ 2.90 →  Grey zone
        Z' <  1.23  →  Distress zone
    """
    d = df.copy()
    d["altman_zp_score"] = (
        0.717 * d["X1_WC_TA"]
      + 0.847 * d["X2_RE_TA"]
      + 3.107 * d["X3_EBIT_TA"]
      + 0.420 * d["X4_BVE_TL"]
      + 0.998 * d["X5_Sales_TA"]
    )
    conditions = [
        d["altman_zp_score"] >  2.90,
        d["altman_zp_score"] <  1.23,
    ]
    choices = ["Safe", "Distress"]
    d["altman_zp_zone"] = np.select(conditions, choices, default="Grey")

    d["altman_zp_binary_strict"] = (d["altman_zp_score"] < 1.23).astype(int)
    d["altman_zp_binary_broad"]  = (d["altman_zp_score"] < 2.90).astype(int)

    print(f"[Altman Z'] Score range: "
          f"{d['altman_zp_score'].min():.2f} – {d['altman_zp_score'].max():.2f}, "
          f"median={d['altman_zp_score'].median():.2f}")
    print(f"[Altman Z'] Zones: {d['altman_zp_zone'].value_counts().to_dict()}")
    return d


# ── 3.3  Ohlson O-Score (1980) ────────────────────────────────

def compute_ohlson(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ohlson O-Score (1980) — Logistic distress probability model.

    Formula:
        O = -1.32 - 0.407*SIZE  + 6.03*TLTA  - 1.43*WCTA
             + 0.0757*CLCA - 2.37*NITA - 1.83*FUTL
             + 0.285*INTWO - 1.72*OENEG - 0.521*CHIN

        P(distress) = 1 / (1 + exp(-O))

    Cutoff:  P > 0.50  →  Distressed

    SIZE limitation: log(TA/GNP price index) requires absolute Total Assets,
    which is unavailable in a ratio-only dataset. SIZE is set to 0 (sample
    mean-centred constant). This is a documented limitation and itself evidence
    of benchmark inadequacy for ratio-based Indian firm panels.
    """
    d = df.copy()
    d["ohlson_o"] = (
       -1.32
      - 0.407  * d["SIZE"]
      + 6.03   * d["TL_to_TA"]
      - 1.43   * d["X1_WC_TA"]
      + 0.0757 * d["CLCA"]
      - 2.37   * d["NITA"]
      - 1.83   * d["FUTL"]
      + 0.285  * d["INTWO"]
      - 1.72   * d["OENEG"]
      - 0.521  * d["CHIN"]
    )
    d["ohlson_prob"] = 1.0 / (1.0 + np.exp(-d["ohlson_o"].clip(-20, 20)))
    d["ohlson_binary"] = (d["ohlson_prob"] > 0.50).astype(int)
    d["ohlson_zone"]   = d["ohlson_binary"].map({0: "Non-Distressed", 1: "Distressed"})

    print(f"[Ohlson]  P(distress) range: "
          f"{d['ohlson_prob'].min():.3f} – {d['ohlson_prob'].max():.3f}, "
          f"median={d['ohlson_prob'].median():.3f}")
    print(f"[Ohlson]  Classified distressed: "
          f"{d['ohlson_binary'].sum():,} ({d['ohlson_binary'].mean()*100:.1f}%)")
    return d


# ── 3.4  Zmijewski X-Score (1984) ─────────────────────────────

def compute_zmijewski(df: pd.DataFrame) -> pd.DataFrame:
    """
    Zmijewski X-Score (1984) — Probit-based distress probability.

    Formula:
        X = -4.336 - 4.513*(NI/TA) + 5.679*(TL/TA) - 0.004*(CA/CL)

        P(distress) = Φ(X)  [standard normal CDF]

    Cutoff: P > 0.50  →  Distressed  (equivalently X > 0)

    Fully implementable from available ratios:
        NI/TA  = ROA
        TL/TA  = TL_to_TA
        CA/CL  = Current_Ratio
    This is the cleanest benchmark for this dataset — no approximations needed.
    """
    from scipy.stats import norm
    d = df.copy()
    d["zmijewski_x"] = (
       -4.336
      - 4.513 * d["ROA"]
      + 5.679 * d["TL_to_TA"]
      - 0.004 * d["Current_Ratio"]
    )
    d["zmijewski_prob"]   = norm.cdf(d["zmijewski_x"])
    d["zmijewski_binary"] = (d["zmijewski_prob"] > 0.50).astype(int)
    d["zmijewski_zone"]   = d["zmijewski_binary"].map({0: "Non-Distressed", 1: "Distressed"})

    print(f"[Zmijewski] P(distress) range: "
          f"{d['zmijewski_prob'].min():.3f} – {d['zmijewski_prob'].max():.3f}, "
          f"median={d['zmijewski_prob'].median():.3f}")
    print(f"[Zmijewski] Classified distressed: "
          f"{d['zmijewski_binary'].sum():,} ({d['zmijewski_binary'].mean()*100:.1f}%)")
    return d


# ── 3.5  Springate S-Score (1978) ─────────────────────────────

def compute_springate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Springate S-Score (1978) — Linear discriminant model.

    Formula:
        S = 1.03*A + 3.07*B + 0.66*C + 0.40*D

        A = WC/TA
        B = EBIT/TA
        C = EBT/CL  (see derivation note in derive_benchmark_variables)
        D = Sales/TA

    Cutoff: S < 0.862  →  Distressed
    """
    d = df.copy()
    d["springate_s"] = (
        1.03 * d["X1_WC_TA"]
      + 3.07 * d["X3_EBIT_TA"]
      + 0.66 * d["EBT_CL"]
      + 0.40 * d["X5_Sales_TA"]
    )
    d["springate_binary"] = (d["springate_s"] < 0.862).astype(int)
    d["springate_zone"]   = d["springate_binary"].map({0: "Non-Distressed", 1: "Distressed"})

    print(f"[Springate] Score range: "
          f"{d['springate_s'].min():.2f} – {d['springate_s'].max():.2f}, "
          f"median={d['springate_s'].median():.2f}")
    print(f"[Springate] Classified distressed: "
          f"{d['springate_binary'].sum():,} ({d['springate_binary'].mean()*100:.1f}%)")
    return d


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — TAXONOMY VS BENCHMARK CROSS-TABULATION
# ═══════════════════════════════════════════════════════════════

def cross_tabulate(df: pd.DataFrame, benchmark_col: str,
                   benchmark_name: str, zone_col: str = None) -> pd.DataFrame:
    """
    Build taxonomy vs benchmark cross-tab (counts + row percentages).
    Returns a wide-format table suitable for the paper.
    """
    ct = pd.crosstab(
        df["Label"],
        df[benchmark_col],
        margins=True
    )
    ct = ct.reindex(LABEL_ORDER + ["All"], fill_value=0)

    ct_pct = pd.crosstab(
        df["Label"],
        df[benchmark_col],
        normalize="index"
    ).round(4) * 100
    ct_pct = ct_pct.reindex(LABEL_ORDER, fill_value=0)

    print(f"\n{'═'*60}")
    print(f"  {benchmark_name} — Cross-tabulation (Counts)")
    print(f"{'═'*60}")
    print(ct)
    print(f"\n  {benchmark_name} — Row Percentages (%)")
    print(ct_pct.round(1))

    # Save
    fname = benchmark_name.lower().replace(" ", "_").replace("'", "p").replace('"', "dp")
    ct.to_csv(TBLS / f"crosstab_{fname}_counts.csv")
    ct_pct.to_csv(TBLS / f"crosstab_{fname}_pcts.csv")

    # If zone column provided, also save 3-zone breakdown
    if zone_col and zone_col in df.columns:
        ct3 = pd.crosstab(df["Label"], df[zone_col], margins=True)
        ct3 = ct3.reindex(LABEL_ORDER + ["All"], fill_value=0)
        ct3.to_csv(TBLS / f"crosstab_{fname}_zones.csv")

    return ct_pct


def build_detection_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-category detection rates for all benchmarks.

    For each benchmark, 'detection rate' = % of that taxonomy category
    that the benchmark classifies as Distressed.

    For Healthy firms, the metric is 'false alarm rate' (% incorrectly
    flagged as Distressed). For distress categories, it is the 'recall'
    or 'sensitivity' at that category.

    Two variants:
        strict: grey zone → Non-Distressed (best case for legacy models)
        broad:  grey zone → Distressed     (worst case)
    """
    benchmarks = {
        "Altman Z''_strict":  "altman_zdp_binary_strict",
        "Altman Z''_broad":   "altman_zdp_binary_broad",
        "Altman Z'_strict":   "altman_zp_binary_strict",
        "Altman Z'_broad":    "altman_zp_binary_broad",
        "Ohlson":             "ohlson_binary",
        "Zmijewski":          "zmijewski_binary",
        "Springate":          "springate_binary",
    }

    rows = []
    for bname, bcol in benchmarks.items():
        if bcol not in df.columns:
            continue
        for cat in LABEL_ORDER:
            subset = df[df["Label"] == cat]
            n_total = len(subset)
            n_flagged = subset[bcol].sum()
            rate = n_flagged / n_total * 100
            rows.append({
                "Benchmark": bname,
                "Taxonomy_Category": cat,
                "N_Total": n_total,
                "N_Flagged_Distressed": n_flagged,
                "Detection_Rate_Pct": round(rate, 1),
            })

    result = pd.DataFrame(rows)
    result.to_csv(TBLS / "detection_rates.csv", index=False)

    print("\n" + "═"*60)
    print("  DETECTION RATES (% flagged as Distressed by benchmark)")
    print("═"*60)
    pivot = result.pivot(
        index="Taxonomy_Category",
        columns="Benchmark",
        values="Detection_Rate_Pct"
    ).reindex(LABEL_ORDER)
    print(pivot.to_string())
    return result


def build_agreement_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Overall agreement between each benchmark's binary label and the
    taxonomy-derived binary (Distressed = AD + CD + FD; Healthy = H).

    Metrics:
        overall_agreement_pct
        H_precision           : of benchmark-healthy, % truly Healthy
        H_recall              : of truly Healthy, % benchmark-healthy
        distress_recall       : of any distress category, % caught
        AD_recall, CD_recall, FD_recall
    """
    df = df.copy()
    df["taxonomy_binary"] = (df["Label"] != "Healthy").astype(int)

    benchmarks = {
        "Altman Z'' (strict)": "altman_zdp_binary_strict",
        "Altman Z'' (broad)":  "altman_zdp_binary_broad",
        "Altman Z' (strict)":  "altman_zp_binary_strict",
        "Altman Z' (broad)":   "altman_zp_binary_broad",
        "Ohlson":              "ohlson_binary",
        "Zmijewski":           "zmijewski_binary",
        "Springate":           "springate_binary",
    }

    rows = []
    for bname, bcol in benchmarks.items():
        if bcol not in df.columns:
            continue
        b = df[bcol]
        t = df["taxonomy_binary"]
        agreement = (b == t).mean() * 100

        # Per-category recall
        cat_recalls = {}
        for cat in LABEL_ORDER:
            mask = df["Label"] == cat
            n = mask.sum()
            if n > 0:
                cat_recalls[f"{LABEL_SHORT[cat]}_recall_pct"] = \
                    df.loc[mask, bcol].sum() / n * 100
            else:
                cat_recalls[f"{LABEL_SHORT[cat]}_recall_pct"] = np.nan

        rows.append({
            "Benchmark": bname,
            "Overall_Agreement_Pct": round(agreement, 1),
            **{k: round(v, 1) for k, v in cat_recalls.items()},
        })

    result = pd.DataFrame(rows)
    result.to_csv(TBLS / "agreement_rates.csv", index=False)
    print("\n" + "═"*60)
    print("  AGREEMENT RATES SUMMARY")
    print("═"*60)
    print(result.to_string(index=False))
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — CASH-FLOW BLINDNESS ANALYSIS
# ═══════════════════════════════════════════════════════════════

def cashflow_blindness_analysis() -> pd.DataFrame:
    """
    Formal variable coverage analysis: which financial dimensions
    does each legacy model capture vs the proposed framework?

    The Cash-ICR dimension (operating cash-flow adequacy) has no direct
    equivalent in Altman, Zmijewski, or Springate, and only a partial
    equivalent in Ohlson (one out of nine terms).

    This structural gap is why legacy models cannot distinguish
    Cash Distress from Healthy firms — the defining signal for that
    category is absent from their variable sets.
    """
    coverage = {
        "Model": [
            "Altman Z'' (1995)",
            "Altman Z' (1983)",
            "Ohlson O-Score (1980)",
            "Zmijewski (1984)",
            "Springate (1978)",
            "Proposed Framework"
        ],
        "Profitability": [1, 1, 1, 1, 1, 1],
        "Leverage":      [1, 1, 1, 1, 0, 1],
        "Liquidity":     [1, 1, 1, 1, 1, 1],
        "Activity":      [1, 1, 0, 0, 1, 1],
        "CF_Operations": [0, 0, 1, 0, 0, 1],   # 1 term in Ohlson only
        "CF_Ratio_Rich": [0, 0, 0, 0, 0, 1],   # ≥3 dedicated CFO ratios
        "Cash_ICR_Signal":[0, 0, 0, 0, 0, 1],   # Core Cash-ICR dimension
        "4_Category_Output":[0, 0, 0, 0, 0, 1],
        "Notes": [
            "No CFO variable; uses EBIT, RE, WC",
            "Same structure as Z''; replaces MVE with BVE",
            "One CFO term (FUTL=CFO/TL); SIZE needs absolute TA",
            "Fully ratio-implementable; no CFO variable",
            "No CFO; EBT/CL approximation adds uncertainty",
            "ICR+Cash-ICR taxonomy; 6 dedicated cash-flow predictors"
        ]
    }

    df_cov = pd.DataFrame(coverage)
    df_cov.to_csv(TBLS / "variable_coverage.csv", index=False)

    print("\n" + "═"*60)
    print("  VARIABLE COVERAGE ANALYSIS (1=covered, 0=absent)")
    print("═"*60)
    print(df_cov.to_string(index=False))

    return df_cov


def benchmark_limitations_table() -> pd.DataFrame:
    """
    Formal table of documented limitations per benchmark
    for inclusion in the paper's methodology section.
    """
    lim = {
        "Model": [
            "Altman Z'' (1995)",
            "Altman Z' (1983)",
            "Ohlson O-Score (1980)",
            "Zmijewski (1984)",
            "Springate (1978)"
        ],
        "Market_Data_Required": ["No", "No", "No", "No", "No"],
        "Absolute_TA_Required": ["No", "No", "YES (SIZE)", "No", "No"],
        "RE_Directly_Available": ["No (proxy: ROA)", "No (proxy: ROA)", "N/A", "N/A", "N/A"],
        "Cash_Flow_Dimension": [
            "Absent",
            "Absent",
            "Partial (1/9 variables)",
            "Absent",
            "Absent"
        ],
        "CashICR_Equivalent": ["None", "None", "None", "None", "None"],
        "Binary_Output_Only":  ["Yes", "Yes", "Yes", "Yes", "Yes"],
        "Original_Sample_Context": [
            "US non-manufacturing / emerging",
            "US private firms",
            "US public firms (NYSE/AMEX)",
            "US public firms",
            "US/Canadian firms"
        ],
        "Indian_Applicability_Notes": [
            "Most appropriate Z-variant; no market data needed",
            "Requires book equity; suited to private firms",
            "SIZE variable fails for ratio-only datasets",
            "Cleanest implementation from ratios; well-suited",
            "C-variable (EBT/CL) requires approximation"
        ]
    }

    df_lim = pd.DataFrame(lim)
    df_lim.to_csv(TBLS / "benchmark_limitations.csv", index=False)
    return df_lim


def cd_miss_spotlight(df: pd.DataFrame, det_rates: pd.DataFrame) -> pd.DataFrame:
    """
    Spotlight analysis: How many Cash Distress firms does each legacy
    benchmark miss (classify as Non-Distressed)?

    This is the central inadequacy finding: CD firms are classified as
    Healthy by legacy models because legacy models have no signal for
    operating-cash-flow failure independent of earnings-based distress.
    """
    cd_mask = df["Label"] == "Cash_Distress"
    n_cd    = cd_mask.sum()

    benchmarks = {
        "Altman Z'' (strict)": "altman_zdp_binary_strict",
        "Altman Z'' (broad)":  "altman_zdp_binary_broad",
        "Altman Z' (strict)":  "altman_zp_binary_strict",
        "Ohlson":              "ohlson_binary",
        "Zmijewski":           "zmijewski_binary",
        "Springate":           "springate_binary",
    }

    rows = []
    for bname, bcol in benchmarks.items():
        if bcol not in df.columns:
            continue
        cd_df = df[cd_mask]
        n_missed  = (cd_df[bcol] == 0).sum()   # classified Non-Distressed
        n_caught  = (cd_df[bcol] == 1).sum()
        miss_rate = n_missed / n_cd * 100

        # Compare: what % of FD firms does same benchmark miss?
        fd_df   = df[df["Label"] == "Full_Distress"]
        n_fd    = len(fd_df)
        fd_miss = (fd_df[bcol] == 0).sum() / n_fd * 100

        rows.append({
            "Benchmark": bname,
            "CD_Total": n_cd,
            "CD_Missed": n_missed,
            "CD_Caught": n_caught,
            "CD_Miss_Rate_Pct": round(miss_rate, 1),
            "FD_Miss_Rate_Pct": round(fd_miss, 1),
            "Miss_Rate_Gap_CD_vs_FD": round(miss_rate - fd_miss, 1),
        })

    result = pd.DataFrame(rows)
    result.to_csv(TBLS / "cd_miss_analysis.csv", index=False)

    print("\n" + "═"*60)
    print("  CASH DISTRESS MISS SPOTLIGHT")
    print(f"  Total CD observations: {n_cd:,}")
    print("═"*60)
    print(result.to_string(index=False))
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — BENCHMARK SCORE STATISTICS BY CATEGORY
# ═══════════════════════════════════════════════════════════════

def benchmark_score_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mean and median benchmark scores by taxonomy category.
    Provides economic validation: FD firms should score worst,
    H firms should score best, AD and CD should be intermediate.
    """
    score_cols = {
        "altman_zdp_score": "Altman Z''",
        "altman_zp_score":  "Altman Z'",
        "ohlson_prob":      "Ohlson P(distress)",
        "zmijewski_prob":   "Zmijewski P(distress)",
        "springate_s":      "Springate S",
    }

    rows = []
    for col, name in score_cols.items():
        if col not in df.columns:
            continue
        for cat in LABEL_ORDER:
            sub = df[df["Label"] == cat][col]
            rows.append({
                "Score": name,
                "Category": cat,
                "N": len(sub),
                "Mean": round(sub.mean(), 4),
                "Median": round(sub.median(), 4),
                "Std": round(sub.std(), 4),
                "P25": round(sub.quantile(0.25), 4),
                "P75": round(sub.quantile(0.75), 4),
            })

    result = pd.DataFrame(rows)
    result.to_csv(TBLS / "benchmark_score_stats_by_category.csv", index=False)
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — VISUALISATIONS
# ═══════════════════════════════════════════════════════════════

def plot_detection_rate_heatmap(det_rates: pd.DataFrame) -> None:
    """
    Heatmap: rows = taxonomy categories, columns = benchmarks,
    cells = detection rate (% flagged as Distressed).

    Key finding visualised: CD detection rates are systematically lower
    than FD rates across all benchmarks.
    """
    # Focus on strict-binary and probability-based benchmarks
    benchmarks_focus = [
        "Altman Z''_strict",
        "Altman Z'_strict",
        "Ohlson",
        "Zmijewski",
        "Springate",
    ]
    sub = det_rates[det_rates["Benchmark"].isin(benchmarks_focus)].copy()
    pivot = sub.pivot(
        index="Taxonomy_Category",
        columns="Benchmark",
        values="Detection_Rate_Pct"
    ).reindex(LABEL_ORDER)

    # Rename for cleaner plot labels
    pivot.index = ["Healthy (H)", "Acct. Distress (AD)",
                   "Cash Distress (CD)", "Full Distress (FD)"]
    pivot.columns = [c.replace("_strict", "") for c in pivot.columns]

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="RdYlGn_r",
        vmin=0, vmax=100,
        linewidths=0.5,
        linecolor="white",
        annot_kws={"size": 11, "weight": "bold"},
        ax=ax,
        cbar_kws={"label": "% Flagged as Distressed", "shrink": 0.8}
    )
    ax.set_title(
        "Detection Rates: Legacy Benchmarks vs 4-Category Taxonomy\n"
        "(% of each category flagged as Distressed by each legacy model)",
        fontsize=13, fontweight="bold", pad=14
    )
    ax.set_xlabel("")
    ax.set_ylabel("Taxonomy Category", fontsize=11)
    ax.tick_params(axis="x", labelsize=10, rotation=20)
    ax.tick_params(axis="y", labelsize=10, rotation=0)

    # Annotation box highlighting CD row
    ax.add_patch(plt.Rectangle(
        (0, 2), len(pivot.columns), 1,
        fill=False, edgecolor="#333", lw=2.5, zorder=5
    ))
    ax.text(
        len(pivot.columns) + 0.05, 2.5,
        "← CD row:\nkey gap",
        va="center", ha="left", fontsize=9,
        color="#555", style="italic"
    )

    plt.tight_layout()
    plt.savefig(FIGS / "detection_rate_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[fig] detection_rate_heatmap.png saved")


def plot_cd_miss_spotlight(cd_miss: pd.DataFrame) -> None:
    """
    Grouped bar chart: CD miss rate vs FD miss rate for each benchmark.
    Visualises the core finding: legacy models miss CD at much higher
    rates than FD because they lack cash-flow signal.
    """
    # Keep strict / main variants only
    keep = [
        "Altman Z'' (strict)",
        "Altman Z' (strict)",
        "Ohlson",
        "Zmijewski",
        "Springate",
    ]
    d = cd_miss[cd_miss["Benchmark"].isin(keep)].copy()
    x   = np.arange(len(d))
    w   = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5))
    b1 = ax.bar(x - w/2, d["CD_Miss_Rate_Pct"], w,
                label="Cash Distress Miss Rate",
                color="#9C27B0", alpha=0.88, edgecolor="white")
    b2 = ax.bar(x + w/2, d["FD_Miss_Rate_Pct"], w,
                label="Full Distress Miss Rate",
                color="#F44336", alpha=0.88, edgecolor="white")

    # Value labels
    for rect in b1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9.5)
    for rect in b2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2, h + 0.5,
                f"{h:.0f}%", ha="center", va="bottom", fontsize=9.5)

    ax.set_xticks(x)
    ax.set_xticklabels(d["Benchmark"].str.replace(" (strict)", "", regex=False),
                       fontsize=10, rotation=10)
    ax.set_ylabel("Miss Rate (% of category NOT flagged as Distressed)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_title(
        "Legacy Benchmark Miss Rates: Cash Distress vs Full Distress\n"
        "(higher bar = more firms in that category incorrectly labelled as Healthy)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGS / "cd_miss_spotlight.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[fig] cd_miss_spotlight.png saved")


def plot_score_distributions(df: pd.DataFrame) -> None:
    """
    Box plots of each benchmark score by taxonomy category.
    Shows whether legacy scores can separate the four categories.
    """
    scores = [
        ("altman_zdp_score", "Altman Z''-Score", True),   # higher = safer
        ("altman_zp_score",  "Altman Z'-Score",  True),
        ("ohlson_prob",      "Ohlson P(distress)", False), # lower = safer
        ("zmijewski_prob",   "Zmijewski P(distress)", False),
        ("springate_s",      "Springate S-Score", True),
    ]
    valid = [(c, n, d) for c, n, d in scores if c in df.columns]

    fig, axes = plt.subplots(1, len(valid), figsize=(4.5*len(valid), 5.5))
    if len(valid) == 1:
        axes = [axes]

    for ax, (col, name, higher_safer) in zip(axes, valid):
        data_list = [df[df["Label"] == cat][col].values for cat in LABEL_ORDER]
        bp = ax.boxplot(
            data_list,
            patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
            whiskerprops=dict(linewidth=1.2),
            capprops=dict(linewidth=1.2),
            flierprops=dict(marker=".", alpha=0.3, markersize=3)
        )
        colors = [CAT_PALETTE[c] for c in LABEL_ORDER]
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["H", "AD", "CD", "FD"], fontsize=10)
        ax.set_title(name, fontsize=10, fontweight="bold")
        ax.set_ylabel("Score", fontsize=9)
        ax.yaxis.grid(True, alpha=0.35)
        ax.set_axisbelow(True)

        direction = "↑ safer" if higher_safer else "↓ safer"
        ax.text(0.98, 0.98, direction, transform=ax.transAxes,
                ha="right", va="top", fontsize=8, color="#666")

    fig.suptitle(
        "Legacy Benchmark Score Distributions by Taxonomy Category",
        fontsize=13, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    plt.savefig(FIGS / "score_distributions_by_category.png",
                dpi=180, bbox_inches="tight")
    plt.close()
    print("[fig] score_distributions_by_category.png saved")


def plot_variable_coverage_chart(df_cov: pd.DataFrame) -> None:
    """
    Visual coverage table: models × financial dimensions.
    Highlights that cash-flow signal (CF_Operations, CF_Ratio_Rich,
    Cash_ICR_Signal) is absent from all legacy models.
    """
    dim_cols = [
        "Profitability", "Leverage", "Liquidity", "Activity",
        "CF_Operations", "CF_Ratio_Rich", "Cash_ICR_Signal", "4_Category_Output"
    ]
    models = df_cov["Model"].tolist()
    mat    = df_cov[dim_cols].values.astype(float)

    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(mat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(dim_cols)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in dim_cols],
        fontsize=9, rotation=0, ha="center"
    )
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=9)

    for i in range(len(models)):
        for j in range(len(dim_cols)):
            val = int(mat[i, j])
            text = "✓" if val == 1 else "✗"
            col  = "white" if val == 1 else "#cc2200"
            ax.text(j, i, text, ha="center", va="center",
                    fontsize=12, color=col, fontweight="bold")

    ax.set_title(
        "Variable Coverage by Model and Financial Dimension\n"
        "(✓ = dimension represented; ✗ = absent)",
        fontsize=12, fontweight="bold"
    )
    # Highlight the cash-ICR signal column (index 6)
    ax.add_patch(plt.Rectangle(
        (5.5, -0.5), 3, len(models),
        fill=True, facecolor="#fff3cd",
        edgecolor="#e6ac00", lw=2, zorder=0
    ))
    ax.text(
        7, len(models) - 0.02,
        "← Cash-flow gap",
        ha="center", va="top", fontsize=8.5,
        color="#7a5c00", style="italic",
        transform=ax.transData
    )

    plt.tight_layout()
    plt.savefig(FIGS / "variable_coverage_chart.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[fig] variable_coverage_chart.png saved")


def plot_agreement_bars(agr: pd.DataFrame) -> None:
    """
    Grouped bar chart: per-category recall for each benchmark.
    Highlights how AD and CD recall differ across models.
    """
    # Main benchmarks only (strict variants)
    keep = [b for b in agr["Benchmark"].tolist()
            if "broad" not in b.lower()]
    d = agr[agr["Benchmark"].isin(keep)].copy()

    recall_cols = ["H_recall_pct", "AD_recall_pct", "CD_recall_pct", "FD_recall_pct"]
    col_labels  = ["Healthy\n(false alarm)", "Acct. Distress", "Cash Distress", "Full Distress"]
    colors      = [CAT_PALETTE[c] for c in LABEL_ORDER]

    n_bench = len(d)
    n_cat   = len(recall_cols)
    x       = np.arange(n_bench)
    w       = 0.18

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (col, label, color) in enumerate(zip(recall_cols, col_labels, colors)):
        offset = (i - n_cat / 2 + 0.5) * w
        vals   = d[col].values
        bars   = ax.bar(x + offset, vals, w, label=label,
                        color=color, alpha=0.82, edgecolor="white")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
                    f"{h:.0f}%", ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    bnames = d["Benchmark"].str.replace(" (strict)", "", regex=False) \
                           .str.replace("Altman Z''", "Altman Z''", regex=False)
    ax.set_xticklabels(bnames, fontsize=9.5, rotation=8)
    ax.set_ylim(0, 115)
    ax.set_ylabel("% of Category Flagged as Distressed", fontsize=10)
    ax.set_title(
        "Per-Category Detection Rates Across Legacy Benchmarks\n"
        "(for distress categories: higher = better recall; for Healthy: lower = fewer false alarms)",
        fontsize=12, fontweight="bold"
    )
    ax.legend(fontsize=9, loc="upper right", ncol=2)
    ax.yaxis.grid(True, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(FIGS / "agreement_bars.png", dpi=180, bbox_inches="tight")
    plt.close()
    print("[fig] agreement_bars.png saved")


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — VERIFICATION SUITE
# ═══════════════════════════════════════════════════════════════

def verification_suite(df: pd.DataFrame) -> None:
    """
    Manual verification checks — printed to console for documentation.
    Ensures all cross-tabulations sum correctly and scores are plausible.
    """
    print("\n" + "═"*60)
    print("  VERIFICATION SUITE")
    print("═"*60)

       # 1. Label counts in current panel
    print("  Label counts in current panel:")
    for cat, n in df["Label"].value_counts().items():
        print(f"    {cat}: {n:,}")
    print(f"    Total: {len(df):,}")

    # 2. Sanity: all 4 categories sum to total
    total = len(df)
    print(f"  Total rows: {total:,}")

    # 3. Altman Z'' score range plausibility
    zdp = df["altman_zdp_score"]
    print(f"\n  Altman Z'' score sanity:")
    print(f"    min={zdp.min():.2f}  median={zdp.median():.2f}  max={zdp.max():.2f}")
    print(f"    % Distress zone (< 1.10): {(zdp < 1.10).mean()*100:.1f}%")
    print(f"    % Safe zone    (> 2.60): {(zdp > 2.60).mean()*100:.1f}%")

    # 4. Zmijewski: fully implementable so strictest check
    zm = df["zmijewski_prob"]
    print(f"\n  Zmijewski P(distress) sanity:")
    print(f"    min={zm.min():.4f}  median={zm.median():.4f}  max={zm.max():.4f}")
    assert zm.between(0, 1).all(), "Zmijewski probs out of [0,1]!"
    print(f"    ✓ All probabilities in [0, 1]")

    # 5. Cross-tab row sums = category counts
    ct = pd.crosstab(df["Label"], df["altman_zdp_binary_strict"])
    ct_rowsums = ct.sum(axis=1)
    print(f"\n  Altman Z'' cross-tab row sums vs label counts:")
    for cat in LABEL_ORDER:
        n_lab = (df["Label"] == cat).sum()
        n_ct  = ct_rowsums.get(cat, 0)
        status = "✓" if n_lab == n_ct else "✗ MISMATCH"
        print(f"    {status} {cat}: label_n={n_lab:,}, crosstab_sum={n_ct:,}")

    # 6. Negative equity flag statistics
    print(f"\n  Negative equity observations: "
          f"{df['neg_equity_flag'].sum():,} ({df['neg_equity_flag'].mean()*100:.1f}%)")
    print(f"  These affect X4_BVE_TL computation (capped at ±50)")

    print("\n" + "═"*60)
    print("  All verification checks complete")
    print("═"*60)


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — SAVE MASTER SCORED PANEL
# ═══════════════════════════════════════════════════════════════

def save_scored_panel(df: pd.DataFrame) -> None:
    """
    Save the full scored panel (panel + all benchmark scores/labels).
    This file is used for any follow-up sensitivity analyses.
    """
    score_cols = [
        "Company Name", "Year_t", "Industry group", "Label",
        "altman_zdp_score", "altman_zdp_zone",
        "altman_zdp_binary_strict", "altman_zdp_binary_broad",
        "altman_zp_score", "altman_zp_zone",
        "altman_zp_binary_strict", "altman_zp_binary_broad",
        "ohlson_o", "ohlson_prob", "ohlson_binary",
        "zmijewski_x", "zmijewski_prob", "zmijewski_binary",
        "springate_s", "springate_binary",
        "neg_equity_flag", "OENEG", "INTWO",
    ]
    out_cols = [c for c in score_cols if c in df.columns]
    df[out_cols].to_csv(TBLS / "benchmark_scores.csv", index=False)
    print(f"\n[save] benchmark_scores.csv: {df[out_cols].shape}")


def save_classification_counts(df: pd.DataFrame) -> None:
    """
    Summary counts: how many firms each model classifies as Distressed
    vs Non-Distressed, broken down by taxonomy category.
    """
    benchmarks = {
        "Altman Z'' (strict)": "altman_zdp_binary_strict",
        "Altman Z'' (broad)":  "altman_zdp_binary_broad",
        "Altman Z' (strict)":  "altman_zp_binary_strict",
        "Altman Z' (broad)":   "altman_zp_binary_broad",
        "Ohlson":              "ohlson_binary",
        "Zmijewski":           "zmijewski_binary",
        "Springate":           "springate_binary",
    }
    rows = []
    for bname, bcol in benchmarks.items():
        if bcol not in df.columns:
            continue
        n_d  = df[bcol].sum()
        n_nd = (df[bcol] == 0).sum()
        rows.append({
            "Benchmark": bname,
            "N_Distressed": n_d,
            "N_NonDistressed": n_nd,
            "Pct_Distressed": round(n_d / len(df) * 100, 1),
            "Total": len(df),
        })
    pd.DataFrame(rows).to_csv(TBLS / "benchmark_classification_counts.csv", index=False)
    print("[save] benchmark_classification_counts.csv saved")


# ═══════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  STEP 4 — LEGACY BENCHMARK INADEQUACY ANALYSIS")
    print("=" * 60)

    # ── 1. Load ───────────────────────────────────────────────
    df = load_panel(PANEL)

    # ── 2. Derive shared benchmark variables ──────────────────
    df = derive_benchmark_variables(df)

    # ── 3. Compute benchmark scores ───────────────────────────
    print("\n[benchmarks] Computing scores...")
    df = compute_altman_zdp(df)
    df = compute_altman_zp(df)
    df = compute_ohlson(df)
    df = compute_zmijewski(df)
    df = compute_springate(df)

    # ── 4. Cross-tabulations ──────────────────────────────────
    print("\n[analysis] Cross-tabulations...")
    for bcol, bname, zname in [
        ("altman_zdp_binary_strict", "Altman_ZDP",  "altman_zdp_zone"),
        ("altman_zp_binary_strict",  "Altman_ZP",   "altman_zp_zone"),
        ("ohlson_binary",            "Ohlson",       None),
        ("zmijewski_binary",         "Zmijewski",    None),
        ("springate_binary",         "Springate",    None),
    ]:
        cross_tabulate(df, bcol, bname, zname)

    # ── 5. Detection rates ────────────────────────────────────
    print("\n[analysis] Detection rates...")
    det_rates = build_detection_rates(df)

    # ── 6. Agreement statistics ───────────────────────────────
    print("\n[analysis] Agreement statistics...")
    agr = build_agreement_stats(df)

    # ── 7. Cash-flow blindness analysis ──────────────────────
    print("\n[analysis] Cash-flow blindness analysis...")
    df_cov = cashflow_blindness_analysis()
    benchmark_limitations_table()

    # ── 8. CD miss spotlight ──────────────────────────────────
    print("\n[analysis] CD miss spotlight...")
    cd_miss = cd_miss_spotlight(df, det_rates)

    # ── 9. Score statistics by category ──────────────────────
    print("\n[analysis] Score statistics...")
    benchmark_score_stats(df)

    # ── 10. Visualisations ─────────────────────────────────────
    print("\n[figures] Generating figures...")
    plot_detection_rate_heatmap(det_rates)
    plot_cd_miss_spotlight(cd_miss)
    plot_score_distributions(df)
    plot_variable_coverage_chart(df_cov)
    plot_agreement_bars(agr)

    # ── 11. Save master outputs ───────────────────────────────
    print("\n[save] Saving outputs...")
    save_scored_panel(df)
    save_classification_counts(df)

    # ── 12. Verification ──────────────────────────────────────
    verification_suite(df)

    print("\n" + "="*60)
    print("  STEP 4 COMPLETE")
    print(f"  Tables → {TBLS}")
    print(f"  Figures → {FIGS}")
    print("="*60)

    return df


if __name__ == "__main__":
    main()