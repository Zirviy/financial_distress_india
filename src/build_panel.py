"""
STEP 1 — Build Core Panel Dataset for Financial Distress Prediction
====================================================================
Research design:
- Predictor years (t): 20182023
- Label years (t+1):  20192024
- Labels defined via ICR and Cash-ICR at t+1 (excluded from predictors)
- Missing value handling: industry-year median imputation
- Outlier treatment: winsorize at 1st99th percentile within each year
- Output: one clean firm  year panel table
"""

import pandas as pd
import numpy as np
import re
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
INPUT_PATH  = "data/raw/ratios_final.csv"
OUTPUT_PATH = "data/processed/panel_dataset.csv"

# ── constants ──────────────────────────────────────────────────────────────
PREDICTOR_YEARS   = list(range(2018, 2024))   # 2018–2023
LABEL_YEARS       = [y + 1 for y in PREDICTOR_YEARS]   # 2019–2024

LABEL_ICR_BASE   = "Interest_Coverage"
LABEL_CASH_BASE  = "Cash_Interest_Coverage"

# Ratios excluded from predictors (used to define labels)
LEAKAGE_BASES = {"Interest_Coverage", "Cash_Interest_Coverage", "DSCR"}

# Growth / trend / lagged bases excluded per research design (Phase 1)
GROWTH_BASES = {
    "Revenue_Growth", "EBITDA_Growth", "Asset_Growth",
    "Liability_Growth", "Working_Capital_Growth",
}

MISSING_THRESHOLD = 0.30   # drop row if > 30 % of predictors missing
WINSOR_LOW        = 0.01
WINSOR_HIGH       = 0.99

ID_COLS = ["Company Name", "Industry group", "Industry Class", "Industry type"]


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[load]  Raw shape: {df.shape}")
    print(f"[load]  Unique firms: {df['Company Name'].nunique()}")
    return df


# ══════════════════════════════════════════════════════════════════════════
# 2. IDENTIFY PREDICTOR BASES
# ══════════════════════════════════════════════════════════════════════════
def get_predictor_bases(df: pd.DataFrame) -> list:
    """Return sorted list of ratio base names to use as predictors."""
    all_bases = set()
    for col in df.columns:
        m = re.match(r"^(.+)_(\d{4})$", col)
        if m:
            all_bases.add(m.group(1))

    excluded = LEAKAGE_BASES | GROWTH_BASES
    predictor_bases = sorted(all_bases - excluded)

    print(f"\n[features]  All ratio bases found : {len(all_bases)}")
    print(f"[features]  Excluded (leakage)    : {sorted(LEAKAGE_BASES)}")
    print(f"[features]  Excluded (growth)     : {sorted(GROWTH_BASES)}")
    print(f"[features]  Final predictor bases : {len(predictor_bases)}")
    print(f"[features]  Predictor list        : {predictor_bases}")
    return predictor_bases


# ══════════════════════════════════════════════════════════════════════════
# 3. ASSIGN DISTRESS LABEL
# ══════════════════════════════════════════════════════════════════════════
def assign_label(icr: float, cash_icr: float) -> str:
    """
    FD : ICR < 1  AND Cash-ICR < 1
    AD : ICR < 1  AND Cash-ICR >= 1
    CD : ICR >= 1 AND Cash-ICR < 0
    H  : ICR >= 1 AND Cash-ICR >= 0
    """
    if pd.isna(icr) or pd.isna(cash_icr):
        return np.nan
    if icr < 1 and cash_icr < 1:
        return "Full_Distress"
    if icr < 1 and cash_icr >= 1:
        return "Acct_Distress"
    if icr >= 1 and cash_icr < 0:
        return "Cash_Distress"
    return "Healthy"


# ══════════════════════════════════════════════════════════════════════════
# 4. BUILD LONG-FORMAT PANEL (one row per firm × predictor year)
# ══════════════════════════════════════════════════════════════════════════
def build_panel(df: pd.DataFrame, predictor_bases: list) -> pd.DataFrame:
    rows = []

    for t in PREDICTOR_YEARS:
        t1 = t + 1

        # ── label columns at t+1 ─────────────────────────────────────────
        icr_col  = f"{LABEL_ICR_BASE}_{t1}"
        cash_col = f"{LABEL_CASH_BASE}_{t1}"

        if icr_col not in df.columns or cash_col not in df.columns:
            print(f"[panel]  Skipping t={t}: label columns for {t1} not found.")
            continue

        # ── predictor columns at t ───────────────────────────────────────
        pred_cols = []
        for base in predictor_bases:
            col = f"{base}_{t}"
            if col in df.columns:
                pred_cols.append(col)

        # ── slice: firm identifiers + predictors + label inputs ──────────
        keep = ID_COLS + pred_cols + [icr_col, cash_col]
        keep = [c for c in keep if c in df.columns]
        slice_df = df[keep].copy()

        # ── assign label ─────────────────────────────────────────────────
        slice_df["Label"] = slice_df.apply(
            lambda r: assign_label(r[icr_col], r[cash_col]), axis=1
        )

        # ── drop rows where label is missing (ICR/Cash-ICR null at t+1) ──
        before = len(slice_df)
        slice_df = slice_df.dropna(subset=["Label"])
        after  = len(slice_df)
        dropped_label = before - after

        # ── drop label-construction columns (not predictors) ─────────────
        slice_df = slice_df.drop(columns=[icr_col, cash_col])

        # ── rename predictor columns: remove year suffix ─────────────────
        rename_map = {f"{base}_{t}": base for base in predictor_bases
                      if f"{base}_{t}" in slice_df.columns}
        slice_df = slice_df.rename(columns=rename_map)

        # ── add year identifier ───────────────────────────────────────────
        slice_df.insert(1, "Year_t", t)

        print(f"[panel]  t={t} → t+1={t1} | firms before label drop: {before} "
              f"| dropped (missing label): {dropped_label} | kept: {after}")

        rows.append(slice_df)

    panel = pd.concat(rows, ignore_index=True)
    print(f"\n[panel]  Combined panel shape (before cleaning): {panel.shape}")
    return panel, [r for r in predictor_bases
                   if r not in {LABEL_ICR_BASE, LABEL_CASH_BASE}]


# ══════════════════════════════════════════════════════════════════════════
# 5. DROP ROWS WITH > 30% MISSING PREDICTORS
# ══════════════════════════════════════════════════════════════════════════
def drop_high_missing_rows(panel: pd.DataFrame, pred_bases: list) -> pd.DataFrame:
    pred_cols = [c for c in pred_bases if c in panel.columns]
    missing_frac = panel[pred_cols].isna().mean(axis=1)
    mask = missing_frac <= MISSING_THRESHOLD
    before = len(panel)
    panel  = panel[mask].copy()
    after  = len(panel)
    print(f"\n[missing-row]  Dropped {before - after} rows "
          f"(>{int(MISSING_THRESHOLD*100)}% predictors missing). "
          f"Kept: {after}")
    return panel


# ══════════════════════════════════════════════════════════════════════════
# 6. INDUSTRY-YEAR MEDIAN IMPUTATION
# ══════════════════════════════════════════════════════════════════════════
def impute_industry_year_median(panel: pd.DataFrame, pred_bases: list) -> pd.DataFrame:
    """
    For each predictor, fill missing values with the median of firms in the
    same Industry Class and same Year_t. Fall back to year-wide median if
    the industry-year group has no non-null values.
    """
    pred_cols = [c for c in pred_bases if c in panel.columns]
    total_missing_before = panel[pred_cols].isna().sum().sum()

    for col in pred_cols:
        # industry-year median
        panel[col] = panel.groupby(["Industry Class", "Year_t"])[col] \
                          .transform(lambda x: x.fillna(x.median()))
        # fallback: year-wide median
        panel[col] = panel.groupby("Year_t")[col] \
                          .transform(lambda x: x.fillna(x.median()))

    total_missing_after = panel[pred_cols].isna().sum().sum()
    print(f"\n[impute]  Missing cells before: {total_missing_before}")
    print(f"[impute]  Missing cells after : {total_missing_after}")
    if total_missing_after > 0:
        still_missing = panel[pred_cols].isna().sum()
        still_missing = still_missing[still_missing > 0]
        print(f"[impute]  Columns still missing:\n{still_missing}")
    return panel


# ══════════════════════════════════════════════════════════════════════════
# 7. WINSORIZE WITHIN EACH YEAR
# ══════════════════════════════════════════════════════════════════════════
def winsorize_within_year(panel: pd.DataFrame, pred_bases: list) -> pd.DataFrame:
    """
    Clip each predictor at its 1st and 99th percentile computed separately
    within each Year_t. Preserves cross-year comparability while removing
    within-year extremes.
    """
    pred_cols = [c for c in pred_bases if c in panel.columns]

    def clip_group(group):
        for col in pred_cols:
            if col in group.columns:
                lo = group[col].quantile(WINSOR_LOW)
                hi = group[col].quantile(WINSOR_HIGH)
                group[col] = group[col].clip(lower=lo, upper=hi)
        return group

    year_t_backup = panel["Year_t"].values
    panel = panel.groupby("Year_t", group_keys=False).apply(clip_group)
    if "Year_t" not in panel.columns:
        panel.insert(1, "Year_t", year_t_backup)
    panel = panel.reset_index(drop=True)
    print(f"\n[winsorize]  Applied {int(WINSOR_LOW*100)}th–"
          f"{int(WINSOR_HIGH*100)}th percentile clipping within each Year_t.")
    return panel


# ══════════════════════════════════════════════════════════════════════════
# 8. FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════
def report(panel: pd.DataFrame, pred_bases: list):
    pred_cols = [c for c in pred_bases if c in panel.columns]

    print("\n" + "═"*60)
    print("FINAL PANEL DATASET — SUMMARY REPORT")
    print("═"*60)

    print(f"\n{'Shape':<35} {panel.shape}")
    print(f"{'Unique firms':<35} {panel['Company Name'].nunique()}")
    print(f"{'Predictor year range':<35} {sorted(panel['Year_t'].unique())}")
    print(f"{'Number of predictor columns':<35} {len(pred_cols)}")

    print("\n── Label distribution ──────────────────────────────────────")
    label_order = ["Full_Distress", "Acct_Distress", "Cash_Distress", "Healthy"]
    counts = panel["Label"].value_counts()
    total  = len(panel)
    for lbl in label_order:
        n   = counts.get(lbl, 0)
        pct = 100 * n / total
        print(f"  {lbl:<20} {n:>7,}   ({pct:5.1f}%)")
    sanity = sum(counts.get(l, 0) for l in label_order)
    print(f"  {'─'*40}")
    print(f"  {'TOTAL':<20} {total:>7,}")
    print(f"  {'SUM of 4 categories':<20} {sanity:>7,}")
    print(f"  {'Sanity check':<20} {'✓ PASS' if sanity == total else '✗ FAIL'}")

    print("\n── Label distribution by Year_t ────────────────────────────")
    yt = panel.groupby(["Year_t", "Label"]).size().unstack(fill_value=0)
    yt = yt.reindex(columns=label_order, fill_value=0)
    yt["Total"] = yt.sum(axis=1)
    print(yt.to_string())

    print("\n── Missing values in final panel ───────────────────────────")
    miss = panel[pred_cols].isna().sum()
    miss = miss[miss > 0]
    if len(miss) == 0:
        print("  No missing values remaining in predictor columns.")
    else:
        print(miss.to_string())

    print("\n── Predictor columns used ──────────────────────────────────")
    for i, c in enumerate(pred_cols, 1):
        print(f"  {i:>2}. {c}")

    print("\n── Descriptive stats (first 5 predictors) ──────────────────")
    print(panel[pred_cols[:5]].describe().round(4).to_string())
    print("═"*60)


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    # 1. load
    df = load_data(INPUT_PATH)

    # 2. identify predictors
    pred_bases = get_predictor_bases(df)

    # 3. build long panel
    panel, pred_bases = build_panel(df, pred_bases)

    # 4. drop high-missing rows
    panel = drop_high_missing_rows(panel, pred_bases)

    # 5. impute
    panel = impute_industry_year_median(panel, pred_bases)

    # 6. winsorize
    panel = winsorize_within_year(panel, pred_bases)

    # 7. reset index cleanly
    panel = panel.reset_index(drop=True)

    # 8. report
    report(panel, pred_bases)

    # 9. save
    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"\n[save]  Panel saved → {OUTPUT_PATH}")
    print(f"[save]  Final shape: {panel.shape}")


if __name__ == "__main__":
    main()