"""
STEP 3 — Taxonomy Validation & Descriptive Statistics
=======================================================
Research project: Financial Distress Prediction for Indian Firms

Objectives:
  3.1  Category-wise descriptive statistics
  3.2  Statistical distinctiveness tests (Kruskal-Wallis + Mann-Whitney AD vs CD)
  3.3  AD vs CD deep comparison
  3.4  VIF / multicollinearity analysis
  3.5  Correlation analysis
  3.6  Visualizations

Run from project root:
    python src/taxonomy_validation.py

Outputs:
    outputs/tables/  — all CSV tables
    outputs/figures/ — all publication-ready figures
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data" / "processed" / "panel_dataset.csv"
OUT_TABLES  = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_ORDER  = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
LABEL_SHORT  = {"Healthy": "H", "Acct_Distress": "AD",
                "Cash_Distress": "CD", "Full_Distress": "FD"}
LABEL_COLORS = {"Healthy": "#2ecc71", "Acct_Distress": "#f39c12",
                "Cash_Distress": "#3498db", "Full_Distress": "#e74c3c"}

# Ratio family mapping — every predictor assigned to one family
RATIO_FAMILIES = {
    "Profitability": [
        "ROA", "ROE", "ROCE", "EBITDA_Margin",
        "Operating_Profit_Margin", "Net_Profit_Margin",
    ],
    "Leverage": [
        "Debt_to_Assets", "Debt_to_Equity", "LT_Debt_to_Assets",
        "TL_to_TA", "TA_to_Equity", "Net_Debt_to_EBITDA",
    ],
    "Liquidity": [
        "Current_Ratio", "Quick_Ratio", "Cash_to_CL",
    ],
    "Cash_Flow": [
        "CFO_to_CL", "CFO_to_Debt", "CFO_to_Net_Income",
        "NCF_to_Debt", "NCF_to_TA",
    ],
    "Turnover_Efficiency": [
        "Total_Asset_Turnover", "Inventory_Turnover",
        "Receivables_Turnover", "Payables_Turnover", "WC_Turnover",
    ],
    "Working_Capital": [
        "WC_to_TA", "Cash_Conversion_Cycle",
    ],
}

# Flat predictor list
ALL_PREDICTORS = [c for cols in RATIO_FAMILIES.values() for c in cols]


# ══════════════════════════════════════════════════════════════════════════════
# 0. LOAD & VALIDATE
# ══════════════════════════════════════════════════════════════════════════════

def load_panel(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert set(LABEL_ORDER).issubset(set(df["Label"].unique())), \
        "Label mismatch — check panel dataset"
    missing_cols = [c for c in ALL_PREDICTORS if c not in df.columns]
    if missing_cols:
        print(f"  [warn] Predictors missing from panel: {missing_cols}")
    df["Label"] = pd.Categorical(df["Label"], categories=LABEL_ORDER, ordered=True)
    print(f"[load] Panel shape: {df.shape}")
    print(f"[load] Label distribution:\n{df['Label'].value_counts()}\n")
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 3.1  CATEGORY-WISE DESCRIPTIVE STATISTICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_descriptive_stats(df: pd.DataFrame) -> dict:
    """
    For each ratio family, compute mean / median / std by category.
    Returns dict of DataFrames keyed by family name.
    Also saves one master CSV and one CSV per family.
    """
    print("=" * 60)
    print("3.1  CATEGORY-WISE DESCRIPTIVE STATISTICS")
    print("=" * 60)

    available = [c for c in ALL_PREDICTORS if c in df.columns]
    agg_funcs  = ["mean", "median", "std"]
    results    = {}

    # ── Company-year count table ───────────────────────────────────────────
    cy_table = (
        df.groupby(["Year_t", "Label"], observed=True)
          .size()
          .unstack(fill_value=0)
          .reindex(columns=LABEL_ORDER)
    )
    cy_table["Total"] = cy_table.sum(axis=1)
    cy_table.to_csv(OUT_TABLES / "company_year_counts.csv")
    print("[3.1] Company-year counts by year and category:")
    print(cy_table.to_string())
    print()

    # ── Master stats table ────────────────────────────────────────────────
    master_rows = []
    for family, cols in RATIO_FAMILIES.items():
        avail_cols = [c for c in cols if c in df.columns]
        for col in avail_cols:
            row = {"Family": family, "Variable": col}
            for lbl in LABEL_ORDER:
                subset = df.loc[df["Label"] == lbl, col].dropna()
                row[f"{LABEL_SHORT[lbl]}_mean"]   = round(subset.mean(), 4)
                row[f"{LABEL_SHORT[lbl]}_median"]  = round(subset.median(), 4)
                row[f"{LABEL_SHORT[lbl]}_std"]     = round(subset.std(), 4)
            master_rows.append(row)

    master_df = pd.DataFrame(master_rows)
    master_df.to_csv(OUT_TABLES / "descriptive_stats_master.csv", index=False)
    print(f"[3.1] Master descriptive stats saved → {OUT_TABLES / 'descriptive_stats_master.csv'}")

    # ── Per-family tables ─────────────────────────────────────────────────
    for family, cols in RATIO_FAMILIES.items():
        avail_cols = [c for c in cols if c in df.columns]
        if not avail_cols:
            continue
        grp    = df.groupby("Label", observed=True)[avail_cols]
        tbl    = grp.agg(agg_funcs).round(4)
        tbl.to_csv(OUT_TABLES / f"desc_{family.lower()}.csv")
        results[family] = tbl

    print("[3.1] Per-family tables saved.\n")

    # ── Category composition summary ──────────────────────────────────────
    summary_rows = []
    for lbl in LABEL_ORDER:
        sub = df[df["Label"] == lbl]
        summary_rows.append({
            "Category"        : lbl,
            "Short"           : LABEL_SHORT[lbl],
            "N_observations"  : len(sub),
            "N_unique_firms"  : sub["Company Name"].nunique(),
            "Pct_of_total"    : round(100 * len(sub) / len(df), 2),
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_TABLES / "category_composition.csv", index=False)
    print("[3.1] Category composition:")
    print(summary.to_string(index=False))
    print()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 3.2  STATISTICAL DISTINCTIVENESS — Kruskal-Wallis + Mann-Whitney
# ══════════════════════════════════════════════════════════════════════════════

def run_kruskal_wallis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kruskal-Wallis test across all 4 categories for every predictor.
    Null hypothesis: distributions are identical across all 4 groups.
    """
    print("=" * 60)
    print("3.2  KRUSKAL-WALLIS — ALL PREDICTORS")
    print("=" * 60)

    rows = []
    groups = [df.loc[df["Label"] == lbl, :] for lbl in LABEL_ORDER]

    available = [c for c in ALL_PREDICTORS if c in df.columns]
    for col in available:
        samples = [g[col].dropna().values for g in groups]
        if any(len(s) < 5 for s in samples):
            continue
        stat, pval = stats.kruskal(*samples)
        rows.append({
            "Variable"  : col,
            "Family"    : next(f for f, cs in RATIO_FAMILIES.items() if col in cs),
            "H_stat"    : round(stat, 4),
            "p_value"   : round(pval, 6),
            "Sig_0.05"  : "Yes" if pval < 0.05 else "No",
            "Sig_0.01"  : "Yes" if pval < 0.01 else "No",
        })

    kw_df = pd.DataFrame(rows).sort_values("p_value")
    kw_df.to_csv(OUT_TABLES / "kruskal_wallis_results.csv", index=False)

    n_sig_05 = (kw_df["p_value"] < 0.05).sum()
    n_sig_01 = (kw_df["p_value"] < 0.01).sum()
    print(f"  Variables tested       : {len(kw_df)}")
    print(f"  Significant at p<0.05  : {n_sig_05} / {len(kw_df)}")
    print(f"  Significant at p<0.01  : {n_sig_01} / {len(kw_df)}")
    print()
    print("  Top 10 most distinguishing variables:")
    print(kw_df.head(10)[["Variable", "Family", "H_stat", "p_value"]].to_string(index=False))
    print()

    # ── Verification check ────────────────────────────────────────────────
    assert n_sig_05 > 0, "FAIL: no predictors distinguish categories — check data"
    print(f"  [CHECK] {n_sig_05} predictors pass p<0.05 — taxonomy has statistical support.\n")

    return kw_df


def run_mann_whitney_ad_cd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mann-Whitney U tests: Accounting Distress vs Cash Distress.
    This is the core test proving AD and CD are economically different.
    Priority variables are cash-flow and liquidity ratios.
    """
    print("=" * 60)
    print("3.2  MANN-WHITNEY — ACCOUNTING DISTRESS vs CASH DISTRESS")
    print("=" * 60)

    # Priority variables: cash-flow, coverage, liquidity, working-capital
    PRIORITY = [
        "CFO_to_CL", "CFO_to_Debt", "CFO_to_Net_Income",
        "NCF_to_Debt", "NCF_to_TA",
        "Current_Ratio", "Quick_Ratio", "Cash_to_CL",
        "WC_to_TA", "Cash_Conversion_Cycle",
    ]
    available = [c for c in ALL_PREDICTORS if c in df.columns]

    ad = df[df["Label"] == "Acct_Distress"]
    cd = df[df["Label"] == "Cash_Distress"]

    rows = []
    for col in available:
        ad_vals = ad[col].dropna().values
        cd_vals = cd[col].dropna().values
        if len(ad_vals) < 5 or len(cd_vals) < 5:
            continue
        stat, pval = stats.mannwhitneyu(ad_vals, cd_vals, alternative="two-sided")
        family = next(f for f, cs in RATIO_FAMILIES.items() if col in cs)
        rows.append({
            "Variable"    : col,
            "Family"      : family,
            "AD_median"   : round(np.median(ad_vals), 4),
            "CD_median"   : round(np.median(cd_vals), 4),
            "Higher_in"   : "AD" if np.median(ad_vals) > np.median(cd_vals) else "CD",
            "U_stat"      : round(stat, 2),
            "p_value"     : round(pval, 6),
            "Sig_0.05"    : "Yes" if pval < 0.05 else "No",
            "Priority_var": "Yes" if col in PRIORITY else "No",
        })

    mw_df = pd.DataFrame(rows).sort_values("p_value")
    mw_df.to_csv(OUT_TABLES / "mann_whitney_ad_vs_cd.csv", index=False)

    print("  Mann-Whitney results for priority cash-flow / liquidity variables:")
    priority_df = mw_df[mw_df["Priority_var"] == "Yes"]
    print(priority_df[["Variable", "Family", "AD_median", "CD_median",
                        "Higher_in", "p_value", "Sig_0.05"]].to_string(index=False))
    print()

    n_sig = (priority_df["p_value"] < 0.05).sum()
    print(f"  [CHECK] {n_sig}/{len(priority_df)} priority cash-flow/liquidity vars "
          f"significantly different between AD and CD (p<0.05)\n")

    return mw_df


# ══════════════════════════════════════════════════════════════════════════════
# 3.3  AD vs CD DEEP COMPARISON TABLE
# ══════════════════════════════════════════════════════════════════════════════

def ad_cd_deep_comparison(df: pd.DataFrame, mw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Focused ratio-by-ratio comparison of Accounting Distress vs Cash Distress.
    Highlights which dimensions separate them and which don't.
    """
    print("=" * 60)
    print("3.3  AD vs CD DEEP COMPARISON")
    print("=" * 60)

    ad = df[df["Label"] == "Acct_Distress"]
    cd = df[df["Label"] == "Cash_Distress"]

    rows = []
    available = [c for c in ALL_PREDICTORS if c in df.columns]
    for col in available:
        family   = next(f for f, cs in RATIO_FAMILIES.items() if col in cs)
        ad_med   = ad[col].median()
        cd_med   = cd[col].median()
        ad_mean  = ad[col].mean()
        cd_mean  = cd[col].mean()
        pval_row = mw_df.loc[mw_df["Variable"] == col, "p_value"]
        pval     = pval_row.values[0] if len(pval_row) else np.nan
        rows.append({
            "Variable"  : col,
            "Family"    : family,
            "AD_median" : round(ad_med, 4),
            "CD_median" : round(cd_med, 4),
            "AD_mean"   : round(ad_mean, 4),
            "CD_mean"   : round(cd_mean, 4),
            "Diff_pct"  : round(100 * (cd_med - ad_med) / (abs(ad_med) + 1e-9), 1),
            "p_value"   : round(pval, 6) if not np.isnan(pval) else np.nan,
            "Sig_0.05"  : "Yes" if pval < 0.05 else "No",
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUT_TABLES / "ad_cd_deep_comparison.csv", index=False)

    # Print per-family summary
    for family in RATIO_FAMILIES:
        sub = comp_df[comp_df["Family"] == family]
        if sub.empty:
            continue
        print(f"\n  Family: {family}")
        print(sub[["Variable", "AD_median", "CD_median",
                   "Diff_pct", "p_value", "Sig_0.05"]].to_string(index=False))

    print()

    # Key finding: which family best separates AD vs CD?
    sig_by_family = (
        comp_df[comp_df["Sig_0.05"] == "Yes"]
        .groupby("Family")
        .size()
        .sort_values(ascending=False)
    )
    print("  Significant separating variables by family:")
    print(sig_by_family.to_string())
    print()

    return comp_df


# ══════════════════════════════════════════════════════════════════════════════
# 3.4  VIF — MULTICOLLINEARITY
# ══════════════════════════════════════════════════════════════════════════════

def compute_vif(df: pd.DataFrame) -> pd.DataFrame:
    """
    VIF for all predictors on the full panel (no label split).
    DO NOT remove features — document only.
    VIF > 10 flags high collinearity.
    VIF > 5  flags moderate concern.
    """
    print("=" * 60)
    print("3.4  VIF — MULTICOLLINEARITY ANALYSIS")
    print("=" * 60)

    available = [c for c in ALL_PREDICTORS if c in df.columns]
    X = df[available].dropna()

    vif_rows = []
    for i, col in enumerate(available):
        vif_val = variance_inflation_factor(X.values, i)
        family  = next(f for f, cs in RATIO_FAMILIES.items() if col in cs)
        flag    = (
            "HIGH — collinear (>10)"     if vif_val > 10 else
            "MODERATE — watch (5–10)"    if vif_val > 5  else
            "OK (<5)"
        )
        vif_rows.append({
            "Variable" : col,
            "Family"   : family,
            "VIF"      : round(vif_val, 2),
            "Flag"     : flag,
        })

    vif_df = pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)
    vif_df.to_csv(OUT_TABLES / "vif_table.csv", index=False)

    print(vif_df.to_string(index=False))
    n_high = (vif_df["VIF"] > 10).sum()
    n_mod  = ((vif_df["VIF"] > 5) & (vif_df["VIF"] <= 10)).sum()
    print(f"\n  [CHECK] VIF > 10 (high collinearity): {n_high} variables")
    print(f"  [CHECK] VIF 5–10 (moderate):           {n_mod} variables")
    print("  NOTE: No features removed at this stage — document only.\n")

    return vif_df


# ══════════════════════════════════════════════════════════════════════════════
# 3.5  CORRELATION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def compute_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full predictor correlation matrix + clustered heatmap.
    Ordered by ratio family for readability.
    """
    print("=" * 60)
    print("3.5  CORRELATION ANALYSIS")
    print("=" * 60)

    available = [c for c in ALL_PREDICTORS if c in df.columns]
    corr_df   = df[available].corr(method="spearman").round(3)
    corr_df.to_csv(OUT_TABLES / "correlation_matrix.csv")

    # Identify strongly correlated pairs
    pairs = []
    for i, c1 in enumerate(available):
        for c2 in available[i+1:]:
            r = corr_df.loc[c1, c2]
            if abs(r) >= 0.75:
                f1 = next(f for f, cs in RATIO_FAMILIES.items() if c1 in cs)
                f2 = next(f for f, cs in RATIO_FAMILIES.items() if c2 in cs)
                pairs.append({"Var1": c1, "Var2": c2,
                               "Family1": f1, "Family2": f2,
                               "Spearman_r": round(r, 3)})

    pairs_df = pd.DataFrame(pairs).sort_values("Spearman_r", key=abs, ascending=False)
    pairs_df.to_csv(OUT_TABLES / "high_correlation_pairs.csv", index=False)
    print(f"  Highly correlated pairs (|r| ≥ 0.75): {len(pairs_df)}")
    print(pairs_df.head(15).to_string(index=False))
    print()

    # ── Clustered heatmap ─────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr_df, dtype=bool))
    sns.heatmap(
        corr_df, mask=mask, cmap="RdYlGn", center=0,
        vmin=-1, vmax=1, linewidths=0.3,
        annot=False, ax=ax,
        cbar_kws={"label": "Spearman r", "shrink": 0.7},
    )
    ax.set_title("Predictor Correlation Matrix (Spearman)", fontsize=14, fontweight="bold", pad=12)

    # Add family boundary lines
    boundaries = []
    count = 0
    for family, cols in RATIO_FAMILIES.items():
        avail = [c for c in cols if c in available]
        count += len(avail)
        boundaries.append(count)
    for b in boundaries[:-1]:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "correlation_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] correlation_heatmap.png saved\n")

    return corr_df


# ══════════════════════════════════════════════════════════════════════════════
# 3.6  VISUALIZATIONS
# ══════════════════════════════════════════════════════════════════════════════

def plot_category_counts_by_year(df: pd.DataFrame):
    """Bar chart: category counts by year — explicitly requested in Way Ahead doc."""
    print("[3.6] Plotting category counts by year ...")

    cy = (
        df.groupby(["Year_t", "Label"], observed=True)
          .size()
          .reset_index(name="Count")
    )
    cy["Label"] = cy["Label"].map(LABEL_SHORT)

    fig, ax = plt.subplots(figsize=(12, 6))
    years   = sorted(df["Year_t"].unique())
    width   = 0.2
    offsets = np.linspace(-0.3, 0.3, 4)
    short   = list(LABEL_SHORT.values())
    colors  = list(LABEL_COLORS.values())

    for i, (lbl_key, lbl_short) in enumerate(LABEL_SHORT.items()):
        counts = [cy.loc[(cy["Year_t"] == y) & (cy["Label"] == lbl_short), "Count"].sum()
                  for y in years]
        bars = ax.bar([y + offsets[i] for y in years], counts,
                      width=width, label=lbl_short,
                      color=LABEL_COLORS[lbl_key], edgecolor="white", linewidth=0.5)
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, h + 40,
                        f"{int(h)}", ha="center", va="bottom",
                        fontsize=6.5, color="grey")

    ax.set_xlabel("Predictor Year (t)", fontsize=11)
    ax.set_ylabel("Number of Firm-Year Observations", fontsize=11)
    ax.set_title("Distress Category Distribution by Year", fontsize=13, fontweight="bold")
    ax.set_xticks(years)
    ax.legend(title="Category", fontsize=10, title_fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "category_counts_by_year.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       → category_counts_by_year.png saved\n")


def plot_key_boxplots(df: pd.DataFrame):
    """
    Boxplots: one panel of 12 key variables across all 4 categories.
    Select one high-importance variable from each ratio family.
    """
    print("[3.6] Plotting key variable boxplots ...")

    KEY_VARS = [
        "ROA", "Net_Profit_Margin",         # profitability
        "Debt_to_Assets", "TL_to_TA",       # leverage
        "Current_Ratio", "Quick_Ratio",     # liquidity
        "CFO_to_Debt", "CFO_to_CL",         # cash flow                         
        "Total_Asset_Turnover",             # turnover
        "WC_to_TA", "Cash_Conversion_Cycle" # working capital
    ]
    KEY_VARS = [c for c in KEY_VARS if c in df.columns]

    n_cols = 4
    n_rows = int(np.ceil(len(KEY_VARS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    axes = axes.flatten()

    palette = {LABEL_SHORT[k]: LABEL_COLORS[k] for k in LABEL_ORDER}
    df_plot = df.copy()
    df_plot["Cat"] = df_plot["Label"].map(LABEL_SHORT)
    cat_order = [LABEL_SHORT[k] for k in LABEL_ORDER]

    for i, var in enumerate(KEY_VARS):
        ax = axes[i]
        sns.boxplot(
            data=df_plot, x="Cat", y=var,
            order=cat_order, palette=palette,
            width=0.55, linewidth=0.8,
            flierprops=dict(marker=".", markersize=2, alpha=0.3),
            ax=ax,
        )
        ax.set_title(var.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

    for j in range(len(KEY_VARS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Key Financial Ratios by Distress Category",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "boxplots_by_category.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       → boxplots_by_category.png saved\n")


def plot_ad_cd_violins(df: pd.DataFrame):
    """
    Violin plots: AD vs CD comparison on cash-flow and liquidity variables.
    Core visual for the taxonomy validation argument.
    """
    print("[3.6] Plotting AD vs CD violin comparison ...")

    VIOLIN_VARS = [
        "CFO_to_Debt", "CFO_to_CL", "CFO_to_Net_Income",
        "NCF_to_Debt",
        "Current_Ratio", "Quick_Ratio", "Cash_to_CL",
        "WC_to_TA", "Cash_Conversion_Cycle",
    ]
    VIOLIN_VARS = [c for c in VIOLIN_VARS if c in df.columns]

    sub = df[df["Label"].isin(["Acct_Distress", "Cash_Distress"])].copy()
    sub["Cat"] = sub["Label"].map(LABEL_SHORT)

    n_cols = 4
    n_rows = int(np.ceil(len(VIOLIN_VARS) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 4 * n_rows))
    axes = axes.flatten()

    palette_2 = {"AD": LABEL_COLORS["Acct_Distress"],
                 "CD": LABEL_COLORS["Cash_Distress"]}

    for i, var in enumerate(VIOLIN_VARS):
        ax = axes[i]
        sns.violinplot(
            data=sub, x="Cat", y=var, order=["AD", "CD"],
            palette=palette_2, linewidth=0.8,
            inner="quartile", ax=ax,
        )
        ax.set_title(var.replace("_", " "), fontsize=9, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=8)

        # Annotate medians
        for j, cat in enumerate(["AD", "CD"]):
            med = sub.loc[sub["Cat"] == cat, var].median()
            ax.text(j, ax.get_ylim()[0], f"  med={med:.2f}",
                    fontsize=6.5, color="black", va="bottom")

    for j in range(len(VIOLIN_VARS), len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("AD vs CD: Cash-Flow & Liquidity Ratio Distributions",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "violin_ad_vs_cd.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       → violin_ad_vs_cd.png saved\n")


def plot_vif_bars(vif_df: pd.DataFrame):
    """Horizontal bar chart of VIF values — colored by severity."""
    print("[3.6] Plotting VIF bar chart ...")

    vif_sorted = vif_df.sort_values("VIF", ascending=True)
    colors_vif = vif_sorted["VIF"].apply(
        lambda v: "#e74c3c" if v > 10 else "#f39c12" if v > 5 else "#2ecc71"
    )

    fig, ax = plt.subplots(figsize=(8, max(5, len(vif_sorted) * 0.35)))
    bars = ax.barh(vif_sorted["Variable"], vif_sorted["VIF"],
                   color=colors_vif, edgecolor="white", linewidth=0.4)
    ax.axvline(5,  color="#f39c12", linestyle="--", linewidth=1.2, label="VIF=5")
    ax.axvline(10, color="#e74c3c", linestyle="--", linewidth=1.2, label="VIF=10")
    ax.set_xlabel("Variance Inflation Factor (VIF)", fontsize=11)
    ax.set_title("Multicollinearity Check — VIF per Predictor",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "vif_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       → vif_bars.png saved\n")


def plot_kw_significance(kw_df: pd.DataFrame):
    """
    Dot plot of -log10(p-value) for Kruskal-Wallis, colored by ratio family.
    Shows which variables most strongly distinguish the four categories.
    """
    print("[3.6] Plotting Kruskal-Wallis significance ...")

    kw_plot = kw_df.copy()
    kw_plot["neg_log10_p"] = -np.log10(kw_plot["p_value"].clip(lower=1e-300))
    kw_plot = kw_plot.sort_values("neg_log10_p", ascending=True)

    family_colors = {
        "Profitability"      : "#e74c3c",
        "Leverage"           : "#e67e22",
        "Liquidity"          : "#f1c40f",
        "Cash_Flow"          : "#2ecc71",
        "Turnover_Efficiency": "#3498db",
        "Working_Capital"    : "#9b59b6",
    }

    fig, ax = plt.subplots(figsize=(8, max(5, len(kw_plot) * 0.38)))
    for _, row in kw_plot.iterrows():
        color = family_colors.get(row["Family"], "grey")
        ax.barh(row["Variable"], row["neg_log10_p"], color=color, alpha=0.85)

    ax.axvline(-np.log10(0.05), color="black", linestyle="--",
               linewidth=1, label="p=0.05")
    ax.set_xlabel("−log₁₀(p-value)", fontsize=11)
    ax.set_title("Kruskal-Wallis: Category Distinctiveness per Predictor",
                 fontsize=12, fontweight="bold")

    # Legend for families
    handles = [plt.Rectangle((0, 0), 1, 1, color=c)
               for c in family_colors.values()]
    ax.legend(handles, list(family_colors.keys()),
              fontsize=8, title="Ratio Family", loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(OUT_FIGURES / "kw_significance.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"       → kw_significance.png saved\n")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "█" * 60)
    print("  STEP 3 — TAXONOMY VALIDATION & DESCRIPTIVE STATISTICS")
    print("█" * 60 + "\n")

    df = load_panel(DATA_PATH)

    # 3.1 Descriptive stats
    desc_results = compute_descriptive_stats(df)

    # 3.2 Statistical tests
    kw_df = run_kruskal_wallis(df)
    mw_df = run_mann_whitney_ad_cd(df)

    # 3.3 AD vs CD deep comparison
    comp_df = ad_cd_deep_comparison(df, mw_df)

    # 3.4 VIF
    vif_df = compute_vif(df)

    # 3.5 Correlation
    corr_df = compute_correlation(df)

    # 3.6 Visualizations
    plot_category_counts_by_year(df)
    plot_key_boxplots(df)
    plot_ad_cd_violins(df)
    plot_vif_bars(vif_df)
    plot_kw_significance(kw_df)

    # ── Final output manifest ────────────────────────────────────────────────
    print("=" * 60)
    print("STEP 3 COMPLETE — OUTPUT MANIFEST")
    print("=" * 60)
    print("\nTables (outputs/tables/):")
    for f in sorted(OUT_TABLES.iterdir()):
        print(f"  {f.name}")
    print("\nFigures (outputs/figures/):")
    for f in sorted(OUT_FIGURES.iterdir()):
        print(f"  {f.name}")
    print("\nAll Step 3 outputs generated successfully.")


if __name__ == "__main__":
    main()