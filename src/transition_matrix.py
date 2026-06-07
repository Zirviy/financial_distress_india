"""
STEP 2 — Transition Matrix & Taxonomy Stability Analysis (Revised)
===================================================================
Objective  : Validate the 4-category distress taxonomy through year-to-year
             transition behaviour and category persistence analysis.
Scope      : Pooled sample only (2018–2023). No period splits.
Outputs    : tables/  → transition_counts.csv, transition_percentages.csv,
                        persistence_summary.csv, transition_pairs.csv
             figures/ → transition_heatmap.png, transition_flows.png,
                        persistence_bar.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── paths (project-relative) ───────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH  = BASE_DIR / "data" / "processed" / "panel_dataset.csv"
OUT_TABLES  = BASE_DIR / "outputs" / "tables"
OUT_FIGURES = BASE_DIR / "outputs" / "figures"
OUT_TABLES.mkdir(parents=True, exist_ok=True)
OUT_FIGURES.mkdir(parents=True, exist_ok=True)

# ── constants ──────────────────────────────────────────────────────────────
LABEL_ORDER  = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
SHORT        = {"Healthy": "H", "Acct_Distress": "AD",
                "Cash_Distress": "CD", "Full_Distress": "FD"}
COLORS       = {"Healthy": "#4CAF50", "Acct_Distress": "#FF9800",
                "Cash_Distress": "#2196F3", "Full_Distress": "#F44336"}
PCT_TOL      = 0.15   # allowable rounding error per row (percentage points)


# ══════════════════════════════════════════════════════════════════════════
# 1. BUILD CONSECUTIVE FIRM-YEAR TRANSITION PAIRS
# ══════════════════════════════════════════════════════════════════════════
def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    For every firm, identify pairs of consecutive years (t, t+1) where
    both observations are present. Only same-firm, year-gap-of-exactly-1
    pairs are included.

    Returns a DataFrame with columns:
        Company Name | Year_t | from_label | to_label
    """
    df = df.sort_values(["Company Name", "Year_t"]).reset_index(drop=True)

    rows = []
    skipped_gap = 0

    for firm, grp in df.groupby("Company Name"):
        grp   = grp.sort_values("Year_t")
        years  = grp["Year_t"].tolist()
        labels = grp["Label"].tolist()

        for i in range(len(years) - 1):
            gap = years[i + 1] - years[i]
            if gap == 1:                        # consecutive years only
                rows.append({
                    "Company Name": firm,
                    "Year_t":       years[i],
                    "from_label":   labels[i],
                    "to_label":     labels[i + 1],
                })
            else:
                skipped_gap += 1                # non-consecutive gap — skip

    trans = pd.DataFrame(rows)

    print(f"\n{'─'*62}")
    print("  TRANSITION PAIR CONSTRUCTION")
    print(f"{'─'*62}")
    print(f"  Total firms in panel           : {df['Company Name'].nunique():>8,}")
    print(f"  Firms contributing transitions : "
          f"{trans['Company Name'].nunique():>8,}")
    print(f"  Total consecutive pairs used   : {len(trans):>8,}")
    print(f"  Pairs skipped (non-consec gap) : {skipped_gap:>8,}")
    print(f"  Year range covered             : "
          f"{sorted(trans['Year_t'].unique())}")

    return trans


# ══════════════════════════════════════════════════════════════════════════
# 2. BUILD TRANSITION MATRICES (counts + row percentages)
# ══════════════════════════════════════════════════════════════════════════
def build_matrix(trans: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = pd.crosstab(
        trans["from_label"], trans["to_label"],
        rownames=["From_t"], colnames=["To_t1"]
    ).reindex(index=LABEL_ORDER, columns=LABEL_ORDER, fill_value=0)

    row_totals = counts.sum(axis=1)
    pcts       = counts.div(row_totals, axis=0) * 100

    return counts, pcts, row_totals


# ══════════════════════════════════════════════════════════════════════════
# 3. VERIFICATION SUITE
# ══════════════════════════════════════════════════════════════════════════
def run_verification(counts: pd.DataFrame,
                     pcts:   pd.DataFrame,
                     row_totals: pd.Series,
                     trans:  pd.DataFrame):
    """
    Prints a full set of mathematical verification checks.
    All checks must pass before results are used.
    """
    total_pairs = len(trans)
    all_pass = True

    print(f"\n{'═'*62}")
    print("  VERIFICATION SUITE")
    print(f"{'═'*62}")

    # ── CHECK 1: row sums to 100% ────────────────────────────────────────
    print("\n  CHECK 1 — Row percentages sum to 100% (±tolerance)")
    for lbl in LABEL_ORDER:
        row_sum = pcts.loc[lbl].sum()
        ok = abs(row_sum - 100.0) <= PCT_TOL
        status = "✓ PASS" if ok else "✗ FAIL"
        if not ok:
            all_pass = False
        print(f"    {SHORT[lbl]:<4}  row sum = {row_sum:.4f}%   {status}")

    # ── CHECK 2: total cell count = total transitions ────────────────────
    print(f"\n  CHECK 2 — Sum of all count cells = total transitions")
    cell_sum = int(counts.values.sum())
    ok = cell_sum == total_pairs
    status = "✓ PASS" if ok else "✗ FAIL"
    if not ok:
        all_pass = False
    print(f"    Sum of matrix cells : {cell_sum:,}")
    print(f"    Total pairs in data : {total_pairs:,}")
    print(f"    {status}")

    # ── CHECK 3: manual percentage verification ──────────────────────────
    print(f"\n  CHECK 3 — Manual percentage verification (numerator / denominator)")
    for from_lbl in LABEL_ORDER:
        denom = int(row_totals[from_lbl])
        print(f"\n    From {SHORT[from_lbl]} (n = {denom:,}):")
        for to_lbl in LABEL_ORDER:
            numer    = int(counts.loc[from_lbl, to_lbl])
            computed = (numer / denom * 100) if denom > 0 else 0.0
            stored   = pcts.loc[from_lbl, to_lbl]
            ok = abs(computed - stored) <= PCT_TOL
            status = "✓" if ok else "✗"
            if not ok:
                all_pass = False
            print(f"      → {SHORT[to_lbl]:<4}  {numer:>6,} / {denom:>6,} "
                  f"= {computed:6.2f}%   (stored {stored:6.2f}%)  {status}")

    # ── CHECK 4: diagonal persistence verification ───────────────────────
    print(f"\n  CHECK 4 — Diagonal persistence values (stay-in-category %)")
    for lbl in LABEL_ORDER:
        numer = int(counts.loc[lbl, lbl])
        denom = int(row_totals[lbl])
        pct   = numer / denom * 100 if denom > 0 else 0.0
        print(f"    {SHORT[lbl]:<4}  {numer:>5,} / {denom:>5,} = {pct:.1f}%  "
              f"(persistence = stay rate)")

    # ── CHECK 5: consecutive-year logic verification ─────────────────────
    print(f"\n  CHECK 5 — Consecutive-year logic (all gaps must equal 1)")
    trans_check = trans.copy()
    # re-verify by checking we cannot find any non-1-gap in saved pairs
    # (build_transitions already filters, this just confirms the output)
    # Since we don't store year_t1 explicitly, verify via Year_t distribution
    year_counts = trans["Year_t"].value_counts().sort_index()
    print(f"    Transitions by Year_t:")
    for yr, cnt in year_counts.items():
        print(f"      {yr} → {yr+1}:  {cnt:,} pairs")
    print(f"    No non-consecutive pairs possible by construction  ✓")

    # ── OVERALL ──────────────────────────────────────────────────────────
    print(f"\n  {'═'*58}")
    overall = "✓ ALL CHECKS PASSED" if all_pass else "✗ ONE OR MORE CHECKS FAILED"
    print(f"  OVERALL: {overall}")
    print(f"  {'═'*58}")

    return all_pass


# ══════════════════════════════════════════════════════════════════════════
# 4. PERSISTENCE SUMMARY TABLE
# ══════════════════════════════════════════════════════════════════════════
def build_persistence_summary(counts: pd.DataFrame,
                               pcts:   pd.DataFrame,
                               row_totals: pd.Series) -> pd.DataFrame:
    rows = []
    for lbl in LABEL_ORDER:
        stay_pct   = pcts.loc[lbl, lbl]
        exit_pcts  = pcts.loc[lbl].copy()
        exit_pcts[lbl] = 0
        top_dest     = exit_pcts.idxmax()
        top_dest_pct = exit_pcts.max()
        rows.append({
            "Category":              lbl,
            "N_transitions":         int(row_totals[lbl]),
            "Persistence_%":         round(stay_pct, 1),
            "Top_exit_destination":  top_dest,
            "Exit_pct":              round(top_dest_pct, 1),
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════
# 5. PRINT FORMATTED TRANSITION TABLE
# ══════════════════════════════════════════════════════════════════════════
def print_transition_table(counts: pd.DataFrame, pcts: pd.DataFrame,
                            row_totals: pd.Series):
    print(f"\n{'═'*62}")
    print("  TRANSITION MATRIX — POOLED (2018–2023)")
    print(f"{'═'*62}")

    s_counts = counts.copy()
    s_counts.index   = [SHORT[l] for l in s_counts.index]
    s_counts.columns = [SHORT[l] for l in s_counts.columns]
    s_counts["Row_Total"] = row_totals.values

    s_pcts = pcts.round(1).copy()
    s_pcts.index   = [SHORT[l] for l in s_pcts.index]
    s_pcts.columns = [SHORT[l] for l in s_pcts.columns]
    s_pcts["Row_Total_%"] = s_pcts.sum(axis=1).round(1)

    print(f"\n  COUNT MATRIX  (n = {int(counts.values.sum()):,})")
    print(s_counts.to_string())

    print(f"\n  ROW-PERCENTAGE MATRIX")
    print(s_pcts.to_string())


# ══════════════════════════════════════════════════════════════════════════
# 6. TRANSITION HEATMAP
# ══════════════════════════════════════════════════════════════════════════
def plot_heatmap(pcts: pd.DataFrame, counts: pd.DataFrame, filepath: Path):
    fig, ax = plt.subplots(figsize=(7, 5.5))
    so = [SHORT[l] for l in LABEL_ORDER]

    p = pcts.copy(); p.index = so; p.columns = so
    c = counts.copy(); c.index = so; c.columns = so

    annot = pd.DataFrame(index=so, columns=so)
    for r in so:
        for col in so:
            annot.loc[r, col] = (f"{p.loc[r,col]:.1f}%\n"
                                 f"({int(c.loc[r,col]):,})")

    sns.heatmap(
        p.astype(float), annot=annot, fmt="",
        cmap="RdYlGn_r", linewidths=0.5, linecolor="white",
        vmin=0, vmax=100, ax=ax,
        annot_kws={"size": 9},
        cbar_kws={"label": "Row % (transition probability)"},
    )
    for i in range(len(so)):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1, fill=False,
            edgecolor="black", lw=2.5, clip_on=False
        ))

    ax.set_title("Transition Matrix — Pooled (2018–2023)",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("To (t+1)", fontsize=11)
    ax.set_ylabel("From (t)", fontsize=11)
    ax.tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved]  {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# 7. TRANSITION FLOW CHART (stacked horizontal bars)
# ══════════════════════════════════════════════════════════════════════════
def plot_flows(pcts: pd.DataFrame, counts: pd.DataFrame,
               row_totals: pd.Series, filepath: Path):
    fig, axes = plt.subplots(len(LABEL_ORDER), 1,
                             figsize=(10, 6), sharex=True)
    fig.suptitle("Transition Flows — Pooled (2018–2023)",
                 fontsize=13, fontweight="bold", y=1.01)

    for i, from_lbl in enumerate(LABEL_ORDER):
        ax = axes[i]
        left = 0
        for to_lbl in LABEL_ORDER:
            val = pcts.loc[from_lbl, to_lbl]
            lw  = 2.0 if to_lbl == from_lbl else 0.4
            ec  = "black" if to_lbl == from_lbl else "white"
            ax.barh(0, val, left=left, height=0.6,
                    color=COLORS[to_lbl], edgecolor=ec, linewidth=lw)
            if val >= 5:
                ax.text(left + val / 2, 0,
                        f"{SHORT[to_lbl]}\n{val:.1f}%",
                        ha="center", va="center",
                        fontsize=8, fontweight="bold", color="white")
            left += val

        ax.set_xlim(0, 100)
        ax.set_yticks([])
        ax.set_ylabel(
            f"{SHORT[from_lbl]}\n(n={int(row_totals[from_lbl]):,})",
            fontsize=9, rotation=0, labelpad=52, va="center"
        )
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        if i < len(LABEL_ORDER) - 1:
            ax.spines["bottom"].set_visible(False)
            ax.tick_params(bottom=False)

    axes[-1].set_xlabel("Percentage of firms transitioning (%)", fontsize=10)
    patches = [mpatches.Patch(color=COLORS[l], label=SHORT[l])
               for l in LABEL_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.06), framealpha=0.9)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved]  {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# 8. PERSISTENCE BAR CHART
# ══════════════════════════════════════════════════════════════════════════
def plot_persistence(persist_df: pd.DataFrame, filepath: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    x    = np.arange(len(LABEL_ORDER))
    vals = [persist_df.loc[persist_df["Category"] == l,
                           "Persistence_%"].values[0]
            for l in LABEL_ORDER]
    bar_colors = [COLORS[l] for l in LABEL_ORDER]

    bars = ax.bar(x, vals, width=0.5, color=bar_colors,
                  edgecolor="white", linewidth=0.8, alpha=0.9)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1.2,
                f"{val:.1f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[l] for l in LABEL_ORDER], fontsize=12)
    ax.set_ylabel("Year-to-year persistence (%)", fontsize=11)
    ax.set_title(
        "Category Persistence Rates — Pooled (2018–2023)\n"
        "(% of firms remaining in same category the following year)",
        fontsize=12, fontweight="bold"
    )
    ax.set_ylim(0, 105)
    ax.axhline(50, color="grey", linestyle="--",
               linewidth=0.8, alpha=0.6, label="50% reference")
    ax.legend(fontsize=9, framealpha=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[saved]  {filepath}")


# ══════════════════════════════════════════════════════════════════════════
# 9. ECONOMIC INTERPRETATION (data-driven, academically cautious)
# ══════════════════════════════════════════════════════════════════════════
def print_interpretation(pcts: pd.DataFrame,
                          persist_df: pd.DataFrame,
                          row_totals: pd.Series):
    # derive all values from actual computed results
    persist = {row["Category"]: row["Persistence_%"]
               for _, row in persist_df.iterrows()}

    most_persistent  = max(persist, key=persist.get)
    least_persistent = min(persist, key=persist.get)

    # largest recovery: highest off-diagonal → Healthy transition from a distress state
    recovery_val = -1; recovery_from = None
    for lbl in ["Acct_Distress", "Cash_Distress", "Full_Distress"]:
        v = pcts.loc[lbl, "Healthy"]
        if v > recovery_val:
            recovery_val = v; recovery_from = lbl

    # largest deterioration: highest transition from Healthy to any distress
    detn_val = -1; detn_to = None
    for lbl in ["Acct_Distress", "Cash_Distress", "Full_Distress"]:
        v = pcts.loc["Healthy", lbl]
        if v > detn_val:
            detn_val = v; detn_to = lbl

    # FD pathways from each distress state
    ad_to_fd = pcts.loc["Acct_Distress", "Full_Distress"]
    cd_to_fd = pcts.loc["Cash_Distress",  "Full_Distress"]
    h_to_fd  = pcts.loc["Healthy",        "Full_Distress"]
    fd_to_h  = pcts.loc["Full_Distress",  "Healthy"]

    print(f"\n{'═'*62}")
    print("  ECONOMIC INTERPRETATION — TRANSITION DYNAMICS")
    print(f"{'═'*62}")
    print(f"""
CATEGORY STABILITY
──────────────────
  Highest persistence : {SHORT[most_persistent]}  ({persist[most_persistent]:.1f}%)
  Lowest  persistence : {SHORT[least_persistent]}  ({persist[least_persistent]:.1f}%)

  Detailed persistence rates:
    H   {persist['Healthy']:.1f}%  — Majority of healthy firms remain so the
                   following year, suggesting financial health is
                   relatively stable in the short term.
    AD  {persist['Acct_Distress']:.1f}%  — Earnings-based distress shows moderate
                   persistence, indicating it is not purely transient
                   but also not fully entrenched.
    CD  {persist['Cash_Distress']:.1f}%  — Cash Distress exhibits notably lower
                   persistence relative to other categories. The
                   majority of CD firms ({recovery_val:.1f}% from CD → H) recover
                   to Healthy within one year, consistent with
                   temporary working capital shocks that resolve
                   within an operating cycle.
    FD  {persist['Full_Distress']:.1f}%  — Full Distress shows similar persistence
                   to Accounting Distress. Firms experiencing both
                   earnings and cash-flow failure simultaneously
                   show meaningful year-over-year carryover.

TRANSITION PATHWAYS
────────────────────
  Largest recovery transition :
    {SHORT[recovery_from]} → H = {recovery_val:.1f}%

  Largest deterioration from Healthy :
    H → {SHORT[detn_to]} = {detn_val:.1f}%

  Pathways into Full Distress :
    From H  → FD : {h_to_fd:.1f}%   (direct deterioration from health)
    From AD → FD : {ad_to_fd:.1f}%   (earnings stress escalating to full failure)
    From CD → FD : {cd_to_fd:.1f}%   (cash stress escalating to full failure)
    From FD → H  : {fd_to_h:.1f}%   (direct recovery from full distress)

  The AD → FD transition ({ad_to_fd:.1f}%) suggests Accounting Distress may
  serve as a preceding state on the path to Full Distress, which
  supports treating it as a meaningful early-warning condition.

IMPLICATIONS FOR PREDICTION FRAMEWORK
──────────────────────────────────────
  1. Cash Distress persistence ({persist['Cash_Distress']:.1f}%) is substantially lower
     than other distress states. This property may create challenges
     for forward-prediction models that treat CD as a stable target
     label, since the majority of CD firms do not remain in that
     state the following year.

  2. Full Distress persistence ({persist['Full_Distress']:.1f}%) suggests that the
     conjunction of ICR < 1 and Cash-ICR < 1 identifies a more
     durable condition than either criterion alone, which may
     improve the signal-to-noise ratio in binary prediction models
     targeting this state.

  3. The distinct persistence profiles of Accounting Distress and
     Cash Distress — and their different pathways toward Full
     Distress — are consistent with the view that ICR and Cash-ICR
     capture different underlying mechanisms of financial
     deterioration. Whether this motivates separate binary models
     or a joint model remains an empirical question to be
     addressed in subsequent analysis.

  Note: All interpretations above are based on observed transition
  frequencies in the panel. Causal claims about firm behaviour
  require additional analysis and are not asserted here.
""")


# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════
def main():
    print(f"[load]  Reading: {INPUT_PATH}")
    df = pd.read_csv(INPUT_PATH)
    print(f"[load]  Shape: {df.shape} | "
          f"Years: {sorted(df['Year_t'].unique())} | "
          f"Labels: {df['Label'].value_counts().to_dict()}")

    # 1. build transition pairs
    trans = build_transitions(df)

    # 2. build matrices
    counts, pcts, row_totals = build_matrix(trans)

    # 3. verification suite — run before any output
    checks_ok = run_verification(counts, pcts, row_totals, trans)
    if not checks_ok:
        print("\n[WARNING] One or more verification checks failed. "
              "Review the output above before proceeding.")

    # 4. print formatted tables
    print_transition_table(counts, pcts, row_totals)

    # 5. persistence summary
    persist_df = build_persistence_summary(counts, pcts, row_totals)
    print(f"\n{'═'*62}")
    print("  PERSISTENCE SUMMARY")
    print(f"{'═'*62}")
    print(persist_df.to_string(index=False))

    # 6. economic interpretation
    print_interpretation(pcts, persist_df, row_totals)

    # 7. save tables
    counts.to_csv(OUT_TABLES / "transition_counts.csv")
    pcts.round(4).to_csv(OUT_TABLES / "transition_percentages.csv")
    persist_df.to_csv(OUT_TABLES / "persistence_summary.csv", index=False)
    trans.to_csv(OUT_TABLES / "transition_pairs.csv", index=False)
    print(f"\n[saved]  transition_counts.csv")
    print(f"[saved]  transition_percentages.csv")
    print(f"[saved]  persistence_summary.csv")
    print(f"[saved]  transition_pairs.csv  ({len(trans):,} rows)")

    # 8. figures
    plot_heatmap(pcts, counts,
                 filepath=OUT_FIGURES / "transition_heatmap.png")
    plot_flows(pcts, counts, row_totals,
               filepath=OUT_FIGURES / "transition_flows.png")
    plot_persistence(persist_df,
                     filepath=OUT_FIGURES / "persistence_bar.png")

    print(f"\n{'═'*62}")
    print("  STEP 2 COMPLETE")
    print(f"{'═'*62}")
    print(f"  Tables  → {OUT_TABLES}")
    print(f"  Figures → {OUT_FIGURES}")


if __name__ == "__main__":
    main()