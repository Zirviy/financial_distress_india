"""
STEP 5 — Baseline Models (Final Version)
=========================================
Financial Distress Prediction for Indian Firms

Changes from initial version:
  - Random Forest regularised via validation-set hyperparameter search
  - Train AUC / Train BalAcc reported alongside test metrics
  - Train-test gap reported explicitly for every model
  - 4-class RF tuned separately using balanced accuracy on validation set
  - Manifest print bug fixed (| → +)

Models:
  Dummy Classifier     — majority-class floor baseline
  Logistic Regression  — linear statistical baseline (class_weight='balanced')
  Random Forest        — regularised tree baseline   (class_weight='balanced')

Targets:
  Model A — Earnings Distress
            Positive : Acct_Distress + Full_Distress  (ICR < 1 at t+1)
            Negative : Healthy + Cash_Distress

  Model B — Cash-Flow Distress
            Positive : Cash_Distress + Full_Distress  (Cash-ICR < 0 at t+1)
            Negative : Healthy + Acct_Distress

  4-Class — Direct multiclass  (H / AD / CD / FD)

Temporal split:
  Train      : Year_t ∈ {2018, 2019, 2020, 2021}
  Validation : Year_t = 2022  (RF tuning + threshold selection)
  Test       : Year_t = 2023  (all reported metrics)

Run from project root:
    python src/baseline_models.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from pathlib import Path
import warnings
import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.dummy        import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble     import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline     import Pipeline
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data"    / "processed" / "panel_dataset.csv"
OUT_TABLES  = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_MODELS  = ROOT / "outputs" / "models"
for p in [OUT_TABLES, OUT_FIGURES, OUT_MODELS]:
    p.mkdir(parents=True, exist_ok=True)


# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS = [2018, 2019, 2020, 2021]
VAL_YEAR    = 2022
TEST_YEAR   = 2023

LABEL_ORDER = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
LABEL_SHORT = {"Healthy": "H", "Acct_Distress": "AD",
               "Cash_Distress": "CD", "Full_Distress": "FD"}

FORBIDDEN = {"Interest_Coverage", "Cash_Interest_Coverage", "DSCR"}

# Acceptable train-test AUC gap thresholds for publication
GAP_THRESHOLD_BINARY  = 0.03
GAP_THRESHOLD_4CLASS  = 0.10   # balanced accuracy gap

MODEL_COLORS = {
    "Dummy"              : "#95a5a6",
    "Logistic Regression": "#3498db",
    "Random Forest"      : "#2ecc71",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & VERIFY
# ══════════════════════════════════════════════════════════════════════════════
def load_and_verify(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)

    print("═" * 62)
    print("  VERIFICATION SUITE")
    print("═" * 62)

    meta_cols = ["Company Name", "Year_t",
                 "Industry group", "Industry Class", "Industry type", "Label"]
    pred_cols = [c for c in df.columns if c not in meta_cols]

    # 1. Leakage check
    leaked = FORBIDDEN.intersection(set(pred_cols))
    status = "✓ PASS" if not leaked else f"✗ FAIL — found: {leaked}"
    print(f"\n  [1] Forbidden variable check          : {status}")
    if leaked:
        raise ValueError(f"Leakage detected: {leaked}")

    # 2. Predictor count
    ok_count = "✓ PASS" if len(pred_cols) == 27 else "⚠  CHECK"
    print(f"  [2] Predictor count                   : {len(pred_cols)}  {ok_count}")

    # 3. Year split
    years_present = sorted(df["Year_t"].unique())
    expected      = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]
    ok = set(years_present) == set(expected)
    print(f"  [3] Years present                     : {years_present}")
    print(f"      Split check                       : {'✓ PASS' if ok else '✗ FAIL'}")

    # 4. Labels
    ok_lbl = set(df["Label"].unique()) == set(LABEL_ORDER)
    print(f"  [4] Labels present                    : "
          f"{sorted(df['Label'].unique())}")
    print(f"      Label check                       : {'✓ PASS' if ok_lbl else '✗ FAIL'}")

    # 5. Missing values
    n_miss = df[pred_cols].isnull().sum().sum()
    print(f"  [5] Missing values in predictors      : {n_miss}  "
          f"({'✓ PASS' if n_miss == 0 else '✗ FAIL'})")

    # 6. Class distribution
    print(f"\n  [6] Label distribution:")
    for lbl in LABEL_ORDER:
        n   = (df["Label"] == lbl).sum()
        pct = 100 * n / len(df)
        print(f"        {LABEL_SHORT[lbl]:<3} {lbl:<20} {n:>7,}  ({pct:5.1f}%)")
    print(f"        {'Total':<24} {len(df):>7,}")

    # 7. Split sizes
    print(f"\n  [7] Split sizes:")
    for yr in TRAIN_YEARS:
        print(f"        Train  Year_t={yr} : {(df['Year_t']==yr).sum():,}")
    print(f"        Val    Year_t={VAL_YEAR}  : {(df['Year_t']==VAL_YEAR).sum():,}")
    print(f"        Test   Year_t={TEST_YEAR}  : {(df['Year_t']==TEST_YEAR).sum():,}")

    # 8. Firm overlap (expected and acceptable in panel)
    train_firms = set(df.loc[df["Year_t"].isin(TRAIN_YEARS), "Company Name"])
    test_firms  = set(df.loc[df["Year_t"] == TEST_YEAR,      "Company Name"])
    overlap     = train_firms & test_firms
    pct_overlap = 100 * len(overlap) / max(len(test_firms), 1)
    print(f"\n  [8] Firm overlap train ∩ test         : {len(overlap):,} "
          f"({pct_overlap:.1f}% of test firms)")
    print("      Temporal split prevents label leakage despite firm overlap.")

    print("\n  All verification checks complete.\n")
    return df, pred_cols


# ══════════════════════════════════════════════════════════════════════════════
# 2. BUILD TARGETS & SPLITS
# ══════════════════════════════════════════════════════════════════════════════
def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["target_A"] = df["Label"].isin(
        ["Acct_Distress", "Full_Distress"]).astype(int)
    df["target_B"] = df["Label"].isin(
        ["Cash_Distress", "Full_Distress"]).astype(int)
    label_map      = {"Healthy": 0, "Acct_Distress": 1,
                      "Cash_Distress": 2, "Full_Distress": 3}
    df["target_4"] = df["Label"].map(label_map)
    return df


def make_splits(df: pd.DataFrame, pred_cols: list[str]) -> dict:
    splits = {}
    for name, years in [("train", TRAIN_YEARS),
                        ("val",   [VAL_YEAR]),
                        ("test",  [TEST_YEAR])]:
        mask = df["Year_t"].isin(years)
        sub  = df[mask].reset_index(drop=True)
        splits[name] = {
            "X"    : sub[pred_cols].values,
            "y_A"  : sub["target_A"].values,
            "y_B"  : sub["target_B"].values,
            "y_4"  : sub["target_4"].values,
            "feat_names": pred_cols,
        }

    print("Split sizes:")
    for name, s in splits.items():
        print(f"  {name:<6}: {len(s['X']):>6,} rows  |  "
              f"y_A pos: {s['y_A'].sum():,} ({100*s['y_A'].mean():.1f}%)  |  "
              f"y_B pos: {s['y_B'].sum():,} ({100*s['y_B'].mean():.1f}%)")
    print()
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 3. RF TUNING — VALIDATION-SET GRID SEARCH
# ══════════════════════════════════════════════════════════════════════════════
RF_PARAM_GRID = {
    "max_depth"        : [3, 5, 7, 10, 15],
    "min_samples_leaf" : [20, 50, 100],
    "max_features"     : ["sqrt", 0.5],
}


def tune_rf_binary(X_train, y_train, X_val, y_val,
                   random_state: int = 42) -> dict:
    """
    Select RF hyperparameters by AUC on the validation set.
    Returns the best param dict.
    """
    best_auc, best_params = -1.0, {}

    for params in ParameterGrid(RF_PARAM_GRID):
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            **params,
        )
        rf.fit(X_train, y_train)
        auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
        if auc > best_auc:
            best_auc    = auc
            best_params = params

    print(f"    RF tuning → val AUC={best_auc:.4f}  "
          f"max_depth={best_params['max_depth']}  "
          f"min_samples_leaf={best_params['min_samples_leaf']}  "
          f"max_features={best_params['max_features']}")
    return best_params


def tune_rf_4class(X_train, y_train, X_val, y_val,
                   random_state: int = 42) -> dict:
    """
    Select RF hyperparameters by balanced accuracy on the validation set
    for the 4-class problem.
    """
    best_ba, best_params = -1.0, {}

    for params in ParameterGrid(RF_PARAM_GRID):
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
            **params,
        )
        rf.fit(X_train, y_train)
        ba = balanced_accuracy_score(y_val, rf.predict(X_val))
        if ba > best_ba:
            best_ba     = ba
            best_params = params

    print(f"    RF tuning → val BalAcc={best_ba:.4f}  "
          f"max_depth={best_params['max_depth']}  "
          f"min_samples_leaf={best_params['min_samples_leaf']}  "
          f"max_features={best_params['max_features']}")
    return best_params


# ══════════════════════════════════════════════════════════════════════════════
# 4. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_models_binary(rf_params: dict) -> dict:
    return {
        "Dummy": DummyClassifier(
            strategy="most_frequent", random_state=42),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            **rf_params,
        ),
    }


def build_models_4class(rf_params: dict) -> dict:
    return {
        "Dummy": DummyClassifier(
            strategy="most_frequent", random_state=42),

        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(
                multi_class="multinomial",
                class_weight="balanced",
                max_iter=1000,
                solver="lbfgs",
                random_state=42,
            )),
        ]),

        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
            **rf_params,
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5. BINARY EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def eval_binary(model_name: str, model,
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                target_name: str) -> dict:
    """
    Train on train, select threshold on val, report all metrics on test.
    Also reports train AUC so train-test gap is visible.
    """
    model.fit(X_train, y_train)

    def get_prob(X):
        if hasattr(model, "predict_proba"):
            return model.predict_proba(X)[:, 1]
        return model.predict(X).astype(float)

    prob_train = get_prob(X_train)
    prob_val   = get_prob(X_val)
    prob_test  = get_prob(X_test)

    # AUC
    try:
        train_auc = roc_auc_score(y_train, prob_train)
        test_auc  = roc_auc_score(y_test,  prob_test)
    except ValueError:
        train_auc = test_auc = 0.5

    # Threshold selection on validation set (maximise F1)
    if model_name == "Dummy":
        best_thr = 0.5
    else:
        thresholds = np.linspace(0.05, 0.95, 91)
        f1_vals    = [f1_score(y_val, (prob_val >= t).astype(int),
                               zero_division=0)
                      for t in thresholds]
        best_thr   = thresholds[int(np.argmax(f1_vals))]

    y_pred = (prob_test >= best_thr).astype(int)

    # Confusion matrix → Type I / II
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    gap = train_auc - test_auc
    gap_flag = ("✓" if abs(gap) <= GAP_THRESHOLD_BINARY
                else "⚠  GAP EXCEEDS THRESHOLD")

    return {
        "Model"            : model_name,
        "Target"           : target_name,
        "Threshold"        : round(best_thr, 3),
        "Train_AUC"        : round(train_auc, 4),
        "AUC"              : round(test_auc,  4),
        "AUC_Gap"          : round(gap, 4),
        "Gap_Flag"         : gap_flag,
        "Balanced_Accuracy": round(balanced_accuracy_score(y_test, y_pred), 4),
        "F1"               : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "Precision"        : round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall"           : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "Type_I_Error"     : round(type1, 4),
        "Type_II_Error"    : round(type2, 4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "_model_obj"  : model,
        "_prob_test"  : prob_test,
        "_y_test"     : y_test,
        "_cm"         : cm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6. 4-CLASS EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def eval_4class(model_name: str, model,
                X_train, y_train,
                X_test,  y_test) -> dict:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    train_ba = balanced_accuracy_score(y_train, model.predict(X_train))
    test_ba  = balanced_accuracy_score(y_test,  y_pred)
    mac_f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    cm       = confusion_matrix(y_test, y_pred, labels=[0, 1, 2, 3])

    gap      = train_ba - test_ba
    gap_flag = ("✓" if abs(gap) <= GAP_THRESHOLD_4CLASS
                else "⚠  GAP EXCEEDS THRESHOLD")

    return {
        "Model"              : model_name,
        "Target"             : "4-Class",
        "Train_BalAcc"       : round(train_ba, 4),
        "Balanced_Accuracy"  : round(test_ba,  4),
        "BalAcc_Gap"         : round(gap, 4),
        "Gap_Flag"           : gap_flag,
        "Macro_F1"           : round(mac_f1, 4),
        "_cm"                : cm,
        "_model_obj"         : model,
        "_y_test"            : y_test,
        "_y_pred"            : y_pred,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7. OVERFITTING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_overfitting_summary(binary_rows: list, class4_rows: list):
    print("\n" + "═" * 62)
    print("  TRAIN-TEST GAP SUMMARY  (overfitting check)")
    print("═" * 62)
    print(f"  {'Model':<22} {'Target':<10} {'Train':>8} {'Test':>8} "
          f"{'Gap':>8}  Status")
    print(f"  {'-'*22} {'-'*10} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")

    for r in binary_rows:
        print(f"  {r['Model']:<22} {r['Target']:<10} "
              f"{r['Train_AUC']:>8.4f} {r['AUC']:>8.4f} "
              f"{r['AUC_Gap']:>+8.4f}  {r['Gap_Flag']}")

    for r in class4_rows:
        print(f"  {r['Model']:<22} {'4-Class':<10} "
              f"{r['Train_BalAcc']:>8.4f} {r['Balanced_Accuracy']:>8.4f} "
              f"{r['BalAcc_Gap']:>+8.4f}  {r['Gap_Flag']}")

    print(f"\n  Threshold binary  : gap ≤ {GAP_THRESHOLD_BINARY}")
    print(f"  Threshold 4-class : gap ≤ {GAP_THRESHOLD_4CLASS}")


# ══════════════════════════════════════════════════════════════════════════════
# 8. PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_curves(results_list: list, target_name: str,
                    positive_label: str, filepath: Path):
    fig, ax = plt.subplots(figsize=(7, 6))
    for res in results_list:
        if res["Model"] == "Dummy":
            continue
        fpr, tpr, _ = roc_curve(res["_y_test"], res["_prob_test"])
        ax.plot(fpr, tpr,
                label=f"{res['Model']}  "
                      f"(train={res['Train_AUC']:.3f} / test={res['AUC']:.3f})",
                color=MODEL_COLORS[res["Model"]], linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curves — {target_name}\n(Positive: {positive_label})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


def plot_confusion_matrices_binary(results_list: list,
                                   target_name: str,
                                   pos_label: str,
                                   neg_label: str,
                                   filepath: Path):
    n = len(results_list)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]
    for ax, res in zip(axes, results_list):
        cm = res["_cm"]
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[f"Pred {neg_label}", f"Pred {pos_label}"],
            yticklabels=[f"True {neg_label}", f"True {pos_label}"],
            linewidths=0.5, ax=ax, cbar=False, annot_kws={"size": 12},
        )
        auc_str = f"AUC={res['AUC']:.3f}" if "AUC" in res else ""
        ax.set_title(f"{res['Model']}\n{auc_str}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=9)
        ax.set_ylabel("Actual",    fontsize=9)
    fig.suptitle(f"Confusion Matrices — {target_name}",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


def plot_confusion_matrix_4class(res: dict, filepath: Path):
    labels = [LABEL_SHORT[l] for l in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        res["_cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"True {l}" for l in labels],
        linewidths=0.5, ax=ax, cbar=False, annot_kws={"size": 10},
    )
    ax.set_title(
        f"4-Class Confusion Matrix — {res['Model']}\n"
        f"Train BalAcc={res['Train_BalAcc']:.3f}  "
        f"Test BalAcc={res['Balanced_Accuracy']:.3f}  "
        f"Macro F1={res['Macro_F1']:.3f}",
        fontsize=10, fontweight="bold"
    )
    ax.set_xlabel("Predicted", fontsize=9)
    ax.set_ylabel("Actual",    fontsize=9)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


def extract_rf_importance(model, feat_names: list[str]) -> pd.DataFrame:
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    return (pd.DataFrame({"Feature": feat_names,
                          "Importance": clf.feature_importances_})
              .sort_values("Importance", ascending=False)
              .reset_index(drop=True))


def plot_feature_importance(imp_df: pd.DataFrame,
                            title: str, filepath: Path, top_n: int = 15):
    top = imp_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.42)))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color="#2ecc71", edgecolor="white", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.4f}", va="center", fontsize=7.5)
    ax.set_xlabel("Mean Decrease in Impurity (MDI)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


def build_comparison_table(binary_rows: list, class4_rows: list) -> pd.DataFrame:
    bin_cols = ["Model", "Target", "Train_AUC", "AUC", "AUC_Gap",
                "Balanced_Accuracy", "F1", "Precision", "Recall",
                "Type_I_Error", "Type_II_Error"]
    cls_cols = ["Model", "Target", "Train_BalAcc",
                "Balanced_Accuracy", "BalAcc_Gap", "Macro_F1"]

    bin_df = pd.DataFrame([{k: r[k] for k in bin_cols} for r in binary_rows])
    cls_df = pd.DataFrame([{k: r[k] for k in cls_cols} for r in class4_rows])
    return pd.concat([bin_df, cls_df], ignore_index=True, sort=False)


def plot_comparison_bars(comp_df: pd.DataFrame, filepath: Path):
    binary = comp_df[comp_df["Target"] != "4-Class"].copy()
    c4     = comp_df[comp_df["Target"] == "4-Class"].copy()
    models = ["Dummy", "Logistic Regression", "Random Forest"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: AUC for binary models
    ax     = axes[0]
    tgts   = ["Model A", "Model B"]
    x      = np.arange(len(tgts))
    width  = 0.25

    for i, mdl in enumerate(models):
        vals = []
        for tgt in tgts:
            row = binary[(binary["Model"] == mdl) & (binary["Target"] == tgt)]
            vals.append(float(row["AUC"].values[0]) if len(row) else 0.0)
        bars = ax.bar(x + (i - 1) * width, vals, width,
                      label=mdl, color=MODEL_COLORS[mdl],
                      edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(tgts, fontsize=11)
    ax.set_ylabel("Test AUC-ROC", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Binary Models — Test AUC-ROC", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Right: Balanced Accuracy — all models × all targets
    ax2     = axes[1]
    all_tgt = ["Model A", "Model B", "4-Class"]
    x2      = np.arange(len(all_tgt))

    for i, mdl in enumerate(models):
        vals = []
        for tgt in all_tgt:
            if tgt == "4-Class":
                row = c4[c4["Model"] == mdl]
            else:
                row = binary[(binary["Model"] == mdl) & (binary["Target"] == tgt)]
            vals.append(float(row["Balanced_Accuracy"].values[0])
                        if len(row) else 0.0)
        bars = ax2.bar(x2 + (i - 1) * width, vals, width,
                       label=mdl, color=MODEL_COLORS[mdl],
                       edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.005,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x2); ax2.set_xticklabels(all_tgt, fontsize=11)
    ax2.set_ylabel("Balanced Accuracy (Test)", fontsize=11)
    ax2.set_ylim(0, 1.05)
    ax2.set_title("All Models — Balanced Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    fig.suptitle("Step 5 — Baseline Model Comparison  (Test Year: 2023)",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 62)
    print("  STEP 5 — BASELINE MODELS  (Final Version)")
    print("█" * 62 + "\n")

    # ── load & verify ──────────────────────────────────────────────────────
    df, pred_cols = load_and_verify(DATA_PATH)
    df = build_targets(df)
    splits = make_splits(df, pred_cols)

    Xtr  = splits["train"]["X"];  yA_tr = splits["train"]["y_A"]
    yB_tr = splits["train"]["y_B"]; y4_tr = splits["train"]["y_4"]
    Xvl  = splits["val"]["X"];    yA_vl = splits["val"]["y_A"]
    yB_vl = splits["val"]["y_B"];  y4_vl = splits["val"]["y_4"]
    Xte  = splits["test"]["X"];   yA_te = splits["test"]["y_A"]
    yB_te = splits["test"]["y_B"]; y4_te = splits["test"]["y_4"]

    binary_rows = []
    class4_rows = []

    # ══════════════════════════════════════════════════════════════════════
    # MODEL A — Earnings Distress
    # ══════════════════════════════════════════════════════════════════════
    print("═" * 62)
    print("  MODEL A — Earnings Distress  (AD+FD vs H+CD)")
    print("═" * 62)
    print("  Tuning Random Forest on validation set ...")
    rf_params_A = tune_rf_binary(Xtr, yA_tr, Xvl, yA_vl)

    models_A   = build_models_binary(rf_params_A)
    results_A  = []

    for name, mdl in models_A.items():
        res = eval_binary(name, mdl,
                          Xtr, yA_tr, Xvl, yA_vl, Xte, yA_te,
                          target_name="Model A")
        results_A.append(res)
        binary_rows.append(res)
        print(f"  {name:<22}  train={res['Train_AUC']:.4f}  "
              f"test={res['AUC']:.4f}  gap={res['AUC_Gap']:+.4f}  "
              f"F1={res['F1']:.4f}  {res['Gap_Flag']}")

    rf_A = next(r for r in results_A if r["Model"] == "Random Forest")
    imp_A = extract_rf_importance(rf_A["_model_obj"], pred_cols)
    imp_A.to_csv(OUT_TABLES / "feature_importance_model_A.csv", index=False)
    plot_feature_importance(
        imp_A,
        title="Feature Importance — Model A (Earnings Distress)\nRandom Forest",
        filepath=OUT_FIGURES / "feature_importance_model_A.png")
    plot_roc_curves(results_A, "Model A — Earnings Distress",
                    "AD + FD", OUT_FIGURES / "roc_model_A.png")
    plot_confusion_matrices_binary(
        results_A, "Model A — Earnings Distress",
        "Distressed", "Non-Distressed",
        OUT_FIGURES / "cm_model_A.png")

    # ══════════════════════════════════════════════════════════════════════
    # MODEL B — Cash-Flow Distress
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  MODEL B — Cash-Flow Distress  (CD+FD vs H+AD)")
    print("═" * 62)
    print("  Tuning Random Forest on validation set ...")
    rf_params_B = tune_rf_binary(Xtr, yB_tr, Xvl, yB_vl)

    models_B  = build_models_binary(rf_params_B)
    results_B = []

    for name, mdl in models_B.items():
        res = eval_binary(name, mdl,
                          Xtr, yB_tr, Xvl, yB_vl, Xte, yB_te,
                          target_name="Model B")
        results_B.append(res)
        binary_rows.append(res)
        print(f"  {name:<22}  train={res['Train_AUC']:.4f}  "
              f"test={res['AUC']:.4f}  gap={res['AUC_Gap']:+.4f}  "
              f"F1={res['F1']:.4f}  {res['Gap_Flag']}")

    rf_B = next(r for r in results_B if r["Model"] == "Random Forest")
    imp_B = extract_rf_importance(rf_B["_model_obj"], pred_cols)
    imp_B.to_csv(OUT_TABLES / "feature_importance_model_B.csv", index=False)
    plot_feature_importance(
        imp_B,
        title="Feature Importance — Model B (Cash-Flow Distress)\nRandom Forest",
        filepath=OUT_FIGURES / "feature_importance_model_B.png")
    plot_roc_curves(results_B, "Model B — Cash-Flow Distress",
                    "CD + FD", OUT_FIGURES / "roc_model_B.png")
    plot_confusion_matrices_binary(
        results_B, "Model B — Cash-Flow Distress",
        "Cash-Distressed", "Non-Cash-Distressed",
        OUT_FIGURES / "cm_model_B.png")

    # ══════════════════════════════════════════════════════════════════════
    # DIRECT 4-CLASS MODEL
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  DIRECT 4-CLASS MODEL")
    print("═" * 62)
    print("  Tuning Random Forest on validation set ...")
    rf_params_4 = tune_rf_4class(Xtr, y4_tr, Xvl, y4_vl)

    models_4  = build_models_4class(rf_params_4)
    results_4 = []

    for name, mdl in models_4.items():
        res = eval_4class(name, mdl, Xtr, y4_tr, Xte, y4_te)
        results_4.append(res)
        class4_rows.append(res)
        print(f"  {name:<22}  train={res['Train_BalAcc']:.4f}  "
              f"test={res['Balanced_Accuracy']:.4f}  "
              f"gap={res['BalAcc_Gap']:+.4f}  "
              f"MacroF1={res['Macro_F1']:.4f}  {res['Gap_Flag']}")

    for res in results_4:
        plot_confusion_matrix_4class(
            res,
            OUT_FIGURES / (f"cm_4class_"
                           f"{res['Model'].replace(' ','_').lower()}.png"))

    # ══════════════════════════════════════════════════════════════════════
    # OVERFITTING SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print_overfitting_summary(binary_rows, class4_rows)

    # ══════════════════════════════════════════════════════════════════════
    # COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  COMPARISON TABLE")
    print("═" * 62)

    comp_df = build_comparison_table(binary_rows, class4_rows)
    comp_df.to_csv(OUT_TABLES / "baseline_comparison.csv", index=False)

    display_cols = ["Target", "Model", "Train_AUC", "AUC", "AUC_Gap",
                    "Balanced_Accuracy", "F1", "Precision", "Recall",
                    "Type_I_Error", "Type_II_Error",
                    "Train_BalAcc", "BalAcc_Gap", "Macro_F1"]
    display_cols = [c for c in display_cols if c in comp_df.columns]
    print(comp_df[display_cols].to_string(index=False))

    plot_comparison_bars(comp_df, OUT_FIGURES / "baseline_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE MODELS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[save] Saving trained models ...")
    for tag, results in [("A", results_A), ("B", results_B), ("4class", results_4)]:
        for res in results:
            safe = res["Model"].replace(" ", "_").lower()
            fpath = OUT_MODELS / f"model_{tag}_{safe}.pkl"
            joblib.dump(res["_model_obj"], fpath)
            print(f"  {fpath.name}")

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT MANIFEST  (fixed: + concatenation, no | operator)
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  STEP 5 COMPLETE")
    print("═" * 62)

    print("\nTables:")
    for f in (sorted(OUT_TABLES.glob("*baseline*")) +
              sorted(OUT_TABLES.glob("*feature_importance*"))):
        print(f"  {f.name}")

    print("\nFigures:")
    for f in (sorted(OUT_FIGURES.glob("*model_A*")) +
              sorted(OUT_FIGURES.glob("*model_B*")) +
              sorted(OUT_FIGURES.glob("*4class*")) +
              sorted(OUT_FIGURES.glob("*comparison*")) +
              sorted(OUT_FIGURES.glob("*importance*"))):
        print(f"  {f.name}")

    print("\nModels:")
    for f in sorted(OUT_MODELS.iterdir()):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()