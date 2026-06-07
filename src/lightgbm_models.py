"""
STEP 6 — LightGBM Models
=========================
Financial Distress Prediction for Indian Firms

Models:
  LightGBM Binary — Model A  (Earnings Distress:   AD+FD vs H+CD)
  LightGBM Binary — Model B  (Cash-Flow Distress:  CD+FD vs H+AD)
  LightGBM Multiclass        (4-category taxonomy: H / AD / CD / FD)

Key design decisions:
  - Hyperparameter tuning via Optuna Bayesian search on validation year 2022
  - GPU auto-detection (NVIDIA RTX 3050); silent fallback to CPU
  - Threshold selected on validation set (maximise F1); all metrics on test 2023
  - Train AUC reported alongside test AUC for every model
  - Comparison table pulls Step 5 baseline results from saved CSV

Temporal split (unchanged from Step 5):
  Train      : Year_t ∈ {2018, 2019, 2020, 2021}
  Validation : Year_t = 2022   (tuning + threshold selection)
  Test       : Year_t = 2023   (all reported metrics)

Run from project root:
    python src/lightgbm_models.py
"""

# ── Imports ────────────────────────────────────────────────────────────────────
from pathlib import Path
import warnings
import time
import joblib

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from sklearn.metrics import (
    roc_auc_score, roc_curve,
    confusion_matrix,
    f1_score, precision_score, recall_score,
    balanced_accuracy_score,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent.parent
DATA_PATH   = ROOT / "data"    / "processed" / "panel_dataset.csv"
BASELINE_CSV = ROOT / "outputs" / "tables"  / "baseline_comparison.csv"
OUT_TABLES  = ROOT / "outputs" / "tables"
OUT_FIGURES = ROOT / "outputs" / "figures"
OUT_MODELS  = ROOT / "outputs" / "models"
for p in [OUT_TABLES, OUT_FIGURES, OUT_MODELS]:
    p.mkdir(parents=True, exist_ok=True)


# ── Constants ──────────────────────────────────────────────────────────────────
TRAIN_YEARS  = [2018, 2019, 2020, 2021]
VAL_YEAR     = 2022
TEST_YEAR    = 2023
N_TRIALS     = 60        # Optuna trials per model
RANDOM_SEED  = 42
FORBIDDEN    = {"Interest_Coverage", "Cash_Interest_Coverage", "DSCR"}

LABEL_ORDER  = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
LABEL_SHORT  = {"Healthy": "H", "Acct_Distress": "AD",
                "Cash_Distress": "CD", "Full_Distress": "FD"}
LABEL_MAP    = {"Healthy": 0, "Acct_Distress": 1,
                "Cash_Distress": 2, "Full_Distress": 3}

GAP_THRESHOLD_BINARY = 0.03
GAP_THRESHOLD_4CLASS = 0.10

MODEL_COLORS = {
    "Dummy"              : "#95a5a6",
    "Logistic Regression": "#3498db",
    "Random Forest"      : "#2ecc71",
    "LightGBM"           : "#e74c3c",
}


# ══════════════════════════════════════════════════════════════════════════════
# 1. GPU DETECTION
# ══════════════════════════════════════════════════════════════════════════════
def detect_device() -> str:
    """
    Attempt to instantiate a minimal LightGBM model with device_type='gpu'.
    Returns 'gpu' if successful, 'cpu' otherwise.
    Prints a clear status line.
    """
    try:
        probe = lgb.LGBMClassifier(
            n_estimators=1, num_leaves=4,
            device_type="gpu", verbose=-1,
        )
        _dummy_X = np.random.rand(20, 3)
        _dummy_y = np.array([0] * 10 + [1] * 10)
        probe.fit(_dummy_X, _dummy_y)
        print("  [GPU] NVIDIA GPU detected — LightGBM will use GPU training.")
        return "gpu"
    except Exception:
        print("  [GPU] GPU unavailable or LightGBM GPU build not installed "
              "— falling back to CPU.")
        return "cpu"


# ══════════════════════════════════════════════════════════════════════════════
# 2. LOAD & VERIFY
# ══════════════════════════════════════════════════════════════════════════════
def load_and_verify(path: Path) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(path)

    print("═" * 62)
    print("  VERIFICATION SUITE")
    print("═" * 62)

    meta_cols = ["Company Name", "Year_t",
                 "Industry group", "Industry Class", "Industry type", "Label"]
    pred_cols = [c for c in df.columns if c not in meta_cols]

    # 1. Leakage
    leaked = FORBIDDEN.intersection(set(pred_cols))
    status = "✓ PASS" if not leaked else f"✗ FAIL — found: {leaked}"
    print(f"\n  [1] Forbidden variable check          : {status}")
    if leaked:
        raise ValueError(f"Leakage detected: {leaked}")

    # 2. Predictor count
    ok = "✓ PASS" if len(pred_cols) == 27 else "⚠  CHECK"
    print(f"  [2] Predictor count                   : {len(pred_cols)}  {ok}")

    # 3. Years
    years = sorted(df["Year_t"].unique())
    expected = TRAIN_YEARS + [VAL_YEAR, TEST_YEAR]
    print(f"  [3] Years present                     : {years}")
    print(f"      Split check                       : "
          f"{'✓ PASS' if set(years)==set(expected) else '✗ FAIL'}")

    # 4. Labels
    ok_lbl = set(df["Label"].unique()) == set(LABEL_ORDER)
    print(f"  [4] Label check                       : "
          f"{'✓ PASS' if ok_lbl else '✗ FAIL'}")

    # 5. Missing
    n_miss = df[pred_cols].isnull().sum().sum()
    print(f"  [5] Missing predictor values          : {n_miss}  "
          f"({'✓ PASS' if n_miss == 0 else '✗ FAIL'})")

    # 6. Class distribution
    print(f"\n  [6] Label distribution:")
    for lbl in LABEL_ORDER:
        n = (df["Label"] == lbl).sum()
        print(f"        {LABEL_SHORT[lbl]:<3} {lbl:<20} {n:>7,}  "
              f"({100*n/len(df):5.1f}%)")
    print(f"        {'Total':<24} {len(df):>7,}")

    # 7. Split sizes
    print(f"\n  [7] Split sizes:")
    for yr in TRAIN_YEARS:
        print(f"        Train Year_t={yr} : {(df['Year_t']==yr).sum():,}")
    print(f"        Val   Year_t={VAL_YEAR}  : {(df['Year_t']==VAL_YEAR).sum():,}")
    print(f"        Test  Year_t={TEST_YEAR}  : {(df['Year_t']==TEST_YEAR).sum():,}")

    print("\n  All checks complete.\n")
    return df, pred_cols


# ══════════════════════════════════════════════════════════════════════════════
# 3. BUILD TARGETS & SPLITS
# ══════════════════════════════════════════════════════════════════════════════
def build_targets_and_splits(df: pd.DataFrame,
                             pred_cols: list[str]) -> dict:
    df = df.copy()
    df["y_A"] = df["Label"].isin(["Acct_Distress", "Full_Distress"]).astype(int)
    df["y_B"] = df["Label"].isin(["Cash_Distress", "Full_Distress"]).astype(int)
    df["y_4"] = df["Label"].map(LABEL_MAP)

    splits = {}
    for name, years in [("train", TRAIN_YEARS),
                        ("val",   [VAL_YEAR]),
                        ("test",  [TEST_YEAR])]:
        sub = df[df["Year_t"].isin(years)].reset_index(drop=True)
        splits[name] = {
            "X"  : sub[pred_cols].values,
            "y_A": sub["y_A"].values,
            "y_B": sub["y_B"].values,
            "y_4": sub["y_4"].values,
        }

    print("Split summary:")
    for name, s in splits.items():
        print(f"  {name:<6}: {len(s['X']):>6,} rows  |  "
              f"y_A pos: {s['y_A'].sum():,} ({100*s['y_A'].mean():.1f}%)  |  "
              f"y_B pos: {s['y_B'].sum():,} ({100*s['y_B'].mean():.1f}%)")
    print()
    return splits


# ══════════════════════════════════════════════════════════════════════════════
# 4. OPTUNA TUNING — BINARY
# ══════════════════════════════════════════════════════════════════════════════
def tune_binary(X_train, y_train, X_val, y_val,
                device: str, n_trials: int = N_TRIALS,
                target_name: str = "") -> dict:
    """
    Bayesian hyperparameter search for binary LightGBM.
    Objective: maximise AUC on validation set.
    """
    print(f"  Tuning {target_name} ({n_trials} Optuna trials, device={device}) ...")
    t0 = time.time()

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        params = {
            "objective"       : "binary",
            "metric"          : "auc",
            "verbosity"       : -1,
            "random_state"    : RANDOM_SEED,
            "device_type"     : device,
            "is_unbalance"    : True,
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 200),
            "max_depth"       : trial.suggest_int("max_depth", 3, 12),
            "learning_rate"   : trial.suggest_float("learning_rate",
                                                     0.01, 0.3, log=True),
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq"    : 1,
            "lambda_l1"       : trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2"       : trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        mdl = lgb.LGBMClassifier(**params)
        mdl.fit(X_train, y_train)
        prob = mdl.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, prob)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    best_val_auc = study.best_value
    elapsed = time.time() - t0
    print(f"    Best val AUC = {best_val_auc:.4f}  "
          f"(num_leaves={best.get('num_leaves')}  "
          f"max_depth={best.get('max_depth')}  "
          f"lr={best.get('learning_rate'):.4f}  "
          f"n_est={best.get('n_estimators')})  "
          f"[{elapsed:.0f}s]")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 5. OPTUNA TUNING — 4-CLASS
# ══════════════════════════════════════════════════════════════════════════════
def tune_4class(X_train, y_train, X_val, y_val,
                device: str, n_trials: int = N_TRIALS) -> dict:
    """
    Bayesian search for 4-class LightGBM.
    Objective: maximise balanced accuracy on validation set.
    """
    print(f"  Tuning 4-Class ({n_trials} Optuna trials, device={device}) ...")
    t0 = time.time()

    def objective(trial):
        params = {
            "objective"       : "multiclass",
            "num_class"       : 4,
            "metric"          : "multi_logloss",
            "verbosity"       : -1,
            "random_state"    : RANDOM_SEED,
            "device_type"     : device,
            "class_weight"    : "balanced",
            "num_leaves"      : trial.suggest_int("num_leaves", 20, 200),
            "max_depth"       : trial.suggest_int("max_depth", 3, 12),
            "learning_rate"   : trial.suggest_float("learning_rate",
                                                     0.01, 0.3, log=True),
            "n_estimators"    : trial.suggest_int("n_estimators", 100, 800),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq"    : 1,
            "lambda_l1"       : trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2"       : trial.suggest_float("lambda_l2", 0.0, 5.0),
        }
        mdl = lgb.LGBMClassifier(**params)
        try:
            mdl.fit(X_train, y_train)
        except Exception:
            raise optuna.exceptions.TrialPruned()
        return balanced_accuracy_score(y_val, mdl.predict(X_val))

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = study.best_params
    elapsed = time.time() - t0
    print(f"    Best val BalAcc = {study.best_value:.4f}  "
          f"(num_leaves={best.get('num_leaves')}  "
          f"max_depth={best.get('max_depth')}  "
          f"lr={best.get('learning_rate'):.4f})  "
          f"[{elapsed:.0f}s]")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 6. BUILD FINAL BINARY MODEL
# ══════════════════════════════════════════════════════════════════════════════
def build_binary_model(best_params: dict, device: str) -> lgb.LGBMClassifier:
    params = {
        "objective"        : "binary",
        "verbosity"        : -1,
        "random_state"     : RANDOM_SEED,
        "device_type"      : device,
        "is_unbalance"     : True,
        "bagging_freq"     : 1,
        **best_params,
    }
    return lgb.LGBMClassifier(**params)


def build_4class_model(best_params: dict, device: str) -> lgb.LGBMClassifier:
    params = {
        "objective"        : "multiclass",
        "num_class"        : 4,
        "verbosity"        : -1,
        "random_state"     : RANDOM_SEED,
        "device_type"      : device,
        "class_weight"     : "balanced",
        "bagging_freq"     : 1,
        **best_params,
    }
    return lgb.LGBMClassifier(**params)


# ══════════════════════════════════════════════════════════════════════════════
# 7. EVALUATE BINARY
# ══════════════════════════════════════════════════════════════════════════════
def eval_binary(model: lgb.LGBMClassifier,
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test,
                target_name: str) -> dict:

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

    prob_train = model.predict_proba(X_train)[:, 1]
    prob_val   = model.predict_proba(X_val)[:, 1]
    prob_test  = model.predict_proba(X_test)[:, 1]

    train_auc = roc_auc_score(y_train, prob_train)
    test_auc  = roc_auc_score(y_test,  prob_test)

    # Threshold: maximise F1 on validation set
    thresholds = np.linspace(0.05, 0.95, 91)
    f1_vals    = [f1_score(y_val, (prob_val >= t).astype(int), zero_division=0)
                  for t in thresholds]
    best_thr   = thresholds[int(np.argmax(f1_vals))]
    y_pred     = (prob_test >= best_thr).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    type1 = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    type2 = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    gap   = train_auc - test_auc
    gap_flag = "✓" if abs(gap) <= GAP_THRESHOLD_BINARY else "⚠  GAP EXCEEDS THRESHOLD"

    return {
        "Model"            : "LightGBM",
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
        "_model"      : model,
        "_prob_test"  : prob_test,
        "_y_test"     : y_test,
        "_cm"         : cm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8. EVALUATE 4-CLASS
# ══════════════════════════════════════════════════════════════════════════════
def eval_4class(model: lgb.LGBMClassifier,
                X_train, y_train,
                X_val,   y_val,
                X_test,  y_test) -> dict:

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False),
                         lgb.log_evaluation(-1)])

    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)

    train_ba = balanced_accuracy_score(y_train, y_pred_train)
    test_ba  = balanced_accuracy_score(y_test,  y_pred_test)
    mac_f1   = f1_score(y_test, y_pred_test, average="macro", zero_division=0)
    cm       = confusion_matrix(y_test, y_pred_test, labels=[0, 1, 2, 3])

    gap      = train_ba - test_ba
    gap_flag = "✓" if abs(gap) <= GAP_THRESHOLD_4CLASS else "⚠  GAP EXCEEDS THRESHOLD"

    return {
        "Model"            : "LightGBM",
        "Target"           : "4-Class",
        "Train_BalAcc"     : round(train_ba, 4),
        "Balanced_Accuracy": round(test_ba,  4),
        "BalAcc_Gap"       : round(gap, 4),
        "Gap_Flag"         : gap_flag,
        "Macro_F1"         : round(mac_f1, 4),
        "_cm"              : cm,
        "_model"           : model,
        "_y_test"          : y_test,
        "_y_pred"          : y_pred_test,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9. FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def get_importance(model: lgb.LGBMClassifier,
                   feat_names: list[str]) -> pd.DataFrame:
    imp = model.feature_importances_
    return (pd.DataFrame({"Feature": feat_names, "Importance": imp})
              .sort_values("Importance", ascending=False)
              .reset_index(drop=True))


def plot_importance(imp_df: pd.DataFrame,
                    title: str, filepath: Path, top_n: int = 15):
    top = imp_df.head(top_n).copy()
    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.42)))
    bars = ax.barh(top["Feature"][::-1], top["Importance"][::-1],
                   color="#e74c3c", edgecolor="white", linewidth=0.4)
    for bar in bars:
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{bar.get_width():.0f}", va="center", fontsize=7.5)
    ax.set_xlabel("LightGBM Feature Importance (Split Count)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 10. ROC CURVE PLOT (LightGBM vs baselines)
# ══════════════════════════════════════════════════════════════════════════════
def plot_roc_comparison(lgb_res: dict,
                        baseline_csv: Path,
                        target_name: str,
                        pos_label: str,
                        filepath: Path,
                        X_test, y_test,
                        step5_models: dict):
    """
    Overlay LightGBM ROC against Step 5 Logistic Regression and Random Forest.
    step5_models: dict of {model_name: trained_model} loaded from pkl files.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    # Step 5 baselines
    colors_step5 = {"Logistic Regression": "#3498db", "Random Forest": "#2ecc71"}
    for mname, mdl in step5_models.items():
        try:
            prob = mdl.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, prob)
            auc = roc_auc_score(y_test, prob)
            ax.plot(fpr, tpr,
                    label=f"{mname}  (AUC={auc:.3f})",
                    color=colors_step5.get(mname, "grey"),
                    linewidth=1.6, linestyle="--")
        except Exception:
            pass

    # LightGBM
    fpr, tpr, _ = roc_curve(lgb_res["_y_test"], lgb_res["_prob_test"])
    ax.plot(fpr, tpr,
            label=f"LightGBM  (train={lgb_res['Train_AUC']:.3f} / "
                  f"test={lgb_res['AUC']:.3f})",
            color="#e74c3c", linewidth=2.2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate",  fontsize=11)
    ax.set_title(f"ROC Curves — {target_name}\n(Positive: {pos_label})",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONFUSION MATRIX PLOT (4-class)
# ══════════════════════════════════════════════════════════════════════════════
def plot_cm_4class(res: dict, filepath: Path):
    labels = [LABEL_SHORT[l] for l in LABEL_ORDER]
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        res["_cm"], annot=True, fmt="d", cmap="Blues",
        xticklabels=[f"Pred {l}" for l in labels],
        yticklabels=[f"True {l}" for l in labels],
        linewidths=0.5, ax=ax, cbar=False, annot_kws={"size": 10},
    )
    ax.set_title(
        f"4-Class Confusion Matrix — LightGBM\n"
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


# ══════════════════════════════════════════════════════════════════════════════
# 12. FULL COMPARISON TABLE (LightGBM + Step 5 baselines)
# ══════════════════════════════════════════════════════════════════════════════
def build_full_comparison(lgb_binary_rows: list,
                          lgb_4class_row:  dict,
                          baseline_csv:    Path) -> pd.DataFrame:
    """
    Merge LightGBM results with Step 5 baseline_comparison.csv.
    Keeps only the metrics that exist for both binary and 4-class.
    """
    # LightGBM rows
    bin_cols = ["Model", "Target", "Train_AUC", "AUC", "AUC_Gap",
                "Balanced_Accuracy", "F1", "Precision", "Recall",
                "Type_I_Error", "Type_II_Error"]
    cls_cols = ["Model", "Target", "Train_BalAcc",
                "Balanced_Accuracy", "BalAcc_Gap", "Macro_F1"]

    lgb_bin = pd.DataFrame([{k: r[k] for k in bin_cols}
                             for r in lgb_binary_rows])
    lgb_cls = pd.DataFrame([{k: lgb_4class_row[k] for k in cls_cols}])
    lgb_all = pd.concat([lgb_bin, lgb_cls], ignore_index=True, sort=False)

    # Step 5 baselines
    if baseline_csv.exists():
        step5 = pd.read_csv(baseline_csv)
        # keep only columns that exist in both
        common = [c for c in lgb_all.columns if c in step5.columns]
        step5  = step5[common]
        combined = pd.concat([step5, lgb_all[common]], ignore_index=True, sort=False)
    else:
        print("  [warn] baseline_comparison.csv not found — "
              "LightGBM-only table produced.")
        combined = lgb_all

    return combined


def plot_full_comparison(comp_df: pd.DataFrame, filepath: Path):
    """
    Three-panel bar chart: AUC (binary), Balanced Accuracy (all), Macro F1 (4-class).
    """
    binary  = comp_df[comp_df["Target"].isin(["Model A", "Model B"])].copy()
    c4      = comp_df[comp_df["Target"] == "4-Class"].copy()
    models  = ["Dummy", "Logistic Regression", "Random Forest", "LightGBM"]
    targets = ["Model A", "Model B"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 1 — AUC
    ax   = axes[0]
    x    = np.arange(len(targets))
    w    = 0.20
    for i, mdl in enumerate(models):
        vals = []
        for tgt in targets:
            row = binary[(binary["Model"] == mdl) & (binary["Target"] == tgt)]
            vals.append(float(row["AUC"].values[0]) if len(row) and not
                        pd.isna(row["AUC"].values[0]) else 0.0)
        bars = ax.bar(x + (i - 1.5) * w, vals, w, label=mdl,
                      color=MODEL_COLORS.get(mdl, "grey"),
                      edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.004,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(targets, fontsize=11)
    ax.set_ylabel("Test AUC-ROC", fontsize=11)
    ax.set_ylim(0.4, 1.02)
    ax.set_title("Binary — AUC-ROC", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8); ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel 2 — Balanced Accuracy (all)
    ax2      = axes[1]
    all_tgts = ["Model A", "Model B", "4-Class"]
    x2 = np.arange(len(all_tgts))
    for i, mdl in enumerate(models):
        vals = []
        for tgt in all_tgts:
            if tgt == "4-Class":
                row = c4[c4["Model"] == mdl]
            else:
                row = binary[(binary["Model"] == mdl) & (binary["Target"] == tgt)]
            vals.append(float(row["Balanced_Accuracy"].values[0])
                        if len(row) and not
                        pd.isna(row["Balanced_Accuracy"].values[0]) else 0.0)
        bars = ax2.bar(x2 + (i - 1.5) * w, vals, w, label=mdl,
                       color=MODEL_COLORS.get(mdl, "grey"),
                       edgecolor="white", alpha=0.9)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + 0.004,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax2.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
    ax2.set_xticks(x2); ax2.set_xticklabels(all_tgts, fontsize=11)
    ax2.set_ylabel("Balanced Accuracy (Test)", fontsize=11)
    ax2.set_ylim(0, 1.02)
    ax2.set_title("All Models — Balanced Accuracy", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8); ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3 — Macro F1 (4-class only)
    ax3 = axes[2]
    x3  = np.arange(len(models))
    vals3 = []
    for mdl in models:
        row = c4[c4["Model"] == mdl]
        vals3.append(float(row["Macro_F1"].values[0])
                     if len(row) and not
                     pd.isna(row["Macro_F1"].values[0]) else 0.0)
    colors3 = [MODEL_COLORS.get(m, "grey") for m in models]
    bars3 = ax3.bar(x3, vals3, 0.55, color=colors3,
                    edgecolor="white", alpha=0.9)
    for bar, v in zip(bars3, vals3):
        if v > 0:
            ax3.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.004,
                     f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax3.set_xticks(x3)
    ax3.set_xticklabels([m.replace(" ", "\n") for m in models], fontsize=9)
    ax3.set_ylabel("Macro F1 (Test)", fontsize=11)
    ax3.set_ylim(0, 0.8)
    ax3.set_title("4-Class — Macro F1", fontsize=12, fontweight="bold")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)

    fig.suptitle(
        "Step 6 — LightGBM vs Baselines  (Test Year: 2023)",
        fontsize=14, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] {filepath.name}")


# ══════════════════════════════════════════════════════════════════════════════
# 13. OVERFITTING SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
def print_gap_summary(binary_rows: list, cls_row: dict):
    print("\n" + "═" * 62)
    print("  TRAIN-TEST GAP SUMMARY  (LightGBM only)")
    print("═" * 62)
    print(f"  {'Target':<12} {'Train':>8} {'Test':>8} "
          f"{'Gap':>8}  Status")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8}  {'-'*25}")
    for r in binary_rows:
        print(f"  {r['Target']:<12} "
              f"{r['Train_AUC']:>8.4f} {r['AUC']:>8.4f} "
              f"{r['AUC_Gap']:>+8.4f}  {r['Gap_Flag']}")
    print(f"  {'4-Class':<12} "
          f"{cls_row['Train_BalAcc']:>8.4f} "
          f"{cls_row['Balanced_Accuracy']:>8.4f} "
          f"{cls_row['BalAcc_Gap']:>+8.4f}  {cls_row['Gap_Flag']}")
    print(f"\n  Threshold binary  : gap ≤ {GAP_THRESHOLD_BINARY}")
    print(f"  Threshold 4-class : gap ≤ {GAP_THRESHOLD_4CLASS}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "█" * 62)
    print("  STEP 6 — LIGHTGBM MODELS")
    print("█" * 62 + "\n")

    # ── device detection ───────────────────────────────────────────────────
    device = detect_device()
    print()

    # ── load & verify ──────────────────────────────────────────────────────
    df, pred_cols = load_and_verify(DATA_PATH)
    splits = build_targets_and_splits(df, pred_cols)

    Xtr  = splits["train"]["X"]
    yA_tr, yB_tr, y4_tr = (splits["train"]["y_A"],
                            splits["train"]["y_B"],
                            splits["train"]["y_4"])
    Xvl  = splits["val"]["X"]
    yA_vl, yB_vl, y4_vl = (splits["val"]["y_A"],
                            splits["val"]["y_B"],
                            splits["val"]["y_4"])
    Xte  = splits["test"]["X"]
    yA_te, yB_te, y4_te = (splits["test"]["y_A"],
                            splits["test"]["y_B"],
                            splits["test"]["y_4"])

    lgb_binary_rows = []

    # ══════════════════════════════════════════════════════════════════════
    # MODEL A — Earnings Distress
    # ══════════════════════════════════════════════════════════════════════
    print("═" * 62)
    print("  MODEL A — Earnings Distress  (AD+FD vs H+CD)")
    print("═" * 62)
    best_A  = tune_binary(Xtr, yA_tr, Xvl, yA_vl, device,
                          target_name="Model A")
    mdl_A   = build_binary_model(best_A, device)
    res_A   = eval_binary(mdl_A, Xtr, yA_tr, Xvl, yA_vl, Xte, yA_te,
                          target_name="Model A")
    lgb_binary_rows.append(res_A)

    print(f"  LightGBM   train={res_A['Train_AUC']:.4f}  "
          f"test={res_A['AUC']:.4f}  gap={res_A['AUC_Gap']:+.4f}  "
          f"F1={res_A['F1']:.4f}  {res_A['Gap_Flag']}")

    # load Step 5 models for ROC overlay
    step5_A = {}
    for mname, fname in [("Logistic Regression", "model_A_logistic_regression.pkl"),
                          ("Random Forest",        "model_A_random_forest.pkl")]:
        fpath = OUT_MODELS / fname
        if fpath.exists():
            step5_A[mname] = joblib.load(fpath)

    imp_A = get_importance(mdl_A, pred_cols)
    imp_A.to_csv(OUT_TABLES / "lgb_feature_importance_model_A.csv", index=False)
    plot_importance(imp_A,
                    title="LightGBM Feature Importance — Model A (Earnings Distress)",
                    filepath=OUT_FIGURES / "lgb_importance_model_A.png")
    plot_roc_comparison(res_A, BASELINE_CSV, "Model A — Earnings Distress",
                        "AD + FD",
                        OUT_FIGURES / "lgb_roc_model_A.png",
                        Xte, yA_te, step5_A)

    # ══════════════════════════════════════════════════════════════════════
    # MODEL B — Cash-Flow Distress
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  MODEL B — Cash-Flow Distress  (CD+FD vs H+AD)")
    print("═" * 62)
    best_B  = tune_binary(Xtr, yB_tr, Xvl, yB_vl, device,
                          target_name="Model B")
    mdl_B   = build_binary_model(best_B, device)
    res_B   = eval_binary(mdl_B, Xtr, yB_tr, Xvl, yB_vl, Xte, yB_te,
                          target_name="Model B")
    lgb_binary_rows.append(res_B)

    print(f"  LightGBM   train={res_B['Train_AUC']:.4f}  "
          f"test={res_B['AUC']:.4f}  gap={res_B['AUC_Gap']:+.4f}  "
          f"F1={res_B['F1']:.4f}  {res_B['Gap_Flag']}")

    step5_B = {}
    for mname, fname in [("Logistic Regression", "model_B_logistic_regression.pkl"),
                          ("Random Forest",        "model_B_random_forest.pkl")]:
        fpath = OUT_MODELS / fname
        if fpath.exists():
            step5_B[mname] = joblib.load(fpath)

    imp_B = get_importance(mdl_B, pred_cols)
    imp_B.to_csv(OUT_TABLES / "lgb_feature_importance_model_B.csv", index=False)
    plot_importance(imp_B,
                    title="LightGBM Feature Importance — Model B (Cash-Flow Distress)",
                    filepath=OUT_FIGURES / "lgb_importance_model_B.png")
    plot_roc_comparison(res_B, BASELINE_CSV, "Model B — Cash-Flow Distress",
                        "CD + FD",
                        OUT_FIGURES / "lgb_roc_model_B.png",
                        Xte, yB_te, step5_B)

    # ══════════════════════════════════════════════════════════════════════
    # DIRECT 4-CLASS MODEL
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  DIRECT 4-CLASS MODEL")
    print("═" * 62)
    best_4  = tune_4class(Xtr, y4_tr, Xvl, y4_vl, device)
    mdl_4   = build_4class_model(best_4, device)
    res_4   = eval_4class(mdl_4, Xtr, y4_tr, Xvl, y4_vl, Xte, y4_te)

    print(f"  LightGBM   train={res_4['Train_BalAcc']:.4f}  "
          f"test={res_4['Balanced_Accuracy']:.4f}  "
          f"gap={res_4['BalAcc_Gap']:+.4f}  "
          f"MacroF1={res_4['Macro_F1']:.4f}  {res_4['Gap_Flag']}")

    imp_4 = get_importance(mdl_4, pred_cols)
    imp_4.to_csv(OUT_TABLES / "lgb_feature_importance_4class.csv", index=False)
    plot_importance(imp_4,
                    title="LightGBM Feature Importance — 4-Class Model",
                    filepath=OUT_FIGURES / "lgb_importance_4class.png")
    plot_cm_4class(res_4, OUT_FIGURES / "lgb_cm_4class.png")

    # ══════════════════════════════════════════════════════════════════════
    # GAP SUMMARY
    # ══════════════════════════════════════════════════════════════════════
    print_gap_summary(lgb_binary_rows, res_4)

    # ══════════════════════════════════════════════════════════════════════
    # FULL COMPARISON TABLE
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  FULL COMPARISON TABLE  (LightGBM + Step 5 Baselines)")
    print("═" * 62)
    comp_df = build_full_comparison(lgb_binary_rows, res_4, BASELINE_CSV)
    comp_df.to_csv(OUT_TABLES / "lgb_full_comparison.csv", index=False)

    display_cols = ["Target", "Model", "Train_AUC", "AUC", "AUC_Gap",
                    "Balanced_Accuracy", "F1", "Precision", "Recall",
                    "Type_I_Error", "Type_II_Error",
                    "Train_BalAcc", "BalAcc_Gap", "Macro_F1"]
    dc = [c for c in display_cols if c in comp_df.columns]
    print(comp_df[dc].to_string(index=False))

    plot_full_comparison(comp_df, OUT_FIGURES / "lgb_full_comparison.png")

    # ══════════════════════════════════════════════════════════════════════
    # SAVE LightGBM MODELS
    # ══════════════════════════════════════════════════════════════════════
    print("\n[save] Saving LightGBM models ...")
    for tag, mdl in [("A", mdl_A), ("B", mdl_B), ("4class", mdl_4)]:
        fpath = OUT_MODELS / f"lgb_model_{tag}.pkl"
        joblib.dump(mdl, fpath)
        print(f"  {fpath.name}")

    # Save best hyperparameters for reproducibility
    import json
    hp_record = {
        "model_A"   : best_A,
        "model_B"   : best_B,
        "model_4class": best_4,
        "device"    : device,
        "n_trials"  : N_TRIALS,
        "random_seed": RANDOM_SEED,
    }
    hp_path = OUT_TABLES / "lgb_best_hyperparameters.json"
    with open(hp_path, "w") as f:
        json.dump(hp_record, f, indent=2)
    print(f"  {hp_path.name}  (hyperparameters saved)")

    # ══════════════════════════════════════════════════════════════════════
    # OUTPUT MANIFEST
    # ══════════════════════════════════════════════════════════════════════
    print()
    print("═" * 62)
    print("  STEP 6 COMPLETE")
    print("═" * 62)

    print("\nTables:")
    for f in (sorted(OUT_TABLES.glob("lgb_*")) +
              sorted(OUT_TABLES.glob("*comparison*"))):
        print(f"  {f.name}")

    print("\nFigures:")
    for f in (sorted(OUT_FIGURES.glob("lgb_*")) +
              sorted(OUT_FIGURES.glob("*comparison*"))):
        print(f"  {f.name}")

    print("\nModels:")
    for f in sorted(OUT_MODELS.glob("lgb_*")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()