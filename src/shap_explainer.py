"""
shap_explainer.py
=================
SHAP Explainability Module — Financial Distress Prediction Platform
--------------------------------------------------------------------
Provides local (single-company) and global (population-level) explanations
for LightGBM distress predictions.

Supports:
  Model A — Earnings Distress   (positive = AD + FD)
  Model B — Cash-Flow Distress  (positive = CD + FD)
  Dual-binary reconstruction    (A + B → 4-category taxonomy)
  4-class model                 (direct multiclass)

Usage:
  from shap_explainer import DistressExplainer
  explainer = DistressExplainer.load("outputs/models/")
  result    = explainer.explain_company(company_ratios_dict)
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


# ── Constants ──────────────────────────────────────────────────────────────────
LABEL_ORDER = ["Healthy", "Acct_Distress", "Cash_Distress", "Full_Distress"]
LABEL_DISPLAY = {
    "Healthy"       : "Healthy",
    "Acct_Distress" : "Accounting Distress",
    "Cash_Distress" : "Cash-Flow Distress",
    "Full_Distress" : "Full Distress",
}
LABEL_MAP_INV = {0: "Healthy", 1: "Acct_Distress",
                 2: "Cash_Distress", 3: "Full_Distress"}

# Risk colours used in all plots
COLOR = {
    "Healthy"       : "#27ae60",
    "Acct_Distress" : "#e67e22",
    "Cash_Distress" : "#2980b9",
    "Full_Distress" : "#e74c3c",
    "increase"      : "#e74c3c",   # pushes toward distress
    "decrease"      : "#27ae60",   # pushes away from distress
    "neutral"       : "#95a5a6",
}

# The 27 predictors used during model training (in training order)
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

# Human-readable labels for plots
FEATURE_LABELS = {
    "CFO_to_CL"              : "CFO / Current Liabilities",
    "CFO_to_Debt"            : "CFO / Total Debt",
    "CFO_to_Net_Income"      : "CFO / Net Income",
    "Cash_Conversion_Cycle"  : "Cash Conversion Cycle (days)",
    "Cash_to_CL"             : "Cash / Current Liabilities",
    "Current_Ratio"          : "Current Ratio",
    "Debt_to_Assets"         : "Debt / Total Assets",
    "Debt_to_Equity"         : "Debt / Equity",
    "EBITDA_Margin"          : "EBITDA Margin",
    "Inventory_Turnover"     : "Inventory Turnover",
    "LT_Debt_to_Assets"      : "LT Debt / Total Assets",
    "NCF_to_Debt"            : "Net Cash Flow / Debt",
    "NCF_to_TA"              : "Net Cash Flow / Total Assets",
    "Net_Debt_to_EBITDA"     : "Net Debt / EBITDA",
    "Net_Profit_Margin"      : "Net Profit Margin",
    "Operating_Profit_Margin": "Operating Profit Margin",
    "Payables_Turnover"      : "Payables Turnover",
    "Quick_Ratio"            : "Quick Ratio",
    "ROA"                    : "Return on Assets",
    "ROCE"                   : "Return on Capital Employed",
    "ROE"                    : "Return on Equity",
    "Receivables_Turnover"   : "Receivables Turnover",
    "TA_to_Equity"           : "Total Assets / Equity",
    "TL_to_TA"               : "Total Liabilities / Total Assets",
    "Total_Asset_Turnover"   : "Asset Turnover",
    "WC_Turnover"            : "Working Capital Turnover",
    "WC_to_TA"               : "Working Capital / Total Assets",
}


# ══════════════════════════════════════════════════════════════════════════════
# CORE CLASS
# ══════════════════════════════════════════════════════════════════════════════
class DistressExplainer:
    """
    Loads trained LightGBM models and provides SHAP-based explanations.

    Parameters
    ----------
    model_A     : trained LightGBM for Earnings Distress (binary)
    model_B     : trained LightGBM for Cash-Flow Distress (binary)
    model_4class: trained LightGBM for direct 4-class prediction
    output_dir  : where plots are saved

    At minimum one of model_A / model_B / model_4class must be provided.
    """

    def __init__(
        self,
        model_A      = None,
        model_B      = None,
        model_4class = None,
        output_dir   : str | Path = "outputs/shap",
        feature_names: list[str]  = None,
    ):
        self.model_A       = model_A
        self.model_B       = model_B
        self.model_4class  = model_4class
        self.output_dir    = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.feature_names = feature_names or PREDICTOR_NAMES

        # Build SHAP TreeExplainers — TreeExplainer is exact for LightGBM
        self._explainer_A = (
            shap.TreeExplainer(model_A)  if model_A       else None
        )
        self._explainer_B = (
            shap.TreeExplainer(model_B)  if model_B       else None
        )
        self._explainer_4 = (
            shap.TreeExplainer(model_4class) if model_4class else None
        )

    # ── Factory method ─────────────────────────────────────────────────────
    @classmethod
    def load(
        cls,
        model_dir    : str | Path = "outputs/models",
        output_dir   : str | Path = "outputs/shap",
        feature_names: list[str]  = None,
    ) -> "DistressExplainer":
        """
        Load all LightGBM pkl files from model_dir.
        Missing files are silently skipped — at least one must exist.
        """
        model_dir = Path(model_dir)
        models = {}
        for key, fname in [
            ("model_A",       "lgb_model_A.pkl"),
            ("model_B",       "lgb_model_B.pkl"),
            ("model_4class",  "lgb_model_4class.pkl"),
        ]:
            path = model_dir / fname
            if path.exists():
                models[key] = joblib.load(path)
                print(f"[load] {fname} ✓")
            else:
                print(f"[load] {fname} not found — skipping")

        if not models:
            raise FileNotFoundError(
                f"No LightGBM models found in {model_dir}"
            )

        return cls(
            model_A       = models.get("model_A"),
            model_B       = models.get("model_B"),
            model_4class  = models.get("model_4class"),
            output_dir    = output_dir,
            feature_names = feature_names,
        )

    # ── Input preparation ──────────────────────────────────────────────────
    def _prepare_input(
        self,
        company_data: dict | pd.DataFrame | pd.Series,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Convert company data to a (1 × n_features) numpy array.
        Missing features are filled with 0 (neutral imputation for SHAP).
        Returns (array, dataframe_with_feature_names).
        """
        if isinstance(company_data, dict):
            series = pd.Series(company_data)
        elif isinstance(company_data, pd.Series):
            series = company_data
        elif isinstance(company_data, pd.DataFrame):
            series = company_data.iloc[0]
        else:
            raise TypeError(
                "company_data must be dict, pd.Series, or pd.DataFrame"
            )

        # Align to training feature order; fill missing with 0
        aligned = pd.Series(0.0, index=self.feature_names)
        for feat in self.feature_names:
            if feat in series.index:
                aligned[feat] = float(series[feat])

        missing = [f for f in self.feature_names if f not in series.index]
        if missing:
            print(f"  [warn] {len(missing)} features missing, filled with 0: "
                  f"{missing[:5]}{'...' if len(missing) > 5 else ''}")

        X  = aligned.values.reshape(1, -1)
        df = aligned.to_frame().T
        return X, df

    # ── Prediction ─────────────────────────────────────────────────────────
    def predict(
        self,
        company_data: dict | pd.DataFrame | pd.Series,
    ) -> dict:
        """
        Return predicted distress class and probabilities.

        If both model_A and model_B are available, uses dual-binary
        reconstruction (more reliable than direct 4-class).
        Falls back to 4-class model if binary models unavailable.
        """
        X, _ = self._prepare_input(company_data)

        result = {}

        # Dual-binary reconstruction
        if self.model_A and self.model_B:
            prob_A = float(self.model_A.predict_proba(X)[0, 1])
            prob_B = float(self.model_B.predict_proba(X)[0, 1])

            # Default thresholds — override if you tuned different values
            thr_A, thr_B = 0.5, 0.5

            pred_A = int(prob_A >= thr_A)   # 1 = ICR < 1
            pred_B = int(prob_B >= thr_B)   # 1 = Cash-ICR < 0

            # Taxonomy reconstruction
            if   pred_A == 1 and pred_B == 1: category = "Full_Distress"
            elif pred_A == 1 and pred_B == 0: category = "Acct_Distress"
            elif pred_A == 0 and pred_B == 1: category = "Cash_Distress"
            else:                              category = "Healthy"

            # Overall distress probability = max of the two signals
            distress_prob = max(prob_A, prob_B)

            result = {
                "method"          : "dual_binary",
                "predicted_class" : category,
                "display_label"   : LABEL_DISPLAY[category],
                "is_distressed"   : category != "Healthy",
                "distress_probability": round(distress_prob, 4),
                "model_A_prob"    : round(prob_A, 4),
                "model_B_prob"    : round(prob_B, 4),
                "earnings_distress_flag"  : bool(pred_A),
                "cashflow_distress_flag"  : bool(pred_B),
            }

        # Fallback: 4-class model
        elif self.model_4class:
            probs = self.model_4class.predict_proba(X)[0]
            pred  = int(self.model_4class.predict(X)[0])
            category = LABEL_MAP_INV[pred]
            result = {
                "method"           : "4class",
                "predicted_class"  : category,
                "display_label"    : LABEL_DISPLAY[category],
                "is_distressed"    : category != "Healthy",
                "distress_probability": round(1 - float(probs[0]), 4),
                "class_probabilities": {
                    LABEL_DISPLAY[LABEL_MAP_INV[i]]: round(float(p), 4)
                    for i, p in enumerate(probs)
                },
            }
        else:
            raise RuntimeError("No model available for prediction.")

        return result

    # ══════════════════════════════════════════════════════════════════════
    # SHAP VALUES
    # ══════════════════════════════════════════════════════════════════════
    def _get_shap_values_binary(
        self,
        explainer: shap.TreeExplainer,
        X: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Return (shap_values_for_positive_class, expected_value).
        LightGBM binary TreeExplainer returns shape (1, n_features).
        """
        sv = explainer.shap_values(X)
        # For binary: sv is (n_samples, n_features)
        if isinstance(sv, list):
            # Older SHAP returns [class0_sv, class1_sv]
            sv = sv[1]
        shap_row = sv[0]   # shape (n_features,)
        base     = float(explainer.expected_value
                         if np.isscalar(explainer.expected_value)
                         else explainer.expected_value[1])
        return shap_row, base

    def _get_shap_values_4class(
        self,
        X: np.ndarray,
        target_class: int = 3,   # Full_Distress by default
    ) -> tuple[np.ndarray, float]:
        """
        Return SHAP values for a specific class of the 4-class model.
        """
        sv = self._explainer_4.shap_values(X)
        if isinstance(sv, np.ndarray) and len(sv.shape) == 3:
            shap_row = sv[0, :, target_class]
            base = float(self._explainer_4.expected_value[target_class])
            return shap_row, base

    # Old SHAP format:
    # list of class arrays

        shap_row = sv[target_class][0]
        base = float(self._explainer_4.expected_value[target_class])

        return shap_row, base

    # ══════════════════════════════════════════════════════════════════════
    # TOP CONTRIBUTORS
    # ══════════════════════════════════════════════════════════════════════
    def get_top_contributors(
        self,
        shap_values : np.ndarray,
        feature_vals: np.ndarray,
        top_n       : int = 5,
    ) -> dict:
        """
        Return top_n features that increase and decrease distress risk.

        Parameters
        ----------
        shap_values  : 1-D array of SHAP values (one per feature)
        feature_vals : 1-D array of actual feature values
        top_n        : number of contributors to return per direction

        Returns
        -------
        dict with keys 'increasing_risk' and 'decreasing_risk',
        each a list of dicts suitable for JSON serialisation.
        """
        df = pd.DataFrame({
            "feature"      : self.feature_names,
            "display_name" : [FEATURE_LABELS.get(f, f)
                              for f in self.feature_names],
            "shap_value"   : shap_values,
            "feature_value": feature_vals,
        })
        df["abs_shap"] = df["shap_value"].abs()
        df = df.sort_values("abs_shap", ascending=False)

        def to_record(row):
            direction = "increases" if row["shap_value"] > 0 else "decreases"
            return {
                "feature"      : row["feature"],
                "display_name" : row["display_name"],
                "feature_value": round(float(row["feature_value"]), 4),
                "shap_value"   : round(float(row["shap_value"]), 4),
                "direction"    : direction,
                "impact"       : "high" if row["abs_shap"] > 0.1 else
                                 "medium" if row["abs_shap"] > 0.03 else "low",
            }

        increasing = (df[df["shap_value"] > 0]
                      .head(top_n)
                      .apply(to_record, axis=1)
                      .tolist())
        decreasing = (df[df["shap_value"] < 0]
                      .head(top_n)
                      .apply(to_record, axis=1)
                      .tolist())

        return {
            "increasing_risk": increasing,
            "decreasing_risk": decreasing,
        }

    # ══════════════════════════════════════════════════════════════════════
    # WATERFALL PLOT (local explanation)
    # ══════════════════════════════════════════════════════════════════════
    def generate_waterfall_plot(
        self,
        shap_values : np.ndarray,
        feature_vals: np.ndarray,
        base_value  : float,
        company_name: str = "Company",
        model_label : str = "",
        top_n       : int = 12,
        save        : bool = True,
    ) -> Path:
        """
        Horizontal waterfall chart showing how each feature drives the
        prediction away from the base (expected) value.
        """
        df = pd.DataFrame({
            "feature"  : [FEATURE_LABELS.get(f, f) for f in self.feature_names],
            "shap"     : shap_values,
            "value"    : feature_vals,
        }).assign(abs_shap=lambda d: d["shap"].abs())

        # Keep top_n by absolute SHAP, sort by shap ascending for plot
        df = (df.sort_values("abs_shap", ascending=False)
                .head(top_n)
                .sort_values("shap", ascending=True)
                .reset_index(drop=True))

        colors = [COLOR["increase"] if s > 0 else COLOR["decrease"]
                  for s in df["shap"]]

        fig, ax = plt.subplots(figsize=(9, max(5, len(df) * 0.5 + 1.5)))

        bars = ax.barh(df["feature"], df["shap"],
                       color=colors, edgecolor="white", linewidth=0.4,
                       height=0.65)

        for bar, row in zip(bars, df.itertuples()):
            val_str = f"{row.value:.3f}"
            shap_str = f"{row.shap:+.3f}"
            xpos = bar.get_width()
            ha = "left" if xpos >= 0 else "right"
            offset = 0.002 if xpos >= 0 else -0.002
            ax.text(xpos + offset, bar.get_y() + bar.get_height() / 2,
                    f"{shap_str}  [val={val_str}]",
                    va="center", ha=ha, fontsize=8, color="#2c3e50")

        ax.axvline(0, color="#2c3e50", linewidth=0.8)
        ax.set_xlabel("SHAP Value  (impact on distress probability)", fontsize=10)
        ax.set_title(
            f"Local Explanation — {company_name}\n"
            f"{model_label}  |  Base value: {base_value:.3f}",
            fontsize=11, fontweight="bold", pad=12
        )

        pos_patch = mpatches.Patch(color=COLOR["increase"], label="Increases risk")
        neg_patch = mpatches.Patch(color=COLOR["decrease"], label="Decreases risk")
        ax.legend(handles=[pos_patch, neg_patch], fontsize=9,
                  loc="lower right", framealpha=0.9)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        plt.tight_layout()

        safe_name = company_name.replace(" ", "_").replace("/", "-")
        fpath = self.output_dir / f"waterfall_{safe_name}_{model_label}.png"
        if save:
            fig.savefig(fpath, dpi=160, bbox_inches="tight")
            plt.close(fig)
            return fpath
        else:
            plt.show()
            plt.close(fig)
            return fpath

    # ══════════════════════════════════════════════════════════════════════
    # SUMMARY PLOT (global explanation)
    # ══════════════════════════════════════════════════════════════════════
    def generate_summary_plot(
        self,
        X          : np.ndarray,
        model_label: str  = "model_A",
        max_display: int  = 20,
        save       : bool = True,
    ) -> Path:
        """
        Beeswarm SHAP summary plot across a population of companies.

        Parameters
        ----------
        X           : (n_companies, n_features) array — the dataset to explain
        model_label : 'model_A', 'model_B', or '4class'
        max_display : maximum features shown in plot
        """
        if model_label == "model_A" and self._explainer_A:
            sv = self._explainer_A.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
        elif model_label == "model_B" and self._explainer_B:
            sv = self._explainer_B.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
        elif model_label == "4class" and self._explainer_4:
            sv_all = self._explainer_4.shap_values(X)
            # Mean absolute SHAP across all classes
            sv = np.mean([np.abs(s) for s in sv_all], axis=0)
        else:
            raise ValueError(f"No explainer found for model_label='{model_label}'")

        display_names = [FEATURE_LABELS.get(f, f) for f in self.feature_names]

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            sv, X,
            feature_names=display_names,
            max_display=max_display,
            show=False,
            plot_type="dot",
        )
        plt.title(
            f"Global Feature Importance — {model_label.replace('_', ' ').title()}\n"
            f"(SHAP beeswarm — {len(X):,} observations)",
            fontsize=12, fontweight="bold", pad=12
        )
        plt.tight_layout()

        fpath = self.output_dir / f"summary_{model_label}.png"
        if save:
            plt.savefig(fpath, dpi=160, bbox_inches="tight")
            plt.close()
            return fpath
        else:
            plt.show()
            plt.close()
            return fpath

    # ══════════════════════════════════════════════════════════════════════
    # GLOBAL FEATURE IMPORTANCE TABLE
    # ══════════════════════════════════════════════════════════════════════
    def get_global_importance(
        self,
        X          : np.ndarray,
        model_label: str = "model_A",
    ) -> list[dict]:
        """
        Mean absolute SHAP value per feature across a population.
        Returns list of dicts sorted by importance descending.
        """
        if model_label == "model_A" and self._explainer_A:
            sv = self._explainer_A.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
        elif model_label == "model_B" and self._explainer_B:
            sv = self._explainer_B.shap_values(X)
            if isinstance(sv, list): sv = sv[1]
        elif model_label == "4class" and self._explainer_4:
            sv_all = self._explainer_4.shap_values(X)
            sv = np.mean([np.abs(s) for s in sv_all], axis=0)
        else:
            raise ValueError(f"No explainer for '{model_label}'")

        mean_abs = np.abs(sv).mean(axis=0)
        records = [
            {
                "rank"        : i + 1,
                "feature"     : self.feature_names[j],
                "display_name": FEATURE_LABELS.get(self.feature_names[j],
                                                    self.feature_names[j]),
                "mean_abs_shap": round(float(mean_abs[j]), 5),
            }
            for i, j in enumerate(np.argsort(mean_abs)[::-1])
        ]
        return records

    # ══════════════════════════════════════════════════════════════════════
    # MAIN PUBLIC API — explain a single company
    # ══════════════════════════════════════════════════════════════════════
    def explain_company(
        self,
        company_data: dict | pd.DataFrame | pd.Series,
        company_name: str = "Company",
        top_n       : int = 5,
        save_plots  : bool = True,
    ) -> dict:
        """
        Full explanation for a single company.

        Returns a JSON-serialisable dict containing:
          - prediction (class + probabilities)
          - top contributors (increasing / decreasing risk)
          - file paths of generated plots
          - SHAP values (for frontend custom rendering)

        Parameters
        ----------
        company_data : dict of {feature_name: value} or DataFrame row
        company_name : display name for plots
        top_n        : number of top contributors per direction
        save_plots   : whether to save waterfall plots to disk
        """
        X, _ = self._prepare_input(company_data)
        feature_vals = X[0]

        prediction = self.predict(company_data)
        output = {
            "company_name"   : company_name,
            "prediction"     : prediction,
            "contributors"   : {},
            "shap_values"    : {},
            "plots"          : {},
        }

        # ── Model A explanation ────────────────────────────────────────────
        if self._explainer_A:
            sv_A, base_A = self._get_shap_values_binary(
                self._explainer_A, X)
            output["contributors"]["earnings_distress"] = (
                self.get_top_contributors(sv_A, feature_vals, top_n))
            output["shap_values"]["model_A"] = {
                "base_value" : round(base_A, 4),
                "values"     : {
                    self.feature_names[i]: round(float(sv_A[i]), 5)
                    for i in range(len(self.feature_names))
                },
            }
            if save_plots:
                path_A = self.generate_waterfall_plot(
                    sv_A, feature_vals, base_A,
                    company_name=company_name,
                    model_label="Model_A_Earnings",
                    save=True,
                )
                output["plots"]["waterfall_model_A"] = str(path_A)

        # ── Model B explanation ────────────────────────────────────────────
        if self._explainer_B:
            sv_B, base_B = self._get_shap_values_binary(
                self._explainer_B, X)
            output["contributors"]["cashflow_distress"] = (
                self.get_top_contributors(sv_B, feature_vals, top_n))
            output["shap_values"]["model_B"] = {
                "base_value" : round(base_B, 4),
                "values"     : {
                    self.feature_names[i]: round(float(sv_B[i]), 5)
                    for i in range(len(self.feature_names))
                },
            }
            if save_plots:
                path_B = self.generate_waterfall_plot(
                    sv_B, feature_vals, base_B,
                    company_name=company_name,
                    model_label="Model_B_CashFlow",
                    save=True,
                )
                output["plots"]["waterfall_model_B"] = str(path_B)

        # ── 4-class explanation (Full Distress class) ──────────────────────
        if self._explainer_4:
            sv_4, base_4 = self._get_shap_values_4class(X, target_class=3)
            output["contributors"]["full_distress_4class"] = (
                self.get_top_contributors(sv_4, feature_vals, top_n))
            output["shap_values"]["model_4class"] = {
                "base_value" : round(base_4, 4),
                "target_class": "Full_Distress",
                "values"     : {
                    self.feature_names[i]: round(float(sv_4[i]), 5)
                    for i in range(len(self.feature_names))
                },
            }

        return output


# ══════════════════════════════════════════════════════════════════════════════
# BATCH RESEARCH FUNCTIONS (used in Step 7 research outputs)
# ══════════════════════════════════════════════════════════════════════════════
def run_research_shap(
    explainer   : DistressExplainer,
    X           : np.ndarray,
    output_dir  : Path,
    model_label : str = "model_A",
) -> tuple[list[dict], Path]:
    """
    Generate global summary plot + importance table for the research paper.
    Call this on the full test set.
    """
    print(f"[shap] Computing SHAP values for {model_label} "
          f"on {len(X):,} observations ...")
    importance = explainer.get_global_importance(X, model_label)
    summary_path = explainer.generate_summary_plot(X, model_label, save=True)

    # Save importance table
    imp_df = pd.DataFrame(importance)
    csv_path = output_dir / f"shap_importance_{model_label}.csv"
    imp_df.to_csv(csv_path, index=False)
    print(f"[shap] Importance table → {csv_path}")
    print(f"[shap] Summary plot     → {summary_path}")

    return importance, summary_path