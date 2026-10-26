"""
compute_training_metrics.py
----------------------------
Computes BOTH training and testing metrics for all models.
Run this to get the complete train/test comparison for the paper.

Outputs:
  - Prints a clean train vs test comparison table
  - Saves to ml_results/train_test_comparison.json

Run from ~/Documents/Data/:
  python3 compute_training_metrics.py
"""

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, roc_auc_score, average_precision_score,
    f1_score, r2_score, mean_absolute_error
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("[WARN] XGBoost not installed — Model B XGB will be skipped")

DATA_ROOT   = Path(__file__).parent
FEAT_TABLE  = DATA_ROOT / "phase1a_feature_table.csv"
AF_TABLE    = DATA_ROOT / "population_af_table.csv"
RESULTS_DIR = DATA_ROOT / "ml_results"
RANDOM_STATE = 42

# ── Encoding helpers ──────────────────────────────────────────────────────────

VTYPE_MAP  = {"SNV": 0, "INDEL": 1, "MULTIALLELIC": 2, "SPANNING_DEL": 3}
SYSNDD_MAP = {"Definitive": 4, "Moderate": 3, "Limited": 2,
              "Refuted": 0, "not applicable": 0, "": 0}
SFARI_MAP  = {"1": 4, "2": 3, "3": 2, "S": 1, "": 0}


def gt_to_dosage(s):
    def p(g):
        if pd.isna(g) or str(g).strip() in ("", ".", "./.", "None"):
            return np.nan
        g = str(g).replace("|", "/")
        parts = g.split("/")
        try:
            return float(sum(int(x) > 0 for x in parts if x != "."))
        except Exception:
            return np.nan
    return s.apply(p)


def parse_bool(s):
    m = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    return s.map(m)


def encode(df):
    for r in ("father", "mother", "child"):
        if f"{r}_gt" in df.columns:
            df[f"{r}_dosage"] = gt_to_dosage(df[f"{r}_gt"])
    df["variant_type_enc"] = df["variant_type"].map(VTYPE_MAP).fillna(0).astype(int)
    df["sysndd_score"]     = df["sysndd_category"].fillna("").astype(str).str.strip().map(
                                 lambda x: SYSNDD_MAP.get(x, 0))
    df["sfari_score_enc"]  = df["sfari_score"].fillna("").astype(str).str.strip().map(
                                 lambda x: SFARI_MAP.get(x, 0))
    df["ndd_gene_int"]     = df["is_ndd_gene"].map(
                                 {True:1,False:0,"True":1,"False":0}).fillna(0).astype(int)
    df["n_sources"]        = pd.to_numeric(
                                 df.get("n_sources",0), errors="coerce").fillna(0).astype(int)
    for c in ["father_gq","mother_gq","child_gq","father_dp","mother_dp","child_dp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def sep(title=""):
    line = "─" * 65
    print(f"\n{line}")
    if title:
        print(f"  {title}")
        print(line)


def report(label, y_train, y_train_pred, y_train_prob,
                  y_test,  y_test_pred,  y_test_prob,
                  task="classification"):
    """Print train vs test metrics side by side."""
    if task == "classification":
        tr_acc = accuracy_score(y_train, y_train_pred)
        te_acc = accuracy_score(y_test,  y_test_pred)
        tr_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else None
        te_auc = roc_auc_score(y_test,  y_test_prob)  if y_test_prob  is not None else None

        print(f"\n  {label}")
        print(f"    Accuracy  — Train: {tr_acc:.4f}  |  Test: {te_acc:.4f}  "
              f"|  Gap: {tr_acc-te_acc:+.4f}")
        if tr_auc and te_auc:
            print(f"    AUC-ROC   — Train: {tr_auc:.4f}  |  Test: {te_auc:.4f}  "
                  f"|  Gap: {tr_auc-te_auc:+.4f}")
        tr_f1 = f1_score(y_train, y_train_pred, zero_division=0)
        te_f1 = f1_score(y_test,  y_test_pred,  zero_division=0)
        print(f"    F1 score  — Train: {tr_f1:.4f}  |  Test: {te_f1:.4f}  "
              f"|  Gap: {tr_f1-te_f1:+.4f}")

        overfit = tr_acc - te_acc
        label_ov = ("⚠ Overfit" if overfit > 0.05 else
                    "✓ Well-generalised" if overfit < 0.01 else
                    "~ Acceptable gap")
        print(f"    Overfit assessment: {label_ov} (train-test accuracy gap = {overfit:.4f})")

        return {"train_accuracy": round(tr_acc,6), "test_accuracy": round(te_acc,6),
                "train_auc": round(tr_auc,6) if tr_auc else None,
                "test_auc":  round(te_auc,6) if te_auc else None,
                "train_f1": round(tr_f1,6), "test_f1": round(te_f1,6)}
    else:
        tr_r2  = r2_score(y_train, y_train_pred)
        te_r2  = r2_score(y_test,  y_test_pred)
        tr_mae = mean_absolute_error(y_train, y_train_pred)
        te_mae = mean_absolute_error(y_test,  y_test_pred)
        print(f"\n  {label}")
        print(f"    R²        — Train: {tr_r2:.4f}  |  Test: {te_r2:.4f}  "
              f"|  Gap: {tr_r2-te_r2:+.4f}")
        print(f"    MAE       — Train: {tr_mae:.4f}  |  Test: {te_mae:.4f}  "
              f"|  Gap: {te_mae-tr_mae:+.4f}")
        overfit = tr_r2 - te_r2
        label_ov = ("⚠ Overfit" if overfit > 0.02 else
                    "✓ Well-generalised" if overfit < 0.005 else
                    "~ Acceptable gap")
        print(f"    Overfit assessment: {label_ov} (train-test R² gap = {overfit:.4f})")
        return {"train_r2": round(tr_r2,6), "test_r2": round(te_r2,6),
                "train_mae": round(tr_mae,6), "test_mae": round(te_mae,6)}


def main():
    all_results = {}
    sep("TRAIN vs TEST METRICS — All Models")

    # ══════════════════════════════════════════════════════
    # PART 1: Model A and Model B (Zenodo trio feature table)
    # ══════════════════════════════════════════════════════
    if not FEAT_TABLE.exists():
        print(f"\n[SKIP] Feature table not found: {FEAT_TABLE}")
        print("  Run feature_engineering.py first.")
    else:
        print(f"\n[Loading] {FEAT_TABLE.name} ...")
        df = pd.read_csv(FEAT_TABLE, low_memory=False,
                         dtype={"variant_type":str,"sysndd_category":str,
                                "sfari_score":str,"gene_symbol":str})
        df = encode(df)
        df["is_mc"]  = parse_bool(df["is_mendelian_consistent"])
        df["is_dn"]  = parse_bool(df["is_de_novo"])
        print(f"  {len(df):,} rows loaded")

        # Stratified 500k sample on is_de_novo
        df_valid = df.dropna(subset=["is_dn"]).copy()
        n_sample = min(500_000, len(df_valid))
        df_pos = df_valid[df_valid["is_dn"] == 1]
        df_neg = df_valid[df_valid["is_dn"] == 0]
        n_pos  = max(1, round(n_sample * len(df_pos) / len(df_valid)))
        n_neg  = n_sample - n_pos
        df_s = pd.concat([
            df_pos.sample(min(n_pos, len(df_pos)), random_state=RANDOM_STATE),
            df_neg.sample(min(n_neg, len(df_neg)), random_state=RANDOM_STATE),
        ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        sep("MODEL A — Mendelian Consistency (Random Forest)")
        FEAT_A = ["father_dosage","mother_dosage","child_dosage",
                  "variant_type_enc","ndd_gene_int"]
        df_a = df_s.dropna(subset=FEAT_A + ["is_mc"])
        df_a["is_mc"] = df_a["is_mc"].astype(int)
        X_a = df_a[FEAT_A].values;  y_a = df_a["is_mc"].values
        Xtr_a, Xte_a, ytr_a, yte_a = train_test_split(
            X_a, y_a, test_size=0.2, random_state=RANDOM_STATE, stratify=y_a)

        rf_a = RandomForestClassifier(n_estimators=100, max_depth=8,
                                      class_weight="balanced",
                                      random_state=RANDOM_STATE, n_jobs=-1)
        rf_a.fit(Xtr_a, ytr_a)
        all_results["model_a_rf"] = report(
            "Random Forest",
            ytr_a, rf_a.predict(Xtr_a), rf_a.predict_proba(Xtr_a)[:,1],
            yte_a, rf_a.predict(Xte_a), rf_a.predict_proba(Xte_a)[:,1],
        )

        sep("MODEL B — De Novo Classifier")
        FEAT_B_BIO = ["variant_type_enc","ndd_gene_int","sysndd_score",
                      "sfari_score_enc","n_sources"]
        FEAT_B_FULL = FEAT_B_BIO + ["child_gq","child_dp","father_gq",
                                     "mother_gq","father_dp","mother_dp"]

        df_b = df_s.dropna(subset=["is_dn"]).copy()
        df_b["is_dn"] = df_b["is_dn"].astype(int)
        n_pos_b = df_b["is_dn"].sum(); n_neg_b = len(df_b) - n_pos_b
        spw = n_neg_b / max(n_pos_b, 1)

        avail_full = [f for f in FEAT_B_FULL if f in df_b.columns]
        avail_bio  = [f for f in FEAT_B_BIO  if f in df_b.columns]
        df_b = df_b.dropna(subset=avail_full + ["is_dn"])
        X_b_full = df_b[avail_full].fillna(0).values
        X_b_bio  = df_b[avail_bio].fillna(0).values
        y_b      = df_b["is_dn"].astype(int).values

        Xtr_full, Xte_full, ytr_b, yte_b = train_test_split(
            X_b_full, y_b, test_size=0.2, random_state=RANDOM_STATE, stratify=y_b)
        Xtr_bio, Xte_bio, _, _ = train_test_split(
            X_b_bio, y_b, test_size=0.2, random_state=RANDOM_STATE, stratify=y_b)

        if HAS_XGB:
            print("\n  [B.1] XGBoost — FULL features (all 11 incl. GQ/DP)")
            xgb_full = XGBClassifier(n_estimators=300, max_depth=6,
                                     learning_rate=0.1, scale_pos_weight=spw,
                                     eval_metric="aucpr", random_state=RANDOM_STATE,
                                     n_jobs=-1, verbosity=0)
            xgb_full.fit(Xtr_full, ytr_b)
            all_results["model_b_full_xgb"] = report(
                "XGBoost FULL",
                ytr_b, xgb_full.predict(Xtr_full), xgb_full.predict_proba(Xtr_full)[:,1],
                yte_b, xgb_full.predict(Xte_full),  xgb_full.predict_proba(Xte_full)[:,1],
            )

            print("\n  [B.2] XGBoost — BIO-ONLY features (5, no quality metrics)")
            xgb_bio = XGBClassifier(n_estimators=300, max_depth=6,
                                    learning_rate=0.1, scale_pos_weight=spw,
                                    eval_metric="aucpr", random_state=RANDOM_STATE,
                                    n_jobs=-1, verbosity=0)
            xgb_bio.fit(Xtr_bio, ytr_b)
            all_results["model_b_bio_xgb"] = report(
                "XGBoost BIO-ONLY",
                ytr_b, xgb_bio.predict(Xtr_bio), xgb_bio.predict_proba(Xtr_bio)[:,1],
                yte_b, xgb_bio.predict(Xte_bio),  xgb_bio.predict_proba(Xte_bio)[:,1],
            )

    # ══════════════════════════════════════════════════════
    # PART 2: AF Portability Regression (chr1 train → chr2 test)
    # ══════════════════════════════════════════════════════
    sep("AF PORTABILITY REGRESSION (chr1 train → chr2 test)")

    if not AF_TABLE.exists():
        print(f"\n[SKIP] AF table not found: {AF_TABLE}")
        print("  Run population_af_analysis.py first.")
    else:
        print(f"\n[Loading] {AF_TABLE.name} ...")
        af = pd.read_csv(AF_TABLE, low_memory=False,
                         dtype={"chrom":str,"variant_type":str,"gene_symbol":str})
        print(f"  {len(af):,} rows loaded")

        EUR_FEAT  = ["eur_af"]
        FULL_FEAT = ["eur_af","afr_af","eas_af","amr_af","variant_type_enc",
                     "is_ndd_gene","sysndd_score","sfari_score_enc",
                     "n_sources","denovo_db_count"]

        for col in ["sysndd_score","sfari_score_enc","variant_type_enc",
                    "denovo_db_count","is_ndd_gene","n_sources"]:
            if col not in af.columns:
                af[col] = 0
            else:
                af[col] = pd.to_numeric(af[col], errors="coerce").fillna(0)

        chr1 = af[af["chrom_label"] == "1"].dropna(subset=EUR_FEAT + ["sas_af"])
        chr2 = af[af["chrom_label"] == "2"].dropna(subset=EUR_FEAT + ["sas_af"])

        if len(chr1) == 0 or len(chr2) == 0:
            print("  [WARN] Could not split by chrom_label — checking 'chrom' column")
            chr1 = af[af["chrom"].isin(["1","chr1"])].dropna(subset=EUR_FEAT+["sas_af"])
            chr2 = af[af["chrom"].isin(["2","chr2"])].dropna(subset=EUR_FEAT+["sas_af"])

        print(f"  chr1 (train): {len(chr1):,} variants")
        print(f"  chr2 (test):  {len(chr2):,} variants")

        avail_full = [f for f in FULL_FEAT if f in chr1.columns]

        # EUR-only Ridge
        print("\n  [AF.1] Ridge — EUR_AF only")
        ridge = Ridge(alpha=1.0)
        Xtr = chr1[EUR_FEAT].values; ytr = chr1["sas_af"].values
        Xte = chr2[EUR_FEAT].values; yte = chr2["sas_af"].values
        ridge.fit(Xtr, ytr)
        all_results["af_eur_ridge"] = report(
            "Ridge EUR-only", ytr, ridge.predict(Xtr), None,
            yte, ridge.predict(Xte), None, task="regression")

        # EUR-only Random Forest
        print("\n  [AF.2] Random Forest — EUR_AF only")
        rf_af = RandomForestRegressor(n_estimators=200, max_depth=8,
                                      random_state=RANDOM_STATE, n_jobs=-1,
                                      min_samples_leaf=50)
        rf_af.fit(Xtr, ytr)
        all_results["af_eur_rf"] = report(
            "RF EUR-only", ytr, rf_af.predict(Xtr), None,
            yte, rf_af.predict(Xte), None, task="regression")

        # Full Ridge
        if avail_full:
            print(f"\n  [AF.3] Ridge — Full features ({len(avail_full)} features)")
            chr1_f = chr1.dropna(subset=avail_full+["sas_af"])
            chr2_f = chr2.dropna(subset=avail_full+["sas_af"])
            Xtr_f = chr1_f[avail_full].fillna(0).values; ytr_f = chr1_f["sas_af"].values
            Xte_f = chr2_f[avail_full].fillna(0).values; yte_f = chr2_f["sas_af"].values
            ridge_f = Ridge(alpha=1.0)
            ridge_f.fit(Xtr_f, ytr_f)
            all_results["af_full_ridge"] = report(
                "Ridge FULL features", ytr_f, ridge_f.predict(Xtr_f), None,
                yte_f, ridge_f.predict(Xte_f), None, task="regression")

    # ══════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ══════════════════════════════════════════════════════
    sep("COMPLETE TRAIN vs TEST SUMMARY TABLE")

    rows = [
        ("Model A — RF", "Accuracy", "train_accuracy", "test_accuracy", "model_a_rf"),
        ("Model A — RF", "AUC-ROC",  "train_auc",      "test_auc",      "model_a_rf"),
        ("Model B FULL — XGB", "AUC-ROC", "train_auc", "test_auc",      "model_b_full_xgb"),
        ("Model B BIO — XGB",  "AUC-ROC", "train_auc", "test_auc",      "model_b_bio_xgb"),
        ("AF EUR Ridge", "R²", "train_r2", "test_r2", "af_eur_ridge"),
        ("AF EUR RF",    "R²", "train_r2", "test_r2", "af_eur_rf"),
        ("AF FULL Ridge","R²", "train_r2", "test_r2", "af_full_ridge"),
    ]

    print(f"\n  {'Model':<25} {'Metric':<10} {'TRAIN':>8} {'TEST':>8} {'Gap':>8}  Verdict")
    print(f"  {'─'*25} {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*20}")

    for model_name, metric, train_key, test_key, res_key in rows:
        r = all_results.get(res_key, {})
        tr = r.get(train_key)
        te = r.get(test_key)
        if tr is None or te is None:
            print(f"  {model_name:<25} {metric:<10} {'N/A':>8} {'N/A':>8}")
            continue
        gap = tr - te
        verdict = ("⚠ Overfit" if abs(gap) > 0.02 else
                   "✓ Good generalisation" if abs(gap) < 0.005 else
                   "~ Acceptable")
        print(f"  {model_name:<25} {metric:<10} {tr:>8.4f} {te:>8.4f} {gap:>+8.4f}  {verdict}")

    # Save
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "train_test_comparison.json"
    with open(out, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  Saved → {out}")
    sep()


if __name__ == "__main__":
    main()
