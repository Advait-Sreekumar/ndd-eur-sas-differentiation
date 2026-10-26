"""
ml_baseline.py
---------------
Phase 1a ML baseline models trained on phase1a_feature_table.csv.

MODEL A — Mendelian Consistency Classifier
    Task    : Predict whether child GT is a valid Mendelian combination
              of parental genotypes.
    Purpose : Pipeline validation — Mendelian inheritance is deterministic,
              so >99% accuracy is expected and correct. The interesting
              signal is in the residuals (violations = de novo candidates
              or genotyping artifacts).
    Features: father_dosage, mother_dosage, child_dosage, variant_type,
              is_ndd_gene
    Metric  : Accuracy, F1

MODEL B — De Novo Candidate Classifier
    Task    : Predict is_de_novo from sequence-context + NDD-annotation
              features ONLY (no parental genotypes — those would be
              definitional leakage).
    Purpose : Demonstrate that sequence context and gene biological
              context carry signal for de novo enrichment. The cross-
              population transfer of this learned signal is the core
              scientific question of the paper.
    Features: variant_type, is_ndd_gene, sysndd_score, sfari_score,
              n_sources, per-sample GQ and DP
    Metric  : AUC-ROC, AUC-PR (preferred for imbalanced), F1
    Note    : Class imbalance ~0.3% positive — handled via
              class_weight='balanced' (sklearn) and scale_pos_weight (XGB)

Usage:
    python3 ml_baseline.py                   # 500k stratified sample (fast)
    python3 ml_baseline.py --full            # all 5.8M rows (slow RF ~15min)
    python3 ml_baseline.py --sample 200000   # custom sample size
    python3 ml_baseline.py --no-xgb         # skip XGBoost (if not installed)
    python3 ml_baseline.py --model a        # run only Model A
    python3 ml_baseline.py --model b        # run only Model B

Requires: pandas, numpy, scikit-learn, xgboost (optional)
"""

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT      = Path(__file__).parent
FEATURE_TABLE  = DATA_ROOT / "phase1a_feature_table.csv"
RESULTS_DIR    = DATA_ROOT / "ml_results"

DEFAULT_SAMPLE = 500_000
TEST_SIZE      = 0.2
RANDOM_STATE   = 42

# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

VTYPE_MAP = {"SNV": 0, "INDEL": 1, "MULTIALLELIC": 2, "SPANNING_DEL": 3}

SYSNDD_MAP = {
    "Definitive":      4,
    "Moderate":        3,
    "Limited":         2,
    "Refuted":         0,
    "not applicable":  0,
    "":                0,
}

SFARI_MAP = {"1": 4, "2": 3, "3": 2, "S": 1, "": 0}


def gt_to_dosage(gt_series: pd.Series) -> pd.Series:
    """
    Convert a genotype string column ('0/1', '1/1', './.', …) to
    allele dosage (number of alt alleles: 0, 1, or 2).
    Returns NaN for missing / uninterpretable genotypes.
    """
    def _parse(gt):
        if pd.isna(gt) or str(gt).strip() in ("", ".", "./.", "None"):
            return np.nan
        gt = str(gt).replace("|", "/")
        parts = gt.split("/")
        try:
            return float(sum(int(p) > 0 for p in parts if p != "."))
        except (ValueError, TypeError):
            return np.nan
    return gt_series.apply(_parse)


def _parse_bool_col(series: pd.Series) -> pd.Series:
    """
    Robustly parse a True/False/NA column that may come back from CSV
    as strings 'True'/'False', booleans, ints, or empty strings.
    Returns 1/0/NaN.
    """
    mapping = {
        True: 1,  False: 0,
        "True": 1, "False": 0,
        1: 1, 0: 0,
        "1": 1, "0": 0,
    }
    return series.map(mapping)


def encode_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all encoded feature columns to df in-place and return it.
    """
    # ── Genotype dosage ───────────────────────────────────────────────────────
    for role in ("father", "mother", "child"):
        col = f"{role}_gt"
        if col in df.columns:
            df[f"{role}_dosage"] = gt_to_dosage(df[col])

    # ── Variant type ─────────────────────────────────────────────────────────
    df["variant_type_enc"] = (
        df["variant_type"].map(VTYPE_MAP).fillna(0).astype(int)
    )

    # ── SysNDD confidence score ───────────────────────────────────────────────
    df["sysndd_score"] = (
        df["sysndd_category"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(lambda x: SYSNDD_MAP.get(x, 0))
    )

    # ── SFARI score (numeric confidence) ─────────────────────────────────────
    df["sfari_score_enc"] = (
        df["sfari_score"]
        .fillna("")
        .astype(str)
        .str.strip()
        .map(lambda x: SFARI_MAP.get(x, 0))
    )

    # ── NDD gene flag ─────────────────────────────────────────────────────────
    df["ndd_gene_int"] = (
        df["is_ndd_gene"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .fillna(0)
        .astype(int)
    )

    # ── n_sources ─────────────────────────────────────────────────────────────
    df["n_sources"] = (
        pd.to_numeric(df.get("n_sources", 0), errors="coerce").fillna(0).astype(int)
    )

    # ── Quality metrics to float (handle missing gracefully) ─────────────────
    for col in ["father_gq", "mother_gq", "child_gq",
                "father_dp", "mother_dp", "child_dp"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# Metric printer
# ---------------------------------------------------------------------------

def print_metrics(y_true, y_pred, y_prob, label: str, sep_char: str = "─") -> dict:
    """
    Print and return a standard evaluation metric dict.
    y_prob: predicted probability of the positive class.
    """
    sep = sep_char * 55
    print(f"\n  {sep}")
    print(f"  {label}")
    print(f"  {sep}")

    acc   = accuracy_score(y_true, y_pred)
    f1    = f1_score(y_true, y_pred, zero_division=0)
    cm    = confusion_matrix(y_true, y_pred)

    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1 score : {f1:.4f}")

    metrics = {"accuracy": round(acc, 6), "f1": round(f1, 6)}

    # AUC-ROC and AUC-PR (only meaningful when both classes present)
    if len(np.unique(y_true)) > 1 and y_prob is not None:
        auc_roc = roc_auc_score(y_true, y_prob)
        auc_pr  = average_precision_score(y_true, y_prob)
        print(f"  AUC-ROC  : {auc_roc:.4f}")
        print(f"  AUC-PR   : {auc_pr:.4f}  "
              f"(baseline={y_true.mean():.4f})")
        metrics.update({"auc_roc": round(auc_roc, 6),
                        "auc_pr": round(auc_pr, 6)})

    # Confusion matrix
    print(f"\n  Confusion matrix:")
    print(f"                Pred 0    Pred 1")
    print(f"  Actual 0   {cm[0,0]:>8,}  {cm[0,1]:>8,}")
    print(f"  Actual 1   {cm[1,0]:>8,}  {cm[1,1]:>8,}")

    # Positive-class breakdown
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    tp    = cm[1, 1] if cm.shape == (2, 2) else 0
    fp    = cm[0, 1] if cm.shape == (2, 2) else 0
    fn    = cm[1, 0] if cm.shape == (2, 2) else 0
    prec  = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec   = tp / (tp + fn) if (tp + fn) > 0 else 0
    print(f"\n  Class balance: {n_pos:,} positive / {n_neg:,} negative "
          f"({100*n_pos/(n_pos+n_neg):.3f}% positive)")
    print(f"  Precision: {prec:.4f}   Recall: {rec:.4f}")

    metrics.update({"precision": round(prec, 6), "recall": round(rec, 6),
                    "n_positive": int(n_pos), "n_negative": int(n_neg)})
    return metrics


def print_feature_importance(model, feature_names: list, top_n: int = 10):
    """Print top-N feature importances for tree-based models."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp.sort_values(ascending=False).head(top_n)
    print(f"\n  Top {top_n} feature importances:")
    for fname, score in imp.items():
        bar = "█" * int(score * 40)
        print(f"    {fname:<30} {score:.4f}  {bar}")
    return imp


# ---------------------------------------------------------------------------
# MODEL A — Mendelian consistency
# ---------------------------------------------------------------------------

MODEL_A_FEATURES = [
    "father_dosage",
    "mother_dosage",
    "child_dosage",
    "variant_type_enc",
    "ndd_gene_int",
]


def run_model_a(df: pd.DataFrame) -> dict:
    """
    Train and evaluate Mendelian consistency classifiers.

    Feature set deliberately includes parental and child dosage because
    this IS a genotype-to-consistency mapping — the model is learning
    the Mendelian rule, which is the whole point of Model A.
    """
    sep = "═" * 65

    print(f"\n{sep}")
    print("  MODEL A — Mendelian Consistency Classifier")
    print(f"{sep}")
    print("  Target    : is_mendelian_consistent (1 = consistent, 0 = violation)")
    print("  Features  : parental + child dosage, variant_type, ndd_gene")
    print("  Rationale : Mendelian inheritance is deterministic. Model A validates")
    print("              the pipeline captures real biology. Violations = de novo")
    print("              candidates or genotyping artifacts.")

    # Prepare data
    df_a = df.copy()
    df_a["target"] = _parse_bool_col(df_a["is_mendelian_consistent"])
    df_a = df_a.dropna(subset=MODEL_A_FEATURES + ["target"])
    df_a["target"] = df_a["target"].astype(int)

    print(f"\n  Samples after NA-drop : {len(df_a):>10,}")
    print(f"  Consistent (1)        : {df_a['target'].sum():>10,}  "
          f"({100*df_a['target'].mean():.2f}%)")
    print(f"  Violations (0)        : {(df_a['target']==0).sum():>10,}  "
          f"({100*(1-df_a['target'].mean()):.2f}%)")

    X = df_a[MODEL_A_FEATURES].values
    y = df_a["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results_a = {"model_a": {}}

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n  [A.1] Logistic Regression")
    t0 = time.time()
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=500, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred  = lr_pipe.predict(X_test)
    y_prob  = lr_pipe.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    print(f"  Train time : {elapsed:.1f}s")
    m = print_metrics(y_test, y_pred, y_prob, "Logistic Regression — Model A")
    results_a["model_a"]["logistic_regression"] = m

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n  [A.2] Random Forest (100 trees)")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=100, max_depth=8,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    print(f"  Train time : {elapsed:.1f}s")
    m = print_metrics(y_test, y_pred, y_prob, "Random Forest — Model A")
    results_a["model_a"]["random_forest"] = m
    imp_a = print_feature_importance(rf, MODEL_A_FEATURES)

    print(f"\n  KEY INSIGHT: Violations (Mendelian errors) in the test set:")
    violations = np.where(y_test == 0)[0]
    print(f"    Total violations in test set : {len(violations):,}")
    print(f"    RF caught                    : {confusion_matrix(y_test, rf.predict(X_test))[0,0]:,}")
    print(f"    These rows → de novo candidates or genotyping artifacts")

    return results_a


# ---------------------------------------------------------------------------
# MODEL B — De novo candidate classifier
# ---------------------------------------------------------------------------

# CRITICAL: NO parental/child genotype features — those are definitional
# leakage (is_de_novo is defined as both parents 0/0 + child has alt).
# We want to learn from sequence context and gene annotation only.
MODEL_B_FEATURES = [
    "variant_type_enc",   # SNV=0, INDEL=1, MULTI=2, SPAN_DEL=3
    "ndd_gene_int",       # 1 if variant falls in an NDD gene
    "sysndd_score",       # 0–4 ordinal confidence from SysNDD
    "sfari_score_enc",    # 0–4 ordinal confidence from SFARI
    "n_sources",          # 0–4 number of databases corroborating the gene
    "child_gq",           # genotype quality of child call
    "child_dp",           # read depth at child
    "father_gq",          # parental call quality (not genotype)
    "mother_gq",
    "father_dp",
    "mother_dp",
]

MODEL_B_FEATURE_DESCRIPTIONS = {
    "variant_type_enc":  "Variant type (SNV/INDEL/MULTI/SPAN) — different de novo rates",
    "ndd_gene_int":      "Is this variant in a known NDD gene?",
    "sysndd_score":      "SysNDD gene confidence (0=none, 4=Definitive)",
    "sfari_score_enc":   "SFARI gene score (0=none, 4=score 1/highest)",
    "n_sources":         "Number of independent databases flagging this gene",
    "child_gq":          "Genotype quality of child call (data quality, not biology)",
    "child_dp":          "Read depth at child (data quality, not biology)",
    "father_gq":         "Parental call quality (data quality, not biology)",
    "mother_gq":         "Parental call quality (data quality, not biology)",
    "father_dp":         "Parental read depth (data quality, not biology)",
    "mother_dp":         "Parental read depth (data quality, not biology)",
}


def run_model_b(df: pd.DataFrame, use_xgb: bool = True) -> dict:
    """
    Train and evaluate de novo candidate classifiers.

    Uses only sequence-context and NDD-annotation features (no genotype
    data) to avoid definitional leakage. This models "what properties
    of a genomic site are associated with de novo mutation events" —
    the transferable signal we want to test cross-population.
    """
    sep = "═" * 65

    print(f"\n{sep}")
    print("  MODEL B — De Novo Candidate Classifier")
    print(f"{sep}")
    print("  Target    : is_de_novo (1 = de novo candidate, 0 = inherited)")
    print("  Features  : sequence context + NDD gene annotations ONLY")
    print("  NO parental GT features — definitional leakage prevention")
    print("  Rationale : Cross-population transfer of sequence-context signal")
    print("              is the core scientific question of the paper.")

    # Print feature descriptions
    print("\n  Feature set:")
    for i, (fname, desc) in enumerate(MODEL_B_FEATURE_DESCRIPTIONS.items()):
        print(f"    {i+1:>2}. {fname:<22} {desc}")

    # Prepare data
    df_b = df.copy()
    df_b["target"] = _parse_bool_col(df_b["is_de_novo"])

    available = [f for f in MODEL_B_FEATURES if f in df_b.columns]
    missing_features = [f for f in MODEL_B_FEATURES if f not in df_b.columns]
    if missing_features:
        print(f"\n  [WARN] Missing feature columns (will skip): {missing_features}")

    df_b = df_b.dropna(subset=available + ["target"])
    df_b["target"] = df_b["target"].astype(int)

    n_pos = df_b["target"].sum()
    n_neg = len(df_b) - n_pos
    pos_rate = n_pos / len(df_b)

    print(f"\n  Samples after NA-drop : {len(df_b):>10,}")
    print(f"  De novo (1)           : {n_pos:>10,}  ({100*pos_rate:.3f}%)")
    print(f"  Inherited (0)         : {n_neg:>10,}  ({100*(1-pos_rate):.3f}%)")
    print(f"  Imbalance ratio       : 1 : {n_neg//max(n_pos,1)}")

    scale_pos_weight = n_neg / max(n_pos, 1)

    X = df_b[available].values
    y = df_b["target"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    results_b = {"model_b": {}}

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n  [B.1] Logistic Regression")
    t0 = time.time()
    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1,
            solver="lbfgs",
        )),
    ])
    lr_pipe.fit(X_train, y_train)
    y_pred = lr_pipe.predict(X_test)
    y_prob = lr_pipe.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    print(f"  Train time : {elapsed:.1f}s")
    m = print_metrics(y_test, y_pred, y_prob,
                      "Logistic Regression — Model B")
    results_b["model_b"]["logistic_regression"] = m

    # LR coefficients (proxy for feature importance)
    print("\n  LR feature coefficients (proxy for importance):")
    lr_clf = lr_pipe.named_steps["clf"]
    coefs  = pd.Series(lr_clf.coef_[0], index=available).sort_values(
        key=abs, ascending=False
    )
    for fname, coef in coefs.items():
        direction = "↑ de novo" if coef > 0 else "↓ de novo"
        bar = "█" * min(int(abs(coef) * 5), 20)
        print(f"    {fname:<22} {coef:>+.4f}  {direction}  {bar}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n  [B.2] Random Forest (200 trees)")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10,
        class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        min_samples_leaf=10,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    elapsed = time.time() - t0
    print(f"  Train time : {elapsed:.1f}s")
    m = print_metrics(y_test, y_pred, y_prob, "Random Forest — Model B")
    results_b["model_b"]["random_forest"] = m
    imp_b_rf = print_feature_importance(rf, available)
    results_b["model_b"]["rf_feature_importance"] = (
        imp_b_rf.round(6).to_dict() if imp_b_rf is not None else {}
    )

    # ── XGBoost ───────────────────────────────────────────────────────────────
    if use_xgb and HAS_XGB:
        print("\n  [B.3] XGBoost")
        t0 = time.time()
        xgb = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train)
        y_pred = xgb.predict(X_test)
        y_prob = xgb.predict_proba(X_test)[:, 1]
        elapsed = time.time() - t0
        print(f"  Train time : {elapsed:.1f}s")
        m = print_metrics(y_test, y_pred, y_prob, "XGBoost — Model B")
        results_b["model_b"]["xgboost"] = m
        imp_b_xgb = print_feature_importance(xgb, available)
        results_b["model_b"]["xgb_feature_importance"] = (
            imp_b_xgb.round(6).to_dict() if imp_b_xgb is not None else {}
        )
    elif use_xgb and not HAS_XGB:
        print("\n  [B.3] XGBoost — SKIPPED (not installed)")
        print("        Run: pip install xgboost")

    # ── De novo enrichment in NDD genes ──────────────────────────────────────
    print("\n  ── De Novo Enrichment Analysis ──────────────────────────────────")
    df_b_analysis = df_b.copy()

    ndd    = df_b_analysis[df_b_analysis["ndd_gene_int"] == 1]
    non_ndd = df_b_analysis[df_b_analysis["ndd_gene_int"] == 0]

    ndd_dn_rate    = ndd["target"].mean()    if len(ndd)    > 0 else 0
    nonndd_dn_rate = non_ndd["target"].mean() if len(non_ndd) > 0 else 0
    enrichment     = ndd_dn_rate / max(nonndd_dn_rate, 1e-9)

    print(f"  De novo rate in NDD genes     : {100*ndd_dn_rate:.4f}%")
    print(f"  De novo rate outside NDD genes: {100*nonndd_dn_rate:.4f}%")
    print(f"  Enrichment ratio              : {enrichment:.2f}x")
    print(f"  Interpretation : de novos are {enrichment:.1f}x more likely in")
    print(f"  NDD genes — expected (ascertainment, large gene bodies)")

    # SysNDD tier breakdown
    print("\n  De novo rate by SysNDD confidence tier:")
    tier_map_inv = {4: "Definitive", 3: "Moderate", 2: "Limited",
                    1: "Refuted",   0: "Not in SysNDD"}
    for score in [4, 3, 2, 1, 0]:
        sub = df_b_analysis[df_b_analysis["sysndd_score"] == score]
        if len(sub) > 100:
            rate = sub["target"].mean()
            print(f"    {tier_map_inv[score]:<20} {100*rate:.4f}%  "
                  f"(n={len(sub):,})")

    results_b["de_novo_enrichment"] = {
        "ndd_gene_rate":     round(ndd_dn_rate, 8),
        "non_ndd_gene_rate": round(nonndd_dn_rate, 8),
        "enrichment_ratio":  round(enrichment, 4),
    }

    return results_b


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_feature_table(
    n_sample: int | None,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Load phase1a_feature_table.csv, encode all features.
    If n_sample is set, take a stratified sample on is_de_novo
    so that both classes are proportionally represented.
    """
    if not FEATURE_TABLE.exists():
        raise FileNotFoundError(
            f"Feature table not found: {FEATURE_TABLE}\n"
            "Run feature_engineering.py first."
        )

    print(f"[Loading] {FEATURE_TABLE.name}")
    t0 = time.time()

    # Read with dtype hints to control memory
    df = pd.read_csv(
        FEATURE_TABLE,
        dtype={
            "chrom":             str,
            "variant_type":      str,
            "gene_symbol":       str,
            "sysndd_category":   str,
            "sfari_score":       str,
            "sources":           str,
            "transmitted_allele_father": str,
            "transmitted_allele_mother": str,
        },
        low_memory=False,
    )
    elapsed = time.time() - t0
    print(f"  Loaded {len(df):,} rows in {elapsed:.1f}s")
    print(f"  Columns: {list(df.columns)}")

    # Encode features
    print("  Encoding features …")
    df = encode_all_features(df)

    # Parse boolean targets
    df["is_de_novo_parsed"]            = _parse_bool_col(df["is_de_novo"])
    df["is_mendelian_consistent_parsed"] = _parse_bool_col(
        df["is_mendelian_consistent"]
    )

    # Dataset overview
    dn_count = df["is_de_novo_parsed"].sum()
    mc_count = (df["is_mendelian_consistent_parsed"] == 0).sum()
    print(f"  De novo candidates        : {dn_count:,} "
          f"({100*dn_count/len(df):.3f}%)")
    print(f"  Mendelian violations      : {mc_count:,} "
          f"({100*mc_count/len(df):.3f}%)")
    print(f"  Variants in NDD genes     : {df['ndd_gene_int'].sum():,} "
          f"({100*df['ndd_gene_int'].mean():.1f}%)")

    if n_sample is not None and n_sample < len(df):
        print(f"\n  Stratified sampling to {n_sample:,} rows "
              f"(stratify on is_de_novo) …")
        df_valid = df.dropna(subset=["is_de_novo_parsed"]).copy()

        # Manual stratified split — avoids pandas 2.x groupby.apply
        # dropping the key column from the result DataFrame.
        df_pos = df_valid[df_valid["is_de_novo_parsed"] == 1.0]
        df_neg = df_valid[df_valid["is_de_novo_parsed"] == 0.0]

        # Proportional allocation so the natural class ratio is preserved
        n_pos = max(1, round(n_sample * len(df_pos) / len(df_valid)))
        n_neg = n_sample - n_pos

        df_pos_s = df_pos.sample(
            min(n_pos, len(df_pos)), random_state=random_state
        )
        df_neg_s = df_neg.sample(
            min(n_neg, len(df_neg)), random_state=random_state
        )

        df = (
            pd.concat([df_pos_s, df_neg_s])
            .sample(frac=1, random_state=random_state)   # shuffle
            .reset_index(drop=True)
        )

        print(f"  Sample size : {len(df):,}")
        dn_s = df["is_de_novo_parsed"].sum()
        print(f"  De novos in sample : {dn_s:.0f} "
              f"({100*dn_s/len(df):.3f}%)")

    return df


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(all_results: dict) -> None:
    """Save all metric results to ml_results/ directory."""
    RESULTS_DIR.mkdir(exist_ok=True)

    out_json = RESULTS_DIR / "baseline_metrics.json"
    with open(out_json, "w") as fh:
        json.dump(all_results, fh, indent=2, default=str)
    print(f"\n  Saved metrics → {out_json}")

    # Feature importance CSVs
    for model_key in ["model_a", "model_b"]:
        for algo_key in ["rf_feature_importance", "xgb_feature_importance"]:
            imp = all_results.get(model_key, {}).get(algo_key, {})
            if imp:
                imp_df = pd.DataFrame(
                    list(imp.items()), columns=["feature", "importance"]
                ).sort_values("importance", ascending=False)
                out_imp = RESULTS_DIR / f"{model_key}_{algo_key}.csv"
                imp_df.to_csv(out_imp, index=False)
                print(f"  Saved importances → {out_imp}")


# ---------------------------------------------------------------------------
# CLI + main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Phase 1a ML baseline — Models A and B."
    )
    p.add_argument(
        "--full", action="store_true",
        help="Use all rows in the feature table (slow for RF on 5.8M rows).",
    )
    p.add_argument(
        "--sample", type=int, default=None,
        help=f"Stratified sample size. Default={DEFAULT_SAMPLE:,} when "
             "--full not set.",
    )
    p.add_argument(
        "--no-xgb", action="store_true",
        help="Skip XGBoost (runs faster, fewer dependencies).",
    )
    p.add_argument(
        "--model", choices=["a", "b", "both"], default="both",
        help="Which model to run: a, b, or both (default).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    n_sample = None if args.full else (args.sample or DEFAULT_SAMPLE)
    use_xgb  = not args.no_xgb
    run_a    = args.model in ("a", "both")
    run_b    = args.model in ("b", "both")

    sep = "=" * 65
    print(sep)
    print("  Phase 1a ML Baseline")
    print(sep)
    print(f"  Feature table : {FEATURE_TABLE.name}")
    print(f"  Sample size   : {'ALL' if n_sample is None else f'{n_sample:,}'}")
    print(f"  XGBoost       : {'enabled' if use_xgb and HAS_XGB else 'disabled'}")
    print(f"  Models        : {args.model.upper()}")
    if not HAS_XGB and use_xgb:
        print("  [WARN] XGBoost not installed — run: pip install xgboost")

    # Load data
    df = load_feature_table(n_sample)

    # Remap parsed columns for convenience
    df["is_de_novo"]                 = df["is_de_novo_parsed"]
    df["is_mendelian_consistent"]    = df["is_mendelian_consistent_parsed"]

    all_results: dict = {}
    t_total = time.time()

    if run_a:
        res_a = run_model_a(df)
        all_results.update(res_a)

    if run_b:
        res_b = run_model_b(df, use_xgb=use_xgb)
        all_results.update(res_b)

    total_time = time.time() - t_total

    print(f"\n{sep}")
    print(f"  ALL DONE — total runtime: {total_time:.1f}s")
    print(f"  Results directory: {RESULTS_DIR}")
    print(sep)

    save_results(all_results)

    # Paper-ready summary
    print(f"\n  ── PAPER-READY NUMBERS ──────────────────────────────────────")
    if "model_a" in all_results:
        best_a = max(
            all_results["model_a"].items(),
            key=lambda kv: kv[1].get("accuracy", 0)
            if isinstance(kv[1], dict) else 0,
        )
        if isinstance(best_a[1], dict):
            print(f"  Model A best accuracy   : "
                  f"{best_a[1].get('accuracy', 'N/A'):.4f} "
                  f"({best_a[0]})")
    if "model_b" in all_results:
        best_b = max(
            all_results["model_b"].items(),
            key=lambda kv: kv[1].get("auc_roc", 0)
            if isinstance(kv[1], dict) else 0,
        )
        if isinstance(best_b[1], dict):
            print(f"  Model B best AUC-ROC    : "
                  f"{best_b[1].get('auc_roc', 'N/A'):.4f} "
                  f"({best_b[0]})")
            print(f"  Model B best AUC-PR     : "
                  f"{best_b[1].get('auc_pr', 'N/A'):.4f}")
    if "de_novo_enrichment" in all_results:
        enr = all_results["de_novo_enrichment"]["enrichment_ratio"]
        print(f"  De novo enrichment in NDD genes : {enr:.2f}x")
    print(f"  ─────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()