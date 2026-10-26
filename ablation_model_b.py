"""
ablation_model_b.py
---------------------
Ablation study for Model B: compares three feature sets to isolate
how much of the AUC-ROC comes from quality metrics (GQ/DP) vs
genuine biological signal (sequence context + NDD annotation).

Three conditions:
  FULL     — all 11 features (variant_type, NDD annots, GQ, DP)
  BIO-ONLY — 5 features: variant_type, ndd_gene, sysndd, sfari, n_sources
  QC-ONLY  — 6 features: GQ and DP values only (negative control)

If FULL ≈ QC-ONLY >> BIO-ONLY → high AUC is quality-driven (artifact signal)
If FULL ≈ BIO-ONLY >> QC-ONLY → high AUC is biologically meaningful

Run from your Data directory:
    python3 ablation_model_b.py
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_ROOT     = Path(__file__).parent
FEATURE_TABLE = DATA_ROOT / "phase1a_feature_table.csv"
RESULTS_DIR   = DATA_ROOT / "ml_results"
RANDOM_STATE  = 42
SAMPLE_SIZE   = 500_000

# ---------------------------------------------------------------------------
# Feature sets for the three ablation conditions
# ---------------------------------------------------------------------------

FEATURE_SETS = {
    "FULL": [
        "variant_type_enc", "ndd_gene_int", "sysndd_score",
        "sfari_score_enc", "n_sources",
        "child_gq", "child_dp", "father_gq", "mother_gq",
        "father_dp", "mother_dp",
    ],
    "BIO-ONLY": [
        "variant_type_enc", "ndd_gene_int", "sysndd_score",
        "sfari_score_enc", "n_sources",
    ],
    "QC-ONLY": [
        "child_gq", "child_dp", "father_gq", "mother_gq",
        "father_dp", "mother_dp",
    ],
}

VTYPE_MAP   = {"SNV": 0, "INDEL": 1, "MULTIALLELIC": 2, "SPANNING_DEL": 3}
SYSNDD_MAP  = {"Definitive": 4, "Moderate": 3, "Limited": 2,
               "Refuted": 0, "not applicable": 0, "": 0}
SFARI_MAP   = {"1": 4, "2": 3, "3": 2, "S": 1, "": 0}


def _parse_bool(series):
    mapping = {True: 1, False: 0, "True": 1, "False": 0, 1: 1, 0: 0}
    return series.map(mapping)


def encode(df):
    def gt_dosage(s):
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

    for r in ("father", "mother", "child"):
        if f"{r}_gt" in df.columns:
            df[f"{r}_dosage"] = gt_dosage(df[f"{r}_gt"])

    df["variant_type_enc"] = df["variant_type"].map(VTYPE_MAP).fillna(0).astype(int)
    df["sysndd_score"]     = df["sysndd_category"].fillna("").astype(str).str.strip().map(
                                 lambda x: SYSNDD_MAP.get(x, 0))
    df["sfari_score_enc"]  = df["sfari_score"].fillna("").astype(str).str.strip().map(
                                 lambda x: SFARI_MAP.get(x, 0))
    df["ndd_gene_int"]     = df["is_ndd_gene"].map(
                                 {True: 1, False: 0, "True": 1, "False": 0}).fillna(0).astype(int)
    df["n_sources"]        = pd.to_numeric(df.get("n_sources", 0), errors="coerce").fillna(0).astype(int)
    for c in ["father_gq", "mother_gq", "child_gq",
              "father_dp", "mother_dp", "child_dp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def evaluate(X_train, X_test, y_train, y_test, scale_pos):
    """Train XGBoost (or RF fallback) and return AUC-ROC and AUC-PR."""
    if HAS_XGB:
        clf = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=scale_pos, eval_metric="aucpr",
                            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    else:
        clf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                     class_weight="balanced",
                                     random_state=RANDOM_STATE, n_jobs=-1)
    clf.fit(X_train, y_train)
    prob = clf.predict_proba(X_test)[:, 1]
    return roc_auc_score(y_test, prob), average_precision_score(y_test, prob)


def main():
    sep = "=" * 65
    print(sep)
    print("  Model B Ablation Study — Biological vs Quality Signal")
    print(sep)

    # ── Load and encode ───────────────────────────────────────────────────────
    print(f"\n[1] Loading {FEATURE_TABLE.name} …")
    df = pd.read_csv(FEATURE_TABLE, low_memory=False,
                     dtype={"variant_type": str, "sysndd_category": str,
                            "sfari_score": str, "gene_symbol": str,
                            "sources": str})
    print(f"  {len(df):,} rows loaded")
    df = encode(df)
    df["is_de_novo"] = _parse_bool(df["is_de_novo"])
    df = df.dropna(subset=["is_de_novo"])
    df["is_de_novo"] = df["is_de_novo"].astype(int)

    # ── Stratified sample ─────────────────────────────────────────────────────
    print(f"\n[2] Stratified sample → {SAMPLE_SIZE:,} rows …")
    df_pos = df[df["is_de_novo"] == 1]
    df_neg = df[df["is_de_novo"] == 0]
    n_pos  = max(1, round(SAMPLE_SIZE * len(df_pos) / len(df)))
    n_neg  = SAMPLE_SIZE - n_pos
    df_s   = pd.concat([
        df_pos.sample(min(n_pos, len(df_pos)), random_state=RANDOM_STATE),
        df_neg.sample(min(n_neg, len(df_neg)), random_state=RANDOM_STATE),
    ]).sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

    n_pos_s       = df_s["is_de_novo"].sum()
    n_neg_s       = len(df_s) - n_pos_s
    scale_pos     = n_neg_s / max(n_pos_s, 1)
    y             = df_s["is_de_novo"].values

    print(f"  De novos : {n_pos_s:,} ({100*n_pos_s/len(df_s):.3f}%)")
    print(f"  Inherited: {n_neg_s:,}")
    print(f"  scale_pos_weight: {scale_pos:.1f}")

    # ── Train/test split (same split for all conditions) ──────────────────────
    idx = np.arange(len(df_s))
    idx_train, idx_test = train_test_split(
        idx, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    y_train = y[idx_train]
    y_test  = y[idx_test]

    # ── Run all three conditions ──────────────────────────────────────────────
    print(f"\n[3] Running ablation conditions "
          f"({'XGBoost' if HAS_XGB else 'Random Forest'}) …")

    results = {}
    for condition, features in FEATURE_SETS.items():
        available = [f for f in features if f in df_s.columns]
        missing   = set(features) - set(available)
        if missing:
            print(f"\n  [{condition}] Missing: {missing} — skipping")
            continue

        X = df_s[available].fillna(0).values
        X_tr, X_te = X[idx_train], X[idx_test]

        t0 = time.time()
        auc_roc, auc_pr = evaluate(X_tr, X_te, y_train, y_test, scale_pos)
        elapsed = time.time() - t0

        results[condition] = {
            "features":    available,
            "n_features":  len(available),
            "auc_roc":     round(auc_roc, 4),
            "auc_pr":      round(auc_pr, 4),
            "train_time_s": round(elapsed, 1),
        }

        print(f"\n  [{condition}]  {len(available)} features  |  "
              f"AUC-ROC={auc_roc:.4f}  AUC-PR={auc_pr:.4f}  "
              f"({elapsed:.0f}s)")
        print(f"    Features: {', '.join(available)}")

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  ABLATION SUMMARY")
    print(f"{'─'*65}")
    print(f"  {'Condition':<12} {'N features':>10} {'AUC-ROC':>10} "
          f"{'AUC-PR':>10}  Interpretation")
    print(f"  {'─'*10:<12} {'─'*10:>10} {'─'*7:>10} {'─'*7:>10}")
    for cond, r in results.items():
        interp = {
            "FULL":     "combined signal (quality + biology)",
            "BIO-ONLY": "pure biological signal ← THIS is what transfers",
            "QC-ONLY":  "pure quality-artifact signal (negative control)",
        }.get(cond, "")
        print(f"  {cond:<12} {r['n_features']:>10} {r['auc_roc']:>10.4f} "
              f"{r['auc_pr']:>10.4f}  {interp}")

    # ── Interpretation ────────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  INTERPRETATION GUIDE")
    print(f"{'─'*65}")

    if "FULL" in results and "BIO-ONLY" in results and "QC-ONLY" in results:
        full_auc    = results["FULL"]["auc_roc"]
        bio_auc     = results["BIO-ONLY"]["auc_roc"]
        qc_auc      = results["QC-ONLY"]["auc_roc"]
        bio_contrib = bio_auc - 0.5
        qc_contrib  = qc_auc - 0.5

        print(f"\n  AUC above random (0.50):")
        print(f"    FULL     : +{full_auc-0.5:.4f}")
        print(f"    BIO-ONLY : +{bio_contrib:.4f}  ← transferable cross-pop signal")
        print(f"    QC-ONLY  : +{qc_contrib:.4f}  ← non-transferable artifact signal")

        if qc_auc > bio_auc:
            print(f"\n  ⚠  Quality signal dominates biological signal.")
            print(f"  The model primarily learned that lower parental GQ/DP")
            print(f"  predicts artifact de novos — not real mutation biology.")
            print(f"\n  For the paper: report BIO-ONLY AUC ({bio_auc:.4f}) as")
            print(f"  the transferable signal. Full AUC ({full_auc:.4f}) is")
            print(f"  dataset-specific and will not port to South Asian trios.")
        else:
            print(f"\n  ✓  Biological signal dominates quality signal.")
            print(f"  The model has learned real sequence-context patterns.")
            print(f"  Full AUC ({full_auc:.4f}) is largely biologically meaningful.")

    # ── Save ──────────────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / "ablation_results.json"
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"\n  Saved → {out}")
    print(f"\n{sep}")
    print("  Ablation complete.")
    print(sep)


if __name__ == "__main__":
    main()
