"""
population_af_analysis.py
--------------------------
Cross-population portability analysis using 1000G Phase 3 pre-computed
allele frequencies from VCF INFO fields.

WHY THIS APPROACH (instead of trio-based):
  The 1000G Phase 3 VCF (2504 samples) deliberately excludes offspring
  from parent-child trios — samples were selected for unrelatedness.
  The PED file describes family structure but offspring are not in the
  VCF. However, the VCF INFO field contains pre-computed allele
  frequencies for all 5 super-populations from those 2504 individuals:
    EUR_AF, SAS_AF, AFR_AF, EAS_AF, AMR_AF

  We read ONLY the first 8 VCF columns (CHROM–INFO), skipping all 2504
  sample genotype columns, making parsing ~300x faster than genotype
  extraction. Chr1 completes in ~3-5 minutes.

ANALYSIS:
  1. Allele frequency divergence (EUR vs SAS):
       - FST  = (p_EUR - p_SAS)² / (p_EUR*(1-p_EUR) + p_SAS*(1-p_SAS))
       - |ΔAF| = |EUR_AF - SAS_AF|
     Computed genome-wide, then stratified by:
       - NDD gene regions vs background
       - SysNDD confidence tier (Definitive/Moderate/Limited)
       - SFARI score tier
       - DenovoDB mutation burden

  2. ML portability regression:
     Task:   predict SAS_AF from EUR_AF + gene annotation features
     Train:  chr1 variants
     Test:   chr2 variants (held-out chromosome)
     Models: LinearRegression, RandomForestRegressor, XGBRegressor
     Metric: R², Pearson r, MAE
     Key result: compare R² in NDD gene subset vs background
       → lower R² in NDD genes = reduced cross-pop portability in
         the genes that matter most for the paper's claim

  3. DenovoDB-anchored gene analysis:
     Genes with high validated NDD de novo burden — do they show
     greater EUR-SAS frequency divergence? This tests whether
     constrained NDD genes are more population-specific.

PAPER CONTRIBUTION:
  "We characterise EUR-SAS allele frequency divergence specifically
   within NDD gene regions. NDD-gene variants show [higher/lower]
   FST vs background (p<0.001). A regression model trained to predict
   SAS_AF from EUR_AF achieves R²=[full] genome-wide but R²=[ndd]
   within NDD gene regions, demonstrating that EUR-trained models
   do not transfer cleanly to South Asian allele frequency patterns
   in the genes most relevant to NDD risk."

Outputs (saved to ~/Documents/Data/):
  population_af_table.csv         — per-variant AF divergence
  population_af_summary.json      — paper-ready aggregate numbers
  ml_results/cross_pop_af_ml.json — ML portability results

Run:
  python3 population_af_analysis.py          # chr1 train + chr2 test
  python3 population_af_analysis.py --chr 1  # chr1 only (analysis only)
  python3 population_af_analysis.py --n 500000  # cap variants for speed

Requires: pandas, numpy, scikit-learn, scipy, xgboost (optional)
"""

import argparse
import gzip
import json
import sys
import time
import warnings
from bisect import bisect_right
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

try:
    from scipy.stats import pearsonr, mannwhitneyu
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("[WARN] scipy not installed — some stats skipped. Run: pip install scipy")

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

DATA_ROOT = Path(__file__).parent
sys.path.insert(0, str(DATA_ROOT))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

KG_DIR      = DATA_ROOT / "1000 Genomes"
KG2_DIR     = DATA_ROOT / "1000 Genomes 2"
GENE_BED    = DATA_ROOT / "ndd_genes_grch37.bed"
NDD_REF     = DATA_ROOT / "unified_ndd_gene_reference.csv"
DENOVO_GENES= DATA_ROOT / "denovo_db_gene_counts.csv"

CHR_VCFS = {
    "1": KG_DIR  / "ALL.chr1.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
    "2": KG2_DIR / "ALL.chr2.phase3_shapeit2_mvncall_integrated_v5b.20130502.genotypes.vcf.gz",
}

OUT_TABLE   = DATA_ROOT / "population_af_table.csv"
OUT_SUMMARY = DATA_ROOT / "population_af_summary.json"
RESULTS_DIR = DATA_ROOT / "ml_results"

# Encoding maps
VTYPE_MAP  = {"SNV": 0, "INDEL": 1, "MULTIALLELIC": 2, "SPANNING_DEL": 3}
SYSNDD_MAP = {"Definitive": 4, "Moderate": 3, "Limited": 2,
              "Refuted": 0, "not applicable": 0, "": 0}
SFARI_MAP  = {"1": 4, "2": 3, "3": 2, "S": 1, "": 0}

# ---------------------------------------------------------------------------
# Fast VCF INFO-only reader (skips all 2504 sample columns)
# ---------------------------------------------------------------------------

def _parse_info(info_str: str) -> Dict[str, str]:
    """Parse VCF INFO string into dict. '.' → empty dict."""
    if info_str.strip() == ".":
        return {}
    result = {}
    for field in info_str.split(";"):
        if "=" in field:
            k, v = field.split("=", 1)
            result[k] = v
        else:
            result[field] = "1"
    return result


def iter_vcf_info(
    path: Path,
    n: Optional[int] = None,
) -> Iterator[Dict]:
    """
    Stream VCF records reading ONLY the first 8 columns (up to INFO).
    All 2504 sample genotype columns are skipped entirely — this is
    ~300x faster than full-record parsing.

    Yields: {chrom, pos, ref, alt, info: dict_of_info_fields}
    """
    path = Path(path)
    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            if raw.startswith("#"):
                continue
            if n is not None and count >= n:
                break
            # Split only to column 8 — avoids parsing all sample columns
            fields = raw.split("\t", 8)
            if len(fields) < 8:
                continue
            chrom, pos, _, ref, alt = fields[0], fields[1], fields[2], fields[3], fields[4]
            info_dict = _parse_info(fields[7])
            yield {
                "chrom": chrom,
                "pos":   int(pos),
                "ref":   ref,
                "alt":   alt,
                "info":  info_dict,
            }
            count += 1


def classify_variant(ref: str, alt: str) -> str:
    if "*" in alt:      return "SPANNING_DEL"
    if "," in alt:      return "MULTIALLELIC"
    if len(ref) == 1 and len(alt) == 1: return "SNV"
    return "INDEL"


# ---------------------------------------------------------------------------
# Gene interval index (same design as feature_engineering.py)
# ---------------------------------------------------------------------------

class GeneIntervalIndex:
    def __init__(self, bed_df: pd.DataFrame) -> None:
        self._idx: Dict[str, Tuple] = {}
        bed_df = bed_df.copy()
        bed_df["_cn"] = bed_df.iloc[:, 0].astype(str).str.lstrip("chr")
        for chrom, grp in bed_df.groupby("_cn"):
            grp = grp.sort_values(grp.columns[1])
            starts  = grp.iloc[:, 1].astype(int).tolist()
            entries = list(zip(grp.iloc[:, 1].astype(int),
                               grp.iloc[:, 2].astype(int),
                               grp.iloc[:, 3].astype(str)))
            self._idx[str(chrom)] = (starts, entries)

    def query(self, chrom: str, pos: int) -> str:
        cn = chrom.lstrip("chr")
        if cn not in self._idx:
            return ""
        starts, entries = self._idx[cn]
        idx = bisect_right(starts, pos) - 1
        for i in range(idx, max(-1, idx - 50), -1):
            s, e, gene = entries[i]
            if s <= pos <= e:
                return gene
            if pos - s > 3_000_000:
                break
        return ""


# ---------------------------------------------------------------------------
# FST and divergence metrics
# ---------------------------------------------------------------------------

def hudson_fst(p1: float, p2: float) -> float:
    """
    Hudson's FST estimator between two populations.
    FST = (p1-p2)² / (p1*(1-p1) + p2*(1-p2))
    Range [0, 1]: 0 = identical frequencies, 1 = fully differentiated.
    Returns NaN when denominator is zero.
    """
    denom = p1 * (1 - p1) + p2 * (1 - p2)
    if denom < 1e-10:
        return float("nan")
    return (p1 - p2) ** 2 / denom


# ---------------------------------------------------------------------------
# Resource loading
# ---------------------------------------------------------------------------

def load_resources():
    print("[Loading annotation resources]")

    bed_df     = pd.read_csv(GENE_BED, sep="\t", header=None,
                              names=["chrom","start","end","gene_symbol","strand"])
    gene_index = GeneIntervalIndex(bed_df)
    print(f"  Gene BED:         {len(bed_df):,} NDD gene intervals")

    ndd_ref    = pd.read_csv(NDD_REF, dtype=str)
    ndd_lookup = {row["gene_symbol"]: row for _, row in ndd_ref.iterrows()}
    print(f"  NDD reference:    {len(ndd_lookup):,} genes")

    ddb_lookup: Dict[str, int] = {}
    if DENOVO_GENES.exists():
        ddb = pd.read_csv(DENOVO_GENES)
        sym  = next((c for c in ddb.columns if "gene" in c.lower()), None)
        cnt  = next((c for c in ddb.columns if "ndd" in c.lower()), None) or \
               next((c for c in ddb.columns if "total" in c.lower()), None)
        if sym and cnt:
            ddb_lookup = dict(zip(
                ddb[sym].astype(str).str.upper(),
                pd.to_numeric(ddb[cnt], errors="coerce").fillna(0).astype(int),
            ))
            print(f"  DenovoDB:         {len(ddb_lookup):,} genes with de novo counts")
    else:
        print("  DenovoDB gene counts: NOT FOUND (run denovo_db_loader.py)")

    return gene_index, ndd_lookup, ddb_lookup


# ---------------------------------------------------------------------------
# Core extraction: build AF table for one chromosome
# ---------------------------------------------------------------------------

def extract_chr_af_table(
    vcf_path: Path,
    gene_index: GeneIntervalIndex,
    ndd_lookup: dict,
    ddb_lookup: dict,
    chrom_label: str,
    n_max: Optional[int] = None,
) -> pd.DataFrame:
    """
    Single fast pass through a VCF (INFO columns only).
    Builds per-variant table with population AFs + NDD annotation.
    """
    rows: List[dict] = []
    n_seen = 0
    t0 = time.time()

    for rec in iter_vcf_info(vcf_path, n=n_max):
        info = rec["info"]

        # Extract population AFs (missing = NaN)
        def af(key: str) -> float:
            v = info.get(key, ".")
            try:
                return float(v.split(",")[0])   # first allele for multi-allelic
            except (ValueError, AttributeError):
                return float("nan")

        eur_af = af("EUR_AF")
        sas_af = af("SAS_AF")
        afr_af = af("AFR_AF")
        eas_af = af("EAS_AF")
        amr_af = af("AMR_AF")

        # Skip if EUR or SAS frequency is missing
        if np.isnan(eur_af) or np.isnan(sas_af):
            n_seen += 1
            continue

        chrom    = rec["chrom"]
        pos      = rec["pos"]
        ref      = rec["ref"]
        alt      = rec["alt"]
        vtype    = classify_variant(ref, alt)
        vtype_e  = VTYPE_MAP.get(vtype, 0)

        # NDD annotation
        gene_sym = gene_index.query(chrom, pos)
        is_ndd   = gene_sym != ""
        ann      = ndd_lookup.get(gene_sym.upper(), {})
        sfari_e  = SFARI_MAP.get(str(ann.get("sfari_score","")).strip(), 0)
        sysndd_e = SYSNDD_MAP.get(str(ann.get("sysndd_category","")).strip(), 0)
        n_src    = int(ann.get("n_sources", 0) or 0)
        ddb_cnt  = ddb_lookup.get(gene_sym.upper(), 0)

        # Divergence metrics
        abs_diff = abs(eur_af - sas_af)
        fst_val  = hudson_fst(eur_af, sas_af)

        rows.append({
            "chrom":           chrom,
            "pos":             pos,
            "ref":             ref,
            "alt":             alt,
            "chrom_label":     chrom_label,
            "variant_type":    vtype,
            "variant_type_enc": vtype_e,
            # Population AFs
            "eur_af":          eur_af,
            "sas_af":          sas_af,
            "afr_af":          afr_af,
            "eas_af":          eas_af,
            "amr_af":          amr_af,
            # Divergence
            "abs_eur_sas_diff": abs_diff,
            "fst_eur_sas":      fst_val,
            # NDD annotation
            "gene_symbol":      gene_sym,
            "is_ndd_gene":      int(is_ndd),
            "sfari_score_enc":  sfari_e,
            "sysndd_score":     sysndd_e,
            "n_sources":        n_src,
            "denovo_db_count":  ddb_cnt,
        })

        n_seen += 1
        if n_seen % 500_000 == 0:
            print(f"  chr{chrom_label}: {n_seen:>8,} seen | "
                  f"{len(rows):>7,} with valid AFs | "
                  f"{time.time()-t0:.0f}s")

    elapsed = time.time() - t0
    print(f"  chr{chrom_label} complete: {n_seen:>8,} seen | "
          f"{len(rows):>7,} rows | {elapsed:.0f}s")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical analysis
# ---------------------------------------------------------------------------

def divergence_summary(df: pd.DataFrame) -> dict:
    """Compute FST and |ΔAF| statistics for NDD gene vs background."""
    summary = {}

    for subset_name, mask in [
        ("all_variants",       pd.Series([True] * len(df), index=df.index)),
        ("ndd_genes",          df["is_ndd_gene"] == 1),
        ("background",         df["is_ndd_gene"] == 0),
        ("sysndd_definitive",  df["sysndd_score"] == 4),
        ("sysndd_moderate",    df["sysndd_score"] == 3),
        ("sysndd_limited",     df["sysndd_score"] == 2),
        ("sources_3plus",      df["n_sources"] >= 3),
        ("sources_1only",      (df["n_sources"] == 1) & (df["is_ndd_gene"] == 1)),
        ("sfari_score_1_2",    df["sfari_score_enc"] >= 3),
        ("high_denovo_burden", df["denovo_db_count"] >= 5),
    ]:
        sub = df[mask].dropna(subset=["fst_eur_sas", "abs_eur_sas_diff"])
        if len(sub) < 10:
            continue

        entry = {
            "n":              len(sub),
            "mean_fst":       round(sub["fst_eur_sas"].mean(), 5),
            "median_fst":     round(sub["fst_eur_sas"].median(), 5),
            "mean_abs_diff":  round(sub["abs_eur_sas_diff"].mean(), 5),
        }
        if HAS_SCIPY and subset_name not in ("all_variants", "background"):
            bg = df[df["is_ndd_gene"] == 0].dropna(subset=["fst_eur_sas"])
            stat, pval = mannwhitneyu(
                sub["fst_eur_sas"], bg["fst_eur_sas"], alternative="two-sided"
            )
            entry["mannwhitney_p_vs_background"] = float(f"{pval:.2e}")

        summary[subset_name] = entry

    return summary


# ---------------------------------------------------------------------------
# ML portability regression
# ---------------------------------------------------------------------------

# Features: predict SAS_AF from EUR AF + gene context
REGR_FEATURES = [
    "eur_af",            # primary predictor
    "afr_af",            # other pops add context
    "eas_af",
    "amr_af",
    "variant_type_enc",
    "is_ndd_gene",
    "sysndd_score",
    "sfari_score_enc",
    "n_sources",
    "denovo_db_count",
]

EUR_ONLY_FEATURES = ["eur_af"]   # baseline: EUR AF alone


def run_portability_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: List[str],
    label: str = "",
) -> dict:
    """
    Train regression on train_df to predict sas_af.
    Evaluate on test_df genome-wide, NDD-gene subset, background subset.
    Returns dict of metrics.
    """
    available = [f for f in features if f in train_df.columns]
    train_clean = train_df.dropna(subset=available + ["sas_af"])
    test_clean  = test_df.dropna(subset=available + ["sas_af"])

    X_train = train_clean[available].fillna(0).values
    y_train = train_clean["sas_af"].values
    X_test  = test_clean[available].fillna(0).values
    y_test  = test_clean["sas_af"].values

    n_pos  = len(y_train)
    print(f"\n  [{label}] Training on {n_pos:,} variants, "
          f"testing on {len(y_test):,}")

    results = {}

    for model_name, model in [
        ("ridge",  Pipeline([("sc", StandardScaler()),
                              ("m", Ridge(alpha=1.0))])),
        ("rf",     RandomForestRegressor(n_estimators=200, max_depth=8,
                                          random_state=42, n_jobs=-1,
                                          min_samples_leaf=50)),
    ] + ([("xgb", XGBRegressor(n_estimators=300, max_depth=6,
                                 learning_rate=0.1, random_state=42,
                                 n_jobs=-1, verbosity=0))]
         if HAS_XGB else []):

        t0 = time.time()
        model.fit(X_train, y_train)
        elapsed = time.time() - t0

        # Predictions on test set
        y_pred = model.predict(X_test)

        model_res = {}
        for subset_name, mask in [
            ("all",        pd.Series([True] * len(test_clean), index=test_clean.index)),
            ("ndd_genes",  test_clean["is_ndd_gene"] == 1),
            ("background", test_clean["is_ndd_gene"] == 0),
        ]:
            idx = mask[mask].index
            if len(idx) < 20:
                continue
            sub_y_true = test_clean.loc[idx, "sas_af"].values
            sub_y_pred = y_pred[test_clean.index.get_indexer(idx)]
            r2  = r2_score(sub_y_true, sub_y_pred)
            mae = mean_absolute_error(sub_y_true, sub_y_pred)
            corr = pearsonr(sub_y_true, sub_y_pred)[0] if HAS_SCIPY else float("nan")
            model_res[subset_name] = {
                "n":    len(idx),
                "r2":   round(r2, 4),
                "mae":  round(mae, 4),
                "corr": round(corr, 4) if not np.isnan(corr) else None,
            }
            print(f"    {model_name:<6} {label:<20} {subset_name:<12} "
                  f"R²={r2:.4f}  MAE={mae:.4f}  "
                  f"r={corr:.4f}  ({len(idx):,} variants)  [{elapsed:.0f}s]")

        results[model_name] = model_res

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chr", choices=["1","2","both"], default="both", dest="chrom")
    p.add_argument("--n", type=int, default=None,
                   help="Max variants per chromosome (None=all)")
    return p.parse_args()


def main():
    args = parse_args()
    chroms = ["1","2"] if args.chrom == "both" else [args.chrom]

    sep = "=" * 65
    print(sep)
    print("  Population AF Divergence Analysis — EUR vs SAS")
    print("  1000G Phase 3 | INFO-field AFs | NDD gene regions")
    print(sep)
    print(f"\n  Chromosomes : {chroms}")
    print(f"  Max variants: {args.n or 'ALL'} per chromosome")
    print(f"  Approach    : INFO-only parsing (skips sample genotypes)")
    print(f"  Speed       : ~300x faster than full genotype extraction\n")

    # ── Load annotation resources ─────────────────────────────────────────────
    gene_index, ndd_lookup, ddb_lookup = load_resources()

    # ── Extract AF tables ─────────────────────────────────────────────────────
    chr_dfs: Dict[str, pd.DataFrame] = {}
    for chrom in chroms:
        vcf = CHR_VCFS.get(chrom)
        if vcf is None or not vcf.exists():
            print(f"\n[SKIP] chr{chrom} VCF not found: {vcf}")
            continue
        print(f"\n[Extracting chr{chrom}] {vcf.name}")
        chr_dfs[chrom] = extract_chr_af_table(
            vcf, gene_index, ndd_lookup, ddb_lookup, chrom, args.n
        )

    if not chr_dfs:
        print("[ERROR] No data extracted. Check VCF paths.")
        sys.exit(1)

    # Combined DataFrame
    combined = pd.concat(list(chr_dfs.values()), ignore_index=True)
    print(f"\n  Combined: {len(combined):,} variants with valid EUR+SAS AFs")

    # ── Save AF table ─────────────────────────────────────────────────────────
    combined.to_csv(OUT_TABLE, index=False)
    print(f"  Saved → {OUT_TABLE}")

    # ── Divergence statistics ─────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  EUR-SAS ALLELE FREQUENCY DIVERGENCE")
    print(f"{'─'*65}")

    summary = divergence_summary(combined)

    print(f"\n  {'Subset':<25} {'n':>9} {'mean FST':>9} "
          f"{'median FST':>10} {'mean|ΔAF|':>10}")
    print(f"  {'─'*25} {'─'*9} {'─'*9} {'─'*10} {'─'*10}")
    for subset, s in summary.items():
        pval_str = ""
        if "mannwhitney_p_vs_background" in s:
            pval_str = f"  (p={s['mannwhitney_p_vs_background']} vs bg)"
        print(f"  {subset:<25} {s['n']:>9,} {s['mean_fst']:>9.5f} "
              f"{s['median_fst']:>10.5f} {s['mean_abs_diff']:>10.5f}{pval_str}")

    # FST enrichment in NDD genes vs background
    ndd_fst = summary.get("ndd_genes", {}).get("mean_fst")
    bg_fst  = summary.get("background", {}).get("mean_fst")
    if ndd_fst and bg_fst and bg_fst > 0:
        enrichment = ndd_fst / bg_fst
        print(f"\n  NDD gene FST enrichment vs background: {enrichment:.3f}x")
        if enrichment > 1:
            print("  → NDD gene variants are MORE differentiated between "
                  "EUR and SAS than background")
        else:
            print("  → NDD gene variants are LESS differentiated between "
                  "EUR and SAS than background")

    # ── ML portability regression ─────────────────────────────────────────────
    all_ml_results = {}

    if len(chr_dfs) >= 2:
        # Two chromosomes available: chr1=train, chr2=test
        print(f"\n{'─'*65}")
        print("  ML PORTABILITY REGRESSION (chr1 train → chr2 test)")
        print(f"{'─'*65}")
        train_df = chr_dfs["1"]
        test_df  = chr_dfs["2"]

        print(f"\n  BASELINE: EUR_AF alone → predict SAS_AF")
        r1 = run_portability_regression(
            train_df, test_df, EUR_ONLY_FEATURES, "EUR-only baseline"
        )
        all_ml_results["eur_only_baseline"] = r1

        print(f"\n  FULL: EUR_AF + all population AFs + gene features → SAS_AF")
        r2 = run_portability_regression(
            train_df, test_df, REGR_FEATURES, "EUR+gene features"
        )
        all_ml_results["eur_plus_gene_features"] = r2

    elif len(chr_dfs) == 1:
        # Only one chromosome: train/test split on variants
        chrom_key = list(chr_dfs.keys())[0]
        print(f"\n{'─'*65}")
        print(f"  ML PORTABILITY REGRESSION (chr{chrom_key} 80/20 split)")
        print(f"{'─'*65}")
        df_ml = chr_dfs[chrom_key]
        train_df, test_df = train_test_split(
            df_ml, test_size=0.2, random_state=42
        )

        r1 = run_portability_regression(
            train_df, test_df, EUR_ONLY_FEATURES, "EUR-only baseline"
        )
        all_ml_results["eur_only_baseline"] = r1

        r2 = run_portability_regression(
            train_df, test_df, REGR_FEATURES, "EUR+gene features"
        )
        all_ml_results["eur_plus_gene_features"] = r2

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(exist_ok=True)
    full_results = {
        "divergence_summary": summary,
        "ml_portability":     all_ml_results,
        "dataset_stats": {
            "total_variants_with_valid_afs": len(combined),
            "ndd_gene_variants": int((combined["is_ndd_gene"] == 1).sum()),
            "background_variants": int((combined["is_ndd_gene"] == 0).sum()),
            "chromosomes": chroms,
        },
    }
    out_json = RESULTS_DIR / "cross_pop_af_ml.json"
    with open(out_json, "w") as fh:
        json.dump(full_results, fh, indent=2, default=str)
    print(f"\n  Saved summary  → {OUT_SUMMARY}")
    with open(OUT_SUMMARY, "w") as fh:
        json.dump(full_results, fh, indent=2, default=str)
    print(f"  Saved ML results → {out_json}")

    # ── Paper-ready numbers ───────────────────────────────────────────────────
    print(f"\n{sep}")
    print("  PAPER-READY NUMBERS")
    print(sep)

    all_s = summary.get("all_variants", {})
    ndd_s = summary.get("ndd_genes", {})
    bg_s  = summary.get("background", {})
    def_s = summary.get("sysndd_definitive", {})

    print(f"\n  Variants analysed (valid EUR+SAS AF): {len(combined):,}")
    print(f"  NDD gene variants:  {ndd_s.get('n', 'N/A'):,}")
    print(f"  Background variants:{bg_s.get('n', 'N/A'):,}")
    print(f"\n  EUR-SAS FST:")
    print(f"    Genome-wide mean:    {all_s.get('mean_fst', 'N/A'):.5f}")
    print(f"    NDD genes:           {ndd_s.get('mean_fst', 'N/A'):.5f}")
    print(f"    Background:          {bg_s.get('mean_fst', 'N/A'):.5f}")
    print(f"    SysNDD Definitive:   {def_s.get('mean_fst', 'N/A')}")
    if ndd_fst and bg_fst:
        print(f"    NDD/background ratio:{enrichment:.3f}x")
        p = summary.get("ndd_genes",{}).get("mannwhitney_p_vs_background", "N/A")
        print(f"    Mann-Whitney p:      {p}")

    if all_ml_results:
        print(f"\n  ML portability (SAS_AF prediction R²):")
        for cond_name, cond_res in all_ml_results.items():
            best = max(
                [(m, r) for m, r in cond_res.items()],
                key=lambda x: x[1].get("all", {}).get("r2", -1),
                default=(None, {})
            )
            if best[0]:
                all_r2  = best[1].get("all", {}).get("r2", "N/A")
                ndd_r2  = best[1].get("ndd_genes", {}).get("r2", "N/A")
                bg_r2   = best[1].get("background", {}).get("r2", "N/A")
                print(f"    [{cond_name}] best={best[0]} "
                      f"| all R²={all_r2} "
                      f"| NDD R²={ndd_r2} "
                      f"| BG R²={bg_r2}")

    print(f"\n  Paper interpretation:")
    if ndd_fst and bg_fst:
        if enrichment > 1.05:
            print(f"  EUR-SAS allele frequency divergence is {enrichment:.2f}x HIGHER")
            print(f"  in NDD gene regions than the genomic background,")
            print(f"  indicating greater population specificity in")
            print(f"  disease-relevant loci (Mann-Whitney p={p}).")
        elif enrichment < 0.95:
            print(f"  EUR-SAS allele frequency divergence is {enrichment:.2f}x LOWER")
            print(f"  in NDD gene regions, consistent with purifying")
            print(f"  selection constraining population divergence at")
            print(f"  functionally important NDD loci.")
        else:
            print(f"  NDD gene FST ≈ background FST (ratio={enrichment:.2f}x),")
            print(f"  suggesting uniform divergence across gene categories.")
    print(sep)


if __name__ == "__main__":
    main()