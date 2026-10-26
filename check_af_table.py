"""
check_af_table.py
------------------
Three quick diagnostics on population_af_table.csv (run after
population_af_analysis.py). Takes ~1-2 minutes.

1. How many of the 13.5M rows have an undefined FST, and why.
2. EUR-only Ridge portability (chr1 -> chr2) restricted to variants that are
   polymorphic in at least one of EUR/SAS, NDD vs background.
3. Which genes make up the SysNDD Moderate tier (it came out ABOVE background).

Run from ~/Documents/Data/:
    python3 check_af_table.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).parent
CSV = ROOT / "population_af_table.csv"

cols = ["chrom_label", "eur_af", "sas_af", "fst_eur_sas", "is_ndd_gene",
        "sysndd_score", "gene_symbol"]
df = pd.read_csv(CSV, usecols=cols, dtype={"chrom_label": str, "gene_symbol": str})
df["is_ndd_gene"] = pd.to_numeric(df["is_ndd_gene"], errors="coerce").fillna(0).astype(int)
df["sysndd_score"] = pd.to_numeric(df["sysndd_score"], errors="coerce").fillna(0).astype(int)

sep = "-" * 62

# ── 1. Undefined FST ────────────────────────────────────────────────────────
print(sep)
print("1. FST coverage")
print(sep)
n = len(df)
defined = df["fst_eur_sas"].notna()
print(f"  Rows with valid EUR+SAS AF : {n:,}")
print(f"  FST defined                : {defined.sum():,}  ({100*defined.mean():.1f}%)")
und = df[~defined]
both0 = ((und["eur_af"] == 0) & (und["sas_af"] == 0)).sum()
both01 = (und["eur_af"].isin([0, 1]) & und["sas_af"].isin([0, 1])).sum()
print(f"  FST undefined              : {len(und):,}")
print(f"    of which EUR_AF=0 and SAS_AF=0     : {both0:,}")
print(f"    of which both AFs in {{0, 1}}        : {both01:,}")

# ── 2. Restricted portability ───────────────────────────────────────────────
print("\n" + sep)
print("2. EUR-only Ridge, chr1 train -> chr2 test")
print("   (a) all rows   (b) polymorphic in EUR or SAS only")
print(sep)


def run(label, d):
    tr = d[d["chrom_label"] == "1"]
    te = d[d["chrom_label"] == "2"]
    m = Ridge(alpha=1.0).fit(tr[["eur_af"]].values, tr["sas_af"].values)
    pred = m.predict(te[["eur_af"]].values)
    te = te.assign(pred=pred)
    print(f"  {label}: train n={len(tr):,}  test n={len(te):,}")
    for name, sub in (("all", te), ("NDD", te[te.is_ndd_gene == 1]),
                      ("background", te[te.is_ndd_gene == 0])):
        print(f"    {name:<11} n={len(sub):>9,}  R2={r2_score(sub.sas_af, sub.pred):.4f}"
              f"  MAE={mean_absolute_error(sub.sas_af, sub.pred):.4f}")


run("(a) all rows", df)
poly = df[(df["eur_af"] > 0) | (df["sas_af"] > 0)]
run("(b) polymorphic in EUR or SAS", poly)

# ── 3. SysNDD Moderate tier ─────────────────────────────────────────────────
print("\n" + sep)
print("3. SysNDD tiers: genes and per-gene FST")
print(sep)
for label, score in (("Definitive", 4), ("Moderate", 3), ("Limited", 2)):
    t = df[(df.sysndd_score == score) & defined]
    g = (t.groupby("gene_symbol")["fst_eur_sas"]
          .agg(n="size", mean_fst="mean").sort_values("n", ascending=False))
    print(f"\n  {label}: {len(t):,} variants in {len(g):,} genes; "
          f"variant-weighted mean FST {t.fst_eur_sas.mean():.5f}; "
          f"median of per-gene mean FST {g.mean_fst.median():.5f}")
    if label == "Moderate":
        print("  Top 8 genes by variant count:")
        print(g.head(8).round(5).to_string())
        top3 = g.head(3).index
        rest = t[~t.gene_symbol.isin(top3)]
        print(f"  Mean FST excluding top 3 genes: {rest.fst_eur_sas.mean():.5f} "
              f"({len(rest):,} variants)")
print(sep)