"""
gene_level_fst.py
------------------
Gene-level companion to population_af_analysis.py.

WHY: the variant-level Mann-Whitney tests treat millions of neighbouring,
linked variants as independent, so their p-values overstate the evidence.
Here the unit of analysis is the GENE.

METHOD
  1. Per NDD gene (>= MIN_VARIANTS variants with defined FST on chr1/chr2):
     mean per-variant FST.
  2. Null: for each gene with m variants, draw N_WINDOWS random CONTIGUOUS
     windows of m variants from the non-NDD background on the same
     chromosome and take each window's mean FST. This matches gene size (in
     variant count) and keeps local correlation structure like a real gene.
     It does NOT match gene density, recombination rate, mutation rate or
     constraint, so it is a partial control, not a full matched analysis.
  3. diff = gene mean FST - mean of its null windows (negative = lower
     differentiation than a typical background window of the same size).
  4. Summaries per group of genes: median diff, share of genes with diff<0,
     sign test and Wilcoxon signed-rank test across genes.
  5. Between-group comparisons within NDD genes (Mann-Whitney on diff).

Run from ~/Documents/Data/ after population_af_analysis.py:
    python3 gene_level_fst.py
    python3 gene_level_fst.py --min-variants 100 --windows 2000
Output: gene_level_fst.csv (one row per gene) + printed summary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).parent
CSV = ROOT / "population_af_table.csv"
OUT = ROOT / "gene_level_fst.csv"
SEED = 42

ap = argparse.ArgumentParser()
ap.add_argument("--min-variants", type=int, default=50)
ap.add_argument("--windows", type=int, default=1000)
ap.add_argument("--csv", type=str, default=str(CSV))
ap.add_argument("--out", type=str, default=str(OUT))
args = ap.parse_args()
MIN_N, B = args.min_variants, args.windows

cols = ["chrom_label", "pos", "fst_eur_sas", "gene_symbol", "is_ndd_gene",
        "sysndd_score", "sfari_score_enc", "n_sources"]
df = pd.read_csv(args.csv, usecols=cols,
                 dtype={"chrom_label": str, "gene_symbol": str})
df = df[df["fst_eur_sas"].notna()].copy()
for c in ("is_ndd_gene", "sysndd_score", "sfari_score_enc", "n_sources"):
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
df = df.sort_values(["chrom_label", "pos"], kind="mergesort")
print(f"Variants with defined FST: {len(df):,}")

# Background cumulative sums per chromosome (non-NDD variants, position order)
bg = df[df["is_ndd_gene"] == 0]
cums = {ch: np.concatenate([[0.0], np.cumsum(g["fst_eur_sas"].to_numpy())])
        for ch, g in bg.groupby("chrom_label")}
print("Background variants per chromosome:",
      {ch: len(c) - 1 for ch, c in cums.items()})

# Per-gene summary
ndd = df[(df["is_ndd_gene"] == 1) & df["gene_symbol"].notna()
         & (df["gene_symbol"] != "")]
gene = (ndd.groupby("gene_symbol")
           .agg(chrom=("chrom_label", "first"), n_variants=("fst_eur_sas", "size"),
                mean_fst=("fst_eur_sas", "mean"), sysndd=("sysndd_score", "first"),
                sfari=("sfari_score_enc", "first"), n_sources=("n_sources", "first"))
           .reset_index())
n_all = len(gene)
gene = gene[gene["n_variants"] >= MIN_N].reset_index(drop=True)
print(f"NDD genes on analysed chromosomes: {n_all:,}; "
      f"with >= {MIN_N} variants: {len(gene):,}")

# Random contiguous background windows of matching size
rng = np.random.default_rng(SEED)
null_mean, null_med, null_sd, p_low = [], [], [], []
for _, r in gene.iterrows():
    cs = cums[r["chrom"]]
    m, N = int(r["n_variants"]), len(cs) - 1
    if N <= m:
        null_mean.append(np.nan); null_med.append(np.nan); null_sd.append(np.nan); p_low.append(np.nan)
        continue
    starts = rng.integers(0, N - m + 1, size=B)
    w = (cs[starts + m] - cs[starts]) / m
    null_mean.append(w.mean()); null_med.append(np.median(w)); null_sd.append(w.std(ddof=1))
    p_low.append((1 + (w <= r["mean_fst"]).sum()) / (B + 1))
gene["null_mean_fst"] = null_mean
gene["null_median_fst"] = null_med
gene["null_sd"] = null_sd
gene["p_window_low"] = p_low
gene["diff"] = gene["mean_fst"] - gene["null_mean_fst"]          # descriptive only (biased by skew)
gene["diff_med"] = gene["mean_fst"] - gene["null_median_fst"]    # effect size used below
gene["z"] = gene["diff"] / gene["null_sd"]
gene = gene.dropna(subset=["diff"])
gene.to_csv(args.out, index=False)
print(f"Saved per-gene table -> {args.out}  ({len(gene):,} genes)")

groups = {
    "all NDD genes":          gene["diff"].notna(),
    "SysNDD Definitive":      gene["sysndd"] == 4,
    "SysNDD Moderate":        gene["sysndd"] == 3,
    "SysNDD Limited":         gene["sysndd"] == 2,
    "SFARI score 1-2":        gene["sfari"] >= 3,
    "3+ databases":           gene["n_sources"] >= 3,
    "single-database":        gene["n_sources"] == 1,
}

sep = "-" * 92
print("\n" + sep)
print("GENE-LEVEL TESTS use each gene's percentile within its OWN null distribution")
print("(p_window_low = share of size-matched background windows with mean FST <= the")
print("gene's). Under 'no difference' it is uniform on (0,1): about 50% of genes fall")
print("below 0.5. This is valid for skewed FST, unlike comparing to the null MEAN.")
print(sep)
print(f"{'group':<19}{'genes':>6}{'med variants':>13}{'median gene FST':>16}"
      f"{'med diff*':>11}{'% p<0.5':>9}{'sign p':>10}{'Wilcoxon p':>12}")
for name, mask in groups.items():
    g = gene[mask]
    if len(g) < 5:
        print(f"{name:<19}{len(g):>6}  (too few genes)")
        continue
    k = int((g["p_window_low"] < 0.5).sum())
    sign_p = stats.binomtest(k, len(g), 0.5).pvalue
    try:
        wil_p = stats.wilcoxon(g["p_window_low"] - 0.5).pvalue
    except ValueError:
        wil_p = float("nan")
    print(f"{name:<19}{len(g):>6}{g['n_variants'].median():>13.0f}"
          f"{g['mean_fst'].median():>16.5f}{g['diff_med'].median():>11.5f}"
          f"{100 * k / len(g):>8.1f}%{sign_p:>10.2g}{wil_p:>12.2g}")
print("* med diff = median of (gene mean FST - median of its null windows)")

print("\n" + sep)
print("BETWEEN-GROUP (Mann-Whitney on p_window_low; genes are the unit)")
print(sep)
pairs = [("SysNDD Definitive", "SysNDD Limited"),
         ("SysNDD Definitive", "SysNDD Moderate"),
         ("SysNDD Moderate", "SysNDD Limited"),
         ("3+ databases", "single-database")]
for a, b in pairs:
    ga, gb = gene[groups[a]]["p_window_low"], gene[groups[b]]["p_window_low"]
    if len(ga) < 5 or len(gb) < 5:
        print(f"{a} vs {b}: too few genes")
        continue
    p = stats.mannwhitneyu(ga, gb, alternative="two-sided").pvalue
    print(f"{a:<19} (n={len(ga):>4}, median p_low {ga.median():.3f})  vs  "
          f"{b:<17} (n={len(gb):>4}, median p_low {gb.median():.3f})   p = {p:.3g}")
print(sep)
print("Caveat: windows match gene size (variant count) only; gene density,")
print("recombination, mutation rate and constraint are not controlled.")
print("Between-group p-values are uncorrected for multiple comparisons.")