"""
recompute_v9_numbers.py
------------------------
Recomputes, from the REBUILT population_af_table.csv (4,811-gene reference), the
manuscript quantities that depend on the gene set and need only per-variant
frequencies:

  * Table 1: n variants, plug-in FST*, Hudson FST and ratios per gene set
  * chromosome-specific plug-in ratios (V8 text, section 3.3)
  * share of low-MAF variants in NDD vs background (V8 text, section 3.3)
  * MAF-matched plug-in and Hudson ratios (reweighting reimplementation)
  * checks: rows fixed for opposite alleles; software versions

Definitions follow manuscript V8 section 2.4:
  plug-in  FST* = (pE-pS)^2 / [pE(1-pE) + pS(1-pS)], mean over variants where the
                  denominator is positive (this is the table's fst_eur_sas column)
  Hudson   per-variant numerator   (pE-pS)^2 - pE(1-pE)/(n1-1) - pS(1-pS)/(n2-1)
           per-variant denominator pE(1-pS) + pS(1-pE)
           FST = sum(numerator) / sum(denominator) over ALL variants in the set,
           n1 = 1006 (EUR haplotypes), n2 = 978 (SAS haplotypes)

NOT reproduced here (need the V8 analysis package and the gnomAD table): gene/locus
permutation tests, gene-matched controls, LOEUF, TOST, bootstrap intervals, and the
identity-baseline NDD/background split of R2.

MAF-matching here is a REIMPLEMENTATION: pooled MAF (EUR/SAS haplotype-weighted),
0.01 bins, stratified by chromosome; background variants are re-weighted to the NDD
distribution (the expected value of sampling). V8's own script was not available, so
small differences from V8's sampled result are possible. Check the plug-in numbers
against the values printed by population_af_analysis.py (0.00970, 0.01016, ...).

Run from ~/Documents/Data/:   python3 recompute_v9_numbers.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "population_af_table.csv"
N1, N2 = 1006, 978          # haplotype counts, manuscript section 2.4
BIN = 0.01

cols = ["chrom_label", "eur_af", "sas_af", "fst_eur_sas", "is_ndd_gene",
        "sfari_score_enc", "sysndd_score", "denovo_db_count"]
df = pd.read_csv(CSV, usecols=cols, dtype={"chrom_label": str})
for c in ("is_ndd_gene", "sfari_score_enc", "sysndd_score", "denovo_db_count"):
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)
print(f"Rows: {len(df):,}")

pE = df["eur_af"].to_numpy(float)
pS = df["sas_af"].to_numpy(float)
df["h_num"] = (pE - pS) ** 2 - pE * (1 - pE) / (N1 - 1) - pS * (1 - pS) / (N2 - 1)
df["h_den"] = pE * (1 - pS) + pS * (1 - pE)
pool = (N1 * pE + N2 * pS) / (N1 + N2)
df["maf"] = np.minimum(pool, 1 - pool)
df["bin"] = np.minimum((df["maf"] / BIN).astype(int), int(0.5 / BIN) - 1)
df["defined"] = df["fst_eur_sas"].notna()

SETS = {
    "Background (non-NDD)":   df["is_ndd_gene"] == 0,
    "All NDD genes":          df["is_ndd_gene"] == 1,
    "DenovoDB >=5 de novos":  df["denovo_db_count"] >= 5,
    "SFARI score 1-2":        df["sfari_score_enc"] >= 3,
    "SysNDD Definitive":      df["sysndd_score"] == 4,
}


def hudson(d, w=None):
    if w is None:
        return d["h_num"].sum() / d["h_den"].sum()
    return (w * d["h_num"]).sum() / (w * d["h_den"]).sum()


def plugin(d, w=None):
    d = d[d["defined"]]
    if w is None:
        return d["fst_eur_sas"].mean()
    w = w.loc[d.index]
    return (w * d["fst_eur_sas"]).sum() / w.sum()


sep = "-" * 84
print("\n" + sep + "\nTABLE 1 (all variants for n and Hudson; plug-in over variants where defined)\n" + sep)
bg = df[SETS["Background (non-NDD)"]]
bg_pl, bg_h = plugin(bg), hudson(bg)
print(f"{'Gene set':<24}{'n variants':>12}{'plug-in':>10}{'ratio':>8}{'Hudson':>10}{'ratio':>8}")
for name, mask in SETS.items():
    d = df[mask]
    pl, h = plugin(d), hudson(d)
    ratio = "" if name.startswith("Background") else f"{pl / bg_pl:>8.3f}"
    ratio_h = "" if name.startswith("Background") else f"{h / bg_h:>8.3f}"
    print(f"{name:<24}{len(d):>12,}{pl:>10.5f}{ratio:>8}{h:>10.5f}{ratio_h:>8}")

print("\n" + sep + "\nCHROMOSOME-SPECIFIC plug-in ratio (All NDD / background)\n" + sep)
for ch, g in df.groupby("chrom_label"):
    a = plugin(g[g["is_ndd_gene"] == 1]); b = plugin(g[g["is_ndd_gene"] == 0])
    print(f"chromosome {ch}: {a:.5f} / {b:.5f} = {a / b:.3f}")

print("\n" + sep + "\nSHARE OF LOW-MAF VARIANTS (pooled MAF < 0.05)\n" + sep)
for label, sub in (("all variants", df), ("variants with defined plug-in", df[df["defined"]])):
    s = [100 * (sub[sub["is_ndd_gene"] == k]["maf"] < 0.05).mean() for k in (1, 0)]
    print(f"{label:<30} NDD {s[0]:.1f}%   background {s[1]:.1f}%")

print("\n" + sep + "\nMAF-MATCHED ratios, All NDD vs background (re-weighting)\n" + sep)
ndd = df[df["is_ndd_gene"] == 1]
cnt_n = ndd.groupby(["chrom_label", "bin"]).size()
cnt_b = bg.groupby(["chrom_label", "bin"]).size()
wts = (cnt_n / cnt_b).replace([np.inf], np.nan).dropna()
key = pd.MultiIndex.from_frame(bg[["chrom_label", "bin"]])
w = pd.Series(wts.reindex(key).to_numpy(), index=bg.index).fillna(0.0)
print(f"background variants with positive weight: {(w > 0).sum():,} of {len(bg):,}")
print(f"MAF-matched plug-in ratio: {plugin(ndd) / plugin(bg, w):.3f}")
print(f"MAF-matched Hudson  ratio: {hudson(ndd) / hudson(bg, w):.3f}")

print("\n" + sep + "\nCHECKS\n" + sep)
opp = ((df["eur_af"] == 0) & (df["sas_af"] == 1)) | ((df["eur_af"] == 1) & (df["sas_af"] == 0))
both1 = (df["eur_af"] == 1) & (df["sas_af"] == 1)
both0 = (df["eur_af"] == 0) & (df["sas_af"] == 0)
print(f"EUR_AF=0 and SAS_AF=0 : {both0.sum():,}")
print(f"EUR_AF=1 and SAS_AF=1 : {both1.sum():,}")
print(f"fixed for OPPOSITE alleles (contribute to Hudson sums; undefined for plug-in): {opp.sum():,}")
print(f"plug-in undefined rows: {(~df['defined']).sum():,}")
try:
    import importlib.metadata as md
    pk = ["pandas", "numpy", "scipy", "scikit-learn", "xgboost", "matplotlib", "mygene", "openpyxl"]
    print("\nSoftware (record these in Methods 2.8):", sys.version.split()[0])
    for p in pk:
        try:
            print(f"  {p} {md.version(p)}")
        except md.PackageNotFoundError:
            print(f"  {p} (not installed)")
except Exception as e:
    print("version check failed:", e)
