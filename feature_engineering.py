"""
feature_engineering.py
------------------------
Phase 1a core pipeline: builds the ML-ready feature table from:

    Source 1 — Zenodo trio VCF
        chrom, pos, ref, alt, father_gt, mother_gt, child_gt

    Source 2 — ndd_genes_grch37.bed  (from fetch_gene_coordinates.py)
        chrom, start, end, gene_symbol, strand

    Source 3 — unified_ndd_gene_reference.csv  (from gene_annotation_loader.py)
        gene_symbol, sfari_score, sysndd_category, n_sources, sources

Output columns (one row per variant):
    chrom, pos, ref, alt
    father_gt, mother_gt, child_gt
    variant_type                 SNV | INDEL | MULTIALLELIC | SPANNING_DEL
    is_de_novo                   True/False/pd.NA (NA = missing data)
    is_mendelian_consistent      True/False/pd.NA
    transmitted_allele_father    int | 'ambiguous' | None
    transmitted_allele_mother    int | 'ambiguous' | None
    gene_symbol                  str (empty if intergenic)
    is_ndd_gene                  bool
    sfari_score                  str (empty if not in SFARI)
    sysndd_category              str (Definitive/Moderate/Limited/Refuted/empty)
    n_sources                    int (0 if intergenic)
    sources                      str (empty if intergenic)

Run:
    python3 feature_engineering.py                        # test: first 50k variants (quality filtered)
    python3 feature_engineering.py --full                 # full WGS (~4-6M variants)
    python3 feature_engineering.py --n 100000            # custom limit
    python3 feature_engineering.py --no-filter           # disable quality filters (debug only)

Quality filters applied by default (standard de novo analysis thresholds):
    FILTER == PASS  — only variants the caller itself trusted
    GQ >= 20        — ≥99% genotype confidence for every trio member
    DP >= 10        — ≥10 reads per sample at this site

Without these filters the de novo count is ~1000x inflated by
noise calls in low-mappability regions (telomeres, repeats, etc.).

Requires: pandas, vcf_utils.py, Zenodo_trio_io.py, ndd_genes_grch37.bed,
          unified_ndd_gene_reference.csv
"""

import argparse
import sys
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# Add Data directory to path so sibling scripts are importable
DATA_ROOT = Path(__file__).parent
sys.path.insert(0, str(DATA_ROOT))

from vcf_utils import iter_vcf_records
from Zenodo_trio_io import (
    TRIO_VCF,
    identify_trio_roles,
    load_tfam,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

GENE_BED     = DATA_ROOT / "ndd_genes_grch37.bed"
NDD_GENE_CSV = DATA_ROOT / "unified_ndd_gene_reference.csv"
OUTPUT_CSV   = DATA_ROOT / "phase1a_feature_table.csv"
OUTPUT_PARQ  = DATA_ROOT / "phase1a_feature_table.parquet"

DEFAULT_N_VARIANTS = 50_000   # safe default for a quick test run

# ---------------------------------------------------------------------------
# Quality filter thresholds (standard de novo analysis values)
# ---------------------------------------------------------------------------

FILTER_PASS_ONLY = True   # skip variants where FILTER != PASS
MIN_GQ           = 20     # minimum genotype quality per sample (Phred-scaled)
MIN_DP           = 10     # minimum read depth per sample

# FILTER field values that count as "passing":
#   "PASS" — explicitly passed all filters
#   "."    — no filter applied (raw calls; include with a warning)
PASS_VALUES = {"PASS", "."}


# ---------------------------------------------------------------------------
# Genotype utilities
# ---------------------------------------------------------------------------

def parse_gt(gt_str: str) -> Optional[Tuple[int, ...]]:
    """
    Parse a VCF genotype string into a tuple of allele integers.

    '0/1'  → (0, 1)
    '1/1'  → (1, 1)
    '0|1'  → (0, 1)   (phased — treated as unphased)
    './.'  → None      (missing)
    '0/.'  → None      (partially missing — treated as missing)

    The integer represents the allele index:
        0 = reference allele
        1 = first alt allele
        2 = second alt allele  (multi-allelic sites)
    """
    if not gt_str or gt_str in (".", "./.", ".|."):
        return None

    gt_str = gt_str.replace("|", "/")
    parts = gt_str.split("/")

    alleles = []
    for p in parts:
        if p == ".":
            return None      # any missing component → fully missing
        try:
            alleles.append(int(p))
        except ValueError:
            return None

    return tuple(alleles)


def classify_variant(ref: str, alt: str) -> str:
    """
    Classify a variant by type.

    SPANNING_DEL  — alt contains '*' (overlapping deletion)
    MULTIALLELIC  — alt contains ',' (multiple alt alleles)
    SNV           — ref and alt are both single nucleotides
    INDEL         — anything else (insertion or deletion)
    """
    if "*" in alt:
        return "SPANNING_DEL"
    if "," in alt:
        return "MULTIALLELIC"
    if len(ref) == 1 and len(alt) == 1:
        return "SNV"
    return "INDEL"


def compute_is_de_novo(
    father_gt: str,
    mother_gt: str,
    child_gt: str,
) -> object:         # True | False | pd.NA
    """
    Strict de novo definition:
        - Both parents are homozygous reference (0/0)
        - Child carries at least one non-reference allele

    Returns pd.NA when any genotype is missing (cannot determine).
    """
    f = parse_gt(father_gt)
    m = parse_gt(mother_gt)
    c = parse_gt(child_gt)

    if f is None or m is None or c is None:
        return pd.NA

    # Both parents must be purely ref (every allele == 0)
    if not all(a == 0 for a in f) or not all(a == 0 for a in m):
        return False

    # Child must have at least one non-ref allele
    return any(a != 0 for a in c)


def compute_mendelian_consistent(
    father_gt: str,
    mother_gt: str,
    child_gt: str,
) -> object:         # True | False | pd.NA
    """
    Mendelian consistency check:
    Child's diploid genotype (c1, c2) must be explainable as:
        one allele inherited from father AND one allele from mother.

    We check both orientations:
        (c1 ∈ father_alleles  AND  c2 ∈ mother_alleles)
        OR
        (c2 ∈ father_alleles  AND  c1 ∈ mother_alleles)

    Returns pd.NA when any genotype is missing.
    """
    f = parse_gt(father_gt)
    m = parse_gt(mother_gt)
    c = parse_gt(child_gt)

    if f is None or m is None or c is None:
        return pd.NA

    f_alleles = set(f)
    m_alleles = set(m)
    c1, c2 = c[0], c[1]

    return (
        (c1 in f_alleles and c2 in m_alleles) or
        (c2 in f_alleles and c1 in m_alleles)
    )


def compute_transmitted_allele(
    parent_gt: str,
    child_gt: str,
) -> object:    # int | 'ambiguous' | None
    """
    Determine which allele a parent transmitted to the child.

    Homozygous parent (0/0 or 1/1):
        → The single allele they carry; always unambiguous.

    Heterozygous parent (0/1):
        → One of their alleles appears in the child; identify which.
        → If both appear in the child (both 0 and 1 present), return 'ambiguous'.
        → If neither appears, return None (Mendelian violation or error).

    Returns None also when any genotype is missing.
    """
    p = parse_gt(parent_gt)
    c = parse_gt(child_gt)

    if p is None or c is None:
        return None

    p_alleles = set(p)
    c_alleles = set(c)

    # Homozygous parent — always unambiguous
    if len(p_alleles) == 1:
        return list(p_alleles)[0]

    # Heterozygous parent — which of the two alleles does child carry?
    overlap = p_alleles & c_alleles

    if len(overlap) == 1:
        return list(overlap)[0]
    elif len(overlap) > 1:
        return "ambiguous"    # both parent alleles in child (e.g. parent 0/1, child 0/1)
    else:
        return None           # no overlap — Mendelian error or de novo


# ---------------------------------------------------------------------------
# Gene interval index
# ---------------------------------------------------------------------------

class GeneIntervalIndex:
    """
    Efficient O(log n) lookup: given (chrom, pos) → gene symbol.

    Chromosome normalisation:
        VCF may use 'chr1', gene BED from MyGene.info uses '1'.
        The index stores bare chromosome names ('1', 'X', …).
        Queries strip 'chr' prefix before lookup.

    Coordinate convention:
        MyGene.info genomic_pos_hg19 uses 1-based coordinates,
        consistent with VCF POS. No offset conversion needed.
    """

    def __init__(self, bed_df: pd.DataFrame) -> None:
        self._index: Dict[str, Tuple[list, list]] = {}
        bed_df = bed_df.copy()

        # Normalise chrom column (strip 'chr' if present)
        bed_df["_chrom_n"] = bed_df.iloc[:, 0].astype(str).str.lstrip("chr")

        for chrom, grp in bed_df.groupby("_chrom_n"):
            grp = grp.sort_values(grp.columns[1])  # sort by start
            starts = grp.iloc[:, 1].astype(int).tolist()
            entries = list(zip(
                grp.iloc[:, 1].astype(int),   # start
                grp.iloc[:, 2].astype(int),   # end
                grp.iloc[:, 3].astype(str),   # gene_symbol
            ))
            self._index[str(chrom)] = (starts, entries)

    def query(self, chrom: str, pos: int) -> str:
        """
        Return the gene symbol overlapping (chrom, pos), or '' if intergenic.
        When multiple NDD genes overlap, returns the first one found.
        """
        chrom_n = chrom.lstrip("chr")

        if chrom_n not in self._index:
            return ""

        starts, entries = self._index[chrom_n]

        # Rightmost gene starting at or before pos
        idx = bisect_right(starts, pos) - 1

        # Walk backwards: handle overlapping gene regions
        for i in range(idx, max(-1, idx - 50), -1):
            start, end, gene = entries[i]
            if start <= pos <= end:
                return gene
            # Genes are rarely longer than 3 Mb; stop searching further back
            if pos - start > 3_000_000:
                break

        return ""


# ---------------------------------------------------------------------------
# Quality filters
# ---------------------------------------------------------------------------

def _passes_quality(
    rec: dict,
    sample_ids: List[str],
    filter_pass: bool,
    min_gq: int,
    min_dp: int,
) -> tuple:
    """
    Return (passes: bool, reason: str) for a single VCF record.

    Checks (in order):
      1. FILTER field: must be PASS or '.' (unfiltered raw call)
      2. GQ (genotype quality): must be >= min_gq for ALL trio members
      3. DP (read depth):       must be >= min_dp for ALL trio members

    'sample_quality' must be present in `rec` (requires quality_fields=["GQ","DP"]
    in iter_vcf_records).
    """
    # ── FILTER field ──────────────────────────────────────────────────────────
    if filter_pass:
        filt = rec.get("filter", ".")
        if filt not in PASS_VALUES:
            return False, f"FILTER={filt}"

    sq = rec.get("sample_quality", {})
    if not sq:
        return True, "ok"     # no quality info available — let it through

    for sid in sample_ids:
        fields = sq.get(sid, {})

        # ── GQ ────────────────────────────────────────────────────────────────
        gq = fields.get("GQ")
        if gq is not None and gq < min_gq:
            return False, f"GQ={gq}<{min_gq} ({sid})"

        # ── DP ────────────────────────────────────────────────────────────────
        dp = fields.get("DP")
        if dp is not None and dp < min_dp:
            return False, f"DP={dp}<{min_dp} ({sid})"

    return True, "ok"


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def load_gene_bed(path: Path = GENE_BED) -> pd.DataFrame:
    """Load the NDD gene BED file (no header, 5 columns)."""
    if not path.exists():
        raise FileNotFoundError(
            f"Gene BED not found: {path}\n"
            "Run fetch_gene_coordinates.py first."
        )
    df = pd.read_csv(
        path,
        sep="\t",
        header=None,
        names=["chrom", "start", "end", "gene_symbol", "strand"],
        dtype={"chrom": str, "start": int, "end": int,
               "gene_symbol": str, "strand": str},
    )
    return df


def load_ndd_reference(path: Path = NDD_GENE_CSV) -> pd.DataFrame:
    """Load the unified NDD gene reference table."""
    if not path.exists():
        raise FileNotFoundError(
            f"NDD gene reference not found: {path}\n"
            "Run gene_annotation_loader.py first."
        )
    df = pd.read_csv(path, dtype=str)
    df["n_sources"] = pd.to_numeric(df.get("n_sources", 0), errors="coerce").fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def build_feature_table(
    n_variants: Optional[int] = DEFAULT_N_VARIANTS,
    vcf_path: Path = TRIO_VCF,
    filter_pass: bool = FILTER_PASS_ONLY,
    min_gq: int = MIN_GQ,
    min_dp: int = MIN_DP,
) -> pd.DataFrame:
    """
    Stream through the trio VCF and build the feature table.

    Parameters
    ----------
    n_variants    Number of variants to process (before filtering). None = all.
    vcf_path      Path to the trio VCF.gz.
    filter_pass   If True (default), skip variants where FILTER != PASS.
    min_gq        Minimum genotype quality per sample. Default 20.
    min_dp        Minimum read depth per sample. Default 10.

    Returns
    -------
    pd.DataFrame with all feature columns (only quality-passing variants).
    """

    # ── Load supporting data ─────────────────────────────────────────────────
    print("[1] Loading pedigree and identifying trio roles ...")
    roles = identify_trio_roles(load_tfam())
    father_id = roles["father"]
    mother_id = roles["mother"]
    child_id  = roles["child"]
    print(f"    Father={father_id}  Mother={mother_id}  Child={child_id}")

    print("[2] Loading NDD gene BED and building interval index ...")
    bed_df = load_gene_bed()
    gene_index = GeneIntervalIndex(bed_df)
    print(f"    {len(bed_df):,} NDD gene intervals indexed")

    print("[3] Loading unified NDD gene reference ...")
    ndd_ref = load_ndd_reference()
    # Build fast lookup dict: gene_symbol → annotation row
    ndd_lookup: Dict[str, dict] = {
        row["gene_symbol"]: row
        for _, row in ndd_ref.iterrows()
    }
    print(f"    {len(ndd_lookup):,} NDD genes in reference table")

    # ── Stream VCF and build rows ─────────────────────────────────────────────
    limit_str = f"{n_variants:,}" if n_variants else "ALL"
    print(f"\n[4] Streaming VCF: {vcf_path.name}  (limit={limit_str})")
    print(f"    Quality filters: FILTER=PASS={filter_pass}  GQ≥{min_gq}  DP≥{min_dp}")
    print("    Progress reported every 10,000 seen variants ...\n")

    trio_ids   = [father_id, mother_id, child_id]
    rows: List[dict] = []
    n_seen       = 0     # total records read from VCF
    n_kept       = 0     # records passing quality filters
    n_filtered   = 0     # records dropped by quality filters
    n_de_novo    = 0
    n_mend_error = 0

    # Request GQ and DP alongside GT from the VCF reader
    for rec in iter_vcf_records(
        vcf_path,
        n=n_variants,
        samples_of_interest=trio_ids,
        quality_fields=["GQ", "DP"],
    ):
        n_seen += 1

        # ── Quality gate ─────────────────────────────────────────────────────
        ok, reason = _passes_quality(rec, trio_ids, filter_pass, min_gq, min_dp)
        if not ok:
            n_filtered += 1
            if n_seen % 10_000 == 0:
                print(
                    f"    {n_seen:>8,} seen | {n_kept:>7,} kept | "
                    f"{n_filtered:>7,} filtered | de novo: {n_de_novo} | "
                    f"Mend.err: {n_mend_error}"
                )
            continue

        n_kept += 1

        chrom = rec["chrom"]
        pos   = rec["pos"]
        ref   = rec["ref"]
        alt   = rec["alt"]

        fgt = rec["genotypes"].get(father_id, "./.")
        mgt = rec["genotypes"].get(mother_id, "./.")
        cgt = rec["genotypes"].get(child_id,  "./.")

        # Extract quality values for output columns
        sq     = rec.get("sample_quality", {})
        f_gq   = sq.get(father_id, {}).get("GQ")
        m_gq   = sq.get(mother_id, {}).get("GQ")
        c_gq   = sq.get(child_id,  {}).get("GQ")
        f_dp   = sq.get(father_id, {}).get("DP")
        m_dp   = sq.get(mother_id, {}).get("DP")
        c_dp   = sq.get(child_id,  {}).get("DP")

        # ── Variant-level features ────────────────────────────────────────────
        vtype     = classify_variant(ref, alt)
        is_dn     = compute_is_de_novo(fgt, mgt, cgt)
        is_mend   = compute_mendelian_consistent(fgt, mgt, cgt)
        trans_f   = compute_transmitted_allele(fgt, cgt)
        trans_m   = compute_transmitted_allele(mgt, cgt)

        # ── Gene annotation ───────────────────────────────────────────────────
        gene_sym  = gene_index.query(chrom, pos)
        is_ndd    = gene_sym != ""

        ann = ndd_lookup.get(gene_sym, {})
        sfari_sc  = ann.get("sfari_score",     "")
        sysndd_ct = ann.get("sysndd_category", "")
        n_src     = ann.get("n_sources",       0)
        src_list  = ann.get("sources",         "")

        rows.append({
            "chrom":                     chrom,
            "pos":                       pos,
            "ref":                       ref,
            "alt":                       alt,
            "father_gt":                 fgt,
            "mother_gt":                 mgt,
            "child_gt":                  cgt,
            "father_gq":                 f_gq,
            "mother_gq":                 m_gq,
            "child_gq":                  c_gq,
            "father_dp":                 f_dp,
            "mother_dp":                 m_dp,
            "child_dp":                  c_dp,
            "variant_type":              vtype,
            "is_de_novo":                is_dn,
            "is_mendelian_consistent":   is_mend,
            "transmitted_allele_father": trans_f,
            "transmitted_allele_mother": trans_m,
            "gene_symbol":               gene_sym,
            "is_ndd_gene":               is_ndd,
            "sfari_score":               sfari_sc,
            "sysndd_category":           sysndd_ct,
            "n_sources":                 n_src,
            "sources":                   src_list,
        })

        if is_dn is True:
            n_de_novo += 1
        if is_mend is False:
            n_mend_error += 1

        if n_seen % 10_000 == 0:
            print(
                f"    {n_seen:>8,} seen | {n_kept:>7,} kept | "
                f"{n_filtered:>7,} filtered | de novo: {n_de_novo} | "
                f"Mend.err: {n_mend_error}"
            )

    print(f"\n    Done.")
    print(f"    Total seen:     {n_seen:>8,}")
    print(f"    Quality kept:   {n_kept:>8,}  ({100*n_kept/max(n_seen,1):.1f}%)")
    print(f"    Filtered out:   {n_filtered:>8,}  ({100*n_filtered/max(n_seen,1):.1f}%)")

    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def print_summary(df: pd.DataFrame) -> None:
    """Print a concise summary of the feature table."""
    sep = "─" * 65
    print(f"\n{sep}")
    print("  FEATURE TABLE SUMMARY")
    print(sep)
    print(f"  Total variants:              {len(df):>10,}")

    # Variant types
    vt = df["variant_type"].value_counts()
    print(f"\n  Variant types:")
    for vtype, count in vt.items():
        print(f"    {vtype:<20} {count:>10,}  ({100*count/len(df):.1f}%)")

    # De novo
    dn_true  = (df["is_de_novo"] == True).sum()
    dn_false = (df["is_de_novo"] == False).sum()
    dn_na    = df["is_de_novo"].isna().sum()
    # Quality field summary
    if "father_gq" in df.columns:
        print(f"\n  Median GQ  (father / mother / child): "
              f"{df['father_gq'].median():.0f} / "
              f"{df['mother_gq'].median():.0f} / "
              f"{df['child_gq'].median():.0f}")
        print(f"  Median DP  (father / mother / child): "
              f"{df['father_dp'].median():.0f} / "
              f"{df['mother_dp'].median():.0f} / "
              f"{df['child_dp'].median():.0f}")

    print(f"\n  De novo status:")
    print(f"    True  (de novo candidate):  {dn_true:>8,}")
    print(f"    False (inherited):           {dn_false:>8,}")
    print(f"    NA    (missing data):        {dn_na:>8,}")

    # Mendelian consistency
    mc_false = (df["is_mendelian_consistent"] == False).sum()
    mc_na    = df["is_mendelian_consistent"].isna().sum()
    print(f"\n  Mendelian violations:        {mc_false:>10,}")
    print(f"  Mendelian unknown (missing): {mc_na:>10,}")

    # NDD gene annotation
    ndd_vars = df["is_ndd_gene"].sum()
    print(f"\n  Variants in NDD genes:       {ndd_vars:>10,}  ({100*ndd_vars/len(df):.1f}%)")
    if ndd_vars > 0:
        ndd_df = df[df["is_ndd_gene"]]
        print(f"  Unique NDD genes hit:        {ndd_df['gene_symbol'].nunique():>10,}")

        # De novos in NDD genes
        dn_in_ndd = (ndd_df["is_de_novo"] == True).sum()
        print(f"  De novos in NDD genes:       {dn_in_ndd:>10,}")

        # Top NDD genes by variant count
        top_genes = ndd_df["gene_symbol"].value_counts().head(10)
        print(f"\n  Top 10 NDD genes by variant count:")
        for gene, cnt in top_genes.items():
            cat = df.loc[df["gene_symbol"] == gene, "sysndd_category"].iloc[0]
            sfari = df.loc[df["gene_symbol"] == gene, "sfari_score"].iloc[0]
            print(f"    {gene:<15} {cnt:>6,}  SysNDD={cat}  SFARI={sfari}")

    # SysNDD category breakdown (for NDD-gene variants)
    if ndd_vars > 0:
        sc = df[df["is_ndd_gene"]]["sysndd_category"].value_counts()
        print(f"\n  NDD-variant SysNDD category breakdown:")
        for cat, cnt in sc.items():
            print(f"    {cat:<20} {cnt:>8,}")

    print(sep)


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def save_outputs(df: pd.DataFrame) -> None:
    """Save feature table to CSV and Parquet."""
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n  Saved CSV     → {OUTPUT_CSV}")

    try:
        df.to_parquet(OUTPUT_PARQ, index=False)
        print(f"  Saved Parquet → {OUTPUT_PARQ}")
    except ImportError:
        print("  [INFO] pyarrow not installed — skipping Parquet. "
              "Install with: pip install pyarrow")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Build Phase 1a feature table from Zenodo trio VCF."
    )
    p.add_argument(
        "--full", action="store_true",
        help="Process the entire VCF (all chromosomes). Can take 5-15 minutes."
    )
    p.add_argument(
        "--n", type=int, default=None,
        help=f"Process first N raw VCF records (before filtering). "
             f"Default={DEFAULT_N_VARIANTS:,} when --full not set."
    )
    p.add_argument(
        "--no-filter", action="store_true",
        help="Disable quality filters (FILTER/GQ/DP). Debug only — "
             "produces ~1000x more false de novos."
    )
    p.add_argument(
        "--min-gq", type=int, default=MIN_GQ,
        help=f"Minimum genotype quality per sample. Default={MIN_GQ}."
    )
    p.add_argument(
        "--min-dp", type=int, default=MIN_DP,
        help=f"Minimum read depth per sample. Default={MIN_DP}."
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.full:
        n_variants = None
        print("Mode: FULL genome run (this may take 5–15 minutes for a WGS file)")
    elif args.n is not None:
        n_variants = args.n
        print(f"Mode: first {n_variants:,} raw VCF records")
    else:
        n_variants = DEFAULT_N_VARIANTS
        print(f"Mode: test run (first {n_variants:,} raw records). "
              "Use --full or --n N for more.")

    use_filter = not args.no_filter
    if not use_filter:
        print("WARNING: quality filters DISABLED — de novo counts will be inflated.")

    print("\n" + "=" * 65)
    print("  Phase 1a — Feature Engineering Pipeline")
    print("=" * 65 + "\n")

    df = build_feature_table(
        n_variants=n_variants,
        filter_pass=use_filter,
        min_gq=args.min_gq,
        min_dp=args.min_dp,
    )
    print_summary(df)
    save_outputs(df)

    print("\n  Feature table ready for ML pipeline.")
    print("=" * 65)


if __name__ == "__main__":
    main()