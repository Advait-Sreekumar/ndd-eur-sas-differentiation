"""
denovo_db_loader.py
--------------------
Loads and parses DenovoDB v1.6.1 validated de novo variants.

WHY THIS MATTERS:
  Our Phase 1a de novo labels (18,423 candidates from CEPH 1463) are
  ~99% artifact calls — the ablation study showed AUC driven by quality
  metrics, not biology. DenovoDB provides REAL validated de novos from
  actual NDD patients. Integrating it does two things:

  1. Gene-level enrichment: which genes accumulate real validated de novos
     in NDD patients? We add `denovo_db_count` as a biological feature
     that directly captures gene-level mutation burden.

  2. Pipeline validation: can we detect DenovoDB-listed positions in
     the 1000G trio genotypes? (Likely few exact matches, but gene-level
     overlap is testable and meaningful.)

DenovoDB v1.6.1 reference: Turner et al. 2017, Nucleic Acids Research.
File: DenovoDB/denovo-db.non-ssc-samples.variants (1).tsv.gz
  "non-ssc" = Simons Simplex Collection samples excluded (avoids
  data leakage with downstream SPARK/SSC analyses).

NDD phenotypes in DenovoDB:
  autism_spectrum_disorder, intellectual_disability, epilepsy,
  schizophrenia, developmental_delay, and others.

Output:
  denovo_db_variants.csv       — all variants, NDD-filtered
  denovo_db_gene_counts.csv    — per-gene de novo burden table
  (both saved to ~/Documents/Data/)

Run:
  python3 denovo_db_loader.py
"""

import gzip
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT   = Path(__file__).parent
TSV_GZ      = DATA_ROOT / "DenovoDB" / "denovo-db.non-ssc-samples.variants (1).tsv.gz"
VCF_GZ      = DATA_ROOT / "DenovoDB" / "denovo-db.non-ssc-samples.variants (1).vcf.gz"
OUT_VARS    = DATA_ROOT / "denovo_db_variants.csv"
OUT_GENES   = DATA_ROOT / "denovo_db_gene_counts.csv"

# NDD phenotypes to keep (anything neurological / developmental)
NDD_KEYWORDS = {
    "autism", "asd", "intellectual_disability", "intellectual disability",
    "epilepsy", "developmental_delay", "developmental delay",
    "neurodevelopmental", "ndd", "schizophrenia", "adhd",
    "tourette", "rett", "angelman", "kabuki", "charge",
    "cortical_malformation", "brain_malformation",
}

# ---------------------------------------------------------------------------
# TSV parser
# ---------------------------------------------------------------------------

def load_denovo_db_tsv(path: Path = TSV_GZ) -> pd.DataFrame:
    """
    Parse the DenovoDB TSV.gz file.
    Prints actual column names first so we can adapt to any version.
    Returns a DataFrame of all variants with key columns normalised.
    """
    if not path.exists():
        raise FileNotFoundError(f"DenovoDB TSV not found: {path}")

    print(f"  Reading: {path.name}")

    # DenovoDB v1.6.1 file format:
    #   Line 0:  ##version=denovo-db.v.1.6.1   ← metadata comment, NOT header
    #   Line 1:  #SampleID\tStudyName\t...       ← real header (tab-separated, # prefix)
    #   Lines 2+: data rows
    #
    # pandas treats line 0 as the header by default, giving us one "column"
    # named '##version=...'. Fix: skiprows=1 skips the metadata line so
    # pandas reads the real header; then we strip the leading '#' from names.

    # Peek at lines for diagnostic output
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        meta_line  = fh.readline().rstrip("\n")
        header_line = fh.readline().rstrip("\n")
        sample_line = fh.readline().rstrip("\n")

    print(f"  Metadata line : {meta_line[:60]}")
    print(f"  Header line   : {header_line[:120]}...")
    print(f"  Sample row    : {sample_line[:120]}...")

    # Read file: skip the ##version line, use the #SampleID line as header
    df = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        low_memory=False,
        skiprows=1,          # skip ##version line
    )
    # Strip leading '#' from column names  (#SampleID → SampleID)
    df.columns = [c.lstrip("#").strip() for c in df.columns]

    raw_cols = list(df.columns)
    print(f"  Columns ({len(raw_cols)}): {raw_cols}")
    print(f"  Rows loaded: {len(df):,}")

    return df, raw_cols


def _find_col(df: pd.DataFrame, *candidates: str) -> str | None:
    """Return first column name from candidates that exists in df (case-insensitive)."""
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for cand in candidates:
        match = cols_lower.get(cand.lower().strip())
        if match:
            return match
    return None


def normalise_denovo_db(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise column names and types for DenovoDB.
    Identifies key columns flexibly regardless of exact column naming.
    """
    # Map to standard names.
    # DenovoDB v1.6.1 uses these specific column names (after stripping '#'):
    #   Chr, Position, AncestralAllele, DerivedAllele,
    #   HGMD_Symbol (gene), PrimaryPhenotype, Validation,
    #   StudyName, SampleID, Variant (type description)
    # We search broadly so the loader works across minor version differences.
    col_map = {
        "chrom":      _find_col(df, "chr", "chrom", "chromosome"),
        "pos":        _find_col(df, "position", "pos", "start", "chromstart"),
        "ref":        _find_col(df, "ancestralallele", "ancestral",
                                 "refallele", "ref", "ref_allele"),
        "alt":        _find_col(df, "derivedallele", "derived",
                                 "altallele", "alt", "alt_allele", "alternate"),
        "gene":       _find_col(df, "hgmd_symbol", "hgncsymbol", "hgmd",
                                 "gene", "genesymbol", "gene_name", "symbol"),
        "phenotype":  _find_col(df, "primaryphenotype", "primary_phenotype",
                                 "phenotype", "disease", "disorder", "trait"),
        "validation": _find_col(df, "validation", "validation_status", "validated"),
        "study":      _find_col(df, "studyname", "study_name",
                                 "study", "pubmedid", "pubmed_id", "pmid"),
        "sample_id":  _find_col(df, "sampleid", "sample_id", "sample", "individual"),
        "variant_type": _find_col(df, "variant", "varianttype", "variant_type",
                                   "type", "var_type"),
    }

    print("\n  Column mapping:")
    for standard, actual in col_map.items():
        status = "✓" if actual else "✗ NOT FOUND"
        print(f"    {standard:<15} ← {actual or status}")

    out = pd.DataFrame()
    for standard, actual in col_map.items():
        if actual:
            out[standard] = df[actual]
        else:
            out[standard] = ""

    # Normalise chromosome: strip 'chr' prefix → '1', '2', 'X'
    if "chrom" in out.columns:
        out["chrom"] = (
            out["chrom"].astype(str)
            .str.strip()
            .str.lower()
            .str.replace("^chr", "", regex=True)
        )

    # Position to int
    if "pos" in out.columns:
        out["pos"] = pd.to_numeric(out["pos"], errors="coerce")

    # Normalise phenotype: lowercase, underscore-separated
    if "phenotype" in out.columns:
        out["phenotype_norm"] = (
            out["phenotype"].astype(str)
            .str.lower()
            .str.strip()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^a-z0-9_]", "_", regex=True)
        )
    else:
        out["phenotype_norm"] = "unknown"

    return out


def filter_ndd_phenotypes(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only variants from NDD patients."""
    def is_ndd(pheno: str) -> bool:
        pheno_l = str(pheno).lower()
        return any(kw in pheno_l for kw in NDD_KEYWORDS)

    df["is_ndd_phenotype"] = df["phenotype_norm"].apply(is_ndd)
    ndd_df = df[df["is_ndd_phenotype"]].copy()
    return ndd_df, df


def build_gene_enrichment_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-gene de novo burden table from DenovoDB variants.

    Returns DataFrame:
        gene_symbol | total_de_novos | ndd_de_novos | top_phenotype
    Used as: add `denovo_db_count` feature to Model B.
    """
    if "gene" not in df.columns or df["gene"].isna().all():
        print("  [WARN] No gene column — cannot build enrichment table")
        return pd.DataFrame(columns=["gene_symbol", "total_de_novos",
                                     "ndd_de_novos", "top_phenotype"])

    # Total de novos per gene
    total = (
        df.groupby("gene").size()
        .reset_index(name="total_de_novos")
        .rename(columns={"gene": "gene_symbol"})
    )

    # NDD de novos per gene
    ndd = (
        df[df.get("is_ndd_phenotype", pd.Series(True, index=df.index))]
        .groupby("gene").size()
        .reset_index(name="ndd_de_novos")
        .rename(columns={"gene": "gene_symbol"})
    )

    # Top phenotype per gene
    top_pheno = (
        df.groupby("gene")["phenotype_norm"]
        .agg(lambda x: x.value_counts().index[0] if len(x) > 0 else "unknown")
        .reset_index()
        .rename(columns={"gene": "gene_symbol", "phenotype_norm": "top_phenotype"})
    )

    enrichment = (
        total
        .merge(ndd,       on="gene_symbol", how="left")
        .merge(top_pheno, on="gene_symbol", how="left")
    )
    enrichment["ndd_de_novos"] = enrichment["ndd_de_novos"].fillna(0).astype(int)
    enrichment = enrichment.sort_values("ndd_de_novos", ascending=False)

    return enrichment


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  DenovoDB v1.6.1 Loader — NDD Validated De Novos")
    print(sep)

    # ── Load TSV ──────────────────────────────────────────────────────────────
    print("\n[1] Loading DenovoDB TSV...")
    try:
        raw_df, cols = load_denovo_db_tsv()
    except FileNotFoundError as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

    # ── Normalise ─────────────────────────────────────────────────────────────
    print("\n[2] Normalising columns...")
    df = normalise_denovo_db(raw_df)

    # ── Filter NDD ────────────────────────────────────────────────────────────
    print("\n[3] Filtering NDD phenotypes...")
    ndd_df, all_df = filter_ndd_phenotypes(df)

    # Phenotype breakdown
    if "phenotype_norm" in all_df.columns:
        pheno_counts = all_df["phenotype_norm"].value_counts().head(15)
        print("\n  Phenotype breakdown (top 15):")
        for pheno, count in pheno_counts.items():
            flag = " ← NDD" if any(kw in str(pheno).lower() for kw in NDD_KEYWORDS) else ""
            print(f"    {pheno:<45} {count:>6,}{flag}")

    print(f"\n  Total variants:        {len(all_df):>8,}")
    print(f"  NDD variants:          {len(ndd_df):>8,}")
    print(f"  Non-NDD variants:      {len(all_df)-len(ndd_df):>8,}")

    # Chromosome distribution
    if "chrom" in ndd_df.columns:
        chrom_counts = ndd_df["chrom"].value_counts().head(5)
        print(f"\n  Top chromosomes (NDD de novos):")
        for chrom, cnt in chrom_counts.items():
            print(f"    chr{chrom}: {cnt:,}")

    # Variant type breakdown
    if "variant_type" in ndd_df.columns:
        vt_counts = ndd_df["variant_type"].value_counts()
        print(f"\n  Variant types (NDD de novos):")
        for vt, cnt in vt_counts.items():
            print(f"    {vt:<30} {cnt:,}")

    # ── Gene enrichment table ─────────────────────────────────────────────────
    print("\n[4] Building gene enrichment table...")
    gene_table = build_gene_enrichment_table(all_df)

    print(f"\n  Genes with at least 1 de novo: {len(gene_table):,}")
    print(f"\n  Top 20 NDD-enriched genes:")
    print(f"  {'Gene':<20} {'Total DNMs':>10} {'NDD DNMs':>10} {'Top phenotype'}")
    print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*30}")
    for _, row in gene_table.head(20).iterrows():
        print(f"  {str(row['gene_symbol']):<20} {int(row['total_de_novos']):>10,} "
              f"{int(row['ndd_de_novos']):>10,}  {str(row['top_phenotype'])[:30]}")

    # ── Save outputs ──────────────────────────────────────────────────────────
    print("\n[5] Saving outputs...")
    ndd_df.to_csv(OUT_VARS, index=False)
    print(f"  NDD variant table  → {OUT_VARS}  ({len(ndd_df):,} rows)")
    gene_table.to_csv(OUT_GENES, index=False)
    print(f"  Gene enrichment    → {OUT_GENES}  ({len(gene_table):,} genes)")

    # ── Summary for paper ─────────────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print("  PAPER-READY NUMBERS")
    print(f"{'─'*65}")
    print(f"  DenovoDB non-SSC NDD de novos:  {len(ndd_df):,}")
    if "gene" in ndd_df.columns and ndd_df["gene"].ne("").any():
        print(f"  Unique NDD genes with de novos: {ndd_df['gene'].nunique():,}")
    else:
        print("  [gene column not yet mapped — check column output above]")

    if len(ndd_df) > 0 and "chrom" in ndd_df.columns:
        snv_count = (ndd_df.get("variant_type", pd.Series()).str.upper() == "SNV").sum()
        print(f"  SNV fraction:                   {100*snv_count/len(ndd_df):.1f}%")

    print(f"\n  Next step: run trio_extraction_1000G.py to build the")
    print(f"  multi-trio feature table with DenovoDB gene enrichment features.")
    print(sep)


if __name__ == "__main__":
    main()