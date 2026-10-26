"""
fetch_gene_coordinates.py
--------------------------
Fetches GRCh37/hg19 genomic coordinates for all NDD genes in
unified_ndd_gene_reference.csv using the MyGene.info REST API.

Uses the official `mygene` Python package (handles batching, retry,
rate-limiting automatically). Install once with:

    pip install mygene

Output: ndd_genes_grch37.bed (tab-separated, no header)
    chrom | start | end | gene_symbol | strand

Run from your Data directory with venv active:
    python3 fetch_gene_coordinates.py
"""

from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

try:
    import mygene
except ImportError:
    raise ImportError(
        "mygene package not installed.\n"
        "Run: pip install mygene\n"
        "Then re-run this script."
    )

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT    = Path(__file__).parent
NDD_GENE_CSV = DATA_ROOT / "unified_ndd_gene_reference.csv"
OUTPUT_BED   = DATA_ROOT / "ndd_genes_grch37.bed"


# ---------------------------------------------------------------------------
# Coordinate fetcher
# ---------------------------------------------------------------------------

def _best_position(pos_field) -> Optional[dict]:
    """
    genomic_pos_hg19 can be a single dict or a list of dicts (multi-locus genes).
    When multiple positions exist, prefer main chromosomes (1-22, X, Y)
    and take the longest span if still ambiguous.
    """
    if isinstance(pos_field, dict):
        return pos_field

    if isinstance(pos_field, list):
        main = {str(c) for c in range(1, 23)} | {"X", "Y", "MT"}
        candidates = [p for p in pos_field if str(p.get("chr", "")) in main]
        pool = candidates if candidates else pos_field
        return max(pool, key=lambda p: p.get("end", 0) - p.get("start", 0))

    return None


def fetch_ndd_gene_coordinates(symbols: List[str]) -> pd.DataFrame:
    """
    Query MyGene.info for hg19 coordinates of all NDD gene symbols.
    Returns a DataFrame: chrom, start, end, gene_symbol, strand
    """
    mg = mygene.MyGeneInfo()

    print(f"  Querying MyGene.info for {len(symbols):,} genes "
          f"(auto-batched in 1000-gene chunks) ...")

    results = mg.querymany(
        symbols,
        scopes="symbol",
        fields="symbol,genomic_pos_hg19",
        species="human",
        returnall=False,   # suppress duplicate/missing summary to stdout
        verbose=False,
    )

    rows = []
    not_found = []

    for r in results:
        if r.get("notfound") or "genomic_pos_hg19" not in r:
            not_found.append(r.get("query", ""))
            continue

        pos = _best_position(r["genomic_pos_hg19"])
        if pos is None:
            not_found.append(r.get("query", ""))
            continue

        chrom  = str(pos.get("chr",    "")).strip()
        start  = pos.get("start")
        end    = pos.get("end")
        strand = "+" if pos.get("strand", 1) == 1 else "-"
        symbol = str(r.get("symbol", r.get("query", ""))).upper()

        if chrom and start is not None and end is not None:
            rows.append({
                "chrom":       chrom,       # "1", "X" — no chr prefix
                "start":       int(start),
                "end":         int(end),
                "gene_symbol": symbol,
                "strand":      strand,
            })
        else:
            not_found.append(r.get("query", ""))

    bed_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["chrom", "start", "end", "gene_symbol", "strand"]
    )

    # Sort by chromosome then start position
    chrom_order = {str(c): i for i, c in enumerate(list(range(1, 23)) + ["X", "Y", "MT"])}
    bed_df["_order"] = bed_df["chrom"].map(chrom_order).fillna(99)
    bed_df = (bed_df
              .sort_values(["_order", "start"])
              .drop("_order", axis=1)
              .reset_index(drop=True))

    if not_found:
        clean = [g for g in not_found if g]   # drop empty strings
        print(f"\n  [{len(clean)} genes had no hg19 coordinates "
              f"(aliases, retired symbols, or non-coding RNAs)]")
        if len(clean) <= 20:
            print(f"  {clean}")
        else:
            print(f"  First 20: {clean[:20]} ...")

    return bed_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  NDD Gene Coordinate Fetcher (MyGene.info → GRCh37/hg19)")
    print(sep)

    if not NDD_GENE_CSV.exists():
        raise FileNotFoundError(
            f"Unified NDD gene reference not found: {NDD_GENE_CSV}\n"
            "Run gene_annotation_loader.py first."
        )

    ndd_df  = pd.read_csv(NDD_GENE_CSV)
    symbols = ndd_df["gene_symbol"].dropna().str.strip().str.upper().tolist()
    print(f"\n  Input : {len(symbols):,} NDD gene symbols from {NDD_GENE_CSV.name}")
    print(f"  Output: {OUTPUT_BED.name}\n")

    bed_df = fetch_ndd_gene_coordinates(symbols)

    # Save as standard BED (no header)
    bed_df.to_csv(OUTPUT_BED, sep="\t", index=False, header=False)

    # Summary
    print(f"\n{'─'*65}")
    print(f"  SUMMARY")
    print(f"{'─'*65}")
    print(f"  Genes queried:             {len(symbols):>6,}")
    print(f"  Coordinates found (hg19):  {len(bed_df):>6,}")
    print(f"  Chromosomes covered:       {bed_df['chrom'].nunique():>6}")
    print(f"  Saved to: {OUTPUT_BED}\n")

    print("  Preview (10 random NDD genes):")
    print(bed_df.sample(min(10, len(bed_df)), random_state=42).to_string(index=False))

    print(f"\n{sep}")
    print("  Done — run feature_engineering.py next.")
    print(sep)


if __name__ == "__main__":
    main()