"""
check_genehub.py
-----------------
Run this BEFORE rebuilding anything. It inspects NDD_GeneHub_db.csv and tells
you whether the download is complete and whether gene_annotation_loader.py
can read it unchanged.

Run from ~/Documents/Data/:
    python3 check_genehub.py
"""
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).parent
GH = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "NDD_GeneHub_db.csv"
OLD_REF = ROOT / "unified_ndd_gene_reference.csv"

print("=" * 60)
print("  NDD GeneHub download check")
print("=" * 60)

if not GH.exists():
    raise SystemExit(f"Not found: {GH}")
print(f"\n  File      : {GH.name}")

# Try comma, then tab
for sep in (",", "\t"):
    df = pd.read_csv(GH, sep=sep, dtype=str)
    if df.shape[1] > 1:
        break
print(f"\n  Separator : {'tab' if sep == chr(9) else 'comma'}")
print(f"  Rows      : {len(df):,}")
print(f"  Columns   : {list(df.columns)}")
print("\n  First 3 rows:")
print(df.head(3).to_string())

# Gene symbol column (same logic the loader relies on)
sym_col = next((c for c in df.columns if c.strip().lower() in
                ("symbol", "gene", "gene_symbol", "gene symbol", "gene-symbol")), None)
if sym_col is None:
    print("\n  [PROBLEM] No obvious gene-symbol column.")
    print("  Paste the column list above to Claude so the loader can be adapted.")
    raise SystemExit(1)

syms = df[sym_col].dropna().astype(str).str.strip().str.upper()
print(f"\n  Symbol column      : '{sym_col}'")
print(f"  Unique gene symbols: {syms.nunique():,}")
if len(df) != syms.nunique():
    print(f"  Duplicate rows     : {len(df) - syms.nunique():,} "
          "(multi-row per gene; the loader de-duplicates via a set)")

# Tier column, if present
tier_col = next((c for c in df.columns if "tier" in c.lower()), None)
if tier_col:
    print(f"\n  Tier column '{tier_col}' value counts:")
    print(df[tier_col].value_counts(dropna=False).head(10).to_string())

# Completeness verdict
n = syms.nunique()
print("\n" + "-" * 60)
if n <= 100:
    print(f"  VERDICT: still truncated ({n} genes). Re-download the full file")
    print("  from https://ndd-genehub.org/downloads/ (not the on-screen table).")
elif n < 1000:
    print(f"  VERDICT: {n} genes. Larger than before but smaller than the")
    print("  published curated set (~1,500+). Check the downloads page for a")
    print("  full gene-level file.")
else:
    print(f"  VERDICT: {n:,} genes. Looks like the full download.")

# Overlap with the current reference
if OLD_REF.exists():
    ref = pd.read_csv(OLD_REF, dtype=str)
    old = set(ref["gene_symbol"].str.upper())
    new = set(syms)
    print(f"\n  Genes already in current 4,589-gene reference : {len(new & old):,}")
    print(f"  Genes that would be ADDED to the union        : {len(new - old):,}")
print("=" * 60)