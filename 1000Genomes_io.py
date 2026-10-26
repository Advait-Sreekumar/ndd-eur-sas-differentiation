"""
1000Genomes_io.py
------------------
Read-only I/O helpers for 1000 Genomes Phase 3 data.

Expected layout (relative to this script in D:\\Data\\):

    1000 Genomes/
    ├── ALL.chr*.phase3_*.vcf.gz
    ├── ALL.chr*.phase3_*.vcf.gz.tbi
    ├── integrated_call_samples_v3.*.ALL.panel
    ├── integrated_call_samples_v3.*.ALL.ped
    └── 2013050220140625_related_individuals.txt.txt

NO PYSAM: uses Python's built-in gzip via vcf_utils.py.
Requires:  pandas, vcf_utils.py (in same directory)
"""

from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vcf_utils import validate_vcf, get_vcf_samples, iter_vcf_records


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent           # D:\Data\
KG_DIR    = DATA_ROOT / "1000 Genomes"     # D:\Data\1000 Genomes\

# These are discovered dynamically (filenames contain version/date strings)
def _find(pattern: str) -> Optional[Path]:
    """Return first file matching glob pattern in KG_DIR, or None."""
    hits = sorted(KG_DIR.glob(pattern))
    return hits[0] if hits else None


# ---------------------------------------------------------------------------
# Dynamic path getters (call at runtime, not import time)
# ---------------------------------------------------------------------------

def get_vcf_path() -> Path:
    path = _find("*.vcf.gz")
    if path is None:
        raise FileNotFoundError(f"No .vcf.gz found in {KG_DIR}")
    return path

def get_ped_path() -> Path:
    path = _find("*.ped")
    if path is None:
        raise FileNotFoundError(f"No .ped file found in {KG_DIR}")
    return path

def get_panel_path() -> Path:
    path = _find("*.panel")
    if path is None:
        raise FileNotFoundError(f"No .panel file found in {KG_DIR}")
    return path

def get_related_path() -> Optional[Path]:
    path = _find("*related_individuals*")
    return path


# ---------------------------------------------------------------------------
# Pedigree / panel loading
# ---------------------------------------------------------------------------

def load_ped(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse PLINK-style .ped file into:
        family_id | sample_id | father_id | mother_id | sex | phenotype
        [+ extra_0 ... extra_N for the 1000G extended columns]

    FIX 1: Variable column counts — read line-by-line and pad.
    FIX 2: 1000G PED has a text header row (Family / ID / Individual / ...).
            Detected by checking if the first token is not a valid family ID
            (starts with a letter AND matches known header keywords).
    """
    path = path or get_ped_path()
    if not path.exists():
        raise FileNotFoundError(f".ped not found: {path}")

    # Keywords that indicate a header row rather than a real sample
    HEADER_TOKENS = {"family", "familyid", "id", "individual", "paternal", "maternal"}

    rows = []
    skipped = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                skipped += 1
                continue

            # Skip text header rows: first token is a known header keyword
            if parts[0].lower().replace("_", "") in HEADER_TOKENS:
                print(f"  [INFO] Skipped PED header row: {parts[:4]}")
                continue

            rows.append(parts)

    if not rows:
        raise ValueError(f"No valid sample rows found in {path}")

    if skipped:
        print(f"  [WARN] Skipped {skipped} malformed row(s) in PED file.")

    max_cols = max(len(r) for r in rows)
    rows = [r + [""] * (max_cols - len(r)) for r in rows]

    base = ["family_id", "sample_id", "father_id", "mother_id", "sex", "phenotype"]
    n_extra = max(0, max_cols - len(base))
    col_names = base + [f"extra_{i}" for i in range(n_extra)]

    df = pd.DataFrame(rows, columns=col_names)

    for col in ["father_id", "mother_id"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": "0", "nan": "0", "NaN": "0"})
        )

    return df

def load_panel(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Parse the 1000G .panel file (tab-separated with header row).
    Renames columns: sample → sample_id, pop → population, super_pop → super_population.
    """
    path = path or get_panel_path()
    if not path.exists():
        raise FileNotFoundError(f".panel not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)
    rename = {
        "sample":     "sample_id",
        "pop":        "population",
        "super_pop":  "super_population",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


# ---------------------------------------------------------------------------
# VCF access
# ---------------------------------------------------------------------------

def list_vcf_samples(vcf_path: Optional[Path] = None) -> List[str]:
    """Return all sample IDs from the VCF header."""
    return get_vcf_samples(vcf_path or get_vcf_path())


def get_variant_genotypes(
    chrom: str,
    pos: int,
    sample_ids: Iterable[str],
    vcf_path: Optional[Path] = None,
) -> dict:
    """
    Scan the VCF for the first record at (chrom, pos) and return genotypes
    for the requested samples.

    NOTE: Without tabix, this scans sequentially until it finds the position.
    For a quick lookup in a large file, this may be slow — use a specific
    region-limited VCF or stream through it for batch queries.

    Returns: {sample_id: GT_string, ...}
    """
    vcf_path = vcf_path or get_vcf_path()
    sample_ids = set(sample_ids)

    for rec in iter_vcf_records(vcf_path, samples_of_interest=list(sample_ids)):
        if rec["chrom"] == chrom and rec["pos"] == pos:
            return {s: rec["genotypes"].get(s, "./.") for s in sample_ids}
        # Chromosomes are sorted; if we've passed the target chrom, stop.
        if rec["chrom"] > chrom:
            break
    return {s: None for s in sample_ids}


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  1000 Genomes Phase 3 — I/O Validation")
    print(sep)

    if not KG_DIR.exists():
        print(f"\n[ERROR] Directory not found: {KG_DIR}")
        print("  Check that '1000 Genomes' folder (with the space) is inside D:\\Data\\")
        return

    print(f"\n  Directory: {KG_DIR}")
    files_found = list(KG_DIR.iterdir())
    print(f"  Files present ({len(files_found)}):")
    for f in sorted(files_found):
        print(f"    {f.name}")

    # ── Related individuals (plain text) ────────────────────────────────────
    rel = get_related_path()
    if rel and rel.exists():
        print(f"\n[OK] Related individuals file: {rel.name}")
    else:
        print(f"\n[SKIP] Related individuals file not found")

    # ── PED ─────────────────────────────────────────────────────────────────
    print("\n[1] Loading .ped pedigree...")
    try:
        ped_df = load_ped()
        print(f"  {ped_df.shape[0]} rows × {ped_df.shape[1]} columns")
        print(ped_df.head(4).to_string(index=False))

        # Show available trios (samples where both parents are recorded)
        # father_id / mother_id are already normalised to "0" for missing
        trios = ped_df[
            (ped_df["father_id"] != "0") & (ped_df["father_id"] != "") &
            (ped_df["mother_id"] != "0") & (ped_df["mother_id"] != "")
        ]
        print(f"\n  Complete trios (child has both parents listed): {len(trios)}")
        if not trios.empty:
            print(trios[["family_id", "sample_id", "father_id", "mother_id"]].head(5).to_string(index=False))
    except (FileNotFoundError, Exception) as e:
        print(f"  [ERROR] {e}")

    # ── PANEL ────────────────────────────────────────────────────────────────
    print("\n[2] Loading .panel population info...")
    try:
        panel_df = load_panel()
        print(f"  {panel_df.shape[0]} samples × columns: {list(panel_df.columns)}")
        print(panel_df.head(4).to_string(index=False))

        if "super_population" in panel_df.columns:
            counts = panel_df["super_population"].value_counts()
            print(f"\n  Samples per super-population:\n{counts.to_string()}")
    except (FileNotFoundError, Exception) as e:
        print(f"  [ERROR] {e}")

    # ── VCF ─────────────────────────────────────────────────────────────────
    print("\n[3] Validating VCF...")
    try:
        vcf_path = get_vcf_path()
        print(f"  File: {vcf_path.name}")
        validate_vcf(vcf_path)
        samples = list_vcf_samples(vcf_path)
        print(f"  Total samples in VCF: {len(samples)}")
        print(f"  First 5: {samples[:5]}")

        print("\n  First 3 variants:")
        for rec in iter_vcf_records(vcf_path, n=3):
            print(f"    {rec['chrom']}:{rec['pos']}  {rec['ref']}>{rec['alt']}")
    except (FileNotFoundError, IOError, Exception) as e:
        print(f"  [ERROR] {e}")

    print(f"\n{sep}")
    print("  Done — 1000 Genomes I/O check complete.")
    print(sep)


if __name__ == "__main__":
    main()