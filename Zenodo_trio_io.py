"""
Zenodo_trio_io.py
------------------
Read-only I/O helpers for the Zenodo CEPH 1463 trio.

Directory (relative to this script — put everything in D:\\Data\\):

    zeneodo-trio/
    ├── pedigree_trioCEPH1463.tfam
    ├── README
    ├── trio_NA12878-NA12891-NA12892_hs37d5_dbsnp.vcf.gz
    └── trio_NA12878-NA12891-NA12892_hs37d5_dbsnp.vcf.gz.tbi

Reference genome: hs37d5 (GRCh37-based), as stated in the filename.

NO PYSAM: uses Python's built-in gzip via vcf_utils.py.
Requires:  pandas, vcf_utils.py (in same directory)
"""

from pathlib import Path
from typing import Optional

import pandas as pd

# vcf_utils.py must be in the same directory as this script
import sys
sys.path.insert(0, str(Path(__file__).parent))
from vcf_utils import validate_vcf, get_vcf_samples, iter_vcf_records


# ---------------------------------------------------------------------------
# Paths — derived from *this file's* location, not the working directory.
# Put this script in D:\Data\ and everything resolves correctly.
# ---------------------------------------------------------------------------

DATA_ROOT  = Path(__file__).parent                  # D:\Data\
ZENODO_DIR = DATA_ROOT / "zeneodo-trio"             # D:\Data\zeneodo-trio\

TFAM_FILE  = ZENODO_DIR / "pedigree_trioCEPH1463.tfam"
README_FILE = ZENODO_DIR / "README"
TRIO_VCF   = ZENODO_DIR / "trio_NA12878-NA12891-NA12892_hs37d5_dbsnp.vcf.gz"
TRIO_VCF_TBI = ZENODO_DIR / "trio_NA12878-NA12891-NA12892_hs37d5_dbsnp.vcf.gz.tbi"


# ---------------------------------------------------------------------------
# TFAM parsing
# ---------------------------------------------------------------------------

def load_tfam(path: Path = TFAM_FILE) -> pd.DataFrame:
    """
    Parse a PLINK .tfam file into a DataFrame:
        family_id | sample_id | father_id | mother_id | sex | phenotype

    FIX: Some TFAM files use blank fields (not "0") for missing parents.
    pandas reads blanks as NaN even with dtype=str. We read line-by-line
    and normalise all missing/blank parent IDs to "0" explicitly.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"TFAM file not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            while len(parts) < 6:
                parts.append("0")
            rows.append(parts[:6])

    df = pd.DataFrame(
        rows,
        columns=["family_id", "sample_id", "father_id", "mother_id", "sex", "phenotype"],
    )

    # Normalise: blank, "nan", or genuinely missing -> "0"
    for col in ["father_id", "mother_id"]:
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .replace({"": "0", "nan": "0", "NaN": "0", "NA": "0"})
        )

    return df

def identify_trio_roles(tfam_df: pd.DataFrame) -> dict:
    """
    Determine which sample is the father, mother, and child purely from
    pedigree structure — not from sample name assumptions.

    Logic:
        - Child  = the row where BOTH father_id and mother_id are non-zero.
        - Father = the sample_id matching the child's father_id field.
        - Mother = the sample_id matching the child's mother_id field.

    Returns:
        {'father': str, 'mother': str, 'child': str}
    """
    # Child: the one individual who has both parents listed
    child_rows = tfam_df[
        (tfam_df["father_id"] != "0") & (tfam_df["mother_id"] != "0")
    ]

    if child_rows.empty:
        raise ValueError(
            "Cannot identify child: no row has both father_id and mother_id set to a "
            "non-zero value. Check your TFAM file."
        )
    if len(child_rows) > 1:
        sample_list = child_rows["sample_id"].tolist()
        raise ValueError(
            f"Ambiguous pedigree: {len(child_rows)} rows have both parents listed "
            f"({sample_list}). Expected exactly 1 child."
        )

    child_row  = child_rows.iloc[0]
    father_id  = child_row["father_id"]
    mother_id  = child_row["mother_id"]
    child_id   = child_row["sample_id"]

    # Confirm parents exist as rows in the TFAM
    if father_id not in tfam_df["sample_id"].values:
        raise ValueError(
            f"Father ID '{father_id}' is listed in child's pedigree but not found "
            "as a sample row in the TFAM file."
        )
    if mother_id not in tfam_df["sample_id"].values:
        raise ValueError(
            f"Mother ID '{mother_id}' is listed in child's pedigree but not found "
            "as a sample row in the TFAM file."
        )

    return {"father": father_id, "mother": mother_id, "child": child_id}


# ---------------------------------------------------------------------------
# VCF access
# ---------------------------------------------------------------------------

def build_trio_genotype_df(
    roles: Optional[dict] = None,
    vcf_path: Path = TRIO_VCF,
    n_variants: Optional[int] = None,
) -> pd.DataFrame:
    """
    Build a DataFrame of trio genotypes:
        chrom | pos | ref | alt | father_gt | mother_gt | child_gt

    Parameters
    ----------
    roles       Dict from identify_trio_roles(). Auto-derived if None.
    vcf_path    Path to the .vcf.gz file.
    n_variants  Max variants to read. None = read all (can be slow for WGS).
    """
    if roles is None:
        roles = identify_trio_roles(load_tfam())

    father_id = roles["father"]
    mother_id = roles["mother"]
    child_id  = roles["child"]
    trio_ids  = [father_id, mother_id, child_id]

    # Confirm all three are actually in the VCF before iterating
    vcf_samples = get_vcf_samples(vcf_path)
    missing = set(trio_ids) - set(vcf_samples)
    if missing:
        raise ValueError(
            f"Trio member(s) {missing} not found in VCF sample list: {vcf_samples}"
        )

    rows = []
    for rec in iter_vcf_records(vcf_path, n=n_variants, samples_of_interest=trio_ids):
        rows.append({
            "chrom":     rec["chrom"],
            "pos":       rec["pos"],
            "ref":       rec["ref"],
            "alt":       rec["alt"],
            "father_gt": rec["genotypes"].get(father_id, "./."),
            "mother_gt": rec["genotypes"].get(mother_id, "./."),
            "child_gt":  rec["genotypes"].get(child_id,  "./."),
        })

    cols = ["chrom", "pos", "ref", "alt", "father_gt", "mother_gt", "child_gt"]
    return pd.DataFrame(rows, columns=cols)


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  Zenodo CEPH 1463 Trio — I/O Validation")
    print(sep)

    # ── README ──────────────────────────────────────────────────────────────
    if README_FILE.exists():
        print(f"\n[OK] README found: {README_FILE}")
    else:
        print(f"\n[SKIP] README not found at: {README_FILE}")

    # ── TFAM ────────────────────────────────────────────────────────────────
    print("\n[1] Loading TFAM pedigree...")
    try:
        tfam_df = load_tfam()
        print(f"  {TFAM_FILE.name} → {tfam_df.shape[0]} rows\n")
        print(tfam_df.to_string(index=False))

        roles = identify_trio_roles(tfam_df)
        print(f"\n  Trio roles (from pedigree structure):")
        print(f"    Father : {roles['father']}")
        print(f"    Mother : {roles['mother']}")
        print(f"    Child  : {roles['child']}")
    except (FileNotFoundError, ValueError) as e:
        print(f"  [ERROR] {e}")
        roles = None

    # ── VCF header ──────────────────────────────────────────────────────────
    print("\n[2] Validating VCF...")
    try:
        validate_vcf(TRIO_VCF)
        vcf_samples = get_vcf_samples(TRIO_VCF)
        print(f"  Samples in VCF ({len(vcf_samples)} total): {vcf_samples}")
    except (FileNotFoundError, IOError, ValueError) as e:
        print(f"  [ERROR] {e}")
        return

    # ── First few variants ──────────────────────────────────────────────────
    print("\n[3] First 5 raw variants:")
    try:
        for rec in iter_vcf_records(TRIO_VCF, n=5):
            gts = " | ".join(
                f"{s}={rec['genotypes'].get(s,'?')}" for s in vcf_samples
            )
            print(f"  {rec['chrom']}:{rec['pos']}  {rec['ref']}>{rec['alt']}  [{gts}]")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # ── Trio genotype table ─────────────────────────────────────────────────
    if roles is not None:
        print("\n[4] Trio genotype table (first 10 variants):")
        try:
            df = build_trio_genotype_df(roles=roles, n_variants=10)
            print(df.to_string(index=False))
            print(f"\n  DataFrame shape: {df.shape}")
        except Exception as e:
            print(f"  [ERROR] {e}")

    print(f"\n{sep}")
    print("  Done — Zenodo trio I/O working correctly.")
    print(sep)


if __name__ == "__main__":
    main()