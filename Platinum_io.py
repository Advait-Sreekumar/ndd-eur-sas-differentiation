"""
Platinum_io.py
---------------
Read-only I/O helpers for Illumina Platinum Genomes (hg38).

Expected layout (relative to this script in D:\\Data\\):

    Platinum/
    └── pg2017/
        └── hg38/
            ├── hybrid/
            │   ├── hg38.hybrid.vcf.gz
            │   ├── hg38.hybrid.vcf.gz.tbi
            │   ├── hg38.hybrid.bed.gz
            │   └── hg38.hybrid.bed.gz.tbi
            └── small_variants/
                ├── NA12877/
                └── NA12878/

NO PYSAM: uses Python's built-in gzip via vcf_utils.py.
Requires:  pandas, vcf_utils.py (in same directory)
"""

import gzip
import hashlib
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).parent))
from vcf_utils import validate_vcf, get_vcf_samples, iter_vcf_records, iter_bed_records


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_ROOT    = Path(__file__).parent                    # D:\Data\
PLATINUM_DIR = DATA_ROOT / "Platinum"
HG38_DIR     = PLATINUM_DIR / "pg2017" / "hg38"
HYBRID_DIR   = HG38_DIR / "hybrid"
SMALL_VARS   = HG38_DIR / "small_variants"

HYBRID_VCF   = HYBRID_DIR / "hg38.hybrid.vcf.gz"
HYBRID_VCF_TBI = HYBRID_DIR / "hg38.hybrid.vcf.gz.tbi"
HYBRID_BED   = HYBRID_DIR / "hg38.hybrid.bed.gz"
HYBRID_BED_TBI = HYBRID_DIR / "hg38.hybrid.bed.gz.tbi"

# Per-sample small-variants directories — enumerate contents at runtime
NA12877_DIR  = SMALL_VARS / "NA12877"
NA12878_DIR  = SMALL_VARS / "NA12878"


# ---------------------------------------------------------------------------
# BED utilities
# ---------------------------------------------------------------------------

def validate_bed(path: Path = HYBRID_BED) -> bool:
    """Confirm BED.gz file exists and is readable."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            fh.readline()
    except Exception as exc:
        raise IOError(f"Cannot read {path.name}: {exc}") from exc

    tbi = Path(str(path) + ".tbi")
    if tbi.exists():
        print(f"  [OK] .tbi index: {tbi.name}")
    else:
        print(f"  [WARN] No .tbi index for BED — sequential access only")
    return True


def write_bed_snippet(
    path: Path,
    out_path: Path,
    n: int = 100,
) -> Path:
    """
    Write the first `n` BED intervals from a .bed.gz to a plain .bed file
    for manual inspection or downstream tools that don't accept gzip.
    Does NOT decompress the whole file.
    """
    out_path = Path(out_path)
    written = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for fields in iter_bed_records(path, n=n):
            out.write("\t".join(fields) + "\n")
            written += 1
    print(f"  Wrote {written} intervals → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# MD5 verification (streamed — never loads the whole file into memory)
# ---------------------------------------------------------------------------

def verify_md5(
    file_path: Path,
    expected_md5: Optional[str] = None,
    manifest_path: Optional[Path] = None,
) -> bool:
    """
    Verify a file's MD5 checksum.

    Supply either:
      - expected_md5 directly (a hex string), or
      - manifest_path pointing to an md5sum.txt-style file with lines:
            <md5hex>  <filename>

    Returns True on match, raises ValueError on mismatch.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"File to verify not found: {file_path}")

    if expected_md5 is None:
        if manifest_path is None:
            raise ValueError("Supply either expected_md5 or manifest_path.")
        manifest_path = Path(manifest_path)
        if not manifest_path.exists():
            raise FileNotFoundError(f"MD5 manifest not found: {manifest_path}")
        expected_md5 = None
        with open(manifest_path, "r") as fh:
            for line in fh:
                parts = line.strip().split()
                if len(parts) >= 2:
                    fname = parts[1].lstrip("*")    # md5sum prefixes binaries with *
                    if Path(fname).name == file_path.name:
                        expected_md5 = parts[0].strip()
                        break
        if expected_md5 is None:
            raise ValueError(
                f"'{file_path.name}' not found in manifest {manifest_path}"
            )

    hasher = hashlib.md5()
    with open(file_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            hasher.update(chunk)
    actual = hasher.hexdigest()

    if actual.lower() != expected_md5.lower():
        raise ValueError(
            f"MD5 MISMATCH for {file_path.name}\n"
            f"  expected : {expected_md5}\n"
            f"  actual   : {actual}"
        )
    return True


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  Platinum Genomes (hg38) — I/O Validation")
    print(sep)

    if not PLATINUM_DIR.exists():
        print(f"\n[ERROR] Directory not found: {PLATINUM_DIR}")
        return

    # ── Check expected files ─────────────────────────────────────────────────
    expected = [HYBRID_VCF, HYBRID_VCF_TBI, HYBRID_BED, HYBRID_BED_TBI]
    print("\n  Expected files in hybrid/:")
    for f in expected:
        status = "[OK]     " if f.exists() else "[MISSING]"
        print(f"  {status} {f.name}")

    # ── Hybrid VCF ───────────────────────────────────────────────────────────
    if HYBRID_VCF.exists():
        print("\n[1] Validating hybrid VCF...")
        try:
            validate_vcf(HYBRID_VCF)
            samples = get_vcf_samples(HYBRID_VCF)
            print(f"  Samples in VCF: {samples if samples else '(none — reference VCF)'}")

            print("\n  First 5 variants:")
            for rec in iter_vcf_records(HYBRID_VCF, n=5):
                print(
                    f"    {rec['chrom']}:{rec['pos']}  "
                    f"{rec['ref']}>{rec['alt']}  "
                    f"FILTER={rec['filter']}"
                )
        except (FileNotFoundError, IOError) as e:
            print(f"  [ERROR] {e}")
    else:
        print(f"\n[SKIP] Hybrid VCF not found: {HYBRID_VCF}")

    # ── Hybrid BED ───────────────────────────────────────────────────────────
    if HYBRID_BED.exists():
        print("\n[2] Reading confident regions BED...")
        try:
            validate_bed(HYBRID_BED)
            print("  First 5 intervals (chrom, start, end, ...):")
            for fields in iter_bed_records(HYBRID_BED, n=5):
                print(f"    {fields[:3]}")
        except (FileNotFoundError, IOError) as e:
            print(f"  [ERROR] {e}")
    else:
        print(f"\n[SKIP] Hybrid BED not found: {HYBRID_BED}")

    # ── Small variants directories ────────────────────────────────────────────
    print("\n[3] Small variants directories:")
    for name, d in [("NA12877", NA12877_DIR), ("NA12878", NA12878_DIR)]:
        if d.exists():
            contents = list(d.iterdir())
            print(f"  {name}/  ({len(contents)} files):")
            for f in sorted(contents)[:5]:
                print(f"    {f.name}")
        else:
            print(f"  {name}/  [NOT FOUND] — {d}")

    print(f"\n{sep}")
    print("  Done — Platinum I/O check complete.")
    print(sep)


if __name__ == "__main__":
    main()