"""
vcf_utils.py
-------------
Shared, Windows-compatible VCF reading utilities.

WHY NO PYSAM:
    pysam requires htslib (a C library) and cannot be pip-installed on Windows
    in a regular virtual environment. bgzip-compressed VCF files (.vcf.gz) are
    backward-compatible with standard gzip, so Python's built-in gzip module
    reads them perfectly for sequential access — no pysam needed.

    The only thing we lose vs. pysam: tabix-indexed *random* access to a
    specific region without reading the whole file. For the Zenodo trio (our
    primary dev file) and sequential iteration over 1000G, this is fine.

Imported by: Zenodo_trio_io.py, 1000Genomes_io.py, Platinum_io.py
Requires:    pandas (standard), no third-party C extensions
"""

import gzip
from pathlib import Path
from typing import Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_vcf(path: Path) -> bool:
    """
    Confirm a .vcf.gz file:
      - exists on disk
      - starts with a valid VCF fileformat header (##fileformat=VCF...)
      - has a .tbi index present (warns if not, but doesn't raise)

    Returns True on success, raises FileNotFoundError / IOError otherwise.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"VCF not found: {path}")
    if not str(path).endswith(".gz"):
        raise ValueError(f"Expected a .vcf.gz file, got: {path.name}")

    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            first = fh.readline()
        if not first.startswith("##fileformat=VCF"):
            raise IOError(
                f"File does not look like a VCF (first line: {first[:60]!r})"
            )
    except (OSError, EOFError) as exc:
        raise IOError(f"Cannot read {path.name}: {exc}") from exc

    tbi = Path(str(path) + ".tbi")
    if tbi.exists():
        print(f"  [OK] .tbi index found: {tbi.name}")
    else:
        print(f"  [WARN] No .tbi index — sequential access only (fine for iteration)")

    return True


# ---------------------------------------------------------------------------
# Header / sample parsing
# ---------------------------------------------------------------------------

def get_vcf_samples(path: Path) -> List[str]:
    """
    Return the ordered list of sample IDs from the #CHROM header line.
    Samples start at column index 9 (after CHROM POS ID REF ALT QUAL FILTER INFO FORMAT).
    """
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("#CHROM"):
                cols = line.rstrip("\n").split("\t")
                return cols[9:]      # everything after FORMAT
    return []


# ---------------------------------------------------------------------------
# Genotype helpers
# ---------------------------------------------------------------------------

def _normalize_gt(gt: str) -> str:
    """
    Normalize a raw GT string:
      '0|1' → '0/1'   (phased → unphased notation)
      '.'   → './.'   (missing single allele → standard missing diploid)
    """
    return gt.replace("|", "/")


def _parse_sample_gt(sample_field: str, format_keys: List[str]) -> str:
    """
    Extract and normalize the GT value from a sample column.
    Returns './.' if GT is absent or malformed.
    """
    parts = sample_field.split(":")
    sample_dict = dict(zip(format_keys, parts))
    raw_gt = sample_dict.get("GT", "./.")
    return _normalize_gt(raw_gt)


# ---------------------------------------------------------------------------
# Variant iteration
# ---------------------------------------------------------------------------

def _parse_extra_fields(
    sample_field: str,
    format_keys: List[str],
    extra: List[str],
) -> Dict[str, Optional[int]]:
    """
    Parse additional FORMAT fields (e.g. GQ, DP) from a sample column.
    Returns {field_name: int_value_or_None}.
    '.' (missing) and non-numeric values are returned as None.
    """
    parts = sample_field.split(":")
    sample_dict = dict(zip(format_keys, parts))
    result: Dict[str, Optional[int]] = {}
    for f in extra:
        raw = sample_dict.get(f, ".")
        try:
            result[f] = int(raw) if raw not in (".", "", None) else None
        except (ValueError, TypeError):
            result[f] = None
    return result


def iter_vcf_records(
    path: Path,
    n: Optional[int] = None,
    samples_of_interest: Optional[List[str]] = None,
    quality_fields: Optional[List[str]] = None,
) -> Iterator[Dict]:
    """
    Stream variant records from a .vcf.gz file.

    Yields one dict per variant:
        chrom, pos (int), id, ref, alt, qual, filter, genotypes
        [+ sample_quality if quality_fields is given]

    genotypes:      {sample_id: GT_string, ...}
    sample_quality: {sample_id: {field: int_or_None, ...}, ...}
                    Only present when quality_fields is not None.

    Parameters
    ----------
    path                 Path to the .vcf.gz file.
    n                    Stop after yielding this many records (None = all).
    samples_of_interest  If given, only include these samples in 'genotypes'.
                         Pass None to include all samples.
    quality_fields       Optional list of FORMAT fields to extract per sample
                         in addition to GT, e.g. ["GQ", "DP"].
                         When given, the yielded dict contains 'sample_quality'.
                         Existing callers that don't pass this get the old behaviour.
    """
    path = Path(path)
    count = 0
    samples: List[str] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")

            # ── meta-info lines ────────────────────────────────────────────
            if line.startswith("##"):
                continue

            # ── header line ────────────────────────────────────────────────
            if line.startswith("#CHROM"):
                cols = line.split("\t")
                samples = cols[9:]
                continue

            # ── stop if we hit the requested limit ─────────────────────────
            if n is not None and count >= n:
                break

            # ── data line ──────────────────────────────────────────────────
            fields = line.split("\t")
            if len(fields) < 9:
                continue                             # skip malformed rows

            chrom = fields[0]
            pos   = int(fields[1])
            vid   = fields[2]
            ref   = fields[3]
            alt   = fields[4]
            qual  = fields[5]
            filt  = fields[6]
            fmt   = fields[8].split(":")

            # build genotype dict (GT only — unchanged from original)
            gts: Dict[str, str] = {}
            sq:  Dict[str, dict] = {}

            for i, samp in enumerate(samples):
                if samples_of_interest is not None and samp not in samples_of_interest:
                    continue
                col_idx = 9 + i
                raw_sf  = fields[col_idx] if col_idx < len(fields) else "./."
                gts[samp] = _parse_sample_gt(raw_sf, fmt)

                if quality_fields:
                    sq[samp] = _parse_extra_fields(raw_sf, fmt, quality_fields)

            record: Dict = {
                "chrom":     chrom,
                "pos":       pos,
                "id":        vid,
                "ref":       ref,
                "alt":       alt,
                "qual":      qual,
                "filter":    filt,
                "genotypes": gts,
            }
            if quality_fields:
                record["sample_quality"] = sq

            yield record
            count += 1


# ---------------------------------------------------------------------------
# BED.gz iteration (for Platinum ConfidentRegions)
# ---------------------------------------------------------------------------

def iter_bed_records(
    path: Path,
    n: Optional[int] = None,
) -> Iterator[List[str]]:
    """
    Stream intervals from a bgzip BED file (.bed.gz).
    Skips comment/track/browser lines.
    Yields lists of tab-split fields: [chrom, start, end, ...]
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"BED file not found: {path}")

    count = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line or line.startswith(("#", "track", "browser")):
                continue
            if n is not None and count >= n:
                break
            yield line.split("\t")
            count += 1