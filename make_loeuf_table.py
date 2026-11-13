#!/usr/bin/env python3
from pathlib import Path

import pandas as pd


SOURCE = Path.home() / "Documents/Data/gnomad.v2.1.1.lof_metrics.by_gene.txt"
OUTPUT = Path.home() / "Documents/Data/gnomad_lof_metrics.tsv"

df = pd.read_csv(SOURCE, sep="\t", low_memory=False)
required = {"gene", "oe_lof_upper"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"{SOURCE} is missing required columns: {', '.join(sorted(missing))}")

canonical_cols = [column for column in df.columns if "canonical" in column.lower()]
if canonical_cols:
    canonical_col = canonical_cols[0]
    canonical = df[canonical_col].astype(str).str.lower().isin(
        {"true", "1", "yes", "canonical"}
    )
    df = df.loc[canonical]
    print(f"Filtered to canonical transcripts using {canonical_col!r}.")
    if df["gene"].duplicated().any():
        raise ValueError(
            f"Filtering on {canonical_col!r} did not leave exactly one row per gene."
        )
else:
    transcript_cols = [column for column in df.columns if "transcript" in column.lower()]
    print(
        "No canonical-flag column is present; keeping the row with the smallest "
        "'oe_lof_upper' for each gene."
    )
    if transcript_cols:
        print(f"Transcript-related columns are not canonical flags: {transcript_cols}")

usable = df.dropna(subset=["gene", "oe_lof_upper"]).copy()
raw_duplicate_rows = df.loc[
    df["gene"].notna() & df["gene"].duplicated(keep=False)
]
if not raw_duplicate_rows.empty:
    ranges = raw_duplicate_rows.groupby("gene")["oe_lof_upper"].agg(
        min="min", max="max", usable_values="count"
    )
    comparable = ranges.loc[ranges["usable_values"] > 1]
    relative_range = (
        comparable["max"] - comparable["min"]
    ) / comparable["min"].abs() * 100
    print(
        f"Found {len(ranges)} duplicated source gene symbols; "
        f"{len(comparable)} have multiple usable LOEUF values. "
        f"Among those, {int((relative_range <= 1).sum())} with <=1% relative range; "
        f"{int((relative_range > 1).sum())} with >1%."
    )
    print("Min/max oe_lof_upper per duplicated gene:")
    print(ranges.to_string())

result = (
    usable.sort_values(["gene", "oe_lof_upper"])
    .drop_duplicates(subset=["gene"], keep="first")
    .loc[:, ["gene", "oe_lof_upper"]]
)
if result["gene"].duplicated().any():
    raise ValueError("The output still contains duplicate gene symbols.")
result.to_csv(OUTPUT, sep="\t", index=False)

print(
    f"Wrote {len(result)} rows ({result['gene'].nunique()} unique "
    f"gene symbols) to {OUTPUT}"
)
print(result.head(3).to_string(index=False))
