"""
gene_annotation_loader.py
--------------------------
Loads all NDD gene databases and builds one unified reference table.

Input files (all at D:\\Data\\ — same folder as this script):

    SFARI-Gene_genes_07-12-2026release_08-11-2026export.csv
    DDG2P.tsv
    NDD_GeneHub_db.csv
    sysndd_entity_table.xlsx
    sysndd_gene_table.xlsx

Output: unified_ndd_gene_reference.csv (saved in D:\\Data\\)

    gene_symbol | in_sfari | sfari_score | in_ddg2p | ddg2p_confidence |
    in_sysndd | sysndd_category | in_ndd_genehub | n_sources | sources

Requires: pandas, openpyxl (for .xlsx reading)
"""

from pathlib import Path
from typing import Dict, Optional, Set, Tuple
import pandas as pd


# ---------------------------------------------------------------------------
# Paths — all gene files live at D:\Data\ (same dir as this script)
# ---------------------------------------------------------------------------

DATA_ROOT = Path(__file__).parent

SFARI_CSV        = DATA_ROOT / "SFARI-Gene_genes_07-12-2026release_08-11-2026export.csv"
DDG2P_TSV        = DATA_ROOT / "DDG2P.tsv"
NDD_GENEHUB_CSV  = DATA_ROOT / "NDD_GeneHub_db.csv"
SYSNDD_ENT_XLSX  = DATA_ROOT / "sysndd_entity_table.xlsx"
SYSNDD_GENE_XLSX = DATA_ROOT / "sysndd_gene_table.xlsx"
AUTDB_CSV        = DATA_ROOT / "autdb_human_gene.csv"    # AutDB Human Gene module

OUTPUT_CSV = DATA_ROOT / "unified_ndd_gene_reference.csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_col(df: pd.DataFrame, *keywords: str, required: bool = True) -> Optional[str]:
    """
    Find the first column whose name contains ALL the given keywords
    (case-insensitive substring match). Returns None if not found.
    """
    cols_lower = {c.lower(): c for c in df.columns}
    for col_lower, col_orig in cols_lower.items():
        if all(kw.lower() in col_lower for kw in keywords):
            return col_orig
    if required:
        raise KeyError(
            f"Could not find a column containing all of {keywords} "
            f"in: {list(df.columns)}"
        )
    return None


# ---------------------------------------------------------------------------
# Individual loaders
# ---------------------------------------------------------------------------

def load_sfari(path: Path = SFARI_CSV) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load SFARI Gene CSV.
    Returns: (set of gene symbols, dict {symbol: score})

    SFARI score values: '1', '2', '3', 'S' (syndromic).
    Lower number = higher confidence.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"SFARI CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    print(f"    Columns detected: {list(df.columns)}")

    # Try to find gene-symbol column (SFARI uses 'gene-symbol')
    sym_col = _find_col(df, "gene", "symbol", required=False) or df.columns[0]
    score_col = _find_col(df, "score", required=False)

    symbols = df[sym_col].dropna().str.strip().str.upper()
    if score_col:
        scores = df[score_col].fillna("").astype(str).str.strip()
        score_map = dict(zip(symbols, scores))
    else:
        score_map = {s: "" for s in symbols}

    return set(symbols), score_map


def load_ddg2p(path: Path = DDG2P_TSV) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load DDG2P TSV (from Gene2Phenotype / PanelApp).
    Returns: (set of gene symbols, dict {symbol: highest confidence category})

    Confidence levels (highest to lowest): Definitive > Strong > Moderate > Limited
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"DDG2P TSV not found: {path}")

    df = pd.read_csv(path, sep="\t", dtype=str)
    print(f"    Columns detected: {list(df.columns)}")

    sym_col  = _find_col(df, "gene", required=False)
    # Try a few common column name patterns for the confidence level
    conf_col = (
        _find_col(df, "category",    required=False) or
        _find_col(df, "confidence",  required=False) or
        _find_col(df, "ddd",         required=False)
    )

    if sym_col is None:
        sym_col = df.columns[0]

    # Confidence ordering for deduplication (lower = better)
    _ORDER = {"definitive": 0, "strong": 1, "moderate": 2, "limited": 3}

    records = []
    for _, row in df.iterrows():
        sym = str(row[sym_col]).strip().upper()
        if not sym or sym == "NAN":
            continue
        conf = str(row[conf_col]).strip() if conf_col else "unknown"
        records.append((sym, conf))

    # Keep highest confidence per gene symbol
    best: Dict[str, Tuple[str, int]] = {}
    for sym, conf in records:
        order = _ORDER.get(conf.lower(), 99)
        if sym not in best or order < best[sym][1]:
            best[sym] = (conf, order)

    genes = set(best.keys())
    conf_map = {sym: best[sym][0] for sym in best}
    return genes, conf_map


def load_sysndd_genes(path: Path = SYSNDD_GENE_XLSX) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load SysNDD gene-level data.

    The gene table (sysndd_gene_table.xlsx) only has gene symbols — no
    category column. The confidence category (Definitive/Moderate/Limited/
    Refuted) lives in the entity table (sysndd_entity_table.xlsx).
    We load both and join on gene symbol, taking the best category per gene.
    """
    path = Path(path)
    entity_path = path.parent / "sysndd_entity_table.xlsx"

    if not path.exists():
        raise FileNotFoundError(f"SysNDD gene table not found: {path}")

    # ── Gene table: symbols ───────────────────────────────────────────────────
    df_genes = pd.read_excel(path, dtype=str)
    print(f"    Gene table columns: {list(df_genes.columns)}")

    sym_col = (
        _find_col(df_genes, "symbol", required=False) or
        _find_col(df_genes, "gene",   required=False) or
        df_genes.columns[0]
    )
    symbols = df_genes[sym_col].dropna().str.strip().str.upper()

    # ── Entity table: category per gene ──────────────────────────────────────
    cat_map: Dict[str, str] = {}
    if entity_path.exists():
        df_ent = pd.read_excel(entity_path, dtype=str)
        print(f"    Entity table columns: {list(df_ent.columns)}")

        ent_sym_col = (
            _find_col(df_ent, "symbol",   required=False) or
            _find_col(df_ent, "gene",     required=False)
        )
        ent_cat_col = (
            _find_col(df_ent, "category",    required=False) or
            _find_col(df_ent, "class",       required=False) or
            _find_col(df_ent, "entity_cat",  required=False)
        )

        if ent_sym_col and ent_cat_col:
            print(f"    Category column in entity table: '{ent_cat_col}'")
            unique_cats = df_ent[ent_cat_col].dropna().unique()[:6]
            print(f"    Unique category values (sample): {list(unique_cats)}")

            # Take best (highest confidence) category per gene
            _ORDER = {"definitive": 0, "moderate": 1, "limited": 2, "refuted": 3}
            for _, row in df_ent.iterrows():
                sym = str(row[ent_sym_col]).strip().upper() if ent_sym_col else ""
                cat = str(row[ent_cat_col]).strip() if ent_cat_col else ""
                if not sym or sym == "NAN":
                    continue
                order = _ORDER.get(cat.lower(), 99)
                existing = cat_map.get(sym, "")
                existing_order = _ORDER.get(existing.lower(), 99)
                if order < existing_order:
                    cat_map[sym] = cat
        else:
            print("    [WARN] Could not find symbol/category columns in entity table")
    else:
        print(f"    [WARN] Entity table not found at {entity_path} — "
              "categories will be blank")

    return set(symbols), cat_map


def load_ndd_genehub(path: Path = NDD_GENEHUB_CSV) -> Set[str]:
    """
    Load NDD GeneHub CSV.
    Returns: set of gene symbols.

    (Column structure unknown — we detect the gene symbol column automatically
    and print what we found so you can verify.)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"NDD GeneHub CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    print(f"    Columns detected: {list(df.columns)}")

    # Best-effort: look for a gene/symbol column
    sym_col = (
        _find_col(df, "gene", "symbol", required=False) or
        _find_col(df, "gene", "name",   required=False) or
        _find_col(df, "symbol",          required=False) or
        _find_col(df, "gene",            required=False)
    )

    if sym_col is None:
        sym_col = df.columns[0]
        print(f"    [WARN] Could not detect gene symbol column — using first column: '{sym_col}'")
        print(f"    [WARN] First 5 values: {df[sym_col].head().tolist()}")

    symbols = df[sym_col].dropna().str.strip().str.upper()
    return set(symbols)


def load_autdb(path: Path = AUTDB_CSV) -> Tuple[Set[str], Dict[str, str]]:
    """
    Load AutDB Human Gene module CSV.
    Returns: (set of gene symbols, dict {symbol: primary_disorder})

    AutDB format (columns detected from actual file):
        Gene Symbol | Report Type | Report PMID | Primary Disorder(s) | Associated Disorder(s)

    One gene appears on MULTIPLE rows (one per publication/report type).
    We deduplicate to one row per gene and collect the primary disorder.
    AutDB is independently curated from SFARI and provides additional
    ASD/NDD gene evidence — particularly for genes not yet in other databases.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"AutDB CSV not found: {path}")

    df = pd.read_csv(path, dtype=str)
    print(f"    Columns detected: {list(df.columns)}")
    print(f"    Raw rows (one per report): {len(df):,}")

    # Identify gene symbol column
    sym_col = (
        _find_col(df, "gene", "symbol", required=False) or
        _find_col(df, "gene",           required=False) or
        df.columns[0]
    )

    # Identify primary disorder column
    disorder_col = (
        _find_col(df, "primary", "disorder", required=False) or
        _find_col(df, "disorder",             required=False) or
        _find_col(df, "phenotype",            required=False)
    )

    # Normalise gene symbols
    df["_symbol"] = df[sym_col].fillna("").astype(str).str.strip().str.upper()
    df = df[df["_symbol"] != ""]

    # Deduplicate: one row per gene symbol
    # Keep the primary disorder from the first 'Primary' report type if available
    if disorder_col:
        primary_rows = df[df.get("Report Type", pd.Series(dtype=str)).str.contains(
            "Primary", na=False, case=False
        )] if "Report Type" in df.columns else df
        disorder_map = (
            primary_rows
            .groupby("_symbol")[disorder_col]
            .first()
            .fillna("")
            .to_dict()
        )
    else:
        disorder_map = {}

    symbols = set(df["_symbol"].unique())
    return symbols, disorder_map


# ---------------------------------------------------------------------------
# Unified builder
# ---------------------------------------------------------------------------

def build_unified_ndd_gene_table() -> pd.DataFrame:
    """
    Merge all five gene databases into one unified NDD gene reference table.
    One row per unique gene symbol.

    Columns:
        gene_symbol, in_sfari, sfari_score,
        in_ddg2p, ddg2p_confidence,
        in_sysndd, sysndd_category,
        in_ndd_genehub, in_autdb, autdb_disorder,
        n_sources, sources
    """
    print("[Loading gene databases]\n")

    # ── SFARI ────────────────────────────────────────────────────────────────
    sfari_genes: Set[str] = set()
    sfari_score_map: Dict[str, str] = {}
    try:
        print(f"  SFARI ({SFARI_CSV.name}):")
        sfari_genes, sfari_score_map = load_sfari()
        print(f"    → {len(sfari_genes):,} genes loaded\n")
    except (FileNotFoundError, KeyError) as e:
        print(f"    [SKIP] {e}\n")

    # ── DDG2P ────────────────────────────────────────────────────────────────
    ddg2p_genes: Set[str] = set()
    ddg2p_conf_map: Dict[str, str] = {}
    try:
        print(f"  DDG2P ({DDG2P_TSV.name}):")
        ddg2p_genes, ddg2p_conf_map = load_ddg2p()
        print(f"    → {len(ddg2p_genes):,} genes loaded\n")
    except (FileNotFoundError, KeyError) as e:
        print(f"    [SKIP] {e}\n")

    # ── SysNDD ───────────────────────────────────────────────────────────────
    sysndd_genes: Set[str] = set()
    sysndd_cat_map: Dict[str, str] = {}
    try:
        print(f"  SysNDD genes ({SYSNDD_GENE_XLSX.name}):")
        sysndd_genes, sysndd_cat_map = load_sysndd_genes()
        print(f"    → {len(sysndd_genes):,} genes loaded\n")
    except (FileNotFoundError, KeyError) as e:
        print(f"    [SKIP] {e}\n")

    # ── NDD GeneHub ──────────────────────────────────────────────────────────
    genehub_genes: Set[str] = set()
    try:
        print(f"  NDD GeneHub ({NDD_GENEHUB_CSV.name}):")
        genehub_genes = load_ndd_genehub()
        print(f"    → {len(genehub_genes):,} genes loaded\n")
    except (FileNotFoundError, KeyError) as e:
        print(f"    [SKIP] {e}\n")

    # ── AutDB ────────────────────────────────────────────────────────────────
    autdb_genes: Set[str] = set()
    autdb_disorder_map: Dict[str, str] = {}
    try:
        print(f"  AutDB ({AUTDB_CSV.name}):")
        autdb_genes, autdb_disorder_map = load_autdb()
        print(f"    → {len(autdb_genes):,} unique genes loaded "
              f"(deduplicated from multi-row format)\n")
    except (FileNotFoundError, KeyError) as e:
        print(f"    [SKIP] {e}\n")

    # ── Build unified table ──────────────────────────────────────────────────
    all_genes = sfari_genes | ddg2p_genes | sysndd_genes | genehub_genes | autdb_genes

    if not all_genes:
        raise RuntimeError("No genes loaded from any database. Check file paths and column names.")

    rows = []
    for gene in sorted(all_genes):
        sources = []
        if gene in sfari_genes:    sources.append("SFARI")
        if gene in ddg2p_genes:    sources.append("DDG2P")
        if gene in sysndd_genes:   sources.append("SysNDD")
        if gene in genehub_genes:  sources.append("GeneHub")
        if gene in autdb_genes:    sources.append("AutDB")

        rows.append({
            "gene_symbol":      gene,
            "in_sfari":         gene in sfari_genes,
            "sfari_score":      sfari_score_map.get(gene, ""),
            "in_ddg2p":         gene in ddg2p_genes,
            "ddg2p_confidence": ddg2p_conf_map.get(gene, ""),
            "in_sysndd":        gene in sysndd_genes,
            "sysndd_category":  sysndd_cat_map.get(gene, ""),
            "in_ndd_genehub":   gene in genehub_genes,
            "in_autdb":         gene in autdb_genes,
            "autdb_disorder":   autdb_disorder_map.get(gene, ""),
            "n_sources":        len(sources),
            "sources":          ",".join(sources),
        })

    return pd.DataFrame(rows)


def save_unified_table(df: pd.DataFrame, out_path: Path = OUTPUT_CSV) -> Path:
    """Save the unified gene table to CSV."""
    out_path = Path(out_path)
    df.to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Convenience lookup (for downstream annotation pipeline)
# ---------------------------------------------------------------------------

def is_ndd_gene(gene_symbol: str, unified_df: pd.DataFrame) -> bool:
    """True if gene_symbol appears in any of the NDD databases."""
    return gene_symbol.upper() in unified_df["gene_symbol"].values


def get_gene_annotation(gene_symbol: str, unified_df: pd.DataFrame) -> Optional[dict]:
    """
    Return a dict of NDD annotations for one gene, or None if not in any DB.
    """
    row = unified_df[unified_df["gene_symbol"] == gene_symbol.upper()]
    if row.empty:
        return None
    return row.iloc[0].to_dict()


# ---------------------------------------------------------------------------
# Demo / main
# ---------------------------------------------------------------------------

def main():
    sep = "=" * 65
    print(sep)
    print("  NDD Gene Annotation Loader")
    print(sep)
    print()

    unified_df = build_unified_ndd_gene_table()

    print(f"\n{'─'*65}")
    print(f"  UNIFIED NDD GENE REFERENCE SUMMARY")
    print(f"{'─'*65}")
    print(f"  Total unique genes (union of all sources):  {len(unified_df):>6,}")
    print(f"  In SFARI:                                   {unified_df['in_sfari'].sum():>6,}")
    print(f"  In DDG2P:                                   {unified_df['in_ddg2p'].sum():>6,}")
    print(f"  In SysNDD:                                  {unified_df['in_sysndd'].sum():>6,}")
    print(f"  In NDD GeneHub:                             {unified_df['in_ndd_genehub'].sum():>6,}")
    if 'in_autdb' in unified_df.columns:
        print(f"  In AutDB:                                   {unified_df['in_autdb'].sum():>6,}")
        autdb_only = ((unified_df['in_autdb']) & (unified_df['n_sources'] == 1)).sum()
        print(f"  AutDB-only (unique to AutDB):               {autdb_only:>6,}")
    print(f"  Corroborated by 2+ sources:                 {(unified_df['n_sources']>=2).sum():>6,}")
    print(f"  Corroborated by 3+ sources:                 {(unified_df['n_sources']>=3).sum():>6,}")
    max_src = int(unified_df['n_sources'].max())
    print(f"  In ALL {max_src} sources:                          {(unified_df['n_sources']==max_src).sum():>6,}")

    print(f"\n  Preview (first 10 rows):")
    print(unified_df.head(10).to_string(index=False))

    # Show a few high-confidence NDD genes (in 3+ sources)
    top = unified_df[unified_df["n_sources"] >= 3].head(10)
    if not top.empty:
        print(f"\n  High-confidence genes (in 3+ sources — first 10):")
        print(top[["gene_symbol","sfari_score","ddg2p_confidence","sysndd_category","sources"]].to_string(index=False))

    # Save
    save_unified_table(unified_df)

    # Quick lookup demo
    test_genes = ["SHANK3", "MECP2", "CHD8", "ADCY5", "NOTAREALGENE"]
    print(f"\n  Lookup demo:")
    for g in test_genes:
        ann = get_gene_annotation(g, unified_df)
        if ann:
            print(f"    {g:15s} → sources={ann['sources']}  sfari={ann['sfari_score']}  ddg2p={ann['ddg2p_confidence']}")
        else:
            print(f"    {g:15s} → not in any NDD database")

    print(f"\n{sep}")
    print("  Done — unified NDD gene reference built and saved.")
    print(sep)


if __name__ == "__main__":
    main()