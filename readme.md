# EUR–SAS allele-frequency differentiation at neurodevelopmental disorder (NDD) gene loci

A Python pipeline that (1) builds a unified reference of NDD genes from five public
databases and (2) compares European (EUR) and South Asian (SAS) allele frequencies in
NDD-gene regions against the rest of the analysed variants, using 1000 Genomes Phase 3
data for chromosomes 1 and 2. A small CEPH trio analysis is included to validate the
variant-processing code.

> **Status.** Manuscript in preparation. Results are preliminary, cover chromosomes 1
> and 2 only, and use healthy reference individuals, not patients. Nothing here is a
> clinical tool or evidence of clinical portability.

## What the analysis found

| Question | Result (chromosomes 1 + 2) |
|---|---|
| Gene reference | 4,811 unique genes from five databases (1,455 in ≥3 sources) |
| Variants analysed | 13,549,694 extracted; differentiation defined for 6,228,512 |
| Variant-level difference | Mean differentiation in NDD-gene regions 4.5% below the non-NDD background (0.00970 vs 0.01016) |
| Gene-level difference | 56.7% of 746 NDD genes fall below size-matched background windows (95% CI 53.1–60.3%; sign test p = 0.0003) |
| Confidence tiers | No significant difference between tiers or between 3+ and single-database genes |
| EUR→SAS frequency prediction | R² 0.9412–0.9420 in NDD and background regions alike (difference ≤ 0.001) |

Interpretation: lower differentiation in NDD-gene regions is *consistent with* stronger
evolutionary constraint, but the comparison does not show that the effect is specific to
NDD biology (see Limitations) and does not identify purifying selection as the cause.

## Repository contents

| Script | Purpose |
|---|---|
| `vcf_utils.py` | Shared VCF/BED reader using Python's `gzip` (no pysam) |
| `Zenodo_trio_io.py`, `1000Genomes_io.py`, `Platinum_io.py` | Input readers for the three genomic datasets |
| `gene_annotation_loader.py` | Merges the five gene databases → `unified_ndd_gene_reference.csv` |
| `fetch_gene_coordinates.py` | hg19 coordinates from MyGene.info → `ndd_genes_grch37.bed` |
| `denovo_db_loader.py` | DenovoDB parsing → per-gene de novo counts |
| `population_af_analysis.py` | **Main analysis**: EUR–SAS differentiation per variant, subset comparison, chr1→chr2 prediction models |
| `gene_level_fst.py` | Gene-level comparison against size-matched background windows |
| `check_genehub.py`, `check_af_table.py` | Input and output sanity checks |
| `feature_engineering.py` | Trio feature table (quality filtering, Mendelian status, NDD annotation) |
| `ml_baseline.py`, `ablation_model_b.py` | Trio classifiers and feature-set ablation |
| `compute_training_metrics.py` | Train vs test metrics for the models |
| `make_figures.py` | Figures 2–4 (300 DPI PNG and TIFF) |

## Data (not included)

Download each source separately and check its terms of use. Scripts locate files relative
to the script folder, so keep this layout:

| Data | Source | Expected location |
|---|---|---|
| 1000 Genomes Phase 3 chr1, chr2 VCFs (`.vcf.gz`) | https://www.internationalgenome.org/ | `1000 Genomes/`, `1000 Genomes 2/` |
| CEPH 1463 trio VCF and TFAM | https://zenodo.org/record/3697103 | `zeneodo-trio/` |
| Platinum Genomes (hg38 hybrid VCF + BED) | Illumina | `Platinum/` |
| DenovoDB v1.6.1, non-SSC TSV | http://denovo-db.gs.washington.edu/ | `DenovoDB/` |
| SFARI Gene (CSV) | https://gene.sfari.org/ | project root |
| DDG2P (`DDG2P.tsv`) | https://www.ebi.ac.uk/gene2phenotype/ | project root |
| SysNDD gene and entity tables (`.xlsx`) | https://sysndd.dbmr.unibe.ch/ | project root |
| AutDB human genes (`autdb_human_gene.csv`) | http://autism.mindspec.org/autdb/ | project root |
| NDD GeneHub **full data** (`Full-Data.csv`) | https://ndd-genehub.org/downloads/ | project root, saved as `NDD_GeneHub_db.csv` |

Notes: the SFARI filename is set at the top of `gene_annotation_loader.py`; edit it to
match your download. For NDD GeneHub use the full case table from the downloads page,
not the on-screen table (a single page of 25 rows is truncated); `check_genehub.py`
reports whether the file is complete.

## Setup

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Developed and run on Ubuntu with Python 3.14.

## Running the pipeline

```bash
# 1. Gene reference
python3 check_genehub.py NDD_GeneHub_db.csv   # confirm the file is complete
python3 gene_annotation_loader.py             # -> unified_ndd_gene_reference.csv
python3 fetch_gene_coordinates.py             # -> ndd_genes_grch37.bed
python3 denovo_db_loader.py                   # -> denovo_db_gene_counts.csv

# 2. Population analysis (about 20 minutes, mostly random-forest training)
python3 population_af_analysis.py             # -> population_af_table.csv, *_summary.json, ml_results/
python3 check_af_table.py                     # coverage and restricted-prediction checks
python3 gene_level_fst.py                     # -> gene_level_fst.csv

# 3. Trio validation (secondary)
python3 feature_engineering.py --full         # see warning below
python3 ml_baseline.py
python3 ablation_model_b.py
python3 compute_training_metrics.py

# 4. Figures
python3 make_figures.py                       # -> figures/
```

**Warning.** `feature_engineering.py` without `--full` is a test run on the first 50,000
records and overwrites `phase1a_feature_table.csv`. Always pass `--full` for real runs and
back up earlier results before re-running any step. Use `--help` on each script for
other options.

If the gene reference changes (any database update), rerun everything from step 1.

## Outputs

| File | Contents |
|---|---|
| `unified_ndd_gene_reference.csv` | One row per gene: source membership, SFARI score, SysNDD category, source count |
| `ndd_genes_grch37.bed` | hg19 gene intervals used for variant-to-gene assignment |
| `population_af_table.csv` | Per-variant frequencies, differentiation and annotation (about 13.5 million rows) |
| `population_af_summary.json`, `ml_results/*.json` | Subset statistics and model results |
| `gene_level_fst.csv` | Per-gene mean differentiation, null-window summary and percentile |
| `phase1a_feature_table.csv` | Quality-filtered trio variants with features |
| `figures/` | Figures 2–4 |

## Methods in brief

- **Statistic.** For each variant, D = (p_EUR − p_SAS)² / [p_EUR(1 − p_EUR) + p_SAS(1 − p_SAS)],
  computed from the population frequencies in the VCF INFO field, with no sample-size
  correction, then averaged over variants. This is **not** Hudson's FST estimator; call it
  normalised allele-frequency differentiation.
- **Undefined sites.** D is undefined where both populations are monomorphic (frequency
  0 or 1); 7,321,182 of 13,549,694 rows are excluded for this reason.
- **Gene assignment.** A variant in overlapping gene intervals is assigned to one gene.
- **NDD gene set.** The union of five databases. NDD GeneHub membership means at least
  one reported case. Six genes that SysNDD classes as Refuted (three with no other
  support) are retained.
- **Gene-level test.** Each gene's mean D is ranked within 1,000 random contiguous
  background windows with the same number of variants on the same chromosome.
  Windows match gene size only.
- **Prediction models.** Ridge, Random Forest and XGBoost predict SAS frequency from EUR
  frequency (alone, or with other populations and gene annotations). Trained on
  chromosome 1, tested on chromosome 2.

## Limitations

- Chromosomes 1 and 2 only; no genome-wide claim.
- 1000 Genomes SAS does not represent all South Asian diversity; samples are healthy
  individuals, not NDD cohorts.
- Predicting allele frequency is not variant interpretation. Pathogenicity and phenotype
  portability are not tested.
- The background is mostly non-NDD genic and intergenic sequence. NDD genes may differ
  from it because genes are generally more constrained, so the result does not show NDD
  specificity. A comparison with size-matched non-NDD genes has not been done.
- Variant-level p-values treat linked variants as independent and are descriptive only.
- The association does not identify purifying selection as the cause.
- The CEPH 1463 trio is a pipeline check: its 18,423 de novo candidates are
  artifact-dominated, and its classifiers are not a biological result.

## Not included

`trio_extraction_1000G.py` and `cross_population_ml.py` were abandoned: the 1000 Genomes
Phase 3 callset contains the parents but not the offspring of its trios, so trio-based
analysis in that dataset is not possible.

## License, citation, contact

- License: [to be chosen]
- Citation: [manuscript reference to be added on publication]
- Contact: [email to be added]