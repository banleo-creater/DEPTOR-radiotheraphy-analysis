# DEPTOR uncouples radiosensitivity from immune exclusion in head and neck squamous cell carcinoma

Analysis code and derived tables for the manuscript:
*"DEPTOR uncouples radiosensitivity from immune exclusion in head and neck squamous cell carcinoma"*

## Authors

Xiaobo Wu<sup>1,3</sup>, Gongbiao Lin<sup>1,3</sup>, Hanqing Lin<sup>1,3</sup>, Guangnan Yao<sup>1,3</sup>, Shuping Tang<sup>1,3</sup>, Zhihong Chen<sup>1,3</sup>, Jizhen Wang<sup>2,3,*</sup>

<sup>1</sup>Department of Otorhinolaryngology – Head and Neck Surgery, The First Affiliated Hospital of Fujian Medical University, Fuzhou, China
<sup>2</sup>Department of Hematology, The First Affiliated Hospital of Fujian Medical University, Fuzhou, China
<sup>3</sup>National Regional Medical Center, Binhai Campus of The First Affiliated Hospital, Fujian Medical University, Fuzhou, China

*Corresponding author: Jizhen Wang (banleo@fjmu.edu.cn)

## Data Sources

All primary data used in this study are publicly available:

| Dataset | Type | Source |
|---------|------|--------|
| TCGA-HNSC | Bulk transcriptome & clinical | cBioPortal / GDC |
| GSE65858, GSE41613, GSE139324, GSE198315 | Bulk expression cohorts | GEO |
| GSE164690 | CD45-sorted single-cell RNA-seq | GEO |
| GSE181300 | Visium spatial transcriptomics | GEO |
| GSE300147 | Xenium spatial transcriptomics | GEO |

Raw data must be downloaded from the respective portals. This repository does **not** redistribute primary patient data.

## Repository Contents

| File | Description |
|------|-------------|
| `DEPTOR_HR_analysis.R` | Main survival & interaction HR analysis (TCGA + external cohorts) |
| `spatial_enrichment.py` | Xenium spatial co-localization quantification (NK nest, M2, cDC) |
| `correlation_analysis.R` | Myeloid DEPTOR ↔ NK fraction Spearman correlation |
| `derived_tables.xlsx` | All derived tables: interaction HRs, spatial scores, correlation matrices, external validation summaries |
| `figure_generation/` | R scripts for reproducing main and supplementary figures |

## Requirements

- **R** (≥ 4.2) — scripts `DEPTOR_HR_analysis.R`, `correlation_analysis.R`, and those in `figure_generation/`; required packages are listed in the header of each script.
- **Python** (≥ 3.10) — script `spatial_enrichment.py`; dependencies are listed in the script header.

## Usage

1. Download primary data from TCGA/GEO (see Data Sources above).
2. Update file paths in each script header to point to your local data directory.
3. Run the analysis scripts in the order given below.
4. Derived tables will be written to `./output/`.

```r
source("DEPTOR_HR_analysis.R")
source("correlation_analysis.R")
```

```bash
python spatial_enrichment.py
```

## Derived Tables

All derived tables supporting the findings of this study are provided in `derived_tables.xlsx` and archived at Zenodo: [https://doi.org/10.5281/zenodo.23476871](https://doi.org/10.5281/zenodo.23476871)

## License

This repository is released under the [MIT License](LICENSE).

![Uploading image.png…]()
