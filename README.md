# Predicting Critical Import Concentration in French Trade

**A machine-learning early-warning system for supply-chain vulnerability: classifying, one year in advance, which French import markets are at risk of becoming critically dependent on a single supplier.**

Final project for *Introduction to Statistical Learning*, École Polytechnique (M1 Data & Economics for Public Policy), April 2026.

## Why it matters

The COVID-19 pandemic and the war in Ukraine exposed how concentrated import dependencies threaten economic resilience: semiconductors, pharmaceutical ingredients, energy, critical raw materials. This project builds a classifier that flags, **one year ahead of time**, the HS6 product markets whose import concentration (Herfindahl-Hirschman Index) is about to cross the **HHI > 0.25** threshold used by the European Commission and the US Department of Justice. The tool is meant for diversification policy, strategic stockpiling and trade defense, and speaks directly to the EU strategic autonomy agenda.

## Results

| Model | AUC-ROC (4 temporal folds) | Recall | F1 |
| --- | --- | --- | --- |
| Logistic regression (baseline) | 0.966 ± 0.007 | 0.935 | 0.432 |
| Elastic Net (α = 0.5) | 0.966 ± 0.007 | 0.935 | 0.431 |
| Random Forest | 0.964 ± 0.007 | 0.613 | **0.629** |

**Final model (Random Forest, holdout 2022-2023: 9,330 obs., 3.75% positives): AUC 0.970, precision 0.651, recall 0.529, F1 0.583**

Two takeaways:

- Past market structure carries almost all the signal: lagged HHI values (autocorrelation ≈ 0.83), top-exporter shares and diversification measures are sufficient for linear models to saturate the AUC. The Elastic Net's near-zero optimal λ (≈ 7×10⁻⁵) confirms the logistic specification is already well-specified.
- The models differ on the precision/recall trade-off: the logit is a high-recall "alarm bell" (93.5% of future critical markets flagged, many false alarms), while the Random Forest is the balanced tool when false alarms carry a cost; the relevant choice depends on the policy cost structure.

## Data

All data is public; raw files are too large for Git and must be downloaded:

| Source | Content | Link |
| --- | --- | --- |
| **BACI** (CEPII) | Bilateral HS6 trade flows 1996-2024, HS96 nomenclature, restricted to French imports (country code 251) | [CEPII BACI](https://www.cepii.fr/CEPII/en/bdd_modele/bdd_modele_item.asp?id=37) |
| **WTO** | Annual world merchandise trade values (macro control) | [world_trade_values.xlsx](https://www.wto.org/english/res_e/statis_e/trade_evolution_e/world_trade_values.xlsx) |
| **World Bank Pink Sheet** | Commodity prices matched to HS2 sectors (explored, ultimately excluded: ~75% missing, selection bias) | [CMO Historical Data](https://thedocs.worldbank.org/en/doc/18675f1d1639c7a34d463f59263ba0a2-0050012025/) |

## Method

- **Unit of observation**: HS6 product × year panel, aggregated from millions of bilateral flow rows into market-structure indicators (HHI, top-exporter shares, number of exporters, unit costs).
- **Feature engineering**: HHI lags 1-3, annual change, 5-year rolling trend; log-transforms for heavy-tailed import values and unit costs (diamonds, pharma); crisis dummies (2008-09, 2020-22); HS2 sector codes.
- **Validation**: expanding-window time-series split over 4 folds (train on the past, test on the future). Standard k-fold would leak future information through the lagged features.
- **Class imbalance** (≈ 96/4): inverse-frequency weighting, no resampling.
- **Metrics**: AUC-ROC (primary, threshold-independent), recall and F1 at the 0.5 threshold; accuracy deliberately excluded as uninformative under imbalance.

## Repository structure

```
├── scripts/
│   ├── 01_data_processing.R   # raw BACI -> ML-ready panel + WTO/Pink Sheet merges
│   └── 02_modeling.R          # 3 models, temporal CV, figures, final holdout evaluation
├── report/
│   └── Final_Project_ML_Class.pdf   # full research report (15 pages)
├── LICENSE
└── README.md
```

## Reproduce

1. Download the BACI HS96 annual files (1996-2024) and world_trade_values.xlsx into `data_raw/`.
2. Adapt the two paths at the top of `scripts/01_data_processing.R` (`RAW_DIR`, `PROCESSED_DIR`).
3. Run `Rscript scripts/01_data_processing.R`, then `Rscript scripts/02_modeling.R` (set `OUTPUT_DIR` at the top).

Requires R with: tidyverse, arrow, readxl, zoo, randomForest, caret, glmnet, pROC.

## Author

**Jules Tastet**, École Polytechnique, M1 Data & Economics for Public Policy.
Full methodology, figures and interpretation: see the report in `report/`.
