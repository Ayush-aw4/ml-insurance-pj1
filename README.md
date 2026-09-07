# 🏥 Medical Insurance Cost Analysis

*Repository: `ml-insurance-pj1`*

Exploratory data analysis, data cleaning, feature engineering, and statistical feature selection on the **Medical Cost Personal Dataset** — preparing the data for a future medical insurance charges–prediction model.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-data%20wrangling-150458?logo=pandas&logoColor=white)
![scikit--learn](https://img.shields.io/badge/scikit--learn-preprocessing-F7931E?logo=scikit-learn&logoColor=white)
![Status](https://img.shields.io/badge/status-EDA%20%26%20feature%20engineering-yellow)

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Repository Structure](#repository-structure)
- [Methodology](#methodology)
- [Key Findings](#key-findings)
- [Final Feature Set](#final-feature-set)
- [Getting Started](#getting-started)
- [Tech Stack](#tech-stack)
- [Project Status & Roadmap](#project-status--roadmap)
- [Notes & Recommendations](#notes--recommendations)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Overview

This project explores the classic **Medical Cost Personal Dataset** (1,338 individuals in the US) to understand which demographic and lifestyle factors drive medical insurance charges. Everything lives in a single notebook, `Untitled.ipynb`, which walks through a complete data-preparation pipeline:

1. **Exploratory Data Analysis** — distributions, outliers, correlations
2. **Data cleaning & encoding** — duplicates, categorical → numeric
3. **Feature engineering** — BMI categorization, scaling
4. **Statistical feature selection** — Pearson correlation + Chi-square tests

The result is a clean, model-ready DataFrame (`final_df`) containing the features most strongly associated with `charges`.

|  |  |
|---|---|
| **Raw rows** | 1,338 (1,337 after de-duplication) |
| **Predictors** | 6 raw → 8 engineered/selected |
| **Target** | `charges` (USD, continuous) |
| **Task type** | Regression — data-prep phase; modeling not yet implemented |
| **Environment** | Python 3.11.9, Jupyter Notebook |

> **Note:** This notebook currently covers EDA through feature selection only — no regression model has been trained yet. See [Project Status & Roadmap](#project-status--roadmap).

## Dataset

The data lives in [`insurance.csv`](./insurance.csv) — 1,338 rows × 7 columns, one row per insured individual. It's the widely used **Medical Cost Personal Dataset**, originally compiled for the book *Machine Learning with R* by Brett Lantz and commonly distributed via Kaggle.

| Column | Type | Description | Range / Values |
|---|---|---|---|
| `age` | int | Age of primary beneficiary | 18 – 64 |
| `sex` | category | Gender of the insurance contractee | `female` (662), `male` (675)\* |
| `bmi` | float | Body mass index (kg/m²) | 15.96 – 53.13 |
| `children` | int | Number of dependents covered by the plan | 0 – 5 |
| `smoker` | category | Smoking status | `no` (1,063)\*, `yes` (274)\* |
| `region` | category | US residential region | `northeast`, `northwest`, `southeast`, `southwest` — roughly even split\* |
| `charges` | float | Individual medical costs billed by insurance — **target variable** | \$1,121.87 – \$63,770.43 |

\* counts shown after de-duplication (see [Methodology](#methodology))

There are no missing values in any column. One exact duplicate row exists in the raw file and is dropped during preprocessing (1,338 → 1,337 rows).

## Repository Structure

```
ml-insurance-pj1/
├── Untitled.ipynb    # Main analysis notebook: EDA → cleaning → feature engineering → feature selection
├── insurance.csv     # Raw dataset (1,338 records, 7 columns)
└── README.md         # Project documentation (this file)
```

## Methodology

### 1. Exploratory Data Analysis (EDA)
- Inspected shape, dtypes, summary statistics (`.describe()`), and missing values
- Plotted distributions (histogram + KDE) for `age`, `bmi`, `children`, `charges`
- Count plots for `children`, `sex`, `smoker`
- Box plots on the numeric columns to screen for outliers
- Correlation heatmap across numeric features

### 2. Data Cleaning & Preprocessing
- Copied the raw data into `df_cleaned`, preserving the original `df`
- Removed 1 duplicate row (1,338 → 1,337 rows)
- Re-verified there were no missing values
- Encoded `sex` → binary and renamed to `is_female` (`male` = 0, `female` = 1)
- Encoded `smoker` → binary and renamed to `is_smoker` (`no` = 0, `yes` = 1)
- One-hot encoded `region` (`drop_first=True`), keeping `northeast` as the implicit baseline
- Cast all columns to `int`

### 3. Feature Engineering & Extraction
- Bucketed `bmi` into standard clinical categories:

  | Category | BMI range |
  |---|---|
  | Underweight | < 18.5 |
  | Normal | 18.5 – 24.9 |
  | Overweight | 24.9 – 29.9 |
  | Obese | ≥ 29.9 |

- One-hot encoded `bmi_category` (`drop_first=True`), keeping `Underweight` as the implicit baseline
- Standardized `age`, `bmi`, and `children` with `StandardScaler` (zero mean, unit variance)

### 4. Statistical Feature Selection
- **Pearson correlation** between every numeric/encoded feature and `charges`
- **Chi-square test of independence** (α = 0.05) between each categorical/binary feature and `charges` binned into quartiles, to decide which categorical features carry signal
- Assembled `final_df` from the features that passed the tests above

## Key Findings

**Pearson correlation with `charges`:**

| Feature | Pearson *r* |
|---|---|
| `is_smoker` | **0.787** |
| `age` | 0.298 |
| `bmi_category_Obese` | 0.200 |
| `bmi` | 0.196 |
| `region_southeast` | 0.074 |
| `children` | 0.067 |
| `region_northwest` | -0.039 |
| `region_southwest` | -0.044 |
| `is_female` | -0.058 |
| `bmi_category_Normal` | -0.104 |
| `bmi_category_Overweight` | -0.121 |

**Chi-square test vs. `charges` quartile (α = 0.05):**

| Feature | χ² statistic | p-value | Decision |
|---|---|---|---|
| `is_smoker` | 848.22 | < 0.001 | ✅ Keep |
| `region_southeast` | 16.00 | 0.001 | ✅ Keep |
| `is_female` | 10.26 | 0.016 | ✅ Keep |
| `bmi_category_Obese` | 8.52 | 0.036 | ✅ Keep |
| `region_southwest` | 5.09 | 0.165 | ❌ Drop |
| `bmi_category_Overweight` | 4.25 | 0.236 | ❌ Drop |
| `bmi_category_Normal` | 3.71 | 0.295 | ❌ Drop |
| `region_northwest` | 1.13 | 0.769 | ❌ Drop |

**Takeaways**
- Smoking status is by far the strongest single driver of insurance charges.
- Age and obesity (both raw `bmi` and the `Obese` category) show a moderate positive relationship with charges.
- Number of children and most regions carry only weak signal — the majority of region and BMI-category dummies were statistically indistinguishable from noise at α = 0.05.

## Final Feature Set

`final_df` (1,337 rows × 8 columns) combines the numeric features with the categorical features that passed the Chi-square test:

`age` · `bmi` · `children` · `is_female` · `is_smoker` · `charges` · `region_southeast` · `bmi_category_Obese`

## Getting Started

### Prerequisites
- Python 3.11+ (developed and tested on 3.11.9)
- Jupyter Notebook or JupyterLab

### Installation
```bash
# 1. Clone or download this repository, then move into it
git clone <repository-url>
cd ml-insurance-pj1

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install numpy pandas seaborn matplotlib scipy scikit-learn jupyter
```

### Usage
```bash
jupyter notebook Untitled.ipynb
```
Run all cells top to bottom. Keep `insurance.csv` in the same folder as the notebook — it's loaded with a relative path (`pd.read_csv('insurance.csv')`).

## Tech Stack

| Library | Purpose | Version tested in this notebook |
|---|---|---|
| pandas | Data loading & manipulation | 3.0.5 |
| numpy | Numerical operations | 1.26.4 |
| seaborn | Statistical visualization | 0.13.2 |
| matplotlib | Plotting | 3.10.8 |
| scipy | `pearsonr`, `chi2_contingency` | 1.16.3 |
| scikit-learn | `StandardScaler` | 1.9.0 |

## Project Status & Roadmap

**Done:** EDA, cleaning, encoding, feature engineering, and statistical feature selection down to a model-ready feature set.

**Not yet done — no model has been trained.** Suggested next steps:
- [ ] Train/test split
- [ ] Baseline regression model (Linear Regression)
- [ ] Compare against tree-based models (Random Forest, Gradient Boosting)
- [ ] Evaluate with RMSE, MAE, and R²
- [ ] Cross-validation and hyperparameter tuning
- [ ] Residual analysis
- [ ] Persist `final_df` to disk (e.g. `processed_insurance.csv`) so modeling doesn't require re-running the whole notebook
- [ ] Add a `requirements.txt` / `environment.yml` for reproducible installs
- [ ] Rename `Untitled.ipynb` to something descriptive (e.g. `insurance_eda_and_feature_selection.ipynb`)

## Notes & Recommendations
- The Pearson/Chi-square feature selection here is computed on the **full dataset**. Once a train/test split is introduced, re-run feature selection on the training split only, to avoid leaking test-set information into feature choices.
- `.ipynb_checkpoints/` is a Jupyter autosave folder — consider adding a `.gitignore` so it isn't committed to version control.

## License
No license file is currently included in this repository, so default copyright applies (all rights reserved). If you'd like others to reuse or build on this work, consider adding an open-source license such as [MIT](https://choosealicense.com/licenses/mit/).

## Acknowledgments
- Dataset: **Medical Cost Personal Dataset**, originally compiled for *Machine Learning with R* by Brett Lantz, publicly distributed via [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance).
