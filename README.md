# Data Investigation Toolkit

**v1.0 · Python · pandas · scikit-learn · MIT license · Author: Siddharth Turakhia**

A first look at any tabular dataset in one call. It cleans, checks and summarizes the data, then picks a scaling method for each numeric column, and it **reports the reason behind every decision** so an analyst can review it rather than trust a black box.

## Quick start

```python
# pip install pandas numpy scikit-learn        (pandas 2.0 or later)
# copy data_investigation.py next to your script or notebook

from data_investigation import run_investigation

results = run_investigation("your_file.csv")   # or pass a pandas DataFrame
results["profile"]    # one row per column: type, % missing, unique, example, flag
results["keys"]       # smallest column set that identifies every row
results["scalers"]    # scaler chosen for each numeric column, and why
results["scaled"]     # cleaned data plus the scaled columns
```

Other keys in `results`: `data`, `clean`, `categories`, `numeric_summary`, `range_check`, `outliers`.

## Use one step at a time

```python
from data_investigation import load_data, convert_dates, profile, drop_nulls_report, iqr_outliers

df = convert_dates(load_data("your_file.csv"))   # dates found by testing values; day/month order inferred
profile(df)                                      # flags blank, constant, ID-like and category columns
clean = drop_nulls_report(df, threshold=10)      # warns if more than 10% of rows would be lost
iqr_outliers(clean)                              # Tukey fences per numeric column
```

## What each step does

| Step | Function | What it tells you |
|---|---|---|
| 1 Load | `load_data` | Reads a CSV and trims stray spaces from names and text values. |
| 2 Dates | `convert_dates` | Finds date columns by testing values, not names; infers day-first vs month-first; reports each choice. |
| 3 Profile | `profile`, `fill_rate` | One row per column with a flag: all blank, constant, ID-like, high missing, category. |
| 4 Clean | `constant_columns`, `drop_nulls_report` | Drops empty columns first, then shows which columns cause row losses and which rows a single fill would recover. |
| 5 Keys | `key_candidates`, `find_keys` | Screens column combinations on a sample, confirms on the full data, returns the smallest unique key. |
| 6 Describe | `categorical_counts`, `numeric_summary` | Value counts for every text column; describe() plus skew, kurtosis, % zeros, % negatives. |
| 7 Check | `auto_range_rules`, `range_check`, `iqr_outliers` | Limits inferred from column names (quantities and prices not negative; percentages 0-100), plus your own; outlier fences. |
| 8 Scale | `choose_scaler`, `scale_columns` | Picks Standard, MinMax, Robust or log-then-MinMax from each column's shape; never scales ID columns. |

## Options

```python
run_investigation(source, dayfirst="auto",              # or "day first" / "month first"
                  range_rules={"Price": (0.01, None)},  # your limits override the automatic ones
                  null_threshold=10,                    # % of rows you accept losing to blanks
                  always_minmax=True,                   # also add a 0-1 version of every numeric column
                  show=False)                           # True prints every report as it runs
```

## Authorship

Analysis design, scaling rules and testing: Siddharth Turakhia (built first as a capstone notebook). Packaging into an importable module and documentation were done with AI assistance (Claude, Anthropic), with every output reviewed and checked by the author.

## Design choices

- **Explain, don't just transform.** Every function returns a table with the reason for each decision, ready to review or export.
- **Order matters.** All-blank columns are dropped before blank rows; otherwise one empty column would remove every row.
- **Scale from the evidence.** The scaler follows each column's skew and outlier share, not a single default for all columns.
- **Fast on large files.** Key search rules out combinations with simple counts and a sample before confirming on the full data.

Built and tested in Google Colab and Python 3 (pandas 2+, scikit-learn). Works in any notebook or script; tables display richly in Jupyter and Colab and print as text elsewhere. Free to use and adapt under the MIT license, with credit to the author. Contact: [linkedin.com/in/sidturakhia](https://www.linkedin.com/in/sidturakhia)
