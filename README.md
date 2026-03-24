# PPFS with Threshold Paillier

## Overview
This folder contains a Privacy-Preserving Feature Selection (PPFS) pipeline built on Threshold Paillier encryption.

Main goals:
- Compute feature relevance metrics (Spearman correlation and Mutual Information)
- Compare plain vs encrypted results
- Measure preprocessing and encrypted-computation timings
- Save run reports and plots into an `Output` directory

## Main Files
- `PPFS_Optimized.py`: End-to-end runner for data loading, encryption, encrypted analytics, timing, report generation, and plotting.
- `Threshold_Paillier.py`: Threshold Paillier cryptosystem implementation (key generation, encryption, partial decryption, combining).
- `*_kmeans.csv`, `divorce.csv`: Supported datasets used by the runner.
- `tp_key_cache_2048.json`: Cached key material to avoid regenerating safe primes each run.

## Threshold Paillier (`Threshold_Paillier.py`)
The class `ThresholdPaillierCorrectProtocol` implements a threshold variant of Paillier:

1. Key Setup
- Generates (or loads cached) safe primes `p=2p_bar+1`, `q=2q_bar+1`
- Builds modulus `n=pq`, and related parameters for threshold decryption
- Uses Shamir-style sharing for secret material across parties

2. Encryption
- Encrypts message `M` as:
  - `c = g^M * x^n mod n^2`
- Random `x` is sampled from `Z_n*` each time

3. Threshold Decryption
- Each party computes a decryption share
- `combining_algorithm` combines at least `t` shares to recover plaintext

4. Key Cache
- Key material (`p_bar`, `q_bar`, `p`, `q`) is cached in JSON
- Reduces expensive key generation cost on repeated runs

## PPFS Pipeline (`PPFS_Optimized.py`)
The pipeline runs these high-level stages:

1. Load and Partition Dataset
- Dataset is loaded from CSV
- Optional drop columns are removed
- Columns are split into party-wise partitions based on configured party count

2. Spearman Branch
- Compute ranks and squared ranks
- Encrypt ranked data and squared ranks
- Compute plain Spearman and encrypted Spearman

3. Mutual Information Branch
- Precompute encrypted indicator vectors for MI (`0/1` vectors)
- Compute plain MI and encrypted MI

4. Reporting
- Prints plain vs encrypted result tables
- Prints timing table with:
  - Spearman preprocess
  - Encrypted Spearman compute
  - MI preprocess
  - Encrypted MI compute
- In file output mode, writes a report with dataset metadata and metric tables

5. Plotting
- Saves plots to `Output/` (not interactive `plt.show()`)
- Filenames include dataset name and timestamp

## Dataset Configuration
Current dataset mapping in code:

| Dataset Name | CSV File | Target Column | Drop Columns | Parties |
|---|---|---|---|---|
| beans | `beans_kmeans.csv` | `Class` | none | 4 |
| diabetes | `diabetes_kmeans.csv` | `Outcome` | none | 3 |
| divorce | `divorce.csv` | `Class` | none | 14 |
| parkinsons | `parkinsons_kmeans.csv` | `status` | `name` | 6 |
| rice | `rice_binned_kmeans.csv` | `Class` | none | 2 |
| wdbc | `wdbc_binned_kmeans.csv` | `Diagnosis` | `ID` | 8 |

Threshold used by PPFS runs:
- `threshold = 2` (with dataset-specific party count from table above)

## CLI Usage
Run from this folder:

Single dataset (console output):
```bash
python PPFS_Optimized.py --dataset divorce --mode single
```

Single dataset + write report to `Output/`:
```bash
python PPFS_Optimized.py --dataset divorce --mode single --output
```

All datasets + write report to `Output/`:
```bash
python PPFS_Optimized.py --dataset all --mode all --output
```

Generate plots later from an existing report (without recomputing encrypted metrics):
```bash
python PPFS_Optimized.py --plot-report Output/ppfs_all_datasets_report_YYYYMMDD_HHMMSS.txt
```

## Output Files
When `--output` is used, report files are written to `Output/`:
- Single mode: `ppfs_<dataset>_report_<timestamp>.txt`
- All mode: `ppfs_all_datasets_report_<timestamp>.txt`

Plots are written to `Output/`:
- Spearman: `plot_spearman_<dataset>_<timestamp>.png`
- MI: `plot_mi_<dataset>_<timestamp>.png`

## Dependencies
Install these Python packages before running:
- `numpy`
- `pandas`
- `gmpy2`
- `matplotlib`

Example:
```bash
pip install numpy pandas gmpy2 matplotlib
```

## Notes
- Encrypted and plain results are expected to match up to tiny floating-point differences.
- All-dataset runs can take many hours, especially for MI preprocessing and encrypted MI computation.
