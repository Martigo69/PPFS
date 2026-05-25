# PPFS with Threshold Paillier

Privacy-Preserving Feature Selection (PPFS) is a Python project for evaluating feature relevance under threshold Paillier encryption. It supports Spearman correlation and Mutual Information scoring, timing instrumentation, and report/plot generation across multiple benchmark datasets.

## Features
- Feature selection using Spearman correlation and Mutual Information.
- Threshold Paillier encryption and threshold decryption for secure computation.
- Dataset partitioning across a configurable number of parties.
- Timing measurement for preprocessing and encrypted computation.
- Report and plot generation for each run.

## Project Structure
- `PPFS_Optimized.py`: Main runner for loading data, encrypting inputs, computing metrics, and generating outputs.
- `Threshold_Paillier.py`: Threshold Paillier implementation used by the PPFS pipeline.
- `PPFS_Memory.py`: Variant of the runner with memory profiling instrumentation.
- `Output/`: Generated reports and plots.
- `*.csv`: Supported benchmark datasets.

## Datasets
The runner supports the following datasets:

| Dataset | CSV File | Target Column | Dropped Columns | Parties |
|---|---|---|---|---|
| beans | `beans_kmeans.csv` | `Class` | none | 4 |
| diabetes | `diabetes_kmeans.csv` | `Outcome` | none | 3 |
| divorce | `divorce.csv` | `Class` | none | 14 |
| parkinsons | `parkinsons_kmeans.csv` | `status` | `name` | 6 |
| rice | `rice_binned_kmeans.csv` | `Class` | none | 2 |
| wdbc | `wdbc_binned_kmeans.csv` | `Diagnosis` | `ID` | 8 |

## Requirements
Install the project dependencies with:

```bash
pip install -r requirements.txt
```

If you prefer to install manually, the project uses:
- `numpy`
- `pandas`
- `gmpy2`
- `matplotlib`
- `scikit-learn`

## Usage
Run the project from the `PPFS` directory.

Single dataset:

```bash
python PPFS_Optimized.py --dataset diabetes --mode single
```

Single dataset with saved output:

```bash
python PPFS_Optimized.py --dataset diabetes --mode single --output
```

All datasets with saved output:

```bash
python PPFS_Optimized.py --dataset all --mode all --output
```

## CLI Options
- `--dataset`: Dataset name or `all`.
- `--mode`: `single` for one dataset or `all` for every supported dataset.
- `--output`: Writes the run report to the `Output/` directory.

## Outputs
When output mode is enabled, the project writes:
- `ppfs_<dataset>_report_<timestamp>.txt` for a single dataset run.
- `ppfs_all_datasets_report_<timestamp>.txt` for an all-dataset run.
- `plot_spearman_<dataset>_<timestamp>.png` for Spearman plots.
- `plot_mi_<dataset>_<timestamp>.png` for Mutual Information plots.

## Notes
- All-dataset runs can take a long time, especially for Mutual Information preprocessing and encrypted computation.
- The runner uses the party counts listed above when splitting each dataset into partitions.
