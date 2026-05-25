#!/usr/bin/env python
import numpy as np
import pandas as pd
import gmpy2
import matplotlib.pyplot as plt
from functools import reduce
import Threshold_Paillier as tp
import time
import os
import argparse
from datetime import datetime
import psutil
import threading

gmpy2.get_context().precision = 100

# Get current process for memory tracking
_process = psutil.Process(os.getpid())
MEMORY_SAMPLING_INTERVAL_S = 1.0

def _get_memory_mb():
    """Get current process memory usage in MB."""
    return _process.memory_info().rss / (1024 * 1024)

def _measure(func, *args, **kwargs):
    """Run a function and return (result, seconds)."""
    start = time.perf_counter()
    result = func(*args, **kwargs)
    return result, (time.perf_counter() - start)

def _measure_with_memory(func, *args, **kwargs):
    """Run a function and return (result, seconds, peak_memory_mb, start_memory_mb).
    
    Continuously monitors memory during execution to find true peak.
    """
    peak_memory = [_get_memory_mb()]  # Use list to allow modification in thread
    start_memory = peak_memory[0]
    monitoring = [True]  # Flag to stop monitoring thread
    
    def monitor_memory():
        """Background thread that continuously samples memory."""
        while monitoring[0]:
            current_mem = _get_memory_mb()
            if current_mem > peak_memory[0]:
                peak_memory[0] = current_mem
            time.sleep(MEMORY_SAMPLING_INTERVAL_S)  # Sample every 1 second
    
    # Start monitoring thread
    monitor_thread = threading.Thread(target=monitor_memory, daemon=True)
    monitor_thread.start()
    
    # Execute function
    start = time.perf_counter()
    result = func(*args, **kwargs)
    elapsed = time.perf_counter() - start
    
    # Stop monitoring
    monitoring[0] = False
    monitor_thread.join(timeout=1.0)
    
    return result, elapsed, peak_memory[0], start_memory

# =====================================================================
# Configuration
# =====================================================================

DATASETS = {
    'beans': ('beans_kmeans.csv', 'Class', []),
    'diabetes': ('diabetes_kmeans.csv', 'Outcome', []),
    'divorce': ('divorce.csv', 'Class', []),
    'parkinsons': ('parkinsons_kmeans.csv', 'status', ['name']),
    'rice': ('rice_binned_kmeans.csv', 'Class', []),
    'wdbc': ('wdbc_binned_kmeans.csv', 'Diagnosis', ['ID'])
}


PARTIES = {
    'wdbc': 8,
    'parkinsons': 6,
    'rice': 2,
    'beans': 4,
    'divorce': 14,
    'diabetes': 3
}

# =====================================================================
# Utilities
# =====================================================================

def safe_int(x):
    return int(x) if not isinstance(x, int) else x

def log2_safe(p: gmpy2.mpfr):
    return gmpy2.log2(p) if p > 0 else gmpy2.mpfr(0)

def paillier_add_accumulate(encrypted_values, n_squared):
    """Multiply ciphertexts mod n^2 (homomorphic addition)."""
    acc = 1
    for c in encrypted_values:
        acc = (acc * int(c)) % n_squared
    return acc

def paillier_add_accumulate_parallel(encrypted_values, n_squared):
    values = encrypted_values if isinstance(encrypted_values, list) else list(encrypted_values)
    if not values:
        return 1
    return paillier_add_accumulate(values, n_squared)

def powmod_vector_parallel(bases, exponents, modulus):
    pairs = list(zip(bases, exponents))
    if not pairs:
        return []
    return [gmpy2.powmod(int(base), int(exp), modulus) for base, exp in pairs]

# =====================================================================
# Data Loading
# =====================================================================

def load_csv_data(filename, target_col, drop_cols=None, num_parties=3):
    df = pd.read_csv(filename, delimiter=',')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    df = df.apply(lambda s: s.astype(np.int64) if s.name != target_col else s)
    column_chunks = np.array_split(df.columns, num_parties)
    partitions = [df.loc[:, list(cols)].copy() for cols in column_chunks]
    return partitions

def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    file, target, drops = DATASETS[name]
    num_parties = PARTIES.get(name, 3)
    return load_csv_data(file, target, drops, num_parties=num_parties), target

# =====================================================================
# Feature Selection Helpers
# =====================================================================

def get_min_mutual_info_feature(data_parts, target_col):
    plain_mi_score = compute_mutual_information(data_parts, target_col)
    for part in data_parts:
        if target_col in part.columns:
            feature_cols = [c for c in part.columns if c != target_col]
            if not feature_cols:
                raise ValueError("No feature columns in target-containing part.")
            return min(feature_cols, key=lambda c: plain_mi_score.get(c, float("inf")))
    raise ValueError(f"Target column '{target_col}' not found.")

# =====================================================================
# Ranking & Encryption
# =====================================================================

def compute_ranks(data_parts, target_col):
    ranks = []
    for part in data_parts:
        cols = [c for c in part.columns if c != target_col]
        if cols:
            ranks.append(part[cols].rank().astype(int))
        else:
            ranks.append(pd.DataFrame(index=part.index))
    return ranks

def encrypt_dataframe(df, system):
    if df.empty:
        return df.copy()

    columns = list(df.columns)
    column_arrays = {col: df[col].to_numpy(copy=False) for col in columns}
    data = {}
    for col in columns:
        values = column_arrays[col]
        data[col] = [system.encryption(safe_int(v)) for v in values]
    return pd.DataFrame(data, dtype=object)

def encrypt_data_parts(data_parts, system):
    return [encrypt_dataframe(p, system) for p in data_parts]

def compute_squared_ranks(ranked_parts):
    squared_parts = []
    for rp in ranked_parts:
        if rp.empty:
            squared_parts.append(rp.copy())
            continue

        if hasattr(rp, "map"):
            squared_parts.append(rp.map(lambda x: gmpy2.mpz(x) ** 2))
        else:
            squared_parts.append(rp.applymap(lambda x: gmpy2.mpz(x) ** 2))
    return squared_parts

def precompute_encrypted_mi_indicators(data_parts, target_col, system):
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        raise ValueError(f"Target column '{target_col}' not found for MI precomputation")

    feature_items = []
    for part in data_parts:
        for col in part.columns:
            if col == target_col:
                continue
            X = part[col]
            unique_X = pd.unique(X)
            X_plain_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
            X_enc_ind = {
                xv: [system.encryption(int(bit)) for bit in X_plain_ind[xv]]
                for xv in unique_X
            }
            feature_items.append((col, unique_X, X_enc_ind))
    return target_series, feature_items

# =====================================================================
# Spearman (Plain)
# =====================================================================

def compute_spearman_correlation(ranked_parts, target_col):
    target_rank = None
    for rp in ranked_parts:
        if target_col in rp.columns:
            target_rank = rp[target_col].astype(int)
            break
    if target_rank is None:
        return {}
    n = len(target_rank)
    sum_y = target_rank.sum()
    sum_y2 = (target_rank * target_rank).sum()
    mean_y = sum_y / n
    correlations = {}
    for rp in ranked_parts:
        for col in rp.columns:
            if col == target_col:
                continue
            fr = rp[col].astype(int)
            sum_x = fr.sum()
            sum_x2 = (fr * fr).sum()
            sum_xy = (fr * target_rank).sum()
            mean_x = sum_x / n
            num = sum_xy - n * mean_x * mean_y
            den = np.sqrt((sum_x2 - n * mean_x * mean_x) * (sum_y2 - n * mean_y * mean_y))
            if den != 0:
                correlations[col] = num / den
            else:
                correlations[col] = 0.0
    return correlations

# =====================================================================
# Spearman (Encrypted Approximation)
# =====================================================================

def compute_encrypted_spearman_correlation(enc_ranked_parts, enc_squared_parts, ranked_parts, target_col, system):
    if not enc_ranked_parts or not ranked_parts:
        return {}
    n = len(enc_ranked_parts[0])
    if n == 0:
        return {}
    target_plain = next((rp[target_col] for rp in ranked_parts if target_col in rp.columns), None)
    if target_plain is None:
        return {}

    n_squared = system.n_squared

    enc_sum_y2 = next(
        (paillier_add_accumulate_parallel(sq[target_col].tolist(), n_squared)
         for sq in reversed(enc_squared_parts) if target_col in sq.columns),
        None
    )
    enc_sum_y = next(
        (paillier_add_accumulate_parallel(rp[target_col].tolist(), n_squared)
         for rp in reversed(enc_ranked_parts) if target_col in rp.columns),
        None
    )
    if enc_sum_y is None or enc_sum_y2 is None:
        return {}

    mean_y = gmpy2.mpfr(system.combining_algorithm(enc_sum_y, [1, 2])) / gmpy2.mpfr(n)
    sum_y2 = gmpy2.mpfr(system.combining_algorithm(enc_sum_y2, [1, 2]))
    target_plain_values = target_plain.to_numpy(copy=False)

    args_list = []
    for rp, sq in zip(enc_ranked_parts, enc_squared_parts):
        for col in rp.columns:
            if col != target_col:
                args_list.append((rp, sq, col))

    def process_column(args):
        rp, sq, col = args
        fr = rp[col].to_numpy(copy=False)
        fr2 = sq[col] if col in sq.columns else None
        if fr2 is None:
            return (col, 0.0)
        fr2_values = fr2.to_numpy(copy=False)

        enc_xy = powmod_vector_parallel(fr, target_plain_values, n_squared)
        enc_sum_x = paillier_add_accumulate_parallel(fr.tolist(), n_squared)
        enc_sum_x2 = paillier_add_accumulate_parallel(fr2_values.tolist(), n_squared)
        enc_sum_xy = paillier_add_accumulate_parallel(enc_xy, n_squared)

        mean_x = gmpy2.mpfr(system.combining_algorithm(enc_sum_x, [1, 2])) / gmpy2.mpfr(n)
        sum_x2 = gmpy2.mpfr(system.combining_algorithm(enc_sum_x2, [1, 2]))
        sum_xy = gmpy2.mpfr(system.combining_algorithm(enc_sum_xy, [1, 2]))

        num = sum_xy - gmpy2.mpfr(n) * mean_x * mean_y
        den = gmpy2.sqrt((sum_x2 - gmpy2.mpfr(n) * mean_x ** 2) * (sum_y2 - gmpy2.mpfr(n) * mean_y ** 2))
        corr = float(num / den) if den != 0 else 0.0
        return (col, corr)

    correlations = {}
    for args in args_list:
        col, corr = process_column(args)
        correlations[col] = corr
    return correlations

# =====================================================================
# Mutual Information (Plain)
# =====================================================================

def compute_mutual_information(data_parts, target_col):
    # Locate target column
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        return {}
    n = len(target_series)
    unique_Y = pd.unique(target_series)
    # Precompute Y indicators & entropy H(Y)
    Y_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
    H_y = gmpy2.mpfr(0)
    for y in unique_Y:
        py = gmpy2.mpfr(int(Y_ind[y].sum())) / gmpy2.mpfr(n)
        if py > 0:
            H_y -= py * log2_safe(py)
    mi = {}
    for part in data_parts:
        for col in part.columns:
            if col == target_col:
                continue
            X = part[col]
            unique_X = pd.unique(X)
            X_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
            count_x = {xv: X_ind[xv].sum() for xv in unique_X}
            H_y_given_x = gmpy2.mpfr(0)
            for xv in unique_X:
                cx = count_x[xv]
                if cx == 0:
                    continue
                for y in unique_Y:
                    c_xy = int((X_ind[xv] & Y_ind[y]).sum())
                    if c_xy == 0:
                        continue
                    p_xy = gmpy2.mpfr(c_xy) / gmpy2.mpfr(n)
                    p_y_given_x = gmpy2.mpfr(c_xy) / cx
                    H_y_given_x -= p_xy * log2_safe(p_y_given_x)
            mi[col] = float(H_y - H_y_given_x)
    return mi

# =====================================================================
# Mutual Information (Encrypted Approximation)
# =====================================================================

def compute_encrypted_mutual_information(data_parts, target_col, system, precomputed_indicators=None):
    if precomputed_indicators is None:
        target_series = None
        for p in data_parts:
            if target_col in p.columns:
                target_series = p[target_col]
                break
        if target_series is None:
            return {}

        feature_items = []
        for part in data_parts:
            for col in part.columns:
                if col == target_col:
                    continue
                X = part[col]
                unique_X = pd.unique(X)
                X_plain_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
                X_enc_ind = {
                    xv: [system.encryption(int(bit)) for bit in X_plain_ind[xv]]
                    for xv in unique_X
                }
                feature_items.append((col, unique_X, X_enc_ind))
    else:
        target_series, feature_items = precomputed_indicators

    n = len(target_series)
    unique_Y = pd.unique(target_series)
    Y_plain_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
    Y_indices = {y: np.flatnonzero(Y_plain_ind[y]).tolist() for y in unique_Y}
    # H(Y)
    H_y = gmpy2.mpfr(0)
    for y in unique_Y:
        py = gmpy2.mpfr(int(Y_plain_ind[y].sum())) / gmpy2.mpfr(n)
        if py > 0:
            H_y -= py * log2_safe(py)
    n_squared = system.n_squared
    mi = {}
    for col, unique_X, X_enc_ind in feature_items:
        count_x = {}
        for xv in unique_X:
            chain = paillier_add_accumulate(X_enc_ind[xv], n_squared)
            count_x[xv] = system.combining_algorithm(chain, [1, 2])

        H_y_given_x = gmpy2.mpfr(0)
        for xv in unique_X:
            cx = count_x[xv]
            if cx == 0:
                continue
            enc_vector = X_enc_ind[xv]
            for y in unique_Y:
                sel_product = 1
                for i in Y_indices[y]:
                    sel_product = (sel_product * int(enc_vector[i])) % n_squared
                c_xy = system.combining_algorithm(sel_product, [1, 2])
                if c_xy == 0:
                    continue
                p_xy = gmpy2.mpfr(c_xy) / n
                p_y_given_x = gmpy2.mpfr(c_xy) / cx if cx else gmpy2.mpfr(0)
                if p_y_given_x > 0:
                    H_y_given_x -= p_xy * log2_safe(p_y_given_x)
        mi[col] = float(H_y - H_y_given_x)
    return mi

# =====================================================================
# Reporting / Plotting
# =====================================================================

def print_spearman_results(target_feature, plain_corrs, enc_corrs):
    print(f"Target Feature (Spearman): {target_feature}")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_corrs.items():
        print(f"{f:<25}{v:<22}{enc_corrs.get(f, None)}")

def _format_spearman_results(target_feature, plain_corrs, enc_corrs):
    lines = [
        f"Target Feature (Spearman): {target_feature}",
        f"{'Feature':<25}{'Plain':<22}{'Encrypted'}",
    ]
    for f, v in plain_corrs.items():
        lines.append(f"{f:<25}{v:<22}{enc_corrs.get(f, None)}")
    return "\n".join(lines)

def plot_elbow(values_dict, title, ylabel, dataset_name="unknown"):
    if not values_dict:
        return
    sorted_items = sorted(values_dict.items(), key=lambda x: abs(x[1]), reverse=True)
    labels, scores = zip(*sorted_items)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(scores)), [abs(s) for s in scores], marker='o')
    plt.xticks(range(len(labels)), labels, rotation=50, ha='right')
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Features")
    plt.tight_layout()
    os.makedirs("Output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = str(dataset_name).strip().replace(" ", "_")
    filename = os.path.join("Output", f"plot_spearman_{safe_dataset}_{ts}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

def print_mutual_info_results(target_col, plain_mi, enc_mi):
    print(f"\nMutual Information (target={target_col})")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_mi.items():
        print(f"{f:<25}{v:<22}{enc_mi.get(f, None)}")

def _format_mutual_info_results(target_col, plain_mi, enc_mi):
    lines = [
        f"Mutual Information (target={target_col})",
        f"{'Feature':<25}{'Plain':<22}{'Encrypted'}",
    ]
    for f, v in plain_mi.items():
        lines.append(f"{f:<25}{v:<22}{enc_mi.get(f, None)}")
    return "\n".join(lines)

def _format_timing_table(rows):
    headers = [
        "Dataset",
        "Spearman Preprocess (s)",
        "Enc Spearman Compute (s)",
        "MI Preprocess (s)",
        "Enc MI Compute (s)",
    ]
    body = []
    for r in rows:
        body.append([
            r["dataset"],
            f"{r.get('spearman_preprocess_s', 0.0):.3f}",
            f"{r.get('enc_spearman_s', 0.0):.3f}",
            f"{r.get('mi_preprocess_s', 0.0):.3f}",
            f"{r.get('enc_mi_s', 0.0):.3f}",
        ])

    widths = [len(h) for h in headers]
    for row in body:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def _line(values):
        return " | ".join(values[i].ljust(widths[i]) for i in range(len(values)))

    sep = "-+-".join("-" * w for w in widths)
    lines = [_line(headers), sep]
    lines.extend(_line(row) for row in body)
    return "\n".join(lines)

def _run_single_dataset(dataset_name, make_plots=True, emit_console=True):
    protocol_preprocess_start = time.perf_counter()
    num_parties = PARTIES.get(dataset_name, 3)
    threshold = 2
    system = tp.ThresholdPaillierCorrectProtocol(threshold=threshold, num_parties=num_parties)

    data_parts, target_col = load_dataset(dataset_name)
    target_feature = get_min_mutual_info_feature(data_parts, target_col)
    shared_preprocess_s = time.perf_counter() - protocol_preprocess_start
    row_count = len(data_parts[0]) if data_parts else 0
    total_columns = sum(len(p.columns) for p in data_parts)

    dataset_info_lines = [
        f"Dataset: {dataset_name}",
        f"Parties: {num_parties}",
        f"Threshold: {threshold}",
        f"Rows: {row_count}",
        f"Columns: {total_columns}",
        f"Target Column: {target_col}",
        "",
    ]

    # Spearman preprocess with memory tracking
    spearman_preprocess_start = time.perf_counter()
    ranked_parts = compute_ranks(data_parts, target_col)
    squared_ranks = compute_squared_ranks(ranked_parts)
    enc_ranked_parts, _, enc_ranked_peak_mem, _ = _measure_with_memory(
        encrypt_data_parts,
        ranked_parts,
        system
    )
    enc_squared_parts, _, enc_squared_peak_mem, _ = _measure_with_memory(
        encrypt_data_parts,
        squared_ranks,
        system
    )
    spearman_preprocess_s = shared_preprocess_s + (time.perf_counter() - spearman_preprocess_start)
    spearman_preprocess_peak_mem = max(enc_ranked_peak_mem, enc_squared_peak_mem)

    mi_preprocess_start = time.perf_counter()
    enc_mi_indicators, mi_preprocess_measure_s, mi_preprocess_peak_mem, _ = _measure_with_memory(
        precompute_encrypted_mi_indicators,
        data_parts,
        target_col,
        system
    )
    mi_preprocess_s = shared_preprocess_s + (time.perf_counter() - mi_preprocess_start)
    
    plain_spearman = compute_spearman_correlation(ranked_parts, target_feature)
    enc_spearman, enc_spearman_s_result, enc_spearman_peak_mem, _ = _measure_with_memory(
        compute_encrypted_spearman_correlation,
        enc_ranked_parts,
        enc_squared_parts,
        ranked_parts,
        target_feature,
        system,
    )
    enc_spearman_s = enc_spearman_s_result

    plain_mi = compute_mutual_information(data_parts, target_col)
    enc_mi, enc_mi_s_result, enc_mi_peak_mem, _ = _measure_with_memory(
        compute_encrypted_mutual_information,
        data_parts,
        target_col,
        system,
        enc_mi_indicators,
    )
    enc_mi_s = enc_mi_s_result

    spearman_text = _format_spearman_results(target_feature, plain_spearman, enc_spearman)
    mi_text = _format_mutual_info_results(target_col, plain_mi, enc_mi)

    if emit_console:
        print(spearman_text)
        if make_plots:
            plot_elbow(enc_spearman, f"Encrypted Spearman (Dataset={dataset_name}, Target={target_feature})", "Abs Spearman", dataset_name)
        print("\n" + mi_text)
        if make_plots:
            plot_mi(enc_mi, f"Encrypted MI (Dataset={dataset_name}, Target={target_col})", dataset_name)
        print(f"\n=== Memory Usage ===")
        print(f"Spearman Preprocess Peak: {spearman_preprocess_peak_mem:.2f} MB")
        print(f"Enc Spearman Peak: {enc_spearman_peak_mem:.2f} MB")
        print(f"MI Preprocess Peak: {mi_preprocess_peak_mem:.2f} MB")
        print(f"Enc MI Peak: {enc_mi_peak_mem:.2f} MB")

    return {
        "dataset": dataset_name,
        "spearman_preprocess_s": spearman_preprocess_s,
        "spearman_preprocess_peak_mem": spearman_preprocess_peak_mem,
        "enc_spearman_s": enc_spearman_s,
        "enc_spearman_peak_mem": enc_spearman_peak_mem,
        "mi_preprocess_s": mi_preprocess_s,
        "mi_preprocess_peak_mem": mi_preprocess_peak_mem,
        "enc_mi_s": enc_mi_s,
        "enc_mi_peak_mem": enc_mi_peak_mem,
        "details": [
            f"=== Dataset: {dataset_name} ===",
            *dataset_info_lines,
            spearman_text,
            "",
            mi_text,
            "",
            f"=== Memory Usage ===",
            f"Spearman Preprocess Peak: {spearman_preprocess_peak_mem:.2f} MB",
            f"Enc Spearman Peak: {enc_spearman_peak_mem:.2f} MB",
            f"MI Preprocess Peak: {mi_preprocess_peak_mem:.2f} MB",
            f"Enc MI Peak: {enc_mi_peak_mem:.2f} MB",
            "",
        ],
    }

    
def plot_mi(mi_dict, title, dataset_name="unknown"):
    if not mi_dict:
        return
    sorted_items = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    labels, scores = zip(*sorted_items)
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(scores)), scores, marker='o')
    plt.xticks(range(len(labels)), labels, rotation=50, ha='right')
    plt.title(title)
    plt.ylabel("Mutual Information")
    plt.xlabel("Features")
    plt.tight_layout()
    os.makedirs("Output", exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_dataset = str(dataset_name).strip().replace(" ", "_")
    filename = os.path.join("Output", f"plot_mi_{safe_dataset}_{ts}.png")
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    plt.close()

# =====================================================================
# Main
# =====================================================================

def main(dataset_name='diabetes', run_all=False, output_path=None):
    selected_all = run_all or str(dataset_name).strip().lower() == "all"

    if selected_all:
        if output_path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"ppfs_all_datasets_report_memory_{ts}.txt"

        rows = []
        details_blocks = []
        dataset_names = list(DATASETS.keys())

        # Create the report file early so progress is visible while long runs are in progress.
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("PPFS Benchmark Report (with Memory Profiling)\n")
            f.write(f"Generated at: {datetime.now().isoformat(timespec='seconds')}\n\n")
            f.write("Timing and Memory Summary\n")
            f.write("(in progress)\n\n")

        for idx, name in enumerate(dataset_names, start=1):
            row = _run_single_dataset(name, make_plots=False, emit_console=False)
            rows.append(row)
            details_blocks.extend(row["details"])

            table = _format_timing_table(rows)
            report_lines = [
                "PPFS Benchmark Report (with Memory Profiling)",
                f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
                "",
                "Timing and Memory Summary",
                table,
                "",
                f"Progress: {idx}/{len(dataset_names)} datasets completed",
                "",
            ]
            report_lines.extend(details_blocks)

            # Rewrite full report after each dataset completion for live monitoring.
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))

        return {"mode": "all", "output_path": output_path, "timings": rows}

    row = _run_single_dataset(dataset_name, make_plots=True, emit_console=True)
    print("\nTiming and Memory Summary")
    timing_table = _format_timing_table([row])
    print(timing_table)

    if output_path is not None:
        report_lines = [
            "PPFS Benchmark Report (with Memory Profiling)",
            f"Generated at: {datetime.now().isoformat(timespec='seconds')}",
            "",
            "Timing and Memory Summary",
            timing_table,
            "",
        ]
        report_lines.extend(row["details"])
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(report_lines))
        return {"mode": "single", "output_path": output_path, "timings": [row]}

    return {"mode": "single", "timings": [row]}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PPFS runner with timing and memory profiling.")
    parser.add_argument("--dataset", default="diabetes", help="Dataset name or 'all'.")
    parser.add_argument("--mode", choices=["single", "all"], default="single", help="Run one dataset or all datasets.")
    parser.add_argument("--output", action="store_true", help="Write report to Output directory.")
    args = parser.parse_args()

    run_all = args.mode == "all" or str(args.dataset).strip().lower() == "all"
    output_path = None
    if args.output:
        os.makedirs("Output", exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_all:
            output_path = os.path.join("Output", f"ppfs_all_datasets_report_memory_{ts}.txt")
        else:
            dataset_tag = str(args.dataset).strip().lower().replace(" ", "_")
            output_path = os.path.join("Output", f"ppfs_{dataset_tag}_report_memory_{ts}.txt")

    result = main(dataset_name=args.dataset, run_all=run_all, output_path=output_path)

    if run_all and result.get("output_path"):
        print(f"All-datasets report written to: {result['output_path']}")
