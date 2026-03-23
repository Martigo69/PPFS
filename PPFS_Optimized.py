import numpy as np
import pandas as pd
import gmpy2
import matplotlib.pyplot as plt
from functools import reduce
from sklearn.feature_selection import mutual_info_classif
import Threshold_Paillier as tp
import time
import os

gmpy2.get_context().precision = 100

# Global verbose state
VERBOSE = False
_STARTED_AT = None

def _coerce_bool(value) -> bool:
    """Parse environment variable or string to boolean."""
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}

def _log(message: str):
    """Log message with elapsed time if verbose is enabled."""
    if not VERBOSE or _STARTED_AT is None:
        return
    elapsed = time.perf_counter() - _STARTED_AT
    print(f"[PPFS +{elapsed:8.2f}s] {message}")

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

# =====================================================================
# Data Loading
# =====================================================================

def load_csv_data(filename, target_col, drop_cols=None):
    _log(f"Loading CSV: {filename}")
    df = pd.read_csv(filename, delimiter=',')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    df = df.apply(lambda s: s.astype(np.int64) if s.name != target_col else s)
    # Keep each partition as a DataFrame (np.array_split would coerce to ndarray).
    column_chunks = np.array_split(df.columns, 3)
    partitions = [df.loc[:, list(cols)].copy() for cols in column_chunks if len(cols) > 0]
    _log(f"Data loaded: {len(partitions)} partitions, {len(df)} rows")
    return partitions

def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    file, target, drops = DATASETS[name]
    return load_csv_data(file, target, drops), target

# =====================================================================
# Feature Selection Helpers
# =====================================================================

def get_min_mutual_info_feature(data_parts, target_col):
    for part in data_parts:
        if target_col in part.columns:
            feature_cols = [c for c in part.columns if c != target_col]
            if not feature_cols:
                raise ValueError("No feature columns in target-containing part.")
            X = part[feature_cols].values
            y = part[target_col].values
            mi_scores = mutual_info_classif(X, y, discrete_features=True)
            return min(dict(zip(feature_cols, mi_scores)), key=lambda k: mi_scores[feature_cols.index(k)])
    raise ValueError(f"Target column '{target_col}' not found.")

# =====================================================================
# Ranking & Encryption
# =====================================================================

def compute_ranks(data_parts, target_col):
    _log("Computing ranks")
    ranks = []
    for part in data_parts:
        cols = [c for c in part.columns if c != target_col]
        if cols:
            ranks.append(part[cols].rank().astype(int))
        else:
            ranks.append(pd.DataFrame(index=part.index))
    _log(f"Ranks computed: {len(ranks)} partitions")
    return ranks

def encrypt_dataframe(df, system):
    if df.empty:
        return df.copy()

    columns = list(df.columns)
    column_arrays = {col: df[col].to_numpy(copy=False) for col in columns}
    _log(f"Encrypting {len(columns)} columns with {len(df)} rows")

    data = {}
    for col in columns:
        values = column_arrays[col]
        data[col] = [system.encryption(safe_int(v)) for v in values]

    _log(f"Encryption complete for {len(columns)} columns")
    return pd.DataFrame(data, dtype=object)

def encrypt_data_parts(data_parts, system):
    return [encrypt_dataframe(p, system) for p in data_parts]

def compute_squared_ranks(ranked_parts):
    _log("Computing squared ranks")
    squared_parts = []
    for rp in ranked_parts:
        if rp.empty:
            squared_parts.append(rp.copy())
            continue

        # pandas 3.x removed applymap; use map when available.
        if hasattr(rp, "map"):
            squared_parts.append(rp.map(lambda x: gmpy2.mpz(x) ** 2))
        else:
            squared_parts.append(rp.applymap(lambda x: gmpy2.mpz(x) ** 2))
    _log(f"Squared ranks computed: {len(squared_parts)} partitions")
    return squared_parts

# =====================================================================
# Spearman (Plain)
# =====================================================================

def compute_spearman_correlation(ranked_parts, target_col):
    _log("Computing Spearman correlation (plain)")
    # Find which part (if any) has target ranks; fallback: last part
    target_rank = None
    for rp in ranked_parts:
        if target_col in rp.columns:
            target_rank = rp[target_col].astype(int)
            break
    # If target ranks are not stored separately, cannot proceed
    if target_rank is None:
        # Assume target in last original part (not ranked); cannot compute -> return empty
        _log("Target column not found in ranked parts")
        return {}
    n = len(target_rank)
    denom = n * (n**2 - 1)
    res = {}
    for rp in ranked_parts:
        for col in rp.columns:
            if col == target_col:
                continue
            fr = rp[col].astype(int)
            d2 = (fr - target_rank).apply(lambda d: d * d).sum()
            res[col] = 1 - (6 * d2) / denom
    _log(f"Spearman correlation computed: {len(res)} features")
    return res

# =====================================================================
# Spearman (Encrypted Approximation)
# =====================================================================

def compute_encrypted_spearman_correlation(enc_ranked_parts, enc_squared_parts, ranked_parts, target_col, system):
    _log("Computing Spearman correlation (encrypted)")
    n = len(enc_ranked_parts[0]) if enc_ranked_parts else 0
    if n == 0:
        return {}
    # target rank plaintext (assumed in last ranked part context)
    target_plain = None
    if ranked_parts:
        for rp in ranked_parts:
            if target_col in rp.columns:
                target_plain = rp[target_col]
                break
    if target_plain is None:
        _log("Target column not found in ranked parts")
        return {}
    n_squared = system.n_squared
    target_sq_enc_product = None
    # Locate target column squared encryption (choose last occurrence)
    for idx, sq in enumerate(enc_squared_parts[::-1]):
        if target_col in sq.columns:
            target_sq_enc_product = paillier_add_accumulate(sq[target_col], n_squared)
            break
    if target_sq_enc_product is None:
        return {}
    denom = n * (n**2 - 1)
    correlations = {}
    for part_idx, part in enumerate(enc_ranked_parts):
        sq_part = enc_squared_parts[part_idx]
        for col in part.columns:
            if col == target_col:
                continue
            if col not in sq_part.columns:
                continue
            sum_feature_sq = paillier_add_accumulate(sq_part[col], n_squared)
            sum_feature_target = 1
            feature_enc_series = part[col]
            for i in range(n):
                enc_r1 = int(feature_enc_series.iloc[i])
                r2 = int(target_plain.iloc[i])
                # homomorphic scalar mult: E(x)^k
                enc_scaled = pow(enc_r1, 2 * r2, n_squared)
                sum_feature_target = (sum_feature_target * enc_scaled) % n_squared
            # decrypt
            num_plain = system.combining_algorithm((sum_feature_sq * target_sq_enc_product) % n_squared, [1, 2])
            den_plain = system.combining_algorithm(sum_feature_target, [1, 2])
            d_squared = num_plain - den_plain
            correlations[col] = 1 - (6 * d_squared) / denom
    _log(f"Encrypted Spearman correlation computed: {len(correlations)} features")
    return correlations

# =====================================================================
# Mutual Information (Plain)
# =====================================================================

def compute_mutual_information(data_parts, target_col):
    _log("Computing mutual information (plain)")
    # Locate target column
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        _log("Target column not found")
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

def compute_encrypted_mutual_information(data_parts, target_col, system):
    _log("Computing mutual information (encrypted)")
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        _log("Target column not found")
        return {}
    n = len(target_series)
    unique_Y = pd.unique(target_series)
    Y_plain_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
    # H(Y)
    H_y = gmpy2.mpfr(0)
    for y in unique_Y:
        py = gmpy2.mpfr(int(Y_plain_ind[y].sum())) / gmpy2.mpfr(n)
        if py > 0:
            H_y -= py * log2_safe(py)
    n_squared = system.n_squared
    mi = {}
    for part in data_parts:
        for col in part.columns:
            if col == target_col:
                continue
            X = part[col]
            unique_X = pd.unique(X)
            X_plain_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
            # Encrypt indicators row-wise once
            X_enc_ind = {
                xv: [system.encryption(int(bit)) for bit in X_plain_ind[xv]]
                for xv in unique_X
            }
            # Count_x using homomorphic addition (product of ciphertexts)
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
                    # Multiply encrypted 1_{X=xv} ciphertexts only where Y= y
                    sel_product = 1
                    y_mask = Y_plain_ind[y]
                    for i, bit in enumerate(y_mask):
                        if bit == 1:
                            sel_product = (sel_product * int(enc_vector[i])) % n_squared
                    c_xy = system.combining_algorithm(sel_product, [1, 2])
                    if c_xy == 0:
                        continue
                    p_xy = gmpy2.mpfr(c_xy) / n
                    p_y_given_x = gmpy2.mpfr(c_xy) / cx if cx else gmpy2.mpfr(0)
                    if p_y_given_x > 0:
                        H_y_given_x -= p_xy * log2_safe(p_y_given_x)
            mi[col] = float(H_y - H_y_given_x)
    _log(f"Encrypted mutual information computed: {len(mi)} features")
    return mi

# =====================================================================
# Reporting / Plotting
# =====================================================================

def print_spearman_results(target_feature, plain_corrs, enc_corrs):
    print(f"Target Feature (Spearman): {target_feature}")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_corrs.items():
        print(f"{f:<25}{v:<22}{enc_corrs.get(f, None)}")

def plot_elbow(values_dict, title, ylabel):
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
    plt.show()

def print_mutual_info_results(target_col, plain_mi, enc_mi):
    print(f"\nMutual Information (target={target_col})")
    print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
    for f, v in plain_mi.items():
        print(f"{f:<25}{v:<22}{enc_mi.get(f, None)}")

def plot_mi(mi_dict, title):
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
    plt.show()

# =====================================================================
# Main
# =====================================================================

def main(dataset_name='diabetes', verbose=None):
    global VERBOSE, _STARTED_AT
    
    # Parse verbose flag from parameter or environment
    env_verbose = os.getenv("PPFS_VERBOSE", "0")
    VERBOSE = verbose if verbose is not None else _coerce_bool(env_verbose)
    _STARTED_AT = time.perf_counter()
    
    _log(f"Starting PPFS on dataset: {dataset_name}")
    
    system = tp.ThresholdPaillierCorrectProtocol(threshold=2, num_parties=3, verbose=VERBOSE)
    data_parts, target_col = load_dataset(dataset_name)

    target_feature = get_min_mutual_info_feature(data_parts, target_col)
    _log(f"Target feature: {target_feature}")
    ranked_parts = compute_ranks(data_parts, target_col)
    squared_ranks = compute_squared_ranks(ranked_parts)

    # Encrypt ranks & squared ranks
    _log("Encrypting ranked parts")
    enc_ranked_parts = encrypt_data_parts(ranked_parts, system)
    _log("Encrypting squared ranks")
    enc_squared_parts = encrypt_data_parts(squared_ranks, system)

    # Spearman
    _log("Computing Spearman correlation")
    plain_spearman = compute_spearman_correlation(ranked_parts, target_feature)
    enc_spearman = compute_encrypted_spearman_correlation(
        enc_ranked_parts, enc_squared_parts, ranked_parts, target_feature, system
    )
    print_spearman_results(target_feature, plain_spearman, enc_spearman)
    plot_elbow(enc_spearman, f"Encrypted Spearman (Dataset={dataset_name}, Target={target_feature})", "Abs Spearman")

    # Mutual Information
    _log("Computing mutual information")
    plain_mi = compute_mutual_information(data_parts, target_col)
    enc_mi = compute_encrypted_mutual_information(data_parts, target_col, system)
    print_mutual_info_results(target_col, plain_mi, enc_mi)
    plot_mi(enc_mi, f"Encrypted MI (Dataset={dataset_name}, Target={target_col})")
    
    _log("PPFS analysis complete")

if __name__ == "__main__":
    # Set verbose=True for live progress, or use PPFS_VERBOSE=1 environment variable
    main('divorce', verbose=True)