import numpy as np
import pandas as pd
import gmpy2
import matplotlib.pyplot as plt
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

def measure_time(func, *args, **kwargs):
    """Helper function to measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    duration = end_time - start_time
    print(f"{func.__name__}() took {duration:.4f} seconds")
    return result

# =====================================================================
# Data Loading
# =====================================================================

def load_csv_data(filename, target_col, drop_cols=None, num_parties=3):
    df = pd.read_csv(filename, delimiter=',')
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')
    df = df.apply(lambda s: s.astype(np.int64) if s.name != target_col else s)
    return np.array_split(df, num_parties, axis=1)

def load_dataset(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset: {name}")
    file, target, drops = DATASETS[name]
    num_parties = PARTIES.get(name, 3) 
    return load_csv_data(file, target, drops, num_parties), target


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
            min_feature = min(feature_cols, key=lambda c: plain_mi_score.get(c, float('inf')))
            return min_feature
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
    enc = pd.DataFrame(data, dtype=object)
    return enc

def encrypt_data_parts(data_parts, system):
    if hasattr(system, "precompute_message_powers"):
        values_to_prewarm = set()
        for part in data_parts:
            if part.empty:
                continue
            for col in part.columns:
                unique_values = pd.unique(part[col])
                values_to_prewarm.update(int(v) for v in unique_values)
        if values_to_prewarm:
            system.precompute_message_powers(values_to_prewarm)
    return [encrypt_dataframe(p, system) for p in data_parts]

def compute_squared_ranks(ranked_parts):
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
    return squared_parts

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
    # ρ = (Σxy - n*mx*my) / sqrt((Σx² - n*mx²)*(Σy² - n*my²))
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

    # Precompute encrypted sums for target
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

        # Use gmpy2 for exponentiation and modular arithmetic
        enc_xy = powmod_vector_parallel(
            fr,
            target_plain_values,
            n_squared
        )
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

def compute_encrypted_mutual_information(data_parts, target_col, system):
    target_series = None
    for p in data_parts:
        if target_col in p.columns:
            target_series = p[target_col]
            break
    if target_series is None:
        return {}
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

    def process_feature(part, col):
        X = part[col]
        unique_X = pd.unique(X)
        X_plain_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
        X_enc_ind = {
            xv: [system.encryption(int(bit)) for bit in X_plain_ind[xv]]
            for xv in unique_X
        }
        count_x = {}
        for xv in unique_X:
            chain = paillier_add_accumulate_parallel(
                X_enc_ind[xv], n_squared
            )
            count_x[xv] = system.combining_algorithm(chain, [1, 2])

        H_y_given_x = gmpy2.mpfr(0)
        for xv in unique_X:
            cx = count_x[xv]
            if cx == 0:
                continue
            enc_vector = X_enc_ind[xv]
            for y in unique_Y:
                selected_ciphertexts = [enc_vector[i] for i in Y_indices[y]]
                sel_product = paillier_add_accumulate_parallel(
                    selected_ciphertexts, n_squared
                )
                c_xy = system.combining_algorithm(sel_product, [1, 2])
                if c_xy == 0:
                    continue
                p_xy = gmpy2.mpfr(c_xy) / n
                p_y_given_x = gmpy2.mpfr(c_xy) / cx if cx else gmpy2.mpfr(0)
                if p_y_given_x > 0:
                    H_y_given_x -= p_xy * log2_safe(p_y_given_x)
        return col, float(H_y - H_y_given_x)

    for part in data_parts:
        feature_cols = [col for col in part.columns if col != target_col]
        if not feature_cols:
            continue

        for col in feature_cols:
            col_name, value = process_feature(part, col)
            mi[col_name] = value
    return mi


# =====================================================================
# Gini Impurity (Plain)
# =====================================================================

# def compute_gini_impurity(data_parts, target_col):
#     target_series = None
#     for p in data_parts:
#         if target_col in p.columns:
#             target_series = p[target_col]
#             break
#     if target_series is None:
#         return {}
#     n = len(target_series)
#     unique_Y = pd.unique(target_series)
#     Y_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
#     gini = {}
#     for part in data_parts:
#         for col in part.columns:
#             if col == target_col:
#                 continue
#             X = part[col]
#             unique_X = pd.unique(X)
#             X_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
#             count_x = {xv: X_ind[xv].sum() for xv in unique_X}
#             gini_x = 0.0
#             for xv in unique_X:
#                 cx = count_x[xv]
#                 if cx == 0:
#                     continue
#                 sum_sq = 0.0
#                 for y in unique_Y:
#                     c_xy = int((X_ind[xv] & Y_ind[y]).sum())
#                     p_y_given_x = c_xy / cx if cx else 0.0
#                     sum_sq += p_y_given_x * p_y_given_x
#                 gini_x += (1 - sum_sq) * (cx / n)
#             gini[col] = gini_x
#     return gini

# =====================================================================
# Gini Impurity (Encrypted Approximation)
# =====================================================================

# def compute_encrypted_gini_impurity(data_parts, target_col, system):
#     target_series = None
#     for p in data_parts:
#         if target_col in p.columns:
#             target_series = p[target_col]
#             break
#     if target_series is None:
#         return {}
#     n = len(target_series)
#     unique_Y = pd.unique(target_series)
#     Y_plain_ind = {y: (target_series == y).astype(int).to_numpy() for y in unique_Y}
#     n_squared = system.n_squared
#     gini = {}
#     for part in data_parts:
#         for col in part.columns:
#             if col == target_col:
#                 continue
#             X = part[col]
#             unique_X = pd.unique(X)
#             X_plain_ind = {xv: (X == xv).astype(int).to_numpy() for xv in unique_X}
#             X_enc_ind = {
#                 xv: [system.encryption(int(bit)) for bit in X_plain_ind[xv]]
#                 for xv in unique_X
#             }
#             count_x = {}
#             for xv in unique_X:
#                 chain = paillier_add_accumulate(X_enc_ind[xv], n_squared)
#                 count_x[xv] = system.combining_algorithm(chain, [1, 2])
#             gini_x = gmpy2.mpfr(0)
#             for xv in unique_X:
#                 cx = count_x[xv]
#                 if cx == 0:
#                     continue
#                 sum_sq = gmpy2.mpfr(0)
#                 enc_vector = X_enc_ind[xv]
#                 for y in unique_Y:
#                     sel_product = 1
#                     y_mask = Y_plain_ind[y]
#                     for i, bit in enumerate(y_mask):
#                         if bit == 1:
#                             sel_product = (sel_product * int(enc_vector[i])) % n_squared
#                     c_xy = system.combining_algorithm(sel_product, [1, 2])
#                     p_y_given_x = gmpy2.mpfr(c_xy) / cx if cx else gmpy2.mpfr(0)
#                     sum_sq += p_y_given_x * p_y_given_x
#                 gini_x += (gmpy2.mpfr(1) - sum_sq) * (gmpy2.mpfr(cx) / gmpy2.mpfr(n))
#             gini[col] = float(gini_x)
#     return gini



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
    # Save plot instead of showing
    filename = f"plot_elbow_{int(time.time())}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    _log(f"Elbow plot saved to {filename}")
    plt.close()

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
    # Save plot instead of showing
    filename = f"plot_mi_{int(time.time())}.png"
    plt.savefig(filename, dpi=100, bbox_inches='tight')
    _log(f"MI plot saved to {filename}")
    plt.close()

# def print_gini_results(target_col, plain_gini, enc_gini):
#     print(f"\nGini Impurity (target={target_col})")
#     print(f"{'Feature':<25}{'Plain':<22}{'Encrypted'}")
#     for f, v in plain_gini.items():
#         print(f"{f:<25}{v:<22}{enc_gini.get(f, None)}")

# def plot_gini(gini_dict, title):
#     if not gini_dict:
#         return
#     sorted_items = sorted(gini_dict.items(), key=lambda x: x[1], reverse=True)
#     labels, scores = zip(*sorted_items)
#     plt.figure(figsize=(10, 5))
#     plt.plot(range(len(scores)), scores, marker='o')
#     plt.xticks(range(len(labels)), labels, rotation=50, ha='right')
#     plt.title(title)
#     plt.ylabel("Gini Impurity")
#     plt.xlabel("Features")
#     plt.tight_layout()
#     plt.show()
# =====================================================================
# Main
# =====================================================================

def main(dataset_name='beans', verbose=None):
    global VERBOSE, _STARTED_AT
    
    # Parse verbose flag from parameter or environment
    env_verbose = os.getenv("PPFS_VERBOSE", "0")
    VERBOSE = verbose if verbose is not None else _coerce_bool(env_verbose)
    _STARTED_AT = time.perf_counter()
    
    _log(f"Starting PPFS on dataset: {dataset_name}")
    num_parties = PARTIES.get(dataset_name, 3)
    _log(f"Using {num_parties} parties for dataset '{dataset_name}'")
    
    system = tp.ThresholdPaillierCorrectProtocol(threshold=2, num_parties=num_parties, verbose=VERBOSE)
    _log("Loading dataset")
    data_parts, target_col = load_dataset(dataset_name)

    target_feature = get_min_mutual_info_feature(data_parts, target_col)
    _log(f"Target feature: {target_feature}")
    ranked_parts = measure_time(compute_ranks, data_parts, target_col)
    squared_ranks = measure_time(compute_squared_ranks, ranked_parts)

    # Encrypt ranks & squared ranks
    _log("Encrypting ranked parts")
    enc_ranked_parts = measure_time(encrypt_data_parts, ranked_parts, system)
    _log("Encrypting squared ranks")
    enc_squared_parts = measure_time(encrypt_data_parts, squared_ranks, system)

    # Spearman
    _log("Computing Spearman correlation")
    plain_spearman = measure_time(compute_spearman_correlation, ranked_parts, target_feature)
    enc_spearman = measure_time(compute_encrypted_spearman_correlation,
        enc_ranked_parts, enc_squared_parts, ranked_parts, target_feature, system)
    print_spearman_results(target_feature, plain_spearman, enc_spearman)
    plot_elbow(enc_spearman, f"Encrypted Spearman (Dataset={dataset_name}, Target={target_feature})", "Abs Spearman")

    del enc_ranked_parts
    del enc_squared_parts
    del squared_ranks

    # Mutual Information
    _log("Computing mutual information")
    plain_mi = measure_time(compute_mutual_information, data_parts, target_col)
    enc_mi = measure_time(compute_encrypted_mutual_information, data_parts, target_col, system)
    print_mutual_info_results(target_col, plain_mi, enc_mi)
    plot_mi(enc_mi, f"Encrypted MI (Dataset={dataset_name}, Target={target_col})")
    
    _log("PPFS analysis complete")

    # Gini Impurity
    # plain_gini = measure_time(compute_gini_impurity, data_parts, target_col)
    # enc_gini = measure_time(compute_encrypted_gini_impurity, data_parts, target_col, system)
    # print_gini_results(target_col, plain_gini, enc_gini)
    # plot_gini(enc_gini, f"Encrypted Gini (Dataset={dataset_name}, Target={target_col})")



if __name__ == "__main__":
    # Set verbose=True for live progress, or use PPFS_VERBOSE=1 environment variable
    main('divorce', verbose=True)
