import networkx as nx
import time
import argparse
import csv
from multiprocessing import Pool as ProcessPool
from functools import partial
import glob
import os
import sys
import numpy as np
import gzip
from typing import List, Tuple, Dict, Optional

# Adapted for Java-style API format
# For example, able to match .txt's "android.telephony.SmsManager.sendDataMessage" format
# Computes centrality statistics for all nodes in the graph, selects those in SAPI as features, appends sha256 and label, then outputs as .csv
# Directory traversal updated: recursively searches dataset_path for all subdirectories, only processing those named benign or malware
# Bug fixes:
# 1) Recursively collect all .gexf/.gexf.gz files (case-insensitive) under benign/malware directories
# 2) Single-file errors do not affect the whole processing (handled in map)
# 3) Correctly strips double suffix (.gexf.gz) to obtain sha256
# 4) Compatible with reading gzip-compressed GEXF

def parseargs():
    parser = argparse.ArgumentParser(description='Malware Detection with centrality.')
    parser.add_argument('-d', '--dir', help='The path of a dir contains benign and malware.', default="/mnt/data2/wb2024/Methodology/MyWay/data/Graph-mc")
    parser.add_argument('-o', '--output', help='The dir_path or file path of output, if not exist, auto create', default="/mnt/data2/wb2024/Methodology/MalScan/APIChecker-mc/result")
    parser.add_argument('-c', '--centrality', help='The type of centrality: degree, katz, closeness, harmonic, pagerank, eigenvector, authority', default="degree")
    args = parser.parse_args()
    return args

def resolve_output_path(output_arg: str, centrality_type: str) -> str:
    if output_arg.endswith('/'):
        out_dir = output_arg
        csv_path = os.path.join(out_dir, f'{centrality_type}_features.csv')
    else:
        root, ext = os.path.splitext(output_arg)
        if ext.lower() == '.csv':
            csv_path = output_arg
        else:
            csv_path = os.path.join(output_arg, f'{centrality_type}_features.csv')
    # Ensure parent directory exists
    out_dir = os.path.dirname(csv_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    return csv_path

def obtain_sensitive_apis(file):
    if not os.path.isfile(file):
        print(f"Error: sensitive APIs file not found: {file}. Please create it and list one API per line.", flush=True)
        sys.exit(1)

    sensitive_apis = []
    with open(file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                sensitive_apis.append(line)
    return sensitive_apis

def _strip_gexf_suffixes(path: str) -> str:
    """
    Strip .gexf / .gexf.gz (case-insensitive) from the filename, return the stem (usually sha256)
    """
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith('.gexf.gz'):
        return name[:-(len('.gexf.gz'))]
    if lower.endswith('.gexf'):
        return name[:-(len('.gexf'))]
    # Fallback: only strip the last extension
    return os.path.splitext(name)[0]

def callgraph_extraction(file):
    """
    Support for reading both .gexf and .gexf.gz formats
    """
    try:
        lower = file.lower()
        if lower.endswith('.gz'):
            with gzip.open(file, 'rt', encoding='utf-8', errors='ignore') as f:
                CG = nx.read_gexf(f)
        else:
            CG = nx.read_gexf(file)
        return CG
    except Exception as e:
        print(f"[WARN] Failed to read GEXF: {file} ({e})", flush=True)
        return None

def gexf_node_to_api_format(node_name):
    try:
        if node_name.startswith('L') and ';->' in node_name:
            class_path = node_name[1:node_name.index(';')]
            class_path = class_path.replace('/', '.')
            method_name = node_name.split(';->')[1].split('(')[0]
            if method_name == "<init>":
                method_name = "init"
            return f"{class_path}.{method_name}"
        else:
            return node_name
    except Exception:
        return node_name

def get_vector(node_centrality, sensitive_apis):
    # Build a map from API format to centrality value
    node_api_map = {}
    for node, value in node_centrality.items():
        api_fmt = gexf_node_to_api_format(node)
        node_api_map[api_fmt] = value
    vector = []
    for api in sensitive_apis:
        vector.append(node_api_map.get(api, 0))
    return vector

def degree_centrality_feature(CG, sensitive_apis):
    node_centrality = nx.degree_centrality(CG)
    return get_vector(node_centrality, sensitive_apis)

# --------- Non-recursive Katz implementation (power iteration), avoids recursion limit ----------
def _safe_alpha_for_katz(G, default_alpha=0.01):
    """
    For a given graph, return a safe alpha to ensure Katz centrality convergence.
    Estimate the upper bound of spectral radius using max degree: alpha < 1 / lambda_max <= 1 / d_max
    """
    try:
        if G.is_directed():
            degrees = [d for _, d in G.out_degree()]
        else:
            degrees = [d for _, d in G.degree()]
        dmax = max(degrees) if degrees else 1
        # 0.9/dmax leaves some margin, pick the smaller of that and default for robustness
        return float(min(default_alpha, 0.9 / max(1.0, dmax)))
    except Exception:
        return float(default_alpha)

def _katz_centrality_power(G, alpha=None, beta=1.0, max_iter=10000, tol=1e-6, use_weights=False):
    """
    Compute Katz centrality with power iteration: x_{t+1} = alpha * A * x_t + beta
    - Non-recursive implementation, avoids recursion limit
    - Prefer sparse matrix with SciPy if available, fallback to NumPy dense otherwise
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    if alpha is None:
        alpha = _safe_alpha_for_katz(G, default_alpha=0.01)

    weight = 'weight' if use_weights else None

    # Try using SciPy sparse matrix
    A = None
    use_scipy = False
    try:
        from scipy.sparse import issparse  # noqa: F401
        use_scipy = True
    except Exception:
        use_scipy = False

    if use_scipy:
        try:
            A = nx.to_scipy_sparse_array(G, nodelist=nodes, dtype=float, weight=weight, format='csr')
        except Exception:
            A = None

    # If no SciPy or build fails, use NumPy dense
    if A is None:
        A = nx.to_numpy_array(G, nodelist=nodes, dtype=float, weight=weight)

    x = np.ones(n, dtype=float)
    b = np.ones(n, dtype=float) * beta

    # Iterations
    for _ in range(max_iter):
        if use_scipy and hasattr(A, "dot"):
            x_new = alpha * (A.dot(x)) + b
        else:
            x_new = alpha * (A @ x) + b
        # Use L1 norm for convergence check
        if np.linalg.norm(x_new - x, 1) < tol * n:
            x = x_new
            break
        x = x_new

    # Optional: normalization to avoid large numbers
    s = np.sum(x)
    if s > 0:
        x = x / s

    return {nodes[i]: float(x[i]) for i in range(n)}
# ---------------------------------------------------------------------

def katz_centrality_feature(CG, sensitive_apis):
    try:
        node_centrality = _katz_centrality_power(CG, alpha=None, beta=1.0, max_iter=1000, tol=1e-6, use_weights=False)
    except Exception:
        node_centrality = {node: 0.0 for node in CG.nodes()}
    return get_vector(node_centrality, sensitive_apis)

def closeness_centrality_feature(CG, sensitive_apis):
    node_centrality = nx.closeness_centrality(CG)
    return get_vector(node_centrality, sensitive_apis)

def harmonic_centrality_feature(CG, sensitive_apis):
    node_centrality = nx.harmonic_centrality(CG)
    return get_vector(node_centrality, sensitive_apis)

def pagerank_centrality_feature(CG, sensitive_apis):
    node_centrality = nx.pagerank(CG)
    return get_vector(node_centrality, sensitive_apis)

def eigenvector_centrality_feature(CG, sensitive_apis):
    try:
        node_centrality = nx.eigenvector_centrality(CG, max_iter=1000)
    except nx.NetworkXException:
        # If not converged, return all-zero vector
        node_centrality = {node: 0 for node in CG.nodes()}
    return get_vector(node_centrality, sensitive_apis)

def authority_centrality_feature(CG, sensitive_apis):
    # Use the HITS algorithm to get authority scores
    try:
        _, authority_scores = nx.hits(CG, max_iter=1000)
    except nx.NetworkXException:
        authority_scores = {node: 0 for node in CG.nodes()}
    return get_vector(authority_scores, sensitive_apis)

CENTRALITY_FUNCS = {
    'degree': degree_centrality_feature,
    'katz': katz_centrality_feature,
    'closeness': closeness_centrality_feature,
    'harmonic': harmonic_centrality_feature,
    'pagerank': pagerank_centrality_feature,
    'eigenvector': eigenvector_centrality_feature,
    'authority': authority_centrality_feature,
}

def find_benign_malware_dirs(root_dir: str):
    """
    Recursively traverse root_dir, collect all directory paths named 'benign' and 'malware'.
    """
    benign_dirs = []
    malware_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        base = os.path.basename(dirpath)
        if base == 'benign':
            benign_dirs.append(dirpath)
        elif base == 'malware':
            malware_dirs.append(dirpath)
    return benign_dirs, malware_dirs

def collect_gexf_files(root_dir: str, recursive: bool = True) -> List[str]:
    """
    Collect all .gexf / .gexf.gz files (case-insensitive) under root_dir
    """
    ret = []
    if recursive:
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for fn in filenames:
                lower = fn.lower()
                if lower.endswith('.gexf') or lower.endswith('.gexf.gz'):
                    ret.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root_dir):
            fpath = os.path.join(root_dir, fn)
            if not os.path.isfile(fpath):
                continue
            lower = fn.lower()
            if lower.endswith('.gexf') or lower.endswith('.gexf.gz'):
                ret.append(fpath)
    return ret

def _safe_compute_one(file: str, centrality_type: str, sensitive_apis: List[str]) -> Optional[Tuple[str, List[float]]]:
    """
    Safely compute features for a single file; any exception only affects this file and will not affect others.
    """
    try:
        CG = callgraph_extraction(file)
        if CG is None:
            return None
        func = CENTRALITY_FUNCS.get(centrality_type)
        if func is None:
            print(f"[ERROR] Unknown centrality type: {centrality_type}", flush=True)
            return None
        vector = func(CG, sensitive_apis)
        sha256 = _strip_gexf_suffixes(file)
        return (sha256, vector)
    except Exception as e:
        print(f"[WARN] Failed to compute feature for {file}: {e}", flush=True)
        return None

def obtain_dataset(dataset_path, centrality_type, sensitive_apis):
    Vectors = []
    Labels = []

    if not os.path.exists(dataset_path):
        print(f"Error: dataset path does not exist: {dataset_path}", flush=True)
        return Vectors, Labels

    # Recursively find all directories named benign and malware
    benign_dirs, malware_dirs = find_benign_malware_dirs(dataset_path)

    if not benign_dirs and not malware_dirs:
        print(f"Warning: No 'benign' or 'malware' directories found under: {dataset_path}", flush=True)

    # Recursively collect all gexf files (including .gexf.gz, case-insensitive)
    apps_b = []
    for bdir in benign_dirs:
        apps_b.extend(collect_gexf_files(bdir, recursive=True))
    apps_m = []
    for mdir in malware_dirs:
        apps_m.extend(collect_gexf_files(mdir, recursive=True))

    # Deduplicate and sort for stability
    apps_b = sorted(set(apps_b))
    apps_m = sorted(set(apps_m))

    procs = min(32, os.cpu_count() or 4)
    pool = ProcessPool(processes=procs)
    try:
        worker = partial(_safe_compute_one, centrality_type=centrality_type, sensitive_apis=sensitive_apis)
        results_b = pool.map(worker, apps_b) if apps_b else []
        results_m = pool.map(worker, apps_m) if apps_m else []
    finally:
        pool.close()
        pool.join()

    # Filter out None (failed) results
    results_b = [r for r in results_b if r is not None]
    results_m = [r for r in results_m if r is not None]

    Vectors.extend(results_b)
    Labels.extend([0 for _ in range(len(results_b))])

    Vectors.extend(results_m)
    Labels.extend([1 for _ in range(len(results_m))])

    # Inform if any files were skipped
    skipped_b = len(apps_b) - len(results_b)
    skipped_m = len(apps_m) - len(results_m)
    if skipped_b or skipped_m:
        print(f"[INFO] Processed benign: {len(results_b)}/{len(apps_b)}, malware: {len(results_m)}/{len(apps_m)}. Skipped: {skipped_b + skipped_m}", flush=True)

    return Vectors, Labels

def main():
    sensitive_apis_path = 'APIChecker.txt'
    sensitive_apis = obtain_sensitive_apis(sensitive_apis_path)

    args = parseargs()
    dataset_path = args.dir
    cetrality_type = args.centrality

    Vectors, Labels = obtain_dataset(dataset_path, cetrality_type, sensitive_apis)
    feature_csv = [[] for _ in range(len(Labels)+1)]
    feature_csv[0].append('SHA256')
    feature_csv[0].extend(sensitive_apis)
    feature_csv[0].append('Label')

    for i in range(len(Labels)):
        (sha256, vector) = Vectors[i]
        feature_csv[i+1].append(sha256)
        feature_csv[i+1].extend(vector)
        feature_csv[i+1].append(Labels[i])

    # Resolve output path and ensure directory exists
    csv_path = resolve_output_path(args.output, cetrality_type)

    with open(csv_path, 'w', newline='') as f:
        csvfile = csv.writer(f)
        csvfile.writerows(feature_csv)

if __name__ == '__main__':
    main()