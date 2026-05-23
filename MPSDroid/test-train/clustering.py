#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D clustering (with auto k option and optional MiniBatchKMeans) on "malicious value" vectors for a single features_train.csv / features_test.csv pair.
Progress is displayed live on console; no JSON summary output.

Input:
  features_train.csv: columns sha256, seq_values (JSON array), label
  features_test.csv : same as above

Output (inside --root-dir):
  model.joblib
  train/cluster_summary.csv
  train/file_cluster_distribution.csv
  test/cluster_summary.csv
  test/file_cluster_distribution.csv

Notes:
- Every file's cluster distribution will have one row per sha256; even if a sha's seq_values is empty, a row is written with all zeros for clusters (and the label).
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# Increase CSV single field size limit to avoid "field larger than field limit (131072)"
try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)


def is_tty() -> bool:
    return sys.stdout.isatty()


def tprint(msg: str, end: str = "\n", flush: bool = True):
    sys.stdout.write(msg + end)
    if flush:
        sys.stdout.flush()


def read_feature_csv(path: str, progress: bool = False) -> List[Tuple[str, List[float], int]]:
    """
    Read features_{train|test}.csv and return [(sha, [values...], label), ...]
    """
    rows = []
    start = time.time()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        need = {"sha256", "seq_values", "label"}
        if not need.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"{path} missing required columns: sha256, seq_values, label")
        for i, r in enumerate(rd, start=1):
            sha = (r.get("sha256") or "").strip()
            try:
                label = int((r.get("label") or "").strip())
            except Exception:
                label = -1
            seq_json = (r.get("seq_values") or "").strip()
            if not seq_json:
                vals = []
            else:
                try:
                    vals = json.loads(seq_json)
                    if not isinstance(vals, list):
                        vals = []
                except Exception:
                    vals = []
            # cast to floats only
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            rows.append((sha, vals, label))
            if progress and i % 2000 == 0 and is_tty():
                sys.stdout.write(f"\rRead {os.path.basename(path)}: {i} rows...".ljust(80))
                sys.stdout.flush()
        if progress and is_tty():
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
    if progress:
        tprint(f"Finished reading {os.path.basename(path)}: {len(rows)} rows in {int(time.time()-start)}s")
    return rows


def flatten_values(rows: List[Tuple[str, List[float], int]]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    Flatten (sha, [values], label) to:
    - X: all float values, shape (N,)
    - owners: sha list aligned with X
    - sha_to_label: sha -> label
    """
    vals: List[float] = []
    owners: List[str] = []
    sha_to_label: Dict[str, int] = {}
    for sha, vs, label in rows:
        sha_to_label[sha] = label
        for v in vs:
            vals.append(v)
            owners.append(sha)
    X = np.asarray(vals, dtype=np.float64)
    return X, owners, sha_to_label


def reservoir_sample_1d(X: np.ndarray, k: int, seed: int) -> np.ndarray:
    """Reservoir sampling for 1D np.ndarray. Returns a sample with length <= k."""
    n = X.shape[0]
    if k <= 0 or k >= n:
        return X
    rng = np.random.default_rng(seed)
    sample = X[:k].copy()
    for i in range(k, n):
        j = rng.integers(0, i + 1)
        if j < k:
            sample[j] = X[i]
    return sample


def auto_choose_k_on_scores(
    scores: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int,
    use_minibatch: bool,
    sil_sample_size: int,
    show_progress: bool = False,
) -> Tuple[int, float]:
    """
    Use silhouette score to choose best k in [k_min, k_max].
    The greater the score, the better. Return (best_k, best_score); fallback to k=2 if unable to calculate.
    """
    if scores.size < 3:
        n = scores.size
        return max(1, min(k_min, n)), float("nan")

    X = scores.reshape(-1, 1)

    best_k, best_sc = None, -1.0
    k_min_eff = max(2, k_min)
    k_max_eff = max(k_min_eff, k_max)
    total = max(1, k_max_eff - k_min_eff + 1)

    for idx, k in enumerate(range(k_min_eff, k_max_eff + 1), start=1):
        if k >= X.shape[0]:
            break
        Model = MiniBatchKMeans if use_minibatch else KMeans
        model = Model(n_clusters=k, random_state=random_state, n_init="auto")
        labels = model.fit_predict(X)
        if len(set(labels)) < 2:
            sc = float("nan")
        else:
            try:
                sc = silhouette_score(
                    X,
                    labels,
                    metric="euclidean",
                    sample_size=min(sil_sample_size, X.shape[0]),
                    random_state=random_state,
                )
            except Exception:
                sc = float("nan")

        if not math.isnan(sc) and sc > best_sc:
            best_sc, best_k = sc, k

        if show_progress:
            pct = idx / total * 100.0
            msg = (
                f"  Selecting k: {idx}/{total} ({pct:5.1f}%) current k={k} "
                f"score={('nan' if math.isnan(sc) else f'{sc:.4f}')} "
                f"best k={best_k} "
                f"best score={('nan' if best_sc<0 or math.isnan(best_sc) else f'{best_sc:.4f}')}"
            )
            if is_tty():
                sys.stdout.write("\r" + msg.ljust(120))
                sys.stdout.flush()
            else:
                tprint(msg)
    if show_progress and is_tty():
        sys.stdout.write("\n")
        sys.stdout.flush()

    if best_k is None:
        return max(2, k_min_eff), float("nan")
    return best_k, best_sc


def run_clustering_for_pair(
    root_dir: str,
    n_clusters: int,
    use_minibatch: bool,
    auto_k_min: int,
    auto_k_max: int,
    auto_k_sample: int,
    sil_sample_size: int,
    random_state: int,
    verbose: bool,
    show_progress: bool,
) -> None:
    begin = time.time()
    tprint(f"\n=== Begin processing directory: {root_dir} ===")

    train_csv = os.path.join(root_dir, "features_train.csv")
    test_csv = os.path.join(root_dir, "features_test.csv")
    if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
        raise FileNotFoundError(f"Missing features_train.csv or features_test.csv in: {root_dir}")

    # Read train/test CSVs
    tprint(f"[{root_dir}] Reading train CSV ...")
    train_rows = read_feature_csv(train_csv, progress=show_progress)
    tprint(f"[{root_dir}] Reading test CSV ...")
    test_rows = read_feature_csv(test_csv, progress=show_progress)

    # Flatten train values
    X_train, owners_train, sha_to_label_train = flatten_values(train_rows)
    if X_train.size == 0:
        raise RuntimeError(f"[{root_dir}] No malicious values in train set")
    tprint(f"[{root_dir}] Train values count: {X_train.size}")

    # Choose k (auto or manual)
    if n_clusters and n_clusters > 0:
        k = int(n_clusters)
        tprint(f"[{root_dir}] Using fixed k={k}")
    else:
        tprint(f"[{root_dir}] Auto selecting k, sample cap={auto_k_sample}, range=[{auto_k_min},{auto_k_max}]")
        X_sample = reservoir_sample_1d(X_train, auto_k_sample, random_state)
        k, sil = auto_choose_k_on_scores(
            X_sample,
            k_min=auto_k_min,
            k_max=auto_k_max,
            random_state=random_state,
            use_minibatch=True if use_minibatch else False,
            sil_sample_size=sil_sample_size,
            show_progress=show_progress,
        )
        if math.isnan(sil):
            tprint(f"[{root_dir}] Auto-selected k={k}")
        else:
            tprint(f"[{root_dir}] Auto-selected k={k} (silhouette={sil:.4f})")

    # Fit clustering model
    mode = "MiniBatchKMeans" if use_minibatch else "KMeans"
    t0 = time.time()
    tprint(f"[{root_dir}] Training model {mode} (k={k}, n={X_train.size}) ...")
    Model = MiniBatchKMeans if use_minibatch else KMeans
    kmeans = Model(n_clusters=k, random_state=random_state, n_init="auto")
    kmeans.fit(X_train.reshape(-1, 1))
    tprint(f"[{root_dir}] Model training finished, elapsed {int(time.time()-t0)}s")

    centers = kmeans.cluster_centers_.reshape(-1)

    # Aggregate train set
    tprint(f"[{root_dir}] Predicting and counting train distribution ...")
    t1 = time.time()
    labels_tr = kmeans.predict(X_train.reshape(-1, 1))
    cluster_counts_tr = np.bincount(labels_tr, minlength=k)
    cluster_sum_tr = np.bincount(labels_tr, weights=X_train, minlength=k)
    per_file_counts_tr: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sha, lab in zip(owners_train, labels_tr):
        per_file_counts_tr[sha][int(lab)] += 1
    tprint(f"[{root_dir}] Train distribution finished in {int(time.time()-t1)}s")

    # Write train output (covering all sha, empty file output is all 0s)
    out_train = os.path.join(root_dir, "train")
    os.makedirs(out_train, exist_ok=True)
    tprint(f"[{root_dir}] Writing train output CSV ...")
    with open(os.path.join(out_train, "cluster_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "avg_value", "center"])
        for ci in range(k):
            size = int(cluster_counts_tr[ci])
            avg = (cluster_sum_tr[ci] / size) if size > 0 else float("nan")
            w.writerow([ci, size, f"{avg:.6f}", f"{centers[ci]:.6f}"])

    with open(os.path.join(out_train, "file_cluster_distribution.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sha256"] + [f"cluster_{ci}" for ci in range(k)] + ["label"]
        w.writerow(header)
        all_train_shas = sorted(sha_to_label_train.keys())
        for sha in all_train_shas:
            row = [sha]
            counts = per_file_counts_tr.get(sha, {})  # Empty file => all 0
            for ci in range(k):
                row.append(int(counts.get(ci, 0)))
            row.append(int(sha_to_label_train.get(sha, -1)))
            w.writerow(row)
    tprint(f"[{root_dir}] Train output written")

    # Test set
    X_test, owners_test, sha_to_label_test = flatten_values(test_rows)
    tprint(f"[{root_dir}] Test values count: {X_test.size}")
    tprint(f"[{root_dir}] Predicting and counting test distribution ...")
    t2 = time.time()
    labels_te = np.array([], dtype=int)
    if X_test.size > 0:
        labels_te = kmeans.predict(X_test.reshape(-1, 1))
    cluster_counts_te = np.bincount(labels_te, minlength=k) if labels_te.size else np.zeros(k, dtype=int)
    cluster_sum_te = np.bincount(labels_te, weights=X_test, minlength=k) if labels_te.size else np.zeros(k, dtype=float)
    per_file_counts_te: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sha, lab in zip(owners_test, labels_te):
        per_file_counts_te[sha][int(lab)] += 1
    tprint(f"[{root_dir}] Test distribution finished in {int(time.time()-t2)}s")

    out_test = os.path.join(root_dir, "test")
    os.makedirs(out_test, exist_ok=True)
    tprint(f"[{root_dir}] Writing test output CSV ...")
    with open(os.path.join(out_test, "cluster_summary.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "size", "avg_value", "center"])
        for ci in range(k):
            size = int(cluster_counts_te[ci])
            avg = (cluster_sum_te[ci] / size) if size > 0 else float("nan")
            w.writerow([ci, size, f"{avg:.6f}", f"{centers[ci]:.6f}"])

    with open(os.path.join(out_test, "file_cluster_distribution.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        header = ["sha256"] + [f"cluster_{ci}" for ci in range(k)] + ["label"]
        w.writerow(header)
        all_test_shas = sorted(sha_to_label_test.keys())
        for sha in all_test_shas:
            row = [sha]
            counts = per_file_counts_te.get(sha, {})  # Empty file => all 0
            for ci in range(k):
                row.append(int(counts.get(ci, 0)))
            row.append(int(sha_to_label_test.get(sha, -1)))
            w.writerow(row)
    tprint(f"[{root_dir}] Test output written")

    # Save model
    tprint(f"[{root_dir}] Saving model model.joblib ...")
    dump(kmeans, os.path.join(root_dir, "model.joblib"))
    tprint(f"[{root_dir}] Model saved")

    tprint(f"=== Finished {root_dir} (k={kmeans.n_clusters}), total time {int(time.time()-begin)}s ===")


def main():
    ap = argparse.ArgumentParser(description="1D clustering (auto-k, progress shown) on malicious value of features_train/test.csv pair")
    ap.add_argument(
        "--root-dir",
       default="./statistic",
        help="Directory containing features_train.csv and features_test.csv"
    )
    ap.add_argument("--n-clusters", type=int, default=2000, help="Fixed number of clusters (>0 disables auto-k)")
    ap.add_argument("--auto-k-min", type=int, default=1800, help="Auto-k: min number of clusters")
    ap.add_argument("--auto-k-max", type=int, default=2000, help="Auto-k: max number of clusters")
    ap.add_argument("--auto-k-sample", type=int, default=20000, help="Auto-k: max sample for silhouette scoring (reservoir sample)")
    ap.add_argument("--sil-sample-size", type=int, default=10000, help="Silhouette score max sample size")
    ap.add_argument("--use-minibatch", action="store_true", help="Use MiniBatchKMeans (better for large datasets)")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed")
    ap.add_argument("--verbose", action="store_true", help="Print more detailed logs")
    ap.add_argument("--progress", action="store_true", help="Show progress and timing for each stage")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    os.makedirs(root_dir, exist_ok=True)

    mode = "MiniBatchKMeans" if args.use_minibatch else "KMeans"
    auto_mode = "auto-k" if args.n_clusters <= 0 else f"fixed k={args.n_clusters}"
    tprint(f"root-dir={root_dir}, mode={mode}, {auto_mode}")

    run_clustering_for_pair(
        root_dir=root_dir,
        n_clusters=args.n_clusters,
        use_minibatch=args.use_minibatch,
        auto_k_min=args.auto_k_min,
        auto_k_max=args.auto_k_max,
        auto_k_sample=args.auto_k_sample,
        sil_sample_size=args.sil_sample_size,
        random_state=args.random_state,
        verbose=args.verbose,
        show_progress=args.progress,
    )


if __name__ == "__main__":
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")