#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按每折 features_train.csv / features_test.csv 的恶意值做 1D 聚类（自动选 k，可选 MiniBatchKMeans）。
控制台实时显示处理过程；不输出 JSON 摘要。

输入（每个 fold 目录）:
  features_train.csv: 列 sha256, seq_values(JSON数组), label
  features_test.csv : 同上

输出（每折）:
  model.joblib
  train/cluster_summary.csv
  train/file_cluster_distribution.csv
  test/cluster_summary.csv
  test/file_cluster_distribution.csv

更新：
- 每文件簇分布现在会对所有 sha256 输出一行；即便该 sha 对应的 seq_values 为空数组，也会写入全 0 的簇计数行（并带 label）。
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
from joblib import dump
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

# 提升 CSV 单字段大小上限，避免 "field larger than field limit (131072)"
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
    读取 feature_{train|test}.csv，返回 [(sha, [values...], label), ...]
    """
    rows = []
    start = time.time()
    with open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        need = {"sha256", "seq_values", "label"}
        if not need.issubset(set(rd.fieldnames or [])):
            raise ValueError(f"{path} 缺少必要列: sha256, seq_values, label")
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
            # 过滤成浮点
            vals = [float(v) for v in vals if isinstance(v, (int, float))]
            rows.append((sha, vals, label))
            if progress and i % 2000 == 0:
                if is_tty():
                    sys.stdout.write(f"\r读取 {os.path.basename(path)}: {i} 行...".ljust(80))
                    sys.stdout.flush()
        if progress and is_tty():
            sys.stdout.write("\r" + " " * 80 + "\r")
            sys.stdout.flush()
    if progress:
        tprint(f"读取完成 {os.path.basename(path)}: {len(rows)} 行，用时 {int(time.time()-start)}s")
    return rows


def flatten_values(rows: List[Tuple[str, List[float], int]]) -> Tuple[np.ndarray, List[str], Dict[str, int]]:
    """
    将 (sha, [values], label) 扁平化为:
    - X: 所有恶意值（float64） shape (N,)
    - owners: 与 X 对齐的 sha 列表
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
    """对 1D 向量做蓄水池采样，返回长度<=k 的样本。"""
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
    使用 silhouette score 在 [k_min, k_max] 上选择最佳 k。
    分数越高越好。返回 (best_k, best_score)；若无法计算，回退为 k=2。
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
            msg = f"  选择k进度: {idx}/{total} ({pct:5.1f}%) 当前k={k} 分数={('nan' if math.isnan(sc) else f'{sc:.4f}')} 最优k={best_k} 最优分={('nan' if best_sc<0 or math.isnan(best_sc) else f'{best_sc:.4f}')}"
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


def process_one_fold(
    fold_dir: str,
    n_clusters: int,
    use_minibatch: bool,
    auto_k_min: int,
    auto_k_max: int,
    auto_k_sample: int,
    sil_sample_size: int,
    random_state: int,
    verbose: bool,
    show_progress: bool,
) -> Tuple[bool, str]:
    begin = time.time()
    tprint(f"\n=== 开始处理 {fold_dir} ===")

    train_csv = os.path.join(fold_dir, "features_train.csv")
    test_csv = os.path.join(fold_dir, "features_test.csv")
    if not os.path.isfile(train_csv) or not os.path.isfile(test_csv):
        return False, f"[{fold_dir}] 缺少 features_train.csv 或 features_test.csv"

    # 读取训练与测试 CSV
    tprint(f"[{fold_dir}] 读取训练 CSV ...")
    train_rows = read_feature_csv(train_csv, progress=show_progress)
    tprint(f"[{fold_dir}] 读取测试 CSV ...")
    test_rows = read_feature_csv(test_csv, progress=show_progress)

    # 展平训练值
    X_train, owners_train, sha_to_label_train = flatten_values(train_rows)
    if X_train.size == 0:
        return False, f"[{fold_dir}] 训练集没有任何恶意值"
    tprint(f"[{fold_dir}] 训练恶意值总数: {X_train.size}")

    # 自动选 k（若未指定）
    if n_clusters and n_clusters > 0:
        k = int(n_clusters)
        tprint(f"[{fold_dir}] 使用固定 k={k}")
    else:
        tprint(f"[{fold_dir}] 自动选择 k，采样上限={auto_k_sample}，范围=[{auto_k_min},{auto_k_max}]")
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
            tprint(f"[{fold_dir}] 自动选 k={k}")
        else:
            tprint(f"[{fold_dir}] 自动选 k={k} (silhouette={sil:.4f})")

    # 训练聚类模型
    mode = "MiniBatchKMeans" if use_minibatch else "KMeans"
    t0 = time.time()
    tprint(f"[{fold_dir}] 训练模型 {mode} (k={k}, n={X_train.size}) ...")
    Model = MiniBatchKMeans if use_minibatch else KMeans
    kmeans = Model(n_clusters=k, random_state=random_state, n_init="auto")
    kmeans.fit(X_train.reshape(-1, 1))
    tprint(f"[{fold_dir}] 模型训练完成，用时 {int(time.time()-t0)}s")

    centers = kmeans.cluster_centers_.reshape(-1)

    # 训练集聚合
    tprint(f"[{fold_dir}] 预测并统计训练分布 ...")
    t1 = time.time()
    labels_tr = kmeans.predict(X_train.reshape(-1, 1))
    cluster_counts_tr = np.bincount(labels_tr, minlength=k)
    cluster_sum_tr = np.bincount(labels_tr, weights=X_train, minlength=k)
    per_file_counts_tr: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sha, lab in zip(owners_train, labels_tr):
        per_file_counts_tr[sha][int(lab)] += 1
    tprint(f"[{fold_dir}] 训练分布统计完成，用时 {int(time.time()-t1)}s")

    # 写训练结果（包含所有 sha，空文件输出全 0）
    out_train = os.path.join(fold_dir, "train")
    os.makedirs(out_train, exist_ok=True)
    tprint(f"[{fold_dir}] 写训练输出 CSV ...")
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
        # 遍历所有训练 sha（包括空序列的 sha）
        all_train_shas = sorted(sha_to_label_train.keys())
        for sha in all_train_shas:
            row = [sha]
            counts = per_file_counts_tr.get(sha, {})  # 可能为空：空文件 → 全 0
            for ci in range(k):
                row.append(int(counts.get(ci, 0)))
            row.append(int(sha_to_label_train.get(sha, -1)))
            w.writerow(row)
    tprint(f"[{fold_dir}] 训练输出完成")

    # 测试集
    X_test, owners_test, sha_to_label_test = flatten_values(test_rows)
    tprint(f"[{fold_dir}] 测试恶意值总数: {X_test.size}")
    tprint(f"[{fold_dir}] 预测并统计测试分布 ...")
    t2 = time.time()
    labels_te = np.array([], dtype=int)
    if X_test.size > 0:
        labels_te = kmeans.predict(X_test.reshape(-1, 1))
    cluster_counts_te = np.bincount(labels_te, minlength=k) if labels_te.size else np.zeros(k, dtype=int)
    cluster_sum_te = np.bincount(labels_te, weights=X_test, minlength=k) if labels_te.size else np.zeros(k, dtype=float)
    per_file_counts_te: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for sha, lab in zip(owners_test, labels_te):
        per_file_counts_te[sha][int(lab)] += 1
    tprint(f"[{fold_dir}] 测试分布统计完成，用时 {int(time.time()-t2)}s")

    out_test = os.path.join(fold_dir, "test")
    os.makedirs(out_test, exist_ok=True)
    tprint(f"[{fold_dir}] 写测试输出 CSV ...")
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
        # 遍历所有测试 sha（包括空序列的 sha）
        all_test_shas = sorted(sha_to_label_test.keys())
        for sha in all_test_shas:
            row = [sha]
            counts = per_file_counts_te.get(sha, {})  # 可能为空：空文件 → 全 0
            for ci in range(k):
                row.append(int(counts.get(ci, 0)))
            row.append(int(sha_to_label_test.get(sha, -1)))
            w.writerow(row)
    tprint(f"[{fold_dir}] 测试输出完成")

    # 保存模型
    tprint(f"[{fold_dir}] 保存模型 model.joblib ...")
    dump(kmeans, os.path.join(fold_dir, "model.joblib"))
    tprint(f"[{fold_dir}] 模型保存完成")

    tprint(f"=== 完成 {fold_dir} (k={kmeans.n_clusters})，总用时 {int(time.time()-begin)}s ===")
    return True, f"[{fold_dir}] 成功：k={kmeans.n_clusters}, train_vals={X_train.size}, test_vals={X_test.size}"


def main():
    ap = argparse.ArgumentParser(description="按 features_train/test.csv 的恶意值做 1D 聚类（自动选 k，显示处理过程）")
    ap.add_argument("--folds-root", default="/mnt/data2/wb2024/Methodology/MyWay2.0/fold_outputs-mc", help="包含 fold_01.. 的根目录")
    ap.add_argument("--n-clusters", type=int, default=2000, help="固定聚类数（>0 时跳过自动选 k）")
    ap.add_argument("--auto-k-min", type=int, default=1800, help="自动选 k 最小值")
    ap.add_argument("--auto-k-max", type=int, default=2000, help="自动选 k 最大值")
    ap.add_argument("--auto-k-sample", type=int, default=20000, help="用于自动选 k 的采样上限（蓄水池采样）")
    ap.add_argument("--sil-sample-size", type=int, default=10000, help="silhouette 评分采样上限")
    ap.add_argument("--use-minibatch", action="store_true", help="使用 MiniBatchKMeans（更适合大数据）")
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 8) // 8), help="并行处理折的进程数")
    ap.add_argument("--random-state", type=int, default=42, help="随机种子")
    ap.add_argument("--verbose", action="store_true", help="打印更详细日志")
    ap.add_argument("--progress", action="store_true", help="显示各阶段进度/用时")
    args = ap.parse_args()

    folds = sorted([p for p in glob.glob(os.path.join(args.folds_root, "fold_*")) if os.path.isdir(p)])
    if not folds:
        print(f"未在 {args.folds_root} 下找到任何折目录 fold_*")
        sys.exit(1)

    mode = "MiniBatchKMeans" if args.use_minibatch else "KMeans"
    auto_mode = "auto-k" if args.n_clusters <= 0 else f"fixed k={args.n_clusters}"
    tprint(f"发现 {len(folds)} 个折。jobs={args.jobs}, mode={mode}, {auto_mode}")

    ok, fail = 0, 0
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        fut_map = {
            ex.submit(
                process_one_fold,
                fold_dir,
                args.n_clusters,
                args.use_minibatch,
                args.auto_k_min,
                args.auto_k_max,
                args.auto_k_sample,
                args.sil_sample_size,
                args.random_state,
                args.verbose,
                args.progress,
            ): fold_dir
            for fold_dir in folds
        }
        for fut in as_completed(fut_map):
            fold_dir = fut_map[fut]
            try:
                success, msg = fut.result()
                tprint(msg)
                if success:
                    ok += 1
                else:
                    fail += 1
            except Exception as e:
                tprint(f"[{fold_dir}] 失败: {e}")
                fail += 1

    tprint(f"\n全部完成。成功 {ok}，失败 {fail}。")


if __name__ == "__main__":
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")