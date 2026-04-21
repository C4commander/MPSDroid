#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高并行版本：处理敏感 API 序列数据（single-cell 输出，带进度）。

本版本：
- 不再使用 fold_root/fold_XX 结构；
- 使用单一 root-dir，其中包含 {train_sha.txt, test_sha.txt, weights.csv}；
- 分别指定训练/测试序列根目录 train-data-root / test-data-root，
  它们的目录结构均为：任意层级下包含 malware / benign 子目录，子目录内为 *.txt（文件名为 sha256）。

输入：
- root_dir/train_sha.txt：训练样本 sha 列表
- root_dir/test_sha.txt ：测试样本 sha 列表
- root_dir/weights.csv ：API 权重表（含 API, weight）
- train_data_root 下任意层级包含 malware / benign 的子目录里的 *.txt
- test_data_root  同上

输出（在 root_dir 内）：
- features_train.csv（列：sha256, seq_values(JSON数组), label）
- features_test.csv
- metadata.json（参数与统计）
"""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

# 固定输出文件名前缀
OUTPUT_PREFIX = 'features'


# -----------------------------
# 工具函数与通用输出
# -----------------------------
def _norm_header(s: str) -> str:
    return s.lstrip('\ufeff').strip().lower()


def sniff_dialect(sample: str):
    import csv as _csv
    try:
        return _csv.Sniffer().sniff(sample, delimiters=',;\t')
    except Exception:
        class _D(_csv.Dialect):
            delimiter = ','
            quotechar = '"'
            doublequote = True
            skipinitialspace = True
            lineterminator = '\n'
            quoting = csv.QUOTE_MINIMAL
        return _D()


def format_seconds(sec: float) -> str:
    if sec is None or sec <= 0:
        return "--"
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m}m{s}s"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def is_tty() -> bool:
    return sys.stdout.isatty()


def print_progress(prefix: str, done: int, total: int, start_time: float,
                   recent_durations: deque):
    pct = (done / total * 100) if total else 0
    if recent_durations:
        avg = sum(recent_durations) / len(recent_durations)
        remain = (total - done) * avg
    else:
        avg = None
        remain = None
    rate_info = f"avg={avg:.2f}s" if avg else "avg=--"
    remain_info = f"eta={format_seconds(remain)}"
    line = f"{prefix} {done}/{total} ({pct:5.1f}%) {rate_info} {remain_info}"
    if is_tty():
        sys.stdout.write("\r" + line.ljust(110))
        sys.stdout.flush()
        if done == total:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        print(line)


# -----------------------------
# 权重与敏感 API
# -----------------------------
def load_api_weights(weights_file: Path, encoding='utf-8') -> Dict[str, float]:
    api_weights = {}
    with weights_file.open('r', encoding=encoding, newline='') as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = sniff_dialect(sample)
        reader = csv.DictReader(f, dialect=dialect)
        if not reader.fieldnames:
            raise ValueError(f"无法读取权重表头: {weights_file}")
        norm2orig = {_norm_header(h): h for h in reader.fieldnames if h}
        if 'api' not in norm2orig or 'weight' not in norm2orig:
            raise ValueError(
                f"weights.csv 缺少必须列(API, weight): {weights_file}\n实际表头: {reader.fieldnames}"
            )
        api_col = norm2orig['api']
        weight_col = norm2orig['weight']
        for row in reader:
            name = (row.get(api_col) or '').strip()
            if not name:
                continue
            w_raw = (row.get(weight_col) or '').strip()
            try:
                w = float(w_raw)
            except Exception:
                w = 0.0
            api_weights[name] = w
    return api_weights


def load_sensitive_api_list(file_path: Path, encoding='utf-8') -> List[str]:
    apis = []
    with file_path.open('r', encoding=encoding) as f:
        for line in f:
            name = line.strip()
            if name:
                apis.append(name)
    if not apis:
        raise ValueError("敏感 API 列表文件为空。")
    return apis


# -----------------------------
# 扫描与预加载
# -----------------------------
def _infer_label_from_parts(parts: List[str]) -> int:
    """
    根据路径分段推断标签：
    - 若分段中出现 'malware' 则 1
    - 若分段中出现 'benign'  则 0
    若两者都出现，按更靠近文件的分段优先（即从末尾向前找第一个匹配）。
    若都无，返回 -1。
    """
    for seg in reversed(parts):
        s = seg.lower()
        if s == 'malware':
            return 1
        if s == 'benign':
            return 0
    return -1


def scan_sha_files(data_root: Path) -> Dict[str, Tuple[Path, int]]:
    """
    深度递归扫描 data_root 下任意层级的 malware/benign 子目录里的 *.txt（大小写不敏感）。
    - 文件名（去掉 .txt）作为 sha
    - 标签由路径分段中出现的 'malware' / 'benign' 判定
    - 如同一 sha 在不同位置重复出现，保留首次记录并打印警告
    """
    mapping: Dict[str, Tuple[Path, int]] = {}
    duplicates = 0
    for dirpath, dirnames, filenames in os.walk(data_root):
        p = Path(dirpath)
        parts = list(p.parts)
        label = _infer_label_from_parts(parts)
        if label not in (0, 1):
            continue
        for fn in filenames:
            if not fn.lower().endswith('.txt'):
                continue
            sha = os.path.splitext(fn)[0]
            file_path = p / fn
            if sha in mapping:
                duplicates += 1
                if duplicates <= 10:
                    print(f"[WARN] 重复 sha 检测到，忽略后者: {sha} -> {file_path}")
                continue
            mapping[sha] = (file_path, label)
    print(f"[INFO] 扫描完成：共识别 {len(mapping)} 个 sha（malware/benign 任意层级），重复 {duplicates} 个被忽略。")
    return mapping


def preload_all_sequences(sha_mapping: Dict[str, Tuple[Path, int]],
                          sensitive_api_list: List[str],
                          encoding='utf-8',
                          verbose=False,
                          show_progress=False) -> Dict[str, List[List[int]]]:
    """
    预加载所有 sha 的序列（整数编号列表）。空文件会得到空列表 []，保证后续仍有记录。
    """
    max_api_index = len(sensitive_api_list)
    all_data = {}
    total = len(sha_mapping)
    start = time.time()
    for i, (sha, (file_path, _label)) in enumerate(sha_mapping.items(), start=1):
        sequences: List[List[int]] = []
        try:
            with file_path.open('r', encoding=encoding) as f:
                for line_no, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split()
                    idx_list = []
                    for p in parts:
                        if not p.isdigit():
                            if verbose:
                                print(f"[WARN] {sha}:{line_no} 非数字 '{p}' 忽略")
                            continue
                        idx = int(p)
                        if 1 <= idx <= max_api_index:
                            idx_list.append(idx - 1)  # 0-based
                        else:
                            if verbose:
                                print(f"[WARN] {sha}:{line_no} 编号 {idx} 超出范围 1..{max_api_index} 忽略")
                    if idx_list:
                        sequences.append(idx_list)
            all_data[sha] = sequences  # 空文件 -> []
        except Exception as e:
            if verbose:
                print(f"[ERROR] 预加载 {file_path} 失败: {e}")
            all_data[sha] = []
    if show_progress:
        elapsed = time.time() - start
        print(f"[PRELOAD] 完成，共 {total} 个 sha，elapsed={format_seconds(elapsed)}")
    return all_data


# -----------------------------
# 相似合并核心
# -----------------------------
def compute_sequence_weights_and_sets(raw_sequences: List[List[int]],
                                      api_weight_array: List[float]):
    """
    对每条原始序列：
    - 计算恶意值（序列内允许重复，但对权重求和按出现次数相加）
    - 生成去重后的升序索引列表（用于并集与倒排）
    """
    seq_weights = []
    unique_sets = []
    for seq in raw_sequences:
        w = 0.0
        for idx in seq:
            w += api_weight_array[idx]
        seq_weights.append(w)
        unique_sorted = sorted(set(seq))
        unique_sets.append(unique_sorted)
    return seq_weights, unique_sets


def merge_sequences_union_unique(seq_weights: List[float],
                                 unique_sets: List[List[int]],
                                 api_weight_array: List[float],
                                 similarity_threshold: float) -> List[float]:
    """
    采用倒排表 + 权重区间剪枝的高效合并算法：
    - 相似度 sim(A,B) = 恶意值(A∩B) / max(WA, WB)
    返回各合并簇的恶意值列表（降序）。
    """
    from collections import defaultdict as _dd

    n = len(seq_weights)
    if n == 0:
        return []

    t = float(similarity_threshold)
    if not (0.0 < t < 1.0):
        t = max(1e-9, min(t, 0.999999))

    inverted_index: Dict[int, List[int]] = _dd(list)
    seq_items_nz: List[List[int]] = []
    for sid, items in enumerate(unique_sets):
        nz = [it for it in items if api_weight_array[it] > 0.0]
        nz.sort(key=lambda x: api_weight_array[x], reverse=True)
        seq_items_nz.append(nz)
        for it in nz:
            inverted_index[it].append(sid)

    parent = list(range(n))
    size = [1] * n

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if size[ra] < size[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        size[ra] += size[rb]

    for a in range(n):
        WA = seq_weights[a]
        if WA <= 0.0:
            continue
        lower_WB = t * WA
        upper_WB = WA / t

        inter_acc: Dict[int, float] = {}

        for it in seq_items_nz[a]:
            w_it = api_weight_array[it]
            if w_it <= 0.0:
                continue
            posting = inverted_index.get(it, [])
            for b in posting:
                if b <= a:
                    continue
                WB = seq_weights[b]
                if WB <= 0.0:
                    continue
                if WB < lower_WB or WB > upper_WB:
                    continue
                inter_acc[b] = inter_acc.get(b, 0.0) + w_it

        if not inter_acc:
            continue

        for b, inter_w in inter_acc.items():
            denom = WA if WA >= seq_weights[b] else seq_weights[b]
            if inter_w >= t * denom:
                union(a, b)

    comps = _dd(list)
    for k in range(n):
        comps[find(k)].append(k)

    merged_vals: List[float] = []
    for comp_indices in comps.values():
        api_union = set()
        for sid in comp_indices:
            api_union.update(unique_sets[sid])
        val = sum(api_weight_array[idx] for idx in api_union)
        merged_vals.append(val)

    merged_vals.sort(reverse=True)
    return merged_vals


def process_single_sha(raw_sequences: List[List[int]],
                       api_weight_array: List[float],
                       similarity_threshold: float) -> List[float]:
    seq_weights, unique_sets = compute_sequence_weights_and_sets(raw_sequences, api_weight_array)
    return merge_sequences_union_unique(seq_weights, unique_sets, api_weight_array, similarity_threshold)


# -----------------------------
# 并行处理与 CSV 写出
# -----------------------------
def read_sha_list(path: Path, encoding='utf-8') -> List[str]:
    lst = []
    with path.open('r', encoding=encoding) as f:
        for line in f:
            s = line.strip()
            if s:
                lst.append(s)
    return lst


def compute_all_parallel(shas: List[str],
                         sha_sequences: Dict[str, List[List[int]]],
                         api_weight_array: List[float],
                         similarity_threshold: float,
                         workers: int,
                         prefix: str,
                         show_progress: bool) -> Dict[str, List[float]]:
    results = {}
    total = len(shas)
    if total == 0:
        return results
    start = time.time()
    recent = deque(maxlen=50)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_sha = {}
        for sha in shas:
            raw_seqs = sha_sequences.get(sha, [])
            fut = executor.submit(process_single_sha, raw_seqs, api_weight_array, similarity_threshold)
            future_to_sha[fut] = sha

        done = 0
        for fut in as_completed(future_to_sha):
            sha = future_to_sha[fut]
            t0 = time.time()
            try:
                merged_vals = fut.result()
            except Exception:
                merged_vals = []
            results[sha] = merged_vals
            done += 1
            recent.append(time.time() - t0)
            if show_progress:
                print_progress(prefix, done, total, start, recent)
    return results


def write_single_cell(out_path: Path,
                      sha_list: List[str],
                      feats_global: Dict[str, List[float]],
                      sha_mapping: Dict[str, Tuple[Path, int]],
                      encoding='utf-8',
                      dry_run=False,
                      verbose=False):
    """
    将全局计算好的 feats_global 按指定 sha_list 顺序写出 single-cell CSV。
    """
    if dry_run:
        if verbose:
            print(f"[DRY_RUN] 跳过写入 {out_path}")
        return
    header = ['sha256', 'seq_values', 'label']
    with out_path.open('w', newline='', encoding=encoding) as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for sha in sha_list:
            vals = feats_global.get(sha, [])
            json_vals = json.dumps(vals, ensure_ascii=False)
            label = sha_mapping.get(sha, (None, -1))[1]
            writer.writerow([sha, json_vals, label])


# -----------------------------
# 主入口
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "处理敏感 API 序列数据（single-cell 输出，"
            "使用单个 root-dir + 分别指定 train/test 数据根目录）。"
        )
    )
    parser.add_argument('--root-dir',default="./statistic",
                        help='包含 train_sha.txt, test_sha.txt, weights.csv 的目录（原先单个 fold_XX 的等价物）')
    parser.add_argument('--sensitive-api-file', default="./APIChecker_PScout.txt",
                        help='敏感 API 名称列表文件路径')
    parser.add_argument('--train-data-root', default="./Sequences/train",
                        help='训练集根目录（其下包含 malware / benign 子目录及 .txt 序列文件）')
    parser.add_argument('--test-data-root', default="./Sequences/test",
                        help='测试集根目录（其下包含 malware / benign 子目录及 .txt 序列文件）')
    parser.add_argument('--similarity-threshold', type=float, default=0.3,
                        help='序列相似度阈值')
    parser.add_argument('--workers', type=int, default=120,
                        help='进程数，默认=CPU核心数(上限128)')
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--progress', action='store_true', help='显示进度条形式', default=True)
    args = parser.parse_args()

    root_dir = Path(args.root_dir)
    sensitive_api_file = Path(args.sensitive_api_file)
    train_data_root = Path(args.train_data_root)
    test_data_root = Path(args.test_data_root)

    if not root_dir.exists():
        print(f"[ERROR] root_dir 不存在: {root_dir}")
        sys.exit(1)
    if not sensitive_api_file.exists():
        print(f"[ERROR] 敏感 API 文件不存在: {sensitive_api_file}")
        sys.exit(1)
    if not train_data_root.exists():
        print(f"[ERROR] train_data_root 不存在: {train_data_root}")
        sys.exit(1)
    if not test_data_root.exists():
        print(f"[ERROR] test_data_root 不存在: {test_data_root}")
        sys.exit(1)

    train_sha_file = root_dir / 'train_sha.txt'
    test_sha_file = root_dir / 'test_sha.txt'
    weights_file = root_dir / 'weights.csv'

    if not train_sha_file.exists() or not test_sha_file.exists():
        print(f"[ERROR] 缺少 train_sha.txt 或 test_sha.txt 于 {root_dir}")
        sys.exit(1)
    if not weights_file.exists():
        print(f"[ERROR] 缺少 weights.csv 于 {root_dir}")
        sys.exit(1)

    # 载入敏感 API 与权重
    sensitive_api_list = load_sensitive_api_list(sensitive_api_file, encoding=args.encoding)
    api_weights_map = load_api_weights(weights_file, encoding=args.encoding)
    api_weight_array = [api_weights_map.get(name, 0.0) for name in sensitive_api_list]

    # 深度扫描 train/test 目录（分开扫描与记录标签）
    print("[INFO] 扫描训练目录 ...")
    train_sha_mapping = scan_sha_files(train_data_root)
    print("[INFO] 扫描测试目录 ...")
    test_sha_mapping = scan_sha_files(test_data_root)

    # 预加载 train/test 序列
    train_sequences = preload_all_sequences(
        train_sha_mapping,
        sensitive_api_list,
        encoding=args.encoding,
        verbose=args.verbose,
        show_progress=args.progress
    )
    test_sequences = preload_all_sequences(
        test_sha_mapping,
        sensitive_api_list,
        encoding=args.encoding,
        verbose=args.verbose,
        show_progress=args.progress
    )

    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    total_workers = args.workers if args.workers > 0 else cpu_count
    if total_workers > 128:
        total_workers = 128

    if args.verbose or args.progress:
        print(f"[INFO] CPU={cpu_count} total_workers={total_workers}")

    # 读取 train/test sha 列表
    train_shas = read_sha_list(train_sha_file, encoding=args.encoding)
    test_shas = read_sha_list(test_sha_file, encoding=args.encoding)

    if not train_shas and not test_shas:
        print("[ERROR] train_sha.txt 与 test_sha.txt 皆为空。")
        sys.exit(1)

    # 计算 union shas（用于统计/日志），但实际计算分开做，保证目录/标签不会混
    union_shas = []
    seen = set()
    for s in train_shas:
        if s not in seen:
            seen.add(s)
            union_shas.append(s)
    for s in test_shas:
        if s not in seen:
            seen.add(s)
            union_shas.append(s)

    if args.verbose or args.progress:
        print(f"[INFO] 共有 {len(union_shas)} 个唯一样本 (train={len(train_shas)}, test={len(test_shas)})")

    # 计算训练集特征
    if train_shas:
        if args.verbose or args.progress:
            print(f"[INFO] 开始计算训练集，共 {len(train_shas)} 个样本")
        train_features = compute_all_parallel(
            train_shas,
            train_sequences,
            api_weight_array,
            args.similarity_threshold,
            total_workers,
            prefix="[TRAIN]",
            show_progress=args.progress
        )
    else:
        train_features = {}

    # 计算测试集特征
    if test_shas:
        if args.verbose or args.progress:
            print(f"[INFO] 开始计算测试集，共 {len(test_shas)} 个样本")
        test_features = compute_all_parallel(
            test_shas,
            test_sequences,
            api_weight_array,
            args.similarity_threshold,
            total_workers,
            prefix="[TEST]",
            show_progress=args.progress
        )
    else:
        test_features = {}

    # 写出 train/test features
    write_single_cell(
        root_dir / f'{OUTPUT_PREFIX}_train.csv',
        train_shas,
        train_features,
        train_sha_mapping,
        encoding=args.encoding,
        dry_run=args.dry_run,
        verbose=args.verbose
    )
    write_single_cell(
        root_dir / f'{OUTPUT_PREFIX}_test.csv',
        test_shas,
        test_features,
        test_sha_mapping,
        encoding=args.encoding,
        dry_run=args.dry_run,
        verbose=args.verbose
    )

    # 写出元信息
    meta = {
        'root_dir': str(root_dir),
        'similarity_threshold': args.similarity_threshold,
        'merging_strategy': 'union_unique',
        'train_samples': len(train_shas),
        'test_samples': len(test_shas),
        'weights_file': str(weights_file),
        'sensitive_api_file': str(sensitive_api_file),
        'train_data_root': str(train_data_root),
        'test_data_root': str(test_data_root),
        'output_prefix': OUTPUT_PREFIX,
        'output_layout': 'single-cell',
        'note': '空序列文件也会写入（seq_values=[]）',
        'workers_used': total_workers
    }
    if not args.dry_run:
        with (root_dir / 'metadata.json').open('w', encoding=args.encoding) as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)

    if args.verbose:
        print(f"[DONE] {root_dir}: train={len(train_shas)} test={len(test_shas)}")


if __name__ == '__main__':
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")