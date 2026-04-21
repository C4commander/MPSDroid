#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高并行版本：处理敏感 API 序列数据（single-cell 输出，带进度）。
修复点：
- 更完整地提取 .txt：递归 data_root 下任意层级的 malware/benign 子目录
- 空序列文件也会保留记录（seq_values 为 []）
- 仅进行一次全局相似度合并计算，然后将结果按每折的 train/test 切分复用，避免重复计算
- 移除 fold_jobs 并行扇出逻辑：统一顺序写出每折结果

输入：
- fold_root/fold_XX/{weights.csv, train_sha.txt, test_sha.txt}
- data_root 下的任意路径中包含 malware 或 benign 的子目录里的 *.txt（文件名为 sha256）

输出（每个 fold_XX 内）：
- features_train.csv（列：sha256, seq_values(JSON数组), label）
- features_test.csv
- metadata.json（参数与统计）
"""

import argparse
import csv
import json
import os
import re
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
            raise ValueError(f"weights.csv 缺少必须列(API, weight): {weights_file}\n实际表头: {reader.fieldnames}")
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
# 扫描与预加载（修复扫描不完整）
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
# 相似合并核心（倒排表 + 权重比值剪枝）
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
    - 仅通过 item 倒排生成候选对，并在累计交集时使用必要条件 WB ∈ [t·WA, WA/t] 过滤
    - 达到阈值的对用并查集合并，最后对每个连通分量计算“并集”的恶意值
    返回各合并簇的恶意值列表（降序），与原实现一致的输出格式。
    """
    n = len(seq_weights)
    if n == 0:
        return []

    t = float(similarity_threshold)
    if not (0.0 < t < 1.0):
        # 不改变外部参数接口，但保证健壮性
        t = max(1e-9, min(t, 0.999999))

    # 1) 预处理：过滤出权重>0的 items，构建倒排表 item -> [seq_ids]
    inverted_index: Dict[int, List[int]] = defaultdict(list)
    seq_items_nz: List[List[int]] = []
    for sid, items in enumerate(unique_sets):
        nz = [it for it in items if api_weight_array[it] > 0.0]
        nz.sort(key=lambda x: api_weight_array[x], reverse=True)
        seq_items_nz.append(nz)
        for it in nz:
            inverted_index[it].append(sid)

    # 2) 并查集
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

    # 3) 遍历每个序列作为 A，使用倒排表累计交集，加阈值验证并合并
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

    # 4) 归并连通分量并计算并集恶意值
    comps = defaultdict(list)
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
            raw_seqs = sha_sequences.get(sha, [])  # 空文件将得到 []
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

def write_fold_outputs(fold_dir: Path,
                       train_shas: List[str],
                       test_shas: List[str],
                       feats_global: Dict[str, List[float]],
                       sha_mapping: Dict[str, Tuple[Path, int]],
                       similarity_threshold: float,
                       workers_used: int,
                       encoding='utf-8',
                       dry_run=False,
                       verbose=False):
    # 写出 train/test
    write_single_cell(fold_dir / f'{OUTPUT_PREFIX}_train.csv', train_shas,
                      feats_global, sha_mapping, encoding, dry_run, verbose)
    write_single_cell(fold_dir / f'{OUTPUT_PREFIX}_test.csv', test_shas,
                      feats_global, sha_mapping, encoding, dry_run, verbose)

    # 写出元信息（结构保持不变）
    meta = {
        'fold': fold_dir.name,
        'similarity_threshold': similarity_threshold,
        'merging_strategy': 'union_unique',
        'train_samples': len(train_shas),
        'test_samples': len(test_shas),
        'weights_file': str(fold_dir / 'weights.csv'),
        'sensitive_api_file': 'PRELOADED',
        'output_prefix': OUTPUT_PREFIX,
        'output_layout': 'single-cell',
        'note': '空序列文件也会写入（seq_values=[]）',
        'workers_used': workers_used
    }
    if not dry_run:
        with (fold_dir / 'metadata.json').open('w', encoding=encoding) as mf:
            json.dump(meta, mf, ensure_ascii=False, indent=2)
    if verbose:
        print(f"[DONE] {fold_dir.name}: train={len(train_shas)} test={len(test_shas)}")

# -----------------------------
# 主入口（移除折并行）
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="处理敏感 API 序列数据（single-cell 输出）。")
    parser.add_argument('--fold_root', default="/mnt/data2/wb2024/Methodology/MyWay2.0/fold_outputs-mc", help='包含 fold_01...fold_10 的目录路径')
    parser.add_argument('--sensitive_api_file', default="/mnt/data2/wb2024/Methodology/MyWay/analyze/删除没出现的结果API.txt", help='敏感 API 名称列表文件路径')
    parser.add_argument('--data_root', default="/mnt/data2/wb2024/Methodology/MyWay/data/Sequences-mc", help='包含 malware / benign 子目录的根路径')
    parser.add_argument('--similarity_threshold', type=float, default=0.3, help='序列相似度阈值')
    parser.add_argument('--workers', type=int, default=120, help='进程数，默认=CPU核心数(上限128)')
    parser.add_argument('--encoding', default='utf-8')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--progress', action='store_true', help='显示进度条形式', default=True)
    args = parser.parse_args()

    fold_root = Path(args.fold_root)
    sensitive_api_file = Path(args.sensitive_api_file)
    data_root = Path(args.data_root)

    if not fold_root.exists():
        print(f"[ERROR] fold_root 不存在: {fold_root}")
        sys.exit(1)
    if not sensitive_api_file.exists():
        print(f"[ERROR] 敏感 API 文件不存在: {sensitive_api_file}")
        sys.exit(1)
    if not data_root.exists():
        print(f"[ERROR] data_root 不存在: {data_root}")
        sys.exit(1)

    sensitive_api_list = load_sensitive_api_list(sensitive_api_file, encoding=args.encoding)

    # 深度扫描
    sha_mapping = scan_sha_files(data_root)

    # 预加载所有序列（空文件 => [] 也会保存）
    sha_sequences = preload_all_sequences(
        sha_mapping,
        sensitive_api_list,
        encoding=args.encoding,
        verbose=args.verbose,
        show_progress=args.progress
    )

    # 列出 folds
    fold_dirs = sorted([
        p for p in fold_root.iterdir()
        if p.is_dir() and re.match(r'fold_\d{2}$', p.name)
    ])
    if not fold_dirs:
        print(f"[ERROR] 未找到 fold_XX 目录于 {fold_root}")
        sys.exit(1)

    # 并行 workers 计算
    import multiprocessing
    cpu_count = multiprocessing.cpu_count()
    total_workers = args.workers if args.workers > 0 else cpu_count
    if total_workers > 128:
        total_workers = 128

    if args.verbose or args.progress:
        print(f"[INFO] CPU={cpu_count} total_workers={total_workers}")

    # 读取所有 fold 的 sha 列表，并构建全量 union
    per_fold_shas: Dict[str, Tuple[List[str], List[str]]] = {}
    union_shas = []
    union_seen = set()
    for fd in fold_dirs:
        train_sha_file = fd / 'train_sha.txt'
        test_sha_file = fd / 'test_sha.txt'
        if not train_sha_file.exists() or not test_sha_file.exists():
            print(f"[WARN] 缺少 train/test sha 文件: {fd} 跳过该 fold")
            continue
        train_shas = read_sha_list(train_sha_file, encoding=args.encoding)
        test_shas = read_sha_list(test_sha_file, encoding=args.encoding)
        per_fold_shas[fd.name] = (train_shas, test_shas)
        for s in train_shas:
            if s not in union_seen:
                union_seen.add(s)
                union_shas.append(s)
        for s in test_shas:
            if s not in union_seen:
                union_seen.add(s)
                union_shas.append(s)

    if not union_shas:
        print("[ERROR] 所有 folds 的 train/test 列表皆为空。")
        sys.exit(1)

    # 使用第一个 fold 的 weights.csv 作为全局权重（各折数据一致，避免重复计算）
    ref_fold = fold_dirs[0]
    weights_file = ref_fold / 'weights.csv'
    if not weights_file.exists():
        print(f"[ERROR] 缺少 weights.csv: {weights_file}")
        sys.exit(1)
    api_weights_map = load_api_weights(weights_file, encoding=args.encoding)
    api_weight_array = [api_weights_map.get(name, 0.0) for name in sensitive_api_list]

    # 全局一次性计算所有样本的合并结果
    if args.verbose or args.progress:
        print(f"[INFO] 开始全局计算，共 {len(union_shas)} 个样本（一次性相似度/合并）")
    global_features = compute_all_parallel(
        union_shas,
        sha_sequences,
        api_weight_array,
        args.similarity_threshold,
        total_workers,
        prefix="[GLOBAL]",
        show_progress=args.progress
    )

    # 顺序写出每个 fold 的结果（复用全局计算）
    for fd in fold_dirs:
        if fd.name not in per_fold_shas:
            continue
        train_shas, test_shas = per_fold_shas[fd.name]
        write_fold_outputs(fd,
                           train_shas, test_shas,
                           global_features,
                           sha_mapping,
                           args.similarity_threshold,
                           workers_used=total_workers,
                           encoding=args.encoding,
                           dry_run=args.dry_run,
                           verbose=args.verbose)

if __name__ == '__main__':
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    main()
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")