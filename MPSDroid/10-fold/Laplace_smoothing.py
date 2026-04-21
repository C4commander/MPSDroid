import argparse
from pathlib import Path
import json
import os
import gzip
import traceback
import concurrent.futures as cf
from typing import List, Dict, Tuple, Set

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
import time  # 新增：用于时间统计


# ========== 基础工具 ==========
def find_seq_txt_files(root_dir):
    """
    递归查找 root_dir 下的 .txt 序列文件（来自 gexf_to_java_seq.py 的输出）。
    """
    return list(Path(root_dir).rglob('*.txt'))


def infer_label(path: Path, malware_key='malware', benign_key='benign'):
    parts = [p.lower() for p in path.parts]
    if malware_key.lower() in parts:
        return 'malware'
    elif benign_key.lower() in parts:
        return 'benign'
    else:
        return None


def path_sha_no_ext(p: Path) -> str:
    return p.stem


# ========== 敏感 API 映射与序列解析 ==========
def load_sapi_mapping(sapi_path: Path) -> Tuple[Dict[int, str], Set[str]]:
    """
    读取敏感 API 列表（每行一个 Java 方法签名），构建：
    - id_to_api: 1-based 序号 -> API 名称（与 gexf_to_java_seq.py 的编号一致）
    - sapi_set: 敏感 API 名称集合
    """
    ordered, seen = [], set()
    with open(sapi_path, 'r', encoding='utf-8') as f:
        for line in f:
            api = line.strip()
            if api and api not in seen:
                seen.add(api)
                ordered.append(api)
    id_to_api = {i + 1: api for i, api in enumerate(ordered)}
    return id_to_api, set(ordered)


def extract_sensitive_apis_from_txt(txt_path: Path,
                                    id_to_api: Dict[int, str],
                                    sapi_set: Set[str]) -> List[str]:
    """
    从 gexf_to_java_seq.py 产生的 .txt 序列文件中提取“文件级”敏感 API 使用集合（去重）。
    - 文件每行是一条序列，默认为以空格分隔的 1-based 整数 ID。
    - 若序列中出现了非数字的 token（例如直接是 API 名称），则仅在其属于敏感集合时纳入。
    返回排序后的敏感 API 名称列表。
    """
    used = set()

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                for tok in line.split():
                    if tok.isdigit():
                        idx = int(tok)
                        api = id_to_api.get(idx)
                        if api:
                            used.add(api)
                    else:
                        # 兼容：若直接输出了名称，则只采纳在敏感集合中的名称
                        if tok in sapi_set:
                            used.add(tok)
    except FileNotFoundError:
        # 不存在则返回空
        return []
    except Exception:
        # 任何解析异常，交由上层捕获
        raise

    return sorted(used)


# ========== 缓存 ==========
def write_api_cache(cache_file: Path, apis: List[str]) -> None:
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cache_file, "wt", encoding="utf-8") as f:
        for a in apis:
            f.write(a)
            f.write("\n")


def read_api_cache(cache_file: Path) -> List[str]:
    if not cache_file.exists():
        return []
    with gzip.open(cache_file, "rt", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def preprocess_one_file(args_tuple) -> Dict:
    """
    并行任务：读取一个 .txt 序列文件 -> 根据 SAPI 映射将 ID 还原为敏感 API 名称 -> 去重 -> 写入缓存
    返回字典包括 path, sha, label, cache_path, n_apis, error
    """
    (fpath_str, malware_key, benign_key, cache_dir_str, reuse_cache, id_to_api, sapi_set) = args_tuple

    fpath = Path(fpath_str)
    cache_dir = Path(cache_dir_str)
    sha = path_sha_no_ext(fpath)
    label = infer_label(fpath, malware_key, benign_key)
    out_cache = cache_dir / f"{sha}.txt.gz"

    ret = {
        "path": str(fpath),
        "sha": sha,
        "label": label,
        "cache_path": str(out_cache),
        "n_apis": 0,
        "error": ""
    }

    if label not in ("malware", "benign"):
        return ret

    try:
        if reuse_cache and out_cache.exists():
            # 读取现有缓存获取数量
            apis = read_api_cache(out_cache)
            ret["n_apis"] = len(set(apis))
            return ret

        unique_apis = extract_sensitive_apis_from_txt(fpath, id_to_api, sapi_set)
        write_api_cache(out_cache, unique_apis)
        ret["n_apis"] = len(unique_apis)
    except Exception as e:
        ret["error"] = f"{e}\n{traceback.format_exc()}"
    return ret


# ========== 统计权重（基于缓存） ==========
def compute_weights_from_cache(train_items: List[Tuple[Path, str]],
                               cache_dir: Path,
                               min_count: int,
                               alpha: int = 2,
                               verbose: bool = False):
    """
    基于缓存文件计算恶意/良性中 API 的“文件级出现计数”，并计算熵和权重。
    处理步骤：
      1. 对每个文件的 API 集合去重（布尔出现模型）。
      2. 分别累计在恶意与良性样本中出现该 API 的样本数 c_m(s), c_b(s)。
      3. 过滤 total_count = c_m + c_b < min_count 的 API。
      4. 拉普拉斯平滑：使用 c_m'(s) = c_m(s) + α, c_b'(s) = c_b(s) + α（α 默认为 1）缓解极端 0/1 比例。
         注意：这里保持分母为原始样本总数 N_m, N_b，未采用 (N + 2α) 的标准伯努利平滑形式，以符合您给出的“c_m(s)+α, c_b(s)+α”说明。
         若需标准形式，可改为：
             pm = (c_m + alpha) / (n_malware + 2 * alpha)
             pb = (c_b + alpha) / (n_benign + 2 * alpha)
      5. 计算类内出现频率：
             pm_raw = c_m / N_m
             pb_raw = c_b / N_b
             pm = (c_m + α) / N_m
             pb = (c_b + α) / N_b
         （若某一类样本数为 0，对应频率置 0）
      6. 计算恶意偏好比例：
             p = pm / (pm + pb) （若分母为 0 则置 0）
      7. 信息熵（自然对数）：
             H = -p ln p - (1-p) ln(1-p) （p∈(0,1)），边界 p=0 或 1 时 H=0。
      8. 权重：
             W = p * (1 - H)
         拉普拉斯平滑会使原本接近 0/1 的 p 向中间收缩，从而略微降低高纯度恶意 API 的权重（更保守）。
    返回：
      df: 包含每个 API 及其统计指标的 DataFrame（按 weight 降序）
      stats: 总体统计信息
    """
    from collections import defaultdict

    malware_count = defaultdict(int)
    benign_count = defaultdict(int)
    n_malware = 0
    n_benign = 0

    # 遍历训练样本并计数
    for fpath, label in tqdm(train_items, desc="读取训练缓存并计数API", disable=not verbose):
        sha = path_sha_no_ext(fpath)
        cache_file = cache_dir / f"{sha}.txt.gz"
        apis = read_api_cache(cache_file)
        apis_unique = set(apis)  # 单文件只统计一次

        if label == "malware":
            n_malware += 1
            for api in apis_unique:
                malware_count[api] += 1
        elif label == "benign":
            n_benign += 1
            for api in apis_unique:
                benign_count[api] += 1

    all_apis = set(malware_count.keys()) | set(benign_count.keys())
    rows = []

    for api in all_apis:
        count_m = malware_count.get(api, 0)
        count_b = benign_count.get(api, 0)
        total_count = count_m + count_b
        if total_count < min_count:
            continue

        # 原始频率（未平滑）
        pm_raw = count_m / n_malware if n_malware > 0 else 0.0
        pb_raw = count_b / n_benign if n_benign > 0 else 0.0

        # 拉普拉斯平滑后频率（仅对出现计数加 α，分母保持 N）
        if n_malware > 0:
            pm = (count_m + alpha) / (n_malware + 2 * alpha)
        else:
            pm = 0.0
        if n_benign > 0:
            pb = (count_b + alpha) / (n_benign + 2 * alpha)
        else:
            pb = 0.0

        denom = pm + pb
        p = pm / denom if denom > 0 else 0.0

        H = 0.0
        if 0 < p < 1:
            H = -p * np.log(p) - (1 - p) * np.log(1 - p)

        H_norm = H / np.log(2.0)
        W = p * (1.0 - H_norm)

        rows.append({
            'API': api,
            'count_malware': int(count_m),
            'count_benign': int(count_b),
            'total_count': int(total_count),
            'pm_raw': float(pm_raw),
            'pb_raw': float(pb_raw),
            'pm': float(pm),
            'pb': float(pb),
            'p': float(p),
            'entropy': float(H),
            'weight': float(W),
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values('weight', ascending=False)

    stats = {
        "n_malware_train": int(n_malware),
        "n_benign_train": int(n_benign),
        "n_api_after_filter": int(df.shape[0]),
        "min_count": int(min_count),
        "alpha": int(alpha),
        "smoothing_denominator_adjusted": False,  # 标记目前未调整分母
    }
    return df, stats


def run_one_fold(fold_idx: int,
                 train_files: List[Path],
                 test_files: List[Path],
                 train_labels: List[str],
                 out_root: Path,
                 cache_dir: Path,
                 min_count: int,
                 verbose: bool):
    """
    单折执行：写清单 -> 基于缓存计算权重 -> 写出 CSV/JSON
    """
    fold_dir = out_root / f"fold_{fold_idx:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)

    # 保存训练/测试序列清单
    (fold_dir / "train_seq_list.txt").write_text("\n".join(str(p) for p in train_files), encoding="utf-8")
    (fold_dir / "test_seq_list.txt").write_text("\n".join(str(p) for p in test_files), encoding="utf-8")

    # 保存训练/测试sha列表（用于过滤API序列）
    train_shas = sorted({path_sha_no_ext(p) for p in train_files})
    test_shas = sorted({path_sha_no_ext(p) for p in test_files})
    (fold_dir / "train_sha.txt").write_text("\n".join(train_shas), encoding="utf-8")
    (fold_dir / "test_sha.txt").write_text("\n".join(test_shas), encoding="utf-8")

    # 基于训练集缓存计算权重
    train_items = list(zip(train_files, train_labels))
    df_weights, stats = compute_weights_from_cache(
        train_items=train_items,
        cache_dir=cache_dir,
        min_count=min_count,
        verbose=verbose
    )
    weights_csv = fold_dir / "weights.csv"
    df_weights.to_csv(weights_csv, index=False, encoding="utf-8-sig")

    # 保存meta信息
    meta = {
        "fold": fold_idx,
        "n_train": int(len(train_files)),
        "n_test": int(len(test_files)),
        **stats
    }
    (fold_dir / "stats.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "fold": fold_idx,
        "weights_csv": str(weights_csv),
        "stats": meta
    }


# ========== 主流程 ==========
def main():
    # ---- 新增：记录开始时间 ----
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))
    # ---------------------------

    parser = argparse.ArgumentParser(
        description='敏感API熵和权重统计（基于 gexf_to_java_seq.py 生成的 .txt 序列，按 SAPI 映射还原名称再统计）')
    parser.add_argument('--root', type=str, default="/mnt/data2/wb2024/Methodology/MyWay/data/Sequences-md",
                        help='序列 .txt 数据根目录（gexf_to_java_seq.py 的输出根目录）')
    parser.add_argument('--malware-key', type=str, default='malware', help='恶意目录关键字')
    parser.add_argument('--benign-key', type=str, default='benign', help='良性目录关键字')
    parser.add_argument('--output-dir', type=str, default='/mnt/data2/wb2024/Methodology/MyWay2.0/fold_outputs-md', help='每折输出目录根路径')

    parser.add_argument('--sapi', type=str,
                        default="/mnt/data2/wb2024/Methodology/MyWay/analyze/删除没出现的结果API.txt",
                        help='敏感 API 列表（需与生成 .txt 序列时使用的列表一致，用于将 ID 还原为 API 名称）')

    parser.add_argument('--min-count', type=int, default=2, help='API保留的最小出现次数（基于训练集统计）')
    parser.add_argument('--n-splits', type=int, default=10, help='交叉验证折数')
    parser.add_argument('--random-state', type=int, default=42, help='随机种子')
    parser.add_argument('--verbose', action='store_true', help='显示详细进度', default=True)

    parser.add_argument('--workers', type=int, default=max(8, min(120, (os.cpu_count() or 64))),
                        help='预处理并行线程数（读取 .txt 并解析 ID 为 IO 密集，使用线程池较合适）')
    parser.add_argument('--cache-dir', type=str, default=None, help='API缓存目录（默认在输出目录下的api_cache）')
    parser.add_argument('--reuse-cache', action='store_true', default=True,
                        help='若缓存已存在则复用，跳过 .txt 解析')

    args = parser.parse_args()

    root = Path(args.root)
    out_root = Path(args.output-dir) if False else Path(args.output_dir)  # 避免误改参数名，保持原样
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_root / "api_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 0) 载入敏感 API 映射
    sapi_path = Path(args.sapi)
    if not sapi_path.exists():
        print(f"敏感 API 列表不存在: {sapi_path}")

        # 结束计时并记录日志
        end_ts = time.time()
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
        elapsed = end_ts - start_ts
        print(f"Elapsed: {elapsed:.2f}s")
        log_path = out_root / "run_time.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Start: {start_str} | End: {end_str} | "
                    f"Elapsed: {elapsed:.2f}s | root: {root} | output-dir: {out_root} | note: sapi_not_found\n"
                )
        except Exception as e:
            print(f"WARNING: failed to write time log: {e}")
        return

    id_to_api, sapi_set = load_sapi_mapping(sapi_path)

    # 1) 枚举序列文件并标注标签
    files_all = find_seq_txt_files(root)
    items = []
    for f in files_all:
        label = infer_label(f, args.malware_key, args.benign_key)
        if label in ('malware', 'benign'):
            items.append((Path(f), label))
    if not items:
        print("未找到带有恶意/良性标签的序列 .txt 文件，检查目录与关键字。")

        end_ts = time.time()
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
        elapsed = end_ts - start_ts
        print(f"Elapsed: {elapsed:.2f}s")
        log_path = out_root / "run_time.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Start: {start_str} | End: {end_str} | "
                    f"Elapsed: {elapsed:.2f}s | root: {root} | output-dir: {out_root} | note: no_labeled_items\n"
                )
        except Exception as e:
            print(f"WARNING: failed to write time log: {e}")
        return

    file_paths: List[Path] = [it[0] for it in items]
    labels: List[str] = [it[1] for it in items]
    y = np.array([1 if lab == 'malware' else 0 for lab in labels], dtype=int)

    # 2) 预处理：并行解析全部序列文件，提取敏感 API 并缓存（gzip）
    print(f"[INFO] 预处理序列 .txt -> 缓存API列表至: {cache_dir} ，线程数: {args.workers}，reuse_cache={args.reuse_cache}")
    tasks = []
    for f in file_paths:
        tasks.append((
            str(f),
            args.malware_key,
            args.benign_key,
            str(cache_dir),
            bool(args.reuse_cache),
            id_to_api,
            sapi_set,
        ))

    index_rows = []
    n_errors = 0
    # 使用线程池：IO 密集，避免在每个任务中重复加载 SAPI 映射
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(preprocess_one_file, t) for t in tasks]
        for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="预处理序列 .txt", disable=not args.verbose):
            info = fut.result()
            index_rows.append(info)
            if info.get("error"):
                n_errors += 1
                print(f"[ERROR] 预处理失败: {info['path']}\n{info['error']}")

    # 保存索引（便于排查与复用）
    index_df = pd.DataFrame(index_rows)
    index_csv = out_root / "api_cache_index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存缓存索引: {index_csv}，失败: {n_errors}")

    # 3) 分层交叉划分
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.random_state)
    print(f"共{len(file_paths)}个序列文件，开始运行 {args.n_splits} 折分层交叉验证...")

    # 4) 顺序执行各折（移除了 --fold-workers）
    results = []
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(file_paths)), y), start=1):
        train_files = [file_paths[i] for i in train_idx]
        test_files = [file_paths[i] for i in test_idx]
        train_labels = [labels[i] for i in train_idx]

        r = run_one_fold(
            fold_idx=fold_idx,
            train_files=train_files,
            test_files=test_files,
            train_labels=train_labels,
            out_root=out_root,
            cache_dir=cache_dir,
            min_count=args.min_count,
            verbose=args.verbose
        )
        results.append(r)

    # 汇总提示
    for r in sorted(results, key=lambda x: x["fold"]):
        print(f"[Fold {r['fold']:02d}] 已保存 权重: {r['weights_csv']}  统计: n_train={r['stats']['n_train']} n_test={r['stats']['n_test']} n_api_after_filter={r['stats']['n_api_after_filter']}")

    # ---- 新增：结束时间 & 日志 ----
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")

    log_path = out_root / "run_time.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Start: {start_str} | End: {end_str} | "
                f"Elapsed: {elapsed:.2f}s | root: {root} | output-dir: {out_root} | "
                f"n_splits={args.n_splits} | workers={args.workers} | reuse_cache={args.reuse_cache}\n"
            )
    except Exception as e:
        print(f"WARNING: failed to write time log: {e}")
    # --------------------------------


if __name__ == '__main__':
    main()
