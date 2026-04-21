import argparse
from pathlib import Path
import json
import os
import gzip
import traceback
from typing import List, Dict, Tuple, Set

import pandas as pd
import numpy as np
from tqdm import tqdm
import time


# ========== 基础工具 ==========

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
    - id_to_api: 1-based 序号 -> API 名称
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
    从 .txt 序列文件中提取“文件级”敏感 API 使用集合（去重）。
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
                        if tok in sapi_set:
                            used.add(tok)
    except FileNotFoundError:
        return []
    except Exception:
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


def preprocess_one_file(fpath: Path,
                        malware_key: str,
                        benign_key: str,
                        cache_dir: Path,
                        reuse_cache: bool,
                        id_to_api: Dict[int, str],
                        sapi_set: Set[str]) -> Dict:
    """
    读取一个 .txt 序列文件 -> 根据 SAPI 映射将 ID 还原为敏感 API 名称 -> 去重 -> 写入缓存
    返回字典包括 path, sha, label, cache_path, n_apis, error
    """
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
    """
    from collections import defaultdict

    malware_count = defaultdict(int)
    benign_count = defaultdict(int)
    n_malware = 0
    n_benign = 0

    for fpath, label in tqdm(train_items, desc="读取训练缓存并计数API", disable=not verbose):
        sha = path_sha_no_ext(fpath)
        cache_file = cache_dir / f"{sha}.txt.gz"
        apis = read_api_cache(cache_file)
        apis_unique = set(apis)

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

        pm_raw = count_m / n_malware if n_malware > 0 else 0.0
        pb_raw = count_b / n_benign if n_benign > 0 else 0.0

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
        "smoothing_denominator_adjusted": True,
    }
    return df, stats


# ========== 目录扫描工具 ==========

def collect_txt_files(root_dir: Path) -> List[Path]:
    """
    从给定目��递归收集所有 .txt 文件。
    用于 train-dir / test-dir；目录下包含 malware/benign 子目录。
    """
    if not root_dir.exists():
        return []
    return list(root_dir.rglob("*.txt"))


# ========== 主流程（训练/测试目录） ==========

def main():
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    parser = argparse.ArgumentParser(
        description='敏感API熵和权重统计（训练/测试为两个目录，目录下含 malware/benign 子目录）'
    )
    parser.add_argument('--train-dir', type=str, default="/mnt/data2/wb2024/Methodology/MPSDroid/test-train/Sequences/train",
                        help='训练集根目录（其下包含 malware 和 benign 子目录，里面是 .txt 序列文件）')
    parser.add_argument('--test-dir', type=str, default="/mnt/data2/wb2024/Methodology/MPSDroid/test-train/Sequences/test",
                        help='测试集根目录（其下包含 malware 和 benign 子目录，里面是 .txt 序列文件）')

    parser.add_argument('--malware-key', type=str, default='malware', help='恶意目录关键字')
    parser.add_argument('--benign-key', type=str, default='benign', help='良性目录关键字')
    parser.add_argument('--output-dir', type=str, default="./statistic",
                        help='输出目录根路径')

    parser.add_argument('--sapi', type=str, default="./APIChecker_PScout.txt",
                        help='敏感 API 列表路径（与生成 .txt 序列时使用的列表一致）')

    parser.add_argument('--min-count', type=int, default=2,
                        help='API保留的最小出现次数（基于训练集统计）')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='显示详细进度')

    parser.add_argument('--workers', type=int,
                        default=max(8, min(120, (os.cpu_count() or 64))),
                        help='预处理并行线程数（IO 密集，使用线程池较合适）')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='API缓存目录（默认在输出目录下的api_cache）')
    parser.add_argument('--reuse-cache', action='store_true', default=True,
                        help='若缓存已存在则复用，跳过 .txt 解析')

    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_root / "api_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 0) 载入敏感 API 映射
    sapi_path = Path(args.sapi)
    if not sapi_path.exists():
        print(f"敏感 API 列表不存在: {sapi_path}")
        end_ts = time.time()
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
        elapsed = end_ts - start_ts
        print(f"Elapsed: {elapsed:.2f}s")
        log_path = out_root / "run_time.log"
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Start: {start_str} | End: {end_str} | "
                    f"Elapsed: {elapsed:.2f}s | output-dir: {out_root} | note: sapi_not_found\n"
                )
        except Exception as e:
            print(f"WARNING: failed to write time log: {e}")
        return

    id_to_api, sapi_set = load_sapi_mapping(sapi_path)

    # 1) 扫描训练/测试目录并打标签
    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)

    train_files = collect_txt_files(train_root)
    test_files = collect_txt_files(test_root)

    if not train_files:
        print(f"训练目录下未找到任何 .txt 文件，请检查 --train-dir: {train_root}")
        return
    if not test_files:
        print(f"测试目录下未找到任何 .txt 文件，请检查 --test-dir: {test_root}")
        return

    train_items = []
    for f in train_files:
        label = infer_label(f, args.malware_key, args.benign_key)
        if label in ('malware', 'benign'):
            train_items.append((f, label))
    if not train_items:
        print("训练集中未找到带有恶意/良性标签的序列 .txt 文件，检查目录与关键字。")
        return

    # 保存清单（便于之后复查）
    (out_root / "train_seq_list.txt").write_text(
        "\n".join(str(p) for p in sorted(train_files)), encoding="utf-8"
    )
    (out_root / "test_seq_list.txt").write_text(
        "\n".join(str(p) for p in sorted(test_files)), encoding="utf-8"
    )

    train_shas = sorted({path_sha_no_ext(p) for p in train_files})
    test_shas = sorted({path_sha_no_ext(p) for p in test_files})
    (out_root / "train_sha.txt").write_text("\n".join(train_shas), encoding="utf-8")
    (out_root / "test_sha.txt").write_text("\n".join(test_shas), encoding="utf-8")

    # 2) 预处理：训练集 + 测试集 全部做缓存
    all_files = list({*train_files, *test_files})
    print(f"[INFO] 预处理序列 .txt -> 缓存API列表至: {cache_dir} ，线程数: {args.workers}，reuse_cache={args.reuse_cache}")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    index_rows = []
    n_errors = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                preprocess_one_file,
                f,
                args.malware_key,
                args.benign_key,
                cache_dir,
                bool(args.reuse_cache),
                id_to_api,
                sapi_set,
            ): f for f in all_files
        }
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="预处理序列 .txt", disable=not args.verbose):
            info = fut.result()
            index_rows.append(info)
            if info.get("error"):
                n_errors += 1
                print(f"[ERROR] 预处理失败: {info['path']}\n{info['error']}")

    index_df = pd.DataFrame(index_rows)
    index_csv = out_root / "api_cache_index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] 已保存缓存索引: {index_csv}，失败: {n_errors}")

    # 3) 基于训练集缓存计算权重
    df_weights, stats = compute_weights_from_cache(
        train_items=train_items,
        cache_dir=cache_dir,
        min_count=args.min_count,
        verbose=args.verbose
    )
    weights_csv = out_root / "weights.csv"
    df_weights.to_csv(weights_csv, index=False, encoding="utf-8-sig")

    meta = {
        "n_train": int(len(train_files)),
        "n_test": int(len(test_files)),
        **stats
    }
    (out_root / "stats.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"[INFO] 已保存 权重: {weights_csv}  统计: n_train={meta['n_train']} "
          f"n_test={meta['n_test']} n_api_after_filter={meta['n_api_after_filter']}")

    # ---- 结束时间 & 日志 ----
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")

    log_path = out_root / "run_time.log"
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Start: {start_str} | End: {end_str} | "
                f"Elapsed: {elapsed:.2f}s | output-dir: {out_root} | "
                f"train_dir={args.train_dir} | test_dir={args.test_dir} | "
                f"workers={args.workers} | reuse_cache={args.reuse_cache}\n"
            )
    except Exception as e:
        print(f"WARNING: failed to write time log: {e}")


if __name__ == '__main__':
    main()
