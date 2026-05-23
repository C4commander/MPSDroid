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

# ========== Basic Utilities ==========

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

# ========== Sensitive API Mapping and Sequence Parsing ==========

def load_sapi_mapping(sapi_path: Path) -> Tuple[Dict[int, str], Set[str]]:
    """
    Read the sensitive API list (one Java method signature per line), and construct:
    - id_to_api: 1-based index -> API name
    - sapi_set: set of sensitive API names
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
    Extract a file-level set of used sensitive APIs (deduplicated) from a .txt sequence file.
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

# ========== Cache ==========

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
    Read a .txt sequence file -> restore sensitive API names from IDs using the SAPI mapping -> deduplicate -> write to cache
    Returns a dictionary including path, sha, label, cache_path, n_apis, error
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

# ========== Compute Weights (Based on Cache) ==========

def compute_weights_from_cache(train_items: List[Tuple[Path, str]],
                               cache_dir: Path,
                               min_count: int,
                               alpha: int = 2,
                               verbose: bool = False):
    """
    Compute the 'file-level occurrence count' of sensitive APIs in malware/goodware based on cache files, and calculate entropy and weight.
    """
    from collections import defaultdict

    malware_count = defaultdict(int)
    benign_count = defaultdict(int)
    n_malware = 0
    n_benign = 0

    for fpath, label in tqdm(train_items, desc="Read training cache and count APIs", disable=not verbose):
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

# ========== Directory Scanning Tools ==========

def collect_txt_files(root_dir: Path) -> List[Path]:
    """
    Recursively collect all .txt files from the given directory.
    Used for train-dir / test-dir; these directories contain malware/benign subdirectories.
    """
    if not root_dir.exists():
        return []
    return list(root_dir.rglob("*.txt"))

# ========== Main Procedure (Train/Test Directories) ==========

def main():
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    parser = argparse.ArgumentParser(
        description='Sensitive API entropy and weight statistics (train/test are two directories with malware/benign subdirectories)'
    )
    parser.add_argument('--train-dir', type=str, default="/mnt/data2/wb2024/Methodology/MPSDroid/test-train/Sequences/train",
                        help='Training set root directory (should include malware and benign subdirs containing .txt sequence files)')
    parser.add_argument('--test-dir', type=str, default="/mnt/data2/wb2024/Methodology/MPSDroid/test-train/Sequences/test",
                        help='Test set root directory (should include malware and benign subdirs containing .txt sequence files)')

    parser.add_argument('--malware-key', type=str, default='malware', help='Malware directory keyword')
    parser.add_argument('--benign-key', type=str, default='benign', help='Benign directory keyword')
    parser.add_argument('--output-dir', type=str, default="./statistic",
                        help='Output root directory path')

    parser.add_argument('--sapi', type=str, default="./APIChecker_PScout.txt",
                        help='Sensitive API list path (should match the list used to generate the .txt sequences)')

    parser.add_argument('--min-count', type=int, default=2,
                        help='Minimum API occurrence count to keep (based on training set statistics)')
    parser.add_argument('--verbose', action='store_true', default=True,
                        help='Show detailed progress')

    parser.add_argument('--workers', type=int,
                        default=max(8, min(120, (os.cpu_count() or 64))),
                        help='Number of preprocessing parallel threads (I/O intensive, thread pool is preferred)')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='API cache directory (default is api_cache under output-dir)')
    parser.add_argument('--reuse-cache', action='store_true', default=True,
                        help='Reuse existing cache if available, skip .txt parsing')

    args = parser.parse_args()

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir) if args.cache_dir else (out_root / "api_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # 0) Load sensitive API mapping
    sapi_path = Path(args.sapi)
    if not sapi_path.exists():
        print(f"Sensitive API list not found: {sapi_path}")
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

    # 1) Scan train/test directories and label them
    train_root = Path(args.train_dir)
    test_root = Path(args.test_dir)

    train_files = collect_txt_files(train_root)
    test_files = collect_txt_files(test_root)

    if not train_files:
        print(f"No .txt files found under train directory, check --train-dir: {train_root}")
        return
    if not test_files:
        print(f"No .txt files found under test directory, check --test-dir: {test_root}")
        return

    train_items = []
    for f in train_files:
        label = infer_label(f, args.malware_key, args.benign_key)
        if label in ('malware', 'benign'):
            train_items.append((f, label))
    if not train_items:
        print("No .txt sequence files with malware/benign label found in the training set, check directories and keywords.")
        return

    # Save lists (for further inspection)
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

    # 2) Preprocessing: cache all APIs for both train and test sets
    all_files = list({*train_files, *test_files})
    print(f"[INFO] Preprocessing sequence .txt -> caching API list to: {cache_dir}, threads: {args.workers}, reuse_cache={args.reuse_cache}")

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
                        desc="Preprocessing sequence .txt", disable=not args.verbose):
            info = fut.result()
            index_rows.append(info)
            if info.get("error"):
                n_errors += 1
                print(f"[ERROR] Preprocessing failed: {info['path']}\n{info['error']}")

    index_df = pd.DataFrame(index_rows)
    index_csv = out_root / "api_cache_index.csv"
    index_df.to_csv(index_csv, index=False, encoding="utf-8-sig")
    print(f"[INFO] Saved cache index: {index_csv}, failed: {n_errors}")

    # 3) Compute weights based on training set cache
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

    print(f"[INFO] Saved weights: {weights_csv}  Stats: n_train={meta['n_train']} "
          f"n_test={meta['n_test']} n_api_after_filter={meta['n_api_after_filter']}")

    # ---- End time & Logging ----
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