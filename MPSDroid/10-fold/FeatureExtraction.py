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

# 对java格式的API做了适配
# 例如配对能够.txt的 "android.telephony.SmsManager.sendDataMessage"格式
# 对全图所有的度做中心性统计，筛选处在SAPI中的作为特征，拼接 sha256 和 标签 输出为.csv文件
# 目录获取改为：递归遍历 dataset_path 下所有子目录，找到名为 benign 和 malware 的目录后再处理
# 修复点：
# 1) 递归收集 benign/malware 下所有层级的 .gexf/.gexf.gz 文件（大小写不敏感）
# 2) 单文件异常不影响整体（map 内部捕获）
# 3) 正确剥离双扩展名（.gexf.gz）获取 sha256
# 4) 兼容读取 gzip 压缩的 GEXF

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
    从文件名剥离 .gexf / .gexf.gz（大小写不敏感），返回主干名（通常是 sha256）
    """
    name = os.path.basename(path)
    lower = name.lower()
    if lower.endswith('.gexf.gz'):
        return name[:-(len('.gexf.gz'))]
    if lower.endswith('.gexf'):
        return name[:-(len('.gexf'))]
    # 回退：只去掉最后一个扩展名
    return os.path.splitext(name)[0]

def callgraph_extraction(file):
    """
    兼容读取 .gexf 和 .gexf.gz
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
    # 建立一个从API格式到中心性值的映射
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

# --------- 非递归 Katz 实现（幂迭代），避免递归限制 ----------
def _safe_alpha_for_katz(G, default_alpha=0.01):
    """
    给定图，返回一个较安全的 alpha，用于确保 Katz 中心性收敛。
    使用度的上界来估计谱半径的上界：alpha < 1 / lambda_max <= 1 / d_max
    """
    try:
        if G.is_directed():
            degrees = [d for _, d in G.out_degree()]
        else:
            degrees = [d for _, d in G.degree()]
        dmax = max(degrees) if degrees else 1
        # 0.9/dmax 留一点余量，和默认值取较小值更稳健
        return float(min(default_alpha, 0.9 / max(1.0, dmax)))
    except Exception:
        return float(default_alpha)

def _katz_centrality_power(G, alpha=None, beta=1.0, max_iter=10000, tol=1e-6, use_weights=False):
    """
    使用幂迭代求解 Katz 中心性：x_{t+1} = alpha * A * x_t + beta
    - 非递归实现，避免递归深度限制
    - 若安装 SciPy 则优先使用稀疏矩阵计算，否则回退到 NumPy dense
    """
    nodes = list(G.nodes())
    n = len(nodes)
    if n == 0:
        return {}

    if alpha is None:
        alpha = _safe_alpha_for_katz(G, default_alpha=0.01)

    weight = 'weight' if use_weights else None

    # 尝试使用 SciPy 稀疏矩阵
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

    # 如果没有 SciPy 或构建失败，则使用 NumPy dense
    if A is None:
        A = nx.to_numpy_array(G, nodelist=nodes, dtype=float, weight=weight)

    x = np.ones(n, dtype=float)
    b = np.ones(n, dtype=float) * beta

    # 迭代
    for _ in range(max_iter):
        if use_scipy and hasattr(A, "dot"):
            x_new = alpha * (A.dot(x)) + b
        else:
            x_new = alpha * (A @ x) + b
        # 使用 L1 范数判断收敛
        if np.linalg.norm(x_new - x, 1) < tol * n:
            x = x_new
            break
        x = x_new

    # 可选：归一化，避免数值过大
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
        # 万一不收敛，返回0向量
        node_centrality = {node: 0 for node in CG.nodes()}
    return get_vector(node_centrality, sensitive_apis)

def authority_centrality_feature(CG, sensitive_apis):
    # 使用 hits 算法获取 authority 分数
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
    递归遍历 root_dir，收集所有名为 'benign' 和 'malware' 的目录路径。
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
    收集 root_dir 下所有 .gexf / .gexf.gz 文件（大小写不敏感）
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
    安全地计算单个文件的特征，任何异常都只影响该文件，不影响全局。
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

    # 递归查找所有名为 benign 和 malware 的目录
    benign_dirs, malware_dirs = find_benign_malware_dirs(dataset_path)

    if not benign_dirs and not malware_dirs:
        print(f"Warning: No 'benign' or 'malware' directories found under: {dataset_path}", flush=True)

    # 递归收集所有 gexf 文件（包含 .gexf.gz，大小写不敏感）
    apps_b = []
    for bdir in benign_dirs:
        apps_b.extend(collect_gexf_files(bdir, recursive=True))
    apps_m = []
    for mdir in malware_dirs:
        apps_m.extend(collect_gexf_files(mdir, recursive=True))

    # 去重并排序，保证稳定性
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

    # 过滤掉失败的 None
    results_b = [r for r in results_b if r is not None]
    results_m = [r for r in results_m if r is not None]

    Vectors.extend(results_b)
    Labels.extend([0 for _ in range(len(results_b))])

    Vectors.extend(results_m)
    Labels.extend([1 for _ in range(len(results_m))])

    # 提示是否有文件被跳过
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