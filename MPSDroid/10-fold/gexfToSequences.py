import os
import argparse
import networkx as nx
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import ceil
from utils import dalvik_to_java_method

# 尝试导入 lxml 的异常类型（NetworkX 读 GEXF 通常使用 lxml）
try:
    import lxml.etree as LET
except Exception:
    LET = None

import xml.etree.ElementTree as ET  # 兜底解析异常类型（部分环境会抛 ET.ParseError）
import time  # 新增：用于时间统计


def parse_args():
    p = argparse.ArgumentParser(
        description='从根目录递归提取 GEXF 图中的方法序列：先将 Dalvik 节点转换为 Java 方法名，再按提供的敏感 API 进行缩点（删除非敏感节点并连接其邻居）；随后以线性路径覆盖的方式（O(V+E)）输出序列，避免指数级的全路径枚举。输出保留原相对目录结构。'
    )
    p.add_argument(
        '-f', '--from-root', dest='gexf_root',
        default="/mnt/data2/wb2024/Methodology/MyWay/data/Graph-md",
        help='包含 GEXF 文件的输入根目录（递归遍历子目录）'
    )
    p.add_argument(
        '-o', '--output_root',
        default="/mnt/data2/wb2024/Methodology/MyWay/data/Sequences-md",
        help='输出根目录（会保留输入文件的相对子目录结构）'
    )
    p.add_argument(
        '-s', '--sapi',
        #default="/mnt/data2/wb2024/Methodology/MyWay/analyze/删除没出现的结果API.txt",
        default="/mnt/data2/wb2024/Data/Sensitive_inf/APIChecker_dot.txt",
        help='敏感 API 列表（每行一个，已是 Java 方法名，如 android.telephony.SmsManager.sendTextMessage）。用于缩点与输出 ID 序列（1-based）。'
    )
    p.add_argument(
        '-w', '--workers', type=int, default=max(1, min((os.cpu_count() or 1), 120)),
        help='并行处理的进程数（默认：CPU 核心数）。'
    )
    p.add_argument(
        '--force', action='store_true',
        help='忽略已有的 .txt 序列文件，强制重新生成。',
        default=True
    )
    return p.parse_args()


def load_sapi_map(sapi_path):
    ordered, seen = [], set()
    with open(sapi_path, 'r', encoding='utf-8') as f:
        for line in f:
            api = line.strip()
            if api and api not in seen:
                seen.add(api)
                ordered.append(api)
    # map to 1-based ids
    return {api: i + 1 for i, api in enumerate(ordered)}


def relabel_graph_nodes_to_java(CG: nx.Graph) -> nx.DiGraph:
    """
    将图中的 Dalvik 节点名转换为 Java 方法名。
    - 无法解析的保持原样（后续会在缩点中被删除并连接邻居）。
    - 合并重名节点与并行边（转为 DiGraph）。
    """
    mapping = {}
    for n in CG.nodes():
        name = str(n)
        java = dalvik_to_java_method(name)
        mapping[n] = java if java else name
    CG2 = nx.relabel_nodes(CG, mapping, copy=True)
    return nx.DiGraph(CG2)


def contract_to_sensitive(CG: nx.DiGraph, sensitive_set: set) -> nx.DiGraph:
    """
    内存友好的缩点实现：
    - 仅保留敏感节点；
    - 若存在 u -> ... -> v 的路径（中间只经过非敏感节点），则在结果图中添加边 u -> v；
    - 不为全图构建 pred/succ 副本，不复制大规模邻接结构；
    - 对每个敏感源节点执行一次非递归前向遍历，仅使用局部 visited 集合与栈，显著降低峰值内存。
    """
    if not sensitive_set:
        return CG

    H = nx.DiGraph()

    # 仅加入原图中出现过的敏感节点，保留孤立敏感点
    present_sensitive = [n for n in CG.nodes if n in sensitive_set]
    H.add_nodes_from(present_sensitive)

    # 为加速敏感判定，使用局部引用
    is_sensitive = sensitive_set.__contains__

    for src in present_sensitive:
        # 局部 visited 仅针对 src 的遍历，避免全图级别的 visited 占用
        visited = set()
        stack = list(CG.successors(src))

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)

            if is_sensitive(cur):
                # 命中敏感出口，连边并停止从该点继续扩展（保证中间不穿越其他敏感节点）
                if cur != src:
                    H.add_edge(src, cur)
                continue

            # 非敏感节点，继续向后扩展
            # 使用邻接视图生成器，避免复制成 list
            for nxt in CG.successors(cur):
                if nxt not in visited:
                    stack.append(nxt)

    # 清理可能出现的自环（理论上已避免）
    H.remove_edges_from(nx.selfloop_edges(H))
    return H


def _kmp_build(pattern):
    lps = [0] * len(pattern)
    length = 0
    i = 1
    while i < len(pattern):
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1
    return lps


def _kmp_contains(text, pattern):
    if not pattern:
        return True
    if len(pattern) > len(text):
        return False
    lps = _kmp_build(pattern)
    i = j = 0
    while i < len(text):
        if text[i] == pattern[j]:
            i += 1
            j += 1
            if j == len(pattern):
                return True
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return False


def prune_subsequences_tuples(seq_tuples):
    seq_tuples_sorted = sorted(seq_tuples, key=len, reverse=True)
    kept = []
    for cand in seq_tuples_sorted:
        is_sub = False
        for big in kept:
            if len(big) < len(cand):
                continue
            if _kmp_contains(big, cand):
                is_sub = True
                break
        if not is_sub:
            kept.append(cand)
    return kept


def decompose_paths_linear(CG: nx.DiGraph):
    """
    以 O(V+E) 的复杂度将图分解为一组线性路径，覆盖所有边：
    - 起点选择：入度!=1 或 出度!=1 的节点，及入度为0的节点；
    - 对每个起点，沿唯一后继链条延伸；若当前节点出度>1，则以每条未访问边为分支启动，但不进行组合枚举，
      每条边仅被走一次（用 visited_edges 标记）。
    - 对剩余未覆盖的边（可能在环中或孤立分支），从该边的源启动同样的线性延伸，直到遇到已访问边或无法继续。
    返回节点序列列表（每个序列是节点列表）。
    """
    if CG.number_of_nodes() == 0:
        return []

    # 使用稳定的邻接快照，避免迭代视图在过程中变化带来的开销
    succs = {n: list(CG.successors(n)) for n in CG.nodes}
    preds = {n: list(CG.predecessors(n)) for n in CG.nodes}

    visited_edges = set()  # 存储 (u, v)
    paths = []

    def extend_from(u, first_v=None):
        """
        从节点 u 开始延伸线性路径。
        - 若 first_v 给出，则从边 (u, first_v) 起步；否则从 u 的每条后继边中选择未访问的边依次起步。
        - 仅沿 out_degree==1 的链条进行顺延；遇到分叉时停止本次延伸（分叉由外层调用或稍后补齐）。
        返回构成的一条路径（节点序列）。若无法形成长度>=1的路径（即无边可走），返回包含单节点的路径。
        """
        path = [u]
        # 确定起始后继
        next_candidates = succs[u] if first_v is None else [first_v]
        v_choice = None
        for v in next_candidates:
            e = (u, v)
            if e not in visited_edges:
                visited_edges.add(e)
                v_choice = v
                break

        if v_choice is None:
            # 无可用后继边，返回单节点路径
            return path

        # 进入线性延伸
        cur = v_choice
        path.append(cur)
        while True:
            outs = succs.get(cur, [])
            # 只在 out_degree==1 时继续线性延伸
            if len(outs) != 1:
                break
            nxt = outs[0]
            e = (cur, nxt)
            if e in visited_edges:
                break
            visited_edges.add(e)
            path.append(nxt)
            cur = nxt
        return path

    # 1) 首批起点：入度!=1 或 出度!=1 的节点，以及入度为0的节点
    candidate_starts = set()
    for n in CG.nodes:
        indeg = len(preds[n])
        outdeg = len(succs[n])
        if indeg == 0 or indeg != 1 or outdeg != 1:
            candidate_starts.add(n)

    # 先从“真正起点”（入度为 0）出发，优先覆盖 DAG 的源
    zero_in_nodes = [n for n in CG.nodes if len(preds[n]) == 0]
    for s in zero_in_nodes:
        # 对每条未访问边启动一次线性延伸，确保覆盖所有边
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # 再从其他非1-1结构的节点出发
    for s in candidate_starts:
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # 2) 补齐剩余未访问的边（可能来自环或未涵盖的分支）
    for u in CG.nodes:
        for v in succs[u]:
            if (u, v) not in visited_edges:
                path = extend_from(u, v)
                if path:
                    paths.append(path)

    # 去除仅包含单节点的“路径”（如果节点没有任何边），但保留孤立节点作为单元素路径
    # 为保持原行为，如果图有孤立节点，我们也输出该节点作为一条序列。
    isolated_nodes = [n for n in CG.nodes if len(succs[n]) == 0 and len(preds[n]) == 0]
    paths.extend([[n] for n in isolated_nodes])

    # 合并可能重复的单节点路径
    unique_paths = []
    seen = set()
    for p in paths:
        tup = tuple(p)
        if tup not in seen:
            seen.add(tup)
            unique_paths.append(p)
    return unique_paths


def write_sequences(CG, txt_path, sapi_id_map=None):
    """
    基于线性路径覆盖的序列写出（替代指数级 DFS）。
    """
    if CG.number_of_nodes() == 0:
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        open(txt_path, "w", encoding="utf-8").close()
        print(f"{txt_path}: duplicates=0 total=0 unique=0")
        return

    raw_paths = decompose_paths_linear(CG)

    if sapi_id_map:
        seq_id_tuples = []
        for seq in raw_paths:
            ids = [sapi_id_map[a] for a in seq if a in sapi_id_map]
            if ids:
                seq_id_tuples.append(tuple(ids))
        sequences_all = seq_id_tuples
    else:
        sequences_all = [tuple(seq) for seq in raw_paths]

    total = len(sequences_all)
    unique_set = set(sequences_all)
    duplicates = total - len(unique_set)
    sequences = list(unique_set)

    if sequences:
        sequences = prune_subsequences_tuples(sequences)

    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    out_lines = [" ".join(map(str, seq)) for seq in sequences]
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
        if out_lines:
            f.write("\n")

    print(f"{txt_path}: duplicates={duplicates} total={total} unique={len(unique_set)}")


def compute_txt_path(gexf_path, gexf_root, output_root):
    rel_path = os.path.relpath(gexf_path, gexf_root)
    if rel_path.endswith(".gexf"):
        return os.path.join(output_root, rel_path[:-5] + ".txt")
    else:
        return os.path.join(output_root, rel_path + ".txt")


def _is_unclosed_xml_error(exc: Exception) -> bool:
    """
    判断异常是否为 XML 未闭合（unclosed token）问题。
    同时兼容 lxml 和标准库 ElementTree 的异常类型。
    """
    msg = str(exc).lower()
    if "unclosed token" in msg:
        return True
    if isinstance(exc, ET.ParseError) and "unclosed token" in msg:
        return True
    if LET is not None and isinstance(exc, LET.XMLSyntaxError) and "unclosed token" in msg:
        return True
    return False


def gexf_to_sequences(gexf_path, gexf_root, output_root, sapi_id_map=None, force=False):
    txt_path = compute_txt_path(gexf_path, gexf_root, output_root)

    # 若输出文件已存在且未强制重建则跳过
    if not force and os.path.exists(txt_path):
        print(f"{txt_path}: exists - skip")
        return

    try:
        G = nx.read_gexf(gexf_path)
        G = relabel_graph_nodes_to_java(G)
        if sapi_id_map:
            sensitive_set = set(sapi_id_map.keys())
            G = contract_to_sensitive(G, sensitive_set)
        # 使用线性路径覆盖写出序列
        write_sequences(G, txt_path, sapi_id_map=sapi_id_map)
    except Exception as e:
        # 自动删除“未闭合”的 GEXF 文件
        if _is_unclosed_xml_error(e):
            try:
                os.remove(gexf_path)
                print(f"{txt_path}: failed parsing GEXF (unclosed token) - deleted source {gexf_path}")
            except OSError as oe:
                print(f"{txt_path}: failed parsing GEXF (unclosed token) - could not delete {gexf_path}: {oe}")
        else:
            print(f"{txt_path}: failed - {e}")


def find_all_gexf_files(root):
    gexf_files = []
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.endswith(".gexf"):
                gexf_files.append(os.path.join(dirpath, filename))
    return gexf_files


def main():
    # 记录开始时间（字符串+时间戳）
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    args = parse_args()
    gexf_root = args.gexf_root
    output_root = args.output_root
    sapi_id_map = load_sapi_map(args.sapi) if args.sapi else None

    # 确保输出根目录存在（方便后面写 run_time.log）
    os.makedirs(output_root, exist_ok=True)

    gexf_files = find_all_gexf_files(gexf_root)
    if not gexf_files:
        print("No .gexf files found.")

        # 即使没有文件也记录一次运行时间
        end_ts = time.time()
        end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
        elapsed = end_ts - start_ts
        print(f"Elapsed: {elapsed:.2f}s")

        log_path = os.path.join(output_root, "run_time.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Start: {start_str} | End: {end_str} | "
                    f"Elapsed: {elapsed:.2f}s | GEXF root: {gexf_root}\n"
                )
        except Exception as e:
            print(f"WARNING: failed to write time log: {e}")
        return

    # 预过滤：去除已有结果的文件（除非 force）
    if not args.force:
        original_count = len(gexf_files)
        gexf_files = [
            p for p in gexf_files
            if not os.path.exists(compute_txt_path(p, gexf_root, output_root))
        ]
        skipped = original_count - len(gexf_files)
        if skipped:
            print(f"Skipped {skipped} already processed file(s).")
        if not gexf_files:
            print("All sequences already exist; nothing to do.")

            # 这里同样记录一次运行时间
            end_ts = time.time()
            end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
            elapsed = end_ts - start_ts
            print(f"Elapsed: {elapsed:.2f}s")

            log_path = os.path.join(output_root, "run_time.log")
            try:
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(
                        f"Start: {start_str} | End: {end_str} | "
                        f"Elapsed: {elapsed:.2f}s | GEXF root: {gexf_root}\n"
                    )
            except Exception as e:
                print(f"WARNING: failed to write time log: {e}")
            return

    workers = max(1, args.workers)
    chunksize = max(1, ceil(len(gexf_files) / (workers * 8)))

    with ProcessPool(processes=workers) as pool:
        for _ in pool.imap_unordered(
            partial(
                gexf_to_sequences,
                gexf_root=gexf_root,
                output_root=output_root,
                sapi_id_map=sapi_id_map,
                force=args.force,
            ),
            gexf_files,
            chunksize=chunksize
        ):
            pass

    # 结束时间与耗时记录
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")

    # 写入日志（追加）
    log_path = os.path.join(output_root, "run_time.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"Start: {start_str} | End: {end_str} | "
                f"Elapsed: {elapsed:.2f}s | GEXF root: {gexf_root}\n"
            )
    except Exception as e:
        print(f"WARNING: failed to write time log: {e}")


if __name__ == '__main__':
    main()