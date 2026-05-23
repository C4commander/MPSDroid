import os
import argparse
import networkx as nx
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import ceil
from utils import dalvik_to_java_method

# Try importing lxml's exception type (NetworkX reads GEXF usually using lxml)
try:
    import lxml.etree as LET
except Exception:
    LET = None

import xml.etree.ElementTree as ET  # Fallback for parser exception type (some environments will throw ET.ParseError)
import time  # Used for time statistics


def parse_args():
    p = argparse.ArgumentParser(
        description='Recursively extract method sequences from GEXF graphs under a root dir: convert Dalvik nodes to Java method names, then contract to the provided sensitive APIs (remove non-sensitive nodes and connect their neighbors); then output sequences with linear path cover (O(V+E)), avoiding exponential all-path enumeration. Output retains original relative directory structure.'
    )
    p.add_argument(
        '-f', '--from-root', dest='gexf_root',
        default="/mnt/data5/Temp/",
        help='Input root directory containing GEXF files (recursively traverses subdirectories)'
    )
    p.add_argument(
        '-o', '--output_root',
        default="./Sequences",
        help='Output root directory (keeps input files\' relative subdirectory structure)'
    )
    p.add_argument(
        '-s', '--sapi',
        default="./APIChecker_dot.txt",
        help='List of sensitive APIs (one per line, already in Java method format, e.g., android.telephony.SmsManager.sendTextMessage). Used for contraction and outputting ID sequence (1-based).'
    )
    p.add_argument(
        '-w', '--workers', type=int, default=max(1, min((os.cpu_count() or 1), 120)),
        help='Number of parallel processes (default: number of CPU cores).'
    )
    p.add_argument(
        '--force', action='store_true',
        help='Ignore existing .txt sequence files and force regeneration.',
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
    Convert Dalvik node names in the graph to Java method names.
    - Unparsable nodes are kept unchanged (later removed during contraction).
    - Merge nodes with the same name and parallel edges (convert to DiGraph).
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
    Memory-friendly contraction implementation:
    - Only retain sensitive nodes.
    - If there is a path u -> ... -> v (only passing non-sensitive nodes in between), add edge u -> v to the contracted graph.
    - Do not build pred/succ copies for the whole graph, do not copy large adjacency structures;
    - For each sensitive source, do a non-recursive forward traversal, using only a local visited set and stack, thus lowering peak memory.
    """
    if not sensitive_set:
        return CG

    H = nx.DiGraph()

    # Only add sensitive nodes that exist in the original graph, retain isolated sensitive nodes
    present_sensitive = [n for n in CG.nodes if n in sensitive_set]
    H.add_nodes_from(present_sensitive)

    # Use local reference for faster membership checking
    is_sensitive = sensitive_set.__contains__

    for src in present_sensitive:
        # Local visited is per source, to avoid global visited set usage
        visited = set()
        stack = list(CG.successors(src))

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)

            if is_sensitive(cur):
                # Found sensitive target, add edge and do not continue from here (avoids traversing other sensitive nodes in between)
                if cur != src:
                    H.add_edge(src, cur)
                continue

            # Non-sensitive node, keep going
            # Use adjacency view generator, not list copy
            for nxt in CG.successors(cur):
                if nxt not in visited:
                    stack.append(nxt)

    # Remove possible self loops (theoretically should already be avoided)
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
    Decompose the graph into a set of linear paths covering all edges in O(V+E):
    - Start nodes: indegree != 1 or outdegree != 1, and nodes with indegree 0;
    - From each start, follow the unique outgoing chain; if outdegree > 1, branch for each unvisited edge but do not enumerate combinations,
      each edge is traversed only once (marked by visited_edges).
    - For all unexplored edges (possibly in cycles or isolated branches), start from the source node and linearly extend until no more.
    Returns a list of node sequences (each sequence a node list).
    """
    if CG.number_of_nodes() == 0:
        return []

    # Use stable adjacency snapshots to avoid changes during iteration
    succs = {n: list(CG.successors(n)) for n in CG.nodes}
    preds = {n: list(CG.predecessors(n)) for n in CG.nodes}

    visited_edges = set()  # holds (u, v)
    paths = []

    def extend_from(u, first_v=None):
        """
        Start extending a linear path from node u.
        - If first_v is provided, start from edge (u, first_v); otherwise, for each outgoing edge, start new path.
        - Only continue along out_degree==1 chains; if a branch, stop extension (branches are started anew later).
        Returns the formed path (node sequence). If no edges to walk, returns [u].
        """
        path = [u]
        # Determine the starting successor
        next_candidates = succs[u] if first_v is None else [first_v]
        v_choice = None
        for v in next_candidates:
            e = (u, v)
            if e not in visited_edges:
                visited_edges.add(e)
                v_choice = v
                break

        if v_choice is None:
            # No successor edge, return single node path
            return path

        # Continue extension
        cur = v_choice
        path.append(cur)
        while True:
            outs = succs.get(cur, [])
            # Only continue linearly if outdegree==1
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

    # (1) Initial start nodes: indegree != 1 or outdegree != 1, plus indegree 0 nodes
    candidate_starts = set()
    for n in CG.nodes:
        indeg = len(preds[n])
        outdeg = len(succs[n])
        if indeg == 0 or indeg != 1 or outdeg != 1:
            candidate_starts.add(n)

    # First process true starts (indegree 0), covering DAG sources
    zero_in_nodes = [n for n in CG.nodes if len(preds[n]) == 0]
    for s in zero_in_nodes:
        # For each unvisited outgoing edge, start linear extension to cover all edges
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # Then from all other non 1-1 structured nodes
    for s in candidate_starts:
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # (2) Cover remaining unvisited edges (possibly in cycles or branches)
    for u in CG.nodes:
        for v in succs[u]:
            if (u, v) not in visited_edges:
                path = extend_from(u, v)
                if path:
                    paths.append(path)

    # Remove singleton "paths" (if a node has no edges), but keep isolated nodes as single-element paths
    # To preserve behavior, always emit isolated nodes as single-node sequences.
    isolated_nodes = [n for n in CG.nodes if len(succs[n]) == 0 and len(preds[n]) == 0]
    paths.extend([[n] for n in isolated_nodes])

    # Merge possibly duplicate single-node paths
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
    Sequence output based on linear path cover (replaces exponential DFS).
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
    Determine if the exception is an XML unclosed token problem.
    Compatible with both lxml and stdlib ElementTree exception types.
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

    # Skip if output file already exists and not forcing regeneration
    if not force and os.path.exists(txt_path):
        print(f"{txt_path}: exists - skip")
        return

    try:
        G = nx.read_gexf(gexf_path)
        G = relabel_graph_nodes_to_java(G)
        if sapi_id_map:
            sensitive_set = set(sapi_id_map.keys())
            G = contract_to_sensitive(G, sensitive_set)
        # Output sequences using linear path cover
        write_sequences(G, txt_path, sapi_id_map=sapi_id_map)
    except Exception as e:
        # Automatically delete GEXF files with unclosed XML
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
    # Record start time (string and timestamp)
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    args = parse_args()
    gexf_root = args.gexf_root
    output_root = args.output_root
    sapi_id_map = load_sapi_map(args.sapi) if args.sapi else None

    # Ensure output root directory exists (for run_time.log)
    os.makedirs(output_root, exist_ok=True)

    gexf_files = find_all_gexf_files(gexf_root)
    if not gexf_files:
        print("No .gexf files found.")

        # Record run time even if no files
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

    # Prefilter: remove files that already have results (unless forcing)
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

            # Also record runtime here
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

    # End time and cost recording
    end_ts = time.time()
    end_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_ts))
    elapsed = end_ts - start_ts
    print(f"Elapsed: {elapsed:.2f}s")

    # Write log (append)
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