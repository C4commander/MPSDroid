import os
import argparse
import networkx as nx
from collections import defaultdict
from functools import partial
from multiprocessing import Pool as ProcessPool
from math import ceil
from utils import dalvik_to_java_method

# Try to import lxml's exception types (NetworkX GEXF reading usually uses lxml)
try:
    import lxml.etree as LET
except Exception:
    LET = None

import xml.etree.ElementTree as ET  # Fallback parser exception type (some environments will throw ET.ParseError)
import time  # Used for time statistics


def parse_args():
    p = argparse.ArgumentParser(
        description='Recursively extract method sequences from GEXF graphs: first convert Dalvik nodes to Java method names, then perform contraction according to the provided sensitive APIs (delete non-sensitive nodes and connect their neighbors); finally, output sequences using a linear path cover (O(V+E)) to avoid exponential enumeration of all paths. Output keeps the original relative directory structure.'
    )
    p.add_argument(
        '-f', '--from-root', dest='gexf_root',
        default="/mnt/data2/wb2024/Methodology/MyWay/data/Graph-md",
        help='Input root directory containing GEXF files (recursively find subdirectories)'
    )
    p.add_argument(
        '-o', '--output_root',
        default="/mnt/data2/wb2024/Methodology/MyWay/data/Sequences-md",
        help='Output root directory (will keep the input file\'s relative directory structure)'
    )
    p.add_argument(
        '-s', '--sapi',
        default="/mnt/data2/wb2024/Data/Sensitive_inf/APIChecker_PScout.txt",
        help='Sensitive API list (one per line, already Java method names, e.g., android.telephony.SmsManager.sendTextMessage). Used for contraction and output ID sequences (1-based).'
    )
    p.add_argument(
        '-w', '--workers', type=int, default=min((os.cpu_count() or 1), 120),
        help='Number of parallel processes (default: number of CPU cores).'
    )
    p.add_argument(
        '--force', action='store_true',
        help='Ignore existing .txt sequence files, force regenerate.',
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
    # Map to 1-based ids
    return {api: i + 1 for i, api in enumerate(ordered)}


def relabel_graph_nodes_to_java(CG: nx.Graph) -> nx.DiGraph:
    """
    Convert Dalvik node names in the graph to Java method names.
    - Unparsable nodes stay as-is (later they will be removed during contraction).
    - Merge duplicate node names and parallel edges (convert to DiGraph).
    """
    mapping = {}
    for n in CG.nodes():
        name = str(n)
        java = dalvik_to_java_method(name)
        mapping[n] = java if java else name
    CG2 = nx.relabel_nodes(CG, mapping, copy=True)
    return nx.DiGraph(CG2)


def _xml_local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _mapped_java_name(node_id: str, node_label: str = "") -> str:
    raw_id = str(node_id or "").strip()
    raw_label = str(node_label or "").strip()

    java = dalvik_to_java_method(raw_id)
    if java:
        return java

    if raw_label and raw_label != raw_id:
        java = dalvik_to_java_method(raw_label)
        if java:
            return java

    return raw_id or raw_label


def read_gexf_as_java_digraph(gexf_path: str):
    """
    Stream-parse a GEXF file and directly build the relabelled DiGraph.

    This avoids the expensive path of:
    1. fully loading the original graph with nx.read_gexf(...)
    2. copying it again with nx.relabel_nodes(..., copy=True)
    """
    graph = nx.DiGraph()
    id_to_name = {}
    unresolved_edges = []

    context = ET.iterparse(gexf_path, events=("end",))
    for _, elem in context:
        tag = _xml_local_name(elem.tag)

        if tag == "node":
            node_id = elem.get("id", "")
            node_label = elem.get("label", "")
            mapped = _mapped_java_name(node_id, node_label)
            id_to_name[node_id] = mapped
            graph.add_node(mapped)
        elif tag == "edge":
            source = elem.get("source", "")
            target = elem.get("target", "")
            src_name = id_to_name.get(source)
            dst_name = id_to_name.get(target)
            if src_name is not None and dst_name is not None:
                graph.add_edge(src_name, dst_name)
            else:
                unresolved_edges.append((source, target))

        elem.clear()

    if unresolved_edges:
        for source, target in unresolved_edges:
            src_name = id_to_name.get(source)
            dst_name = id_to_name.get(target)
            if src_name is not None and dst_name is not None:
                graph.add_edge(src_name, dst_name)

    return graph


def contract_to_sensitive(CG: nx.DiGraph, sensitive_set: set) -> nx.DiGraph:
    """
    Memory-friendly contraction:
    - Retain only sensitive nodes;
    - If there is a path u -> ... -> v (only passing non-sensitive nodes), add edge u -> v in the result graph;
    - Do not build pred/succ copies for whole graph, do not copy a large adjacency structure;
    - For each sensitive source node, perform a non-recursive forward traversal using only a local visited set and stack, to significantly reduce peak memory.
    """
    if not sensitive_set:
        return CG

    H = nx.DiGraph()

    # Only add sensitive nodes appearing in the original graph; retain isolated sensitive nodes
    present_sensitive = [n for n in CG.nodes if n in sensitive_set]
    H.add_nodes_from(present_sensitive)

    # Use a local reference to speed up sensitivity checking
    is_sensitive = sensitive_set.__contains__

    for src in present_sensitive:
        # Local visited for each src traversal, avoids global-level visited overhead
        visited = set()
        stack = list(CG.successors(src))

        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)

            if is_sensitive(cur):
                # Found a sensitive target, connect edge and do not continue from here (ensure no other sensitive node is traversed in between)
                if cur != src:
                    H.add_edge(src, cur)
                continue

            # Non-sensitive node, keep extending forward
            # Use adjacency view generator, avoid copying to list
            for nxt in CG.successors(cur):
                if nxt not in visited:
                    stack.append(nxt)

    # Remove possible self-loops (should already be avoided)
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


def _kmp_contains_with_lps(text, pattern, lps):
    if not pattern:
        return True
    if len(pattern) > len(text):
        return False
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
    kept_by_token = defaultdict(list)
    for cand in seq_tuples_sorted:
        is_sub = False
        lps = _kmp_build(cand)
        candidate_indexes = kept_by_token.get(cand[0], []) if cand else range(len(kept))
        seen_indexes = set()
        for idx in candidate_indexes:
            if idx in seen_indexes:
                continue
            seen_indexes.add(idx)
            big = kept[idx]
            if len(big) < len(cand):
                continue
            if _kmp_contains_with_lps(big, cand, lps):
                is_sub = True
                break
        if not is_sub:
            idx = len(kept)
            kept.append(cand)
            for token in set(cand):
                kept_by_token[token].append(idx)
    return kept


def decompose_paths_linear(CG: nx.DiGraph):
    """
    Decompose the graph into O(V+E) linear paths covering all edges:
    - Start nodes: nodes with indegree != 1 or outdegree != 1, and nodes with indegree 0;
    - For each start node, extend along the unique successor chain; if the current node has outdegree > 1, start a new branch for each unvisited edge, but do not enumerate combinations.
      Each edge is traversed only once (marked by visited_edges).
    - For remaining uncovered edges (possibly in cycles or isolated branches), start linear extension from the source until hitting a visited edge or being unable to continue.
    Return a list of node sequences (each is a node list).
    """
    if CG.number_of_nodes() == 0:
        return []

    # Use stable adjacency snapshots to avoid costs from view mutation during iteration
    succs = {n: list(CG.successors(n)) for n in CG.nodes}
    preds = {n: list(CG.predecessors(n)) for n in CG.nodes}

    visited_edges = set()  # Store (u, v)
    paths = []

    def extend_from(u, first_v=None):
        """
        Start extending a linear path from node u.
        - If first_v is given, start from edge (u, first_v); else choose unvisited outgoing edges from u;
        - Only continue along out-degree==1 chains; stop when branching (handled by higher-caller or filled in later).
        Return the path (node sequence). If no path with length >=1 can be formed (no edge), return single-node path.
        """
        path = [u]
        next_candidates = succs[u] if first_v is None else [first_v]
        v_choice = None
        for v in next_candidates:
            e = (u, v)
            if e not in visited_edges:
                visited_edges.add(e)
                v_choice = v
                break

        if v_choice is None:
            # No available successor edge, return single node path
            return path

        # Do linear extension
        cur = v_choice
        path.append(cur)
        while True:
            outs = succs.get(cur, [])
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

    # (1) Initial start nodes: indegree != 1 or outdegree != 1, and nodes with indegree 0
    candidate_starts = set()
    for n in CG.nodes:
        indeg = len(preds[n])
        outdeg = len(succs[n])
        if indeg == 0 or indeg != 1 or outdeg != 1:
            candidate_starts.add(n)

    # First cover DAG sources (nodes with indegree 0)
    zero_in_nodes = [n for n in CG.nodes if len(preds[n]) == 0]
    for s in zero_in_nodes:
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # Then start from other non 1-1 structure nodes
    for s in candidate_starts:
        for v in succs[s]:
            if (s, v) not in visited_edges:
                path = extend_from(s, v)
                if path:
                    paths.append(path)

    # (2) Fill in remaining uncovered edges (may come from cycles or uncovered branches)
    for u in CG.nodes:
        for v in succs[u]:
            if (u, v) not in visited_edges:
                path = extend_from(u, v)
                if path:
                    paths.append(path)

    # Remove singleton "paths" (where a node has no edges), but retain isolated nodes as single-node paths.
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
    Output sequences by linear path cover (replacing exponential DFS enumeration).
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
    Determine if an exception was an XML unclosed token error.
    Compatible with both lxml and ElementTree exception types.
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

    # Skip existing output file unless forced
    if not force and os.path.exists(txt_path):
        print(f"{txt_path}: exists - skip")
        return

    try:
        G = read_gexf_as_java_digraph(gexf_path)
        if sapi_id_map:
            sensitive_set = set(sapi_id_map.keys())
            G = contract_to_sensitive(G, sensitive_set)
        # Output sequences using linear path cover
        write_sequences(G, txt_path, sapi_id_map=sapi_id_map)
    except Exception as e:
        # Auto delete "unclosed" GEXF files
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
    # Record start time (string + timestamp)
    start_ts = time.time()
    start_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_ts))

    args = parse_args()
    gexf_root = args.gexf_root
    output_root = args.output_root
    sapi_id_map = load_sapi_map(args.sapi) if args.sapi else None

    # Ensure output root directory exists (for later writing run_time.log)
    os.makedirs(output_root, exist_ok=True)

    gexf_files = find_all_gexf_files(gexf_root)
    if not gexf_files:
        print("No .gexf files found.")

        # Even if no files, still record run time
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

    # Pre-filter: remove files with existing results (unless forcing)
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

            # Also record run time here
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

    # End time and cost record
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
